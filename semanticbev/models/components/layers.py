import torch
from torch import Tensor
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional, Dict, Tuple
import torch.nn.functional as F
import warnings


from fairscale.nn import checkpoint_wrapper


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation is None:
            activation = nn.GELU()

        width = int(planes * (base_width / 64.0))

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width)
        self.bn2 = norm_layer(width)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.activation(out)

        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class MultiUp(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


def mlp(num_channels: int):
    return Sequential(
        nn.LayerNorm(num_channels),
        nn.Linear(num_channels, num_channels),
        nn.GELU(),
        nn.Linear(num_channels, num_channels),
    )

class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Residual(nn.Module):
    def __init__(self, module: nn.Module, drop_path: float = 0.):
        super().__init__()
        self.module = module

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.drop_path(x) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, average_attn_weights=False,
                              key_padding_mask=pad_mask, attn_mask=attn_mask)[0]

class CrossAttention(nn.Module):
    # Simplified version of cross-attention module described in https://arxiv.org/abs/2103.03206.
    # Here, the embedding dimension is determined by the number of query channels (num_q_channels)
    # whereas in the paper it can be specified separately. This simplification allows re-use of the
    # torch.nn.MultiHeadAttention module whereas a full implementation of the paper would require a
    # custom multi-head attention implementation.
    def __init__(self, num_q_channels: int, num_kv_channels: int, num_heads: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_q_channels, num_kv_channels=num_kv_channels, num_heads=num_heads
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)

class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_q_channels=num_channels, num_kv_channels=num_channels, num_heads=num_heads
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


def cross_attention_layer(
        num_q_channels: int, num_kv_channels: int, num_heads: int, scale_init: float = 0., drop_path: float = 0.,
        activation_checkpoint: bool = False, residual_ca: bool = True
):

    if residual_ca:
        ca = Residual(
            LayerScaleModule(CrossAttention(num_q_channels, num_kv_channels, num_heads), num_q_channels, scale_init),
            drop_path
        )
    else:
        ca = CrossAttention(num_q_channels, num_kv_channels, num_heads)

    layer = Sequential(
        ca,
        Residual(LayerScaleModule(mlp(num_q_channels), num_q_channels, scale_init), drop_path),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer(
        num_channels: int, num_heads: int, scale_init: float = 0., drop_path: float = 0.,
        activation_checkpoint: bool = False
):
    layer = Sequential(
        Residual(LayerScaleModule(SelfAttention(num_channels, num_heads), num_channels, scale_init), drop_path),
        Residual(LayerScaleModule(mlp(num_channels), num_channels, scale_init), drop_path)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block(
        num_layers: int, num_channels: int, num_heads: int, scale_init: float = 0., drop_path: float = 0.,
        activation_checkpoint: bool = False
):
    layers = [
        self_attention_layer(num_channels, num_heads, scale_init, drop_path, activation_checkpoint)
        for _ in range(num_layers)
    ]

    return Sequential(*layers)




# https://github.com/rwightman/pytorch-image-models/blob/475ecdfa3d369b6d482287f2467ce101ce5c276c/timm/models/layers/drop.py
def drop_path_fn(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path_fn(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 0.):
        super().__init__()

        self.gamma = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        return self.gamma * x


class LayerScaleModule(nn.Module):
    def __init__(self, module: nn.Module, dim: int, init_value: float = 0.):
        super().__init__()

        self.module = module

        self.layer_scale = LayerScale(dim, init_value) if init_value > 0. else nn.Identity()

    def forward(self, *args, **kwargs):
        x = self.layer_scale(self.module(*args, **kwargs))
        return x
