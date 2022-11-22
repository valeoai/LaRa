import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import torch

class PyramidDecoder(nn.Module):
    def __init__(self, in_dim, out_dims, pred_dim, activation, scale_factors=None, conv_block='basic',
                 upsample_mode='nearest', blur=True, blur_at_end=True, pre_upconv=True):
        super().__init__()

        out_dims = list(out_dims)
        self.activation = {
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU()
        }[activation]

        self.pre_upconv = pre_upconv

        if scale_factors is None:
            self.scale_factors = [2]*len(out_dims)
        else:
            if len(scale_factors) != len(out_dims):
                raise ValueError(f"len(scale_factors) != len(out_dims)")
            self.scale_factors = list(scale_factors)

        available_upmodes = ['nearest', 'bilinear', 'pixelshuffle', 'res-pixelshuffle', 'deconv']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in {available_upmodes} | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode

        if conv_block == 'basic':
            conv_block = PreActBlock
        elif conv_block == 'bottleneck':
            conv_block = PreActBottleneck
        else:
            raise ValueError(f"conv_block must be in ['basic', 'bottleneck'] | conv_block={conv_block}")


        # decoder
        self.convs = nn.ModuleDict()
        for i in range(len(out_dims)):
            dims = [in_dim] + out_dims

            self.convs[f"resblock_{i}"] = conv_block(dims[i], dims[i], self.activation)
            self.convs[f"upconv_{i}"] = nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=True)

            if 'pixelshuffle' in self.upsample_mode:
                do_blur = blur and (i != len(out_dims)-1 or blur_at_end)
                self.convs[f"pixelshuffle_{i}"] = SubPixelUpsamplingBlock(dims[i+1] if pre_upconv else dims[i],
                                                                          upscale_factor=self.scale_factors[i],
                                                                          blur=do_blur)

            if self.upsample_mode == 'deconv':
                self.convs[f"deconv_{i}"] = nn.Sequential(
                    nn.BatchNorm2d(dims[i + 1] if pre_upconv else dims[i]),
                    nn.ConvTranspose2d(
                        dims[i+1] if pre_upconv else dims[i],
                        dims[i+1] if pre_upconv else dims[i],
                        kernel_size=4, stride=2, padding=1,
                        bias=False
                    ),
                )


        self.res_block = conv_block(out_dims[-1], out_dims[-1], self.activation)

        self.head_conv = nn.Conv2d(out_dims[-1], pred_dim, kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""

        initializer = partial(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):
        self.outputs = {}

        # decoder
        for i in range(len(self.scale_factors)):

            x = self.convs[f"resblock_{i}"](x)

            if self.pre_upconv:
                x = self.convs[f"upconv_{i}"](x)


            if self.upsample_mode == 'pixelshuffle':
                x = self.convs[f"pixelshuffle_{i}"](x)
            if self.upsample_mode == 'res-pixelshuffle':
                near_uped = bilinear_upsample(x, scale_factor=self.scale_factors[i])
                x = self.convs[f"pixelshuffle_{i}"](x) + near_uped
            if self.upsample_mode == 'nearest':
                x = nearest_upsample(x, scale_factor=self.scale_factors[i])
            if self.upsample_mode == 'bilinear':
                x = bilinear_upsample(x, scale_factor=self.scale_factors[i])
            if self.upsample_mode == 'deconv':
                x = self.convs[f"deconv_{i}"](x)

            if not self.pre_upconv:
                x = self.convs[f"upconv_{i}"](x)


        x = self.res_block(x)

        return self.head_conv(x)


class UnshufflePyramidDecoder(nn.Module):
    def __init__(self, in_dim, out_dims, pred_dim, activation, scale_factors=None, conv_block='basic',
                 upsample_mode='nearest', pre_upconv=True):
        super().__init__()

        out_dims = list(out_dims)
        self.activation = {
            'ReLU': nn.ReLU(),
            'GELU': nn.GELU()
        }[activation]

        self.pre_upconv = pre_upconv

        if scale_factors is None:
            self.scale_factors = [2]*len(out_dims)
        else:
            if len(scale_factors) != len(out_dims):
                raise ValueError(f"len(scale_factors) != len(out_dims)")
            self.scale_factors = list(scale_factors)

        available_upmodes = ['bilinear',  'deconv']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in {available_upmodes} | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode

        if conv_block == 'basic':
            conv_block = PreActBlock
        elif conv_block == 'bottleneck':
            conv_block = PreActBottleneck
        else:
            raise ValueError(f"conv_block must be in ['basic', 'bottleneck'] | conv_block={conv_block}")


        self.unshuffle = nn.Sequential(
            nn.Conv2d(in_dim, out_dims[0] * 4, kernel_size=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReplicationPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride=1)
        )

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(len(out_dims)):
            dims = [out_dims[0]] + out_dims

            self.convs[f"resblock_{i}"] = conv_block(dims[i], dims[i], self.activation)
            self.convs[f"upconv_{i}"] = nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, bias=True)

            if self.upsample_mode == 'deconv':
                self.convs[f"deconv_{i}"] = nn.Sequential(
                    nn.BatchNorm2d(dims[i + 1] if pre_upconv else dims[i]),
                    nn.ConvTranspose2d(
                        dims[i+1] if pre_upconv else dims[i],
                        dims[i+1] if pre_upconv else dims[i],
                        kernel_size=4, stride=2, padding=1,
                        bias=False
                    ),
                )


        self.res_block = conv_block(out_dims[-1], out_dims[-1], self.activation)

        self.head_conv = nn.Conv2d(out_dims[-1], pred_dim, kernel_size=1, padding=0)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""

        initializer = partial(nn.init.kaiming_normal_, mode='fan_in', nonlinearity='relu')

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                initializer(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, **kwargs):

        x = self.unshuffle(x)

        # decoder
        for i in range(len(self.scale_factors)):

            x = self.convs[f"resblock_{i}"](x)

            if self.pre_upconv:
                x = self.convs[f"upconv_{i}"](x)

            if self.upsample_mode == 'bilinear':
                x = bilinear_upsample(x, scale_factor=self.scale_factors[i])
            if self.upsample_mode == 'deconv':
                x = self.convs[f"deconv_{i}"](x)

            if not self.pre_upconv:
                x = self.convs[f"upconv_{i}"](x)


        x = self.res_block(x)

        return self.head_conv(x)




def nearest_upsample(x, scale_factor=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode="nearest-exact", align_corners=False)

def bilinear_upsample(x, scale_factor=2):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''

    def __init__(self, in_dim, out_dim, activation):
        super().__init__()
        self.activation = activation
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out += x
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    def __init__(self, in_dim, out_dim, activation, contraction=2):
        super().__init__()

        self.activation = activation
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim//contraction, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim//contraction)
        self.conv2 = nn.Conv2d(out_dim//contraction, out_dim//contraction, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_dim//contraction)
        self.conv3 = nn.Conv2d(out_dim//contraction, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.activation(self.bn1(x)))
        out = self.conv2(self.activation(self.bn2(out)))
        out = self.conv3(self.activation(self.bn3(out)))
        out += x
        return out


def init_subpixel(weight):
    co, ci, h, w = weight.shape
    co2 = co // 4
    # initialize sub kernel
    k = torch.empty([co2, ci, h, w])
    nn.init.kaiming_uniform_(k)
    # repeat 4 times
    k = k.repeat_interleave(4, dim=0)
    weight.data.copy_(k)


class SubPixelUpsamplingBlock(nn.Module):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."
    "useful conversation: https://twitter.com/jeremyphoward/status/1066429286771580928"
    def __init__(self, in_channels, out_channels=None, upscale_factor=2, blur=True):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor * upscale_factor), kernel_size=3,
                              padding=1, bias=True)

        init_subpixel(self.conv.weight)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1,0,1,0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.do_blur = blur

    def forward(self,x):

        x = self.conv(x)
        x = self.pixel_shuffle(x)
        if self.do_blur:
            x = self.pad(x)
            x = self.blur(x)
        return x














