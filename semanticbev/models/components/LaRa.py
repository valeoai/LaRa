from torch import nn
from pathlib import Path

from efficientnet_pytorch import EfficientNet as EfficientNet_extractor
from torchvision.models.resnet import resnet18

from semanticbev.models.components.layers import Bottleneck
from semanticbev.models.components.layers import Up
from semanticbev.models.components.LaRa_encoder import LaRaEncoder
from semanticbev.models.components.LaRa_decoder import LaRaDecoder



class EfficientNet(nn.Module):
    def __init__(self, weights_path=None, version='b0', downsample=16):
        super().__init__()

        ckpt_map = {
            'b0': 'efficientnet-b0-355c32eb.pth',
            'b4': 'efficientnet-b4-6ed6700e.pth',
        }

        if weights_path is not None:
            weights_path = Path(weights_path) / ckpt_map[version]
            if not weights_path.exists():
                print(f'EfficientNet weights file does not exists at path {weights_path}')
                weights_path = None
            else:
                weights_path = str(weights_path)
        else:
            print('EfficientNet weights file not given, downloading...')

        trunk = EfficientNet_extractor.from_pretrained(f"efficientnet-{version}", weights_path=weights_path)

        self._conv_stem, self._bn0, self._swish = trunk._conv_stem, trunk._bn0, trunk._swish
        self.drop_connect_rate = trunk._global_params.drop_connect_rate

        self._blocks = nn.ModuleList()

        for idx, block in enumerate(trunk._blocks):
            if downsample == 4:
                if version == 'b0' and idx > 7 or version == 'b4' and idx > 10:
                    break
            elif downsample == 8:
                if version == 'b0' and idx > 10 or version == 'b4' and idx > 21:
                    break
            self._blocks.append(block)

        del trunk


class CamEncode(nn.Module):
    def __init__(self, num_out_channels, weights_path=None, version='b0', downsample=16):
        super().__init__()

        self.version = version

        params = {
            ('b0', 16): (320 + 112, 2),
            ('b0', 8): (112 + 40, 2),
            ('b0', 4): (40 + 24, 2),
            ('b4', 16): (448 + 160, 2),
            ('b4', 8): (160 + 56, 2),
            ('b4', 4): (56 + 32, 2),
        }
        if downsample not in [4,8,16]:
            raise ValueError(f"downsample must be in [4,8,16] but is: {downsample}")
        self.downsample = downsample

        self.trunk = EfficientNet(weights_path, version, downsample)

        num_in_channels, scale_factor = params[(version, downsample)]
        self.up = Up(num_in_channels, num_out_channels, scale_factor)

    def forward(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints)+1}'] = prev_x
            prev_x = x

        # Head
        endpoints[f'reduction_{len(endpoints)+1}'] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        elif self.downsample == 4:
            input_1, input_2 = endpoints['reduction_3'], endpoints['reduction_2']

        x = self.up(input_1, input_2)
        return x


class BevEncode(nn.Module):
    def __init__(self, num_in_channels, num_out_channels):
        super().__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(num_in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_out_channels, kernel_size=1, padding=0),
        )

    def forward(self, x, need_all_stages=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        xdown1 = self.layer1(x)
        xdown2 = self.layer2(xdown1)
        xdown3 = self.layer3(xdown2)

        xup1 = self.up1(xdown3, xdown1)
        out = self.up2(xup1)

        if need_all_stages:
            return xdown1, xdown2, xdown3, xup1, out

        return out


class LaRa(nn.Module):
    def __init__(self, cam_encoder: nn.Module, latent_encoder: LaRaEncoder, latent_decoder: LaRaDecoder,
                 bev_encoder: nn.Module, input_stride: int, **kwargs):
        super().__init__()

        self.cam_encoder = cam_encoder

        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder
        self.bev_encoder = bev_encoder

        self.input_stride = input_stride


    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape
        x = x.view(B*N, C, imH, imW)
        x = self.cam_encoder(x)
        x = x.view(B, N, -1, imH//self.input_stride, imW//self.input_stride)
        x = x.permute(0, 1, 3, 4, 2)

        return x

    def forward_encode(self, imgs, rots, trans, intrins, **kwargs):
        cam_feats = self.get_cam_feats(imgs)
        latent_array = self.latent_encoder(cam_feats, rots, trans, intrins, self.input_stride)
        return latent_array

    def forward_decode(self, latent_array, *args, **kwargs):
        bev = self.latent_decoder(latent_array, *args, **kwargs)
        bev = self.bev_encoder(bev)
        return bev

    def forward(self, imgs, rots, trans, intrins, **kwargs):
        latent_array = self.forward_encode(imgs, rots, trans, intrins)
        bev = self.forward_decode(latent_array)
        return bev



##############################################
############### For Tiny LaRa ################
##############################################

class SmallCamEncode(nn.Module):
    def __init__(self, num_out_channels, weights_path=None, version='b0', downsample=8):
        super().__init__()

        self.version = version

        params = {
            ('b0', 16): 112,
            ('b0', 8): 40,
            ('b0', 4): 24,
            ('b4', 16): 160,
            ('b4', 8): 56,
            ('b4', 4): 32,
        }
        if downsample not in [4, 8, 16]:
            raise ValueError(f"downsample must be in [8,16] but is: {downsample}")
        self.downsample = downsample

        self.trunk = EfficientNet(weights_path, version, downsample - downsample // 2)

        self.outconv = nn.Sequential(
            nn.Conv2d(params[(version, downsample)], num_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints[f'reduction_{len(endpoints) + 1}'] = prev_x
            prev_x = x

        if self.downsample == 16:
            x = endpoints['reduction_4']
        elif self.downsample == 8:
            x = endpoints['reduction_3']
        elif self.downsample == 4:
            x = endpoints['reduction_2']

        return self.outconv(x)


class SmallBevEncode(nn.Module):
    def __init__(self, num_in_channels: int, num_out_channels: int, planes: int, base_width: int = 64):
        super().__init__()

        self.block1 = Bottleneck(num_in_channels, planes, base_width)
        self.block2 = Bottleneck(planes * self.block1.expansion, planes, base_width)
        self.head_conv = nn.Conv2d(planes * self.block2.expansion, num_out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.head_conv(x)

        return x