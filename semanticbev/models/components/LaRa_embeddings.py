import torch
from torch import nn
from einops import rearrange, repeat
import numpy as np


from semanticbev.models.components.LaRa_encoder import InputEmbedding
from semanticbev.models.components.LaRa_decoder import OutputAdapter, QueryGenerator
from semanticbev.models.components.positional_encodings import meshgrid, position_encodings




class FourierInputEmbedding(InputEmbedding):
    def __init__(self, num_frequency_bands: int):
        self.num_frequency_bands = num_frequency_bands

        position_encoding_channels = 2 * (2 * self.num_frequency_bands + 1)

        super().__init__(num_input_channels=position_encoding_channels)

    def forward(self, x, **kwargs):
        b, n, h, w, c = x.shape

        # create positionnal encodings
        pixel_coords = meshgrid((h, w), indexing='ij').to(x)
        position_encoding = position_encodings(pixel_coords, self.num_frequency_bands) # H, W, C
        position_encoding = repeat(position_encoding, '... -> b n ...', b=b, n=n) # repeat along batch and cam dimension
        position_encoding = rearrange(position_encoding, 'b ... c -> b (...) c') # rearrange input as sequence

        return position_encoding


class LearnedCamInputEmbedding(InputEmbedding):
    def __init__(self, num_cams: int = 6):

        super().__init__(num_input_channels=1)

        self.cam_embedding = nn.Parameter(torch.randn(num_cams))


    def forward(self, x, **kwargs):
        b, n, h, w, c = x.shape

        # repeat position encoding along batch and sequence dimension (also add channel dim)
        cam_embedding = repeat(self.cam_embedding, '... -> b ... k 1', b=b, k=h*w) # B, N, HxW, 1
        cam_embedding = rearrange(cam_embedding, 'b ... c -> b (...) c') # B, NxHxW, 1

        return cam_embedding

class CamMatrixInputEmbedding(InputEmbedding):
    def __init__(self, plucker=False):

        self.plucker = plucker

        super().__init__(num_input_channels=6 + int(plucker)*3)

    def forward(self, x, rots, trans, intrins, input_stride, **kwargs):
        b, n, h, w, c = x.shape

        updated_intrinsics = intrins.clone()

        # Adjust intrinsics scale due to downsizing by input_stride (we take feature maps as input not the raw images)
        updated_intrinsics[:, :, 0, 0] *= 1 / input_stride
        updated_intrinsics[:, :, 0, 2] *= 1 / input_stride
        updated_intrinsics[:, :, 1, 1] *= 1 / input_stride
        updated_intrinsics[:, :, 1, 2] *= 1 / input_stride

        # create positionnal encodings
        pixel_coords = meshgrid((w, h), normalized=False, indexing='xy', device=x.device)
        ones = torch.ones((h, w, 1), device=x.device)

        pixel_coords = torch.cat([pixel_coords, ones], dim=-1)  # [x, y, 1] vectors of pixel coordinates
        pixel_coords = rearrange(pixel_coords, 'h w c -> c (h w)')
        pixel_coords = repeat(pixel_coords, '... -> b n ...', b=b, n=n)
        # pixel_coords.shape = [B, N, 3, K] | N # of cams, K # of pixels
        pixel_coords = rots @ updated_intrinsics.inverse() @ pixel_coords

        # at this point pixel_coords are x,y,z pixel coords in ego-car ref frame at depth=1meter
        # if we want the true 3D position of pixels we would need scale coords by the focal length in meter
        f = 5.5 / 1000  # focal length in meters
        pixel_coords *= f

        normed_dirs = pixel_coords / pixel_coords.norm(dim=2, keepdim=True)
        normed_dirs = rearrange(normed_dirs, 'b n c k -> b (n k) c')

        cam_origins = repeat(trans, 'b n c -> b n c k', k=pixel_coords.shape[-1])
        cam_origins = rearrange(cam_origins, 'b n c k -> b (n k) c')

        output = [cam_origins, normed_dirs]

        if self.plucker:
            cross = torch.cross(cam_origins, normed_dirs)
            output.append(cross)

        return torch.cat(output, dim=-1)


class FrustumInputEmbedding(InputEmbedding):
    def __init__(self, dmin, dmax, n_bins):
        self.dmin = dmin
        self.dmax = dmax
        self.n_bins = n_bins

        super().__init__(num_input_channels=(n_bins+1)*3)

    def forward(self, x, rots, trans, intrins, input_stride, **kwargs):
        b, n, h, w, c = x.shape

        updated_intrinsics = intrins.clone()

        # Adjust intrinsics scale due to resizing
        updated_intrinsics[:, :, 0, 0] *= 1 / input_stride
        updated_intrinsics[:, :, 0, 2] *= 1 / input_stride
        updated_intrinsics[:, :, 1, 1] *= 1 / input_stride
        updated_intrinsics[:, :, 1, 2] *= 1 / input_stride


        # create positionnal encodings
        pixel_coords = meshgrid((w, h), normalized=False, indexing='xy', device=x.device)
        ones = torch.ones((h, w, 1), device=x.device)
        pixel_coords = torch.cat([pixel_coords, ones], dim=-1)  # [x, y, 1] vectors


        i = torch.arange(0, self.n_bins + 1, dtype=torch.float)
        depth_grid = torch.exp(np.log(self.dmin) + np.log(self.dmax/self.dmin) * i/self.n_bins, device=x.device)

        frustum_coords = repeat(pixel_coords, '... -> d ...', d=self.n_bins+1)
        frustum_coords = frustum_coords * repeat(depth_grid, 'd-> d 1 1 1')

        frustum_coords = rearrange(frustum_coords, 'd h w c -> c (d h w)')
        frustum_coords = repeat(frustum_coords, '... -> b n ...', b=b, n=n)

        points = rots @ updated_intrinsics.inverse() @ frustum_coords + trans.unsqueeze(-1)

        points = rearrange(points, ' b n c (d h w)  -> b (n h w) (c d)', h=h, w=w, d=self.n_bins+1)

        return points


class FourierQueryGenerator(QueryGenerator):
    def __init__(self,  grid_conf, num_frequency_bands: int, downscale_factor=1):
        self.num_frequency_bands = num_frequency_bands

        # e.g., xbound=[-50.0, 50.0, 0.5]    min, max, resolution
        axis_bounds = [grid_conf['ybound'], grid_conf['xbound']]
        self.bev_shape = tuple([int((bound[1] - bound[0]) / bound[2] // downscale_factor) for bound in axis_bounds])

        position_encoding_channels = 2 * (2 * self.num_frequency_bands + 1)

        super().__init__(query_seq_len=self.bev_shape[0]*self.bev_shape[1], num_input_channels=position_encoding_channels)

    def forward(self, batch_size, device, **kwargs):
        # create positionnal encodings
        pixel_coords = meshgrid(self.bev_shape, indexing='ij', device=device)
        position_encoding = position_encodings(pixel_coords, self.num_frequency_bands) # H, W, C
        position_encoding = rearrange(position_encoding, '... c -> (...) c') # flatten encodings on spatial dimensions
        position_encoding = repeat(position_encoding, '... -> b ...', b=batch_size) # repeat along batch dimension

        return position_encoding


class CoordConvQueryGenerator(QueryGenerator):
    def __init__(self, grid_conf, with_r=False, scaling=False, downscale_factor=1):

        # e.g., xbound=[-50.0, 50.0, 0.5]    min, max, resolution
        axis_bounds = [grid_conf['xbound'], grid_conf['ybound']]
        self.scale_factor = [bound[2] for bound in axis_bounds]
        self.bev_shape = tuple([int((bound[1] - bound[0]) / bound[2] // downscale_factor) for bound in axis_bounds])
        self.with_r = with_r
        self.scaling = scaling

        num_input_channels = 2 + int(with_r)

        super().__init__(query_seq_len=self.bev_shape[0]*self.bev_shape[1], num_input_channels=num_input_channels)

    def forward(self, batch_size, device, **kwargs):

        # x and y coords in range [-1, 1], one axis encoding by channel
        pixel_coords = meshgrid(self.bev_shape, normalized=True, indexing='ij', device=device)  # (*image_shape, len(image_shape))

        if self.scaling:
            for i in range(pixel_coords.shape[-1]):
                pixel_coords[:,:,0] *= self.scale_factor[0]

        if self.with_r:
            relative_coords = torch.sqrt(torch.pow(pixel_coords, 2).sum(-1, keepdim=True))  # sqrt(x^2+y^2)
            pixel_coords = torch.cat([pixel_coords, relative_coords], dim=-1)

        pixel_coords = rearrange(pixel_coords, '... c -> (...) c')  # flatten encodings on spatial dimensions
        pixel_coords = repeat(pixel_coords, '... -> b ...', b=batch_size)  # K, C -> B, K, C

        return pixel_coords


class LearnedQueryGenerator(QueryGenerator):
    def __init__(self, grid_conf, num_channels: int = 1, downscale_factor=1):

        # e.g., xbound=[-50.0, 50.0, 0.5]    min, max, resolution
        axis_bounds = [grid_conf['xbound'], grid_conf['ybound']]
        self.bev_shape = tuple([int((bound[1] - bound[0]) / bound[2] // downscale_factor) for bound in axis_bounds])
        query_seq_len = self.bev_shape[0] * self.bev_shape[1]

        super().__init__(query_seq_len=query_seq_len, num_input_channels=num_channels)

        self.bev_embedding = nn.Parameter(torch.randn(query_seq_len, num_channels))


    def forward(self, batch_size, device, **kwargs):

        # repeat position encoding along batch and sequence dimension (also add channel dim)
        bev_embedding = repeat(self.bev_embedding, '... -> b ...', b=batch_size) # B, HxW, C

        return bev_embedding




class BEVOutputAdapter(OutputAdapter):
    def __init__(self, grid_conf, num_output_channels: int = 64, downscale_factor: int = 1):

        # e.g., xbound=[-50.0, 50.0, 0.5]    min, max, resolution
        axis_bounds = [grid_conf['xbound'], grid_conf['ybound']]
        self.bev_shape = tuple([int((bound[1] - bound[0]) / bound[2] // downscale_factor) for bound in axis_bounds])
        super().__init__(output_shape=(self.bev_shape[0]*self.bev_shape[1], num_output_channels))

    def forward(self, bev_flattened):

        bev = rearrange(bev_flattened, 'b (h w) c -> b c h w', h=self.bev_shape[0], w=self.bev_shape[1])

        return bev

