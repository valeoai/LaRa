import torch
import math
from typing import Optional, Tuple


from functools import lru_cache

#@lru_cache(maxsize=None)
def meshgrid(spatial_shape, normalized=True, indexing='ij', device=None):
    """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].
    :param v_min: minimum coordinate value per dimension.
    :param v_max: maximum coordinate value per dimension.
    :return: position coordinates tensor of shape (*shape, len(shape)).
    """
    if normalized:
        axis_coords = [torch.linspace(-1., 1., steps=s, device=device) for s in spatial_shape]
    else:
        axis_coords = [torch.linspace(0, s-1, steps=s, device=device) for s in spatial_shape]

    grid_coords = torch.meshgrid(*axis_coords, indexing=indexing)

    return torch.stack(grid_coords, dim=-1)

#@lru_cache(maxsize=None)
def position_encodings(
        p: torch.Tensor,
        num_frequency_bands: int,
        max_frequencies: Optional[Tuple[int, ...]] = None,
        include_positions: bool = True
) -> torch.Tensor:

    """Fourier-encode positions p using self.num_bands frequency bands.
    :param p: positions of shape (*d, c) where c = len(d).
    :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
           2-tuple for images, ...). If `None` values are derived from shape of p.
    :param include_positions: whether to include input positions p in returned encodings tensor.
    :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
    """
    encodings = []

    if max_frequencies is None:
        max_frequencies = p.shape[:-1]

    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands, device=p.device)
        for max_freq in max_frequencies
    ]

    frequency_grids = []

    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])

    if include_positions:
        encodings.append(p)

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

    return torch.cat(encodings, dim=-1)