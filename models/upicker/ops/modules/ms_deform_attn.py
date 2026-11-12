"""Multi-scale deformable attention module (Deformable-DETR).

Reconstructed because the published UPicker tree imports
``models.upicker.ops.modules.MSDeformAttn`` but ships no ``ops`` package, so it
cannot run as distributed on any platform. The attention kernel is the pure
PyTorch ``grid_sample`` formulation in
``models/upicker/ops/functions/ms_deform_attn_func.py``, which removes the need
for the usual hand-written CUDA extension (and therefore any per-architecture
build).
"""

import math
import warnings

import torch
from torch import nn
from torch.nn.init import constant_, xavier_uniform_

from ..functions import ms_deform_attn_core_pytorch

__all__ = ["MSDeformAttn"]


def _is_power_of_2(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Multi-Scale Deformable Attention.

        Args:
            d_model: hidden dimension
            n_levels: number of feature levels
            n_heads: number of attention heads
            n_points: number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        head_dim = d_model // n_heads
        if not _is_power_of_2(head_dim):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make the dimension of each "
                "attention head a power of 2, which is more efficient.",
                stacklevel=2,
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        Args:
            query: (N, Length_query, C)
            reference_points: (N, Length_query, n_levels, 2) normalised to [0, 1],
                top-left (0, 0), bottom-right (1, 1) including padding; or
                (N, Length_query, n_levels, 4) adding the box size.
            input_flatten: (N, sum(H_l * W_l), C)
            input_spatial_shapes: (n_levels, 2) - (H_l, W_l)
            input_level_start_index: (n_levels, ) start index of each level in input_flatten
            input_padding_mask: (N, sum(H_l * W_l)), True for padding elements
        Returns:
            (N, Length_query, C)
        """
        del input_level_start_index  # implied by input_spatial_shapes in the pure-PyTorch kernel
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}"
            )

        output = ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        return self.output_proj(output)
