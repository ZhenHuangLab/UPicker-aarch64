"""Multi-scale deformable attention kernel, pure PyTorch.

UPicker imports ``models.upicker.ops.modules.MSDeformAttn`` but the repository
ships no ``ops`` package at all, so the published tree cannot run on any
platform. Deformable-DETR normally provides ``MultiScaleDeformableAttention`` as
a hand-written CUDA extension; this is the equivalent ``grid_sample``
formulation that Deformable-DETR itself ships as its reference implementation.
It runs on CPU and CUDA, needs no compilation, and therefore needs no
per-architecture build.
"""

import torch
import torch.nn.functional as F

__all__ = ["ms_deform_attn_core_pytorch"]


def _as_shape_list(value_spatial_shapes):
    """Normalise the (n_levels, 2) spatial shapes to a list of (H, W) int pairs.

    ``Tensor.tolist()`` already yields Python ints for an integer tensor, and the
    non-tensor form is a sequence of int pairs, so no numeric parsing is needed.
    """
    if isinstance(value_spatial_shapes, torch.Tensor):
        return [(h, w) for h, w in value_spatial_shapes.tolist()]
    return [(h, w) for h, w in value_spatial_shapes]


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value: (N, S, n_heads, head_dim)
        value_spatial_shapes: (n_levels, 2) - (H, W) per level
        sampling_locations: (N, Lq, n_heads, n_levels, n_points, 2), normalised to [0, 1]
        attention_weights: (N, Lq, n_heads, n_levels, n_points)
    Returns:
        (N, Lq, n_heads * head_dim)
    """
    N_, _, M_, D_ = value.shape
    _, Lq_, _, L_, P_, _ = sampling_locations.shape
    shapes = _as_shape_list(value_spatial_shapes)

    value_list = value.split([h * w for h, w in shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(shapes):
        # (N, H*W, M, D) -> (N*M, D, H, W)
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # (N, Lq, M, P, 2) -> (N*M, Lq, P, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_list.append(
            F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )

    # (N, Lq, M, L, P) -> (N*M, 1, Lq, L*P)
    weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * weights).sum(-1).view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()
