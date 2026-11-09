"""Exact 3D nearest neighbors with a uniform spatial index and CUDA gradients."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="neupig_spatial_knn",
        sources=[str(Path(__file__).resolve().parents[1] / "cuda" / "spatial_knn.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class _Distances(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, points, reference, lower, width, heads, next_indices, sort_queries
    ):
        distances, indices = extension().query(
            points, reference, lower, width, heads, next_indices, sort_queries
        )
        ctx.save_for_backward(points, reference, indices)
        return distances

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_distances):
        points, reference, indices = ctx.saved_tensors
        need_points, need_reference = ctx.needs_input_grad[:2]
        grad_points, grad_reference = extension().backward(
            points,
            reference,
            indices,
            grad_distances.contiguous(),
            need_points,
            need_reference,
        )
        if not need_points:
            grad_points = None
        if not need_reference:
            grad_reference = None
        return grad_points, grad_reference, None, None, None, None, None


class SpatialIndex:
    """Index a finite, immutable CUDA FP32 tensor of shape (B, N, 3).

    Rebuild the index if reference coordinates change. Only first derivatives
    are supported; reference gradients use nondeterministic atomic additions.
    """

    def __init__(self, reference, sort_queries=False):
        if (
            reference.ndim != 3
            or reference.shape[-1] != 3
            or reference.shape[0] == 0
            or reference.shape[1] == 0
        ):
            raise ValueError("Expected a nonempty reference of shape (B, N, 3)")
        if not reference.is_cuda or reference.dtype != torch.float32:
            raise ValueError("Spatial KNN requires CUDA float32 reference points")
        self.sort_queries = sort_queries
        self.reference = reference.contiguous()
        with torch.no_grad():
            self.lower = self.reference.amin(dim=1)
            upper = self.reference.amax(dim=1)
            self.width = ((upper - self.lower).amax(dim=1) / 32).clamp_min(1e-6)
            self.heads, self.next_indices = extension().build(
                self.reference, self.lower, self.width
            )

    def distances(self, points):
        return _Distances.apply(
            points.contiguous(),
            self.reference,
            self.lower,
            self.width,
            self.heads,
            self.next_indices,
            self.sort_queries,
        )
