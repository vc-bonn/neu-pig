"""First-order CUDA operations for fixed-template reconstruction geometry."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load

    return load(
        name="neupig_optimization_ops",
        sources=[
            str(Path(__file__).resolve().parents[1] / "cuda" / "optimization_ops.cu")
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class _Transform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parameters, vertices):
        parameters, vertices = parameters.contiguous(), vertices.contiguous()
        ctx.save_for_backward(parameters, vertices)
        return extension().transform(parameters, vertices)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        parameters, vertices = ctx.saved_tensors
        dp, dv = extension().transform_grad(
            parameters, vertices, gradient.contiguous(), ctx.needs_input_grad[1]
        )
        return dp, dv if ctx.needs_input_grad[1] else None


class _EdgeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, edges, base):
        vertices, edges, base = (
            vertices.contiguous(),
            edges.contiguous(),
            base.contiguous(),
        )
        ctx.save_for_backward(vertices, edges, base)
        return extension().edge_loss(vertices, edges, base)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        vertices, edges, base = ctx.saved_tensors
        return (
            extension().edge_grad(vertices, edges, base, gradient.contiguous()),
            None,
            None,
        )


class _FixedSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, nodes, weights):
        grid = grid.contiguous()
        ctx.save_for_backward(nodes, weights)
        ctx.count = grid.shape[0]
        return extension().sample(grid, nodes, weights)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        nodes, weights = ctx.saved_tensors
        return (
            extension().sample_grad(gradient.contiguous(), nodes, weights, ctx.count),
            None,
            None,
        )


transform_vertices = _Transform.apply
edge_loss = _EdgeLoss.apply
sample_fixed = _FixedSample.apply


@torch.no_grad()
def interpolation_stencil(points, resolution):
    """Eight border-clamped nodes/weights matching align_corners=True sampling."""
    xyz = (points.reshape(-1, 3).clamp(-1, 1) + 1) * ((resolution - 1) / 2)
    lower = xyz.floor().long()
    upper = (lower + 1).clamp_max(resolution - 1)
    fraction = xyz - lower
    nodes, weights = [], []
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                bits = (x, y, z)
                coordinates = [
                    upper[:, a] if bits[a] else lower[:, a] for a in range(3)
                ]
                nodes.append(
                    (coordinates[2] * resolution + coordinates[1]) * resolution
                    + coordinates[0]
                )
                weights.append(
                    torch.stack(
                        [
                            fraction[:, a] if bits[a] else 1 - fraction[:, a]
                            for a in range(3)
                        ],
                        dim=1,
                    ).prod(1)
                )
    return (
        torch.stack(nodes, dim=1).contiguous(),
        torch.stack(weights, dim=1).contiguous(),
    )
