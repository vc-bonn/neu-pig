"""Separable solves for pcgrid's fixed, regular spatial preconditioner.

pcgrid builds M = I + lambda * (Lx + Ly + Lz), where the one-axis
operator has diagonal twice the vertex degree and -1 on adjacent entries.
Diagonalizing the small one-axis operator solves the *same* linear system
without iterative sparse operations or host convergence checks.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F
from src.network.precision import compute_dtype


def axis_eigendecomposition(resolution: int):
    """Match pcgrid's actual degree convention, including its boundary rows."""
    if resolution < 1:
        raise ValueError("Grid resolution must be positive")
    matrix = torch.diag(torch.full((resolution,), 4.0, dtype=torch.float64))
    matrix[0, 0] = matrix[-1, -1] = 2.0
    if resolution == 1:
        matrix.zero_()
    else:
        adjacent = torch.arange(resolution - 1)
        matrix[adjacent, adjacent + 1] = -1
        matrix[adjacent + 1, adjacent] = -1
    return torch.linalg.eigh(matrix)


def transform_axes(values, matrix):
    values = torch.einsum("ij,jklc->iklc", matrix, values)
    values = torch.einsum("ij,kjlc->kilc", matrix, values)
    return torch.einsum("ij,kljc->klic", matrix, values)


class _SymmetricSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, basis, inverse):
        ctx.save_for_backward(basis, inverse)
        return transform_axes(transform_axes(values, basis.T) * inverse, basis)

    @staticmethod
    def backward(ctx, gradient):
        basis, inverse = ctx.saved_tensors
        # M is symmetric: the adjoint is the same solve. Avoid building a
        # backward graph for six separate einsums and their layout copies.
        result = transform_axes(transform_axes(gradient, basis.T) * inverse, basis)
        return result, None, None


class GridLevel(torch.nn.Module):
    def __init__(self, resolution, channels, smoothness, device, dtype=torch.float32):
        super().__init__()
        if channels < 1 or not math.isfinite(smoothness) or smoothness < 0:
            raise ValueError(
                "Channels must be positive and smoothness finite and nonnegative"
            )
        self.resolution = resolution
        self.compute_dtype = dtype
        self.values = torch.nn.Parameter(
            torch.zeros(resolution, resolution, resolution, channels, device=device)
        )
        eigenvalues, eigenvectors = axis_eigendecomposition(resolution)
        denominator = 1 + smoothness * (
            eigenvalues[:, None, None]
            + eigenvalues[None, :, None]
            + eigenvalues[None, None, :]
        )
        self.register_buffer("basis", eigenvectors.float().to(device))
        self.register_buffer(
            "inverse", denominator.reciprocal().float().to(device)[..., None]
        )
        self.register_buffer("compute_basis", self.basis.to(dtype))
        self.register_buffer("compute_inverse", self.inverse.to(dtype))
        self.register_buffer("sampling", None)
        self.prepared_points = None
        self.point_version = None

    @torch.no_grad()
    def prepare(self, points, max_nodes):
        self.sampling = None
        self.prepared_points = None
        self.point_version = None
        nodes = self.resolution**3
        if (
            nodes > max_nodes
            or points.requires_grad
            or tuple(points.shape[:3]) != (1, 1, 1)
        ):
            return
        # S maps a grid to the fixed template vertices. Precompute S M^-1;
        # subsequent forward/adjoint operations are just two dense GEMMs.
        identity = torch.eye(nodes, device=self.values.device).reshape(
            self.resolution, self.resolution, self.resolution, nodes
        )
        sampled = F.grid_sample(
            identity.permute(3, 0, 1, 2)[None],
            points,
            padding_mode="border",
            align_corners=True,
        )
        rhs = sampled[0, :, 0, 0].reshape(
            self.resolution, self.resolution, self.resolution, -1
        )
        solved = _SymmetricSolve.apply(rhs, self.basis, self.inverse)
        self.sampling = solved.reshape(nodes, -1).T.contiguous().to(self.compute_dtype)
        # Retain the tensor so an allocator cannot reuse its pointer. A version
        # check also invalidates the map after an in-place coordinate edit.
        self.prepared_points = points
        self.point_version = points._version

    def solve(self):
        return _SymmetricSolve.apply(
            self.values.to(self.compute_dtype), self.compute_basis, self.compute_inverse
        )

    def forward(self, points):
        if (
            self.sampling is not None
            and points is self.prepared_points
            and not points.requires_grad
            and points._version == self.point_version
        ):
            values = self.values.reshape(-1, self.values.shape[-1]).to(
                self.compute_dtype
            )
            return (self.sampling @ values).float()[None]
        # 3D interpolation, geometry and losses remain FP32 even when the
        # linear solver and dense sampling maps use half precision.
        values = self.solve().float().permute(3, 0, 1, 2)[None]
        return (
            F.grid_sample(values, points, padding_mode="border", align_corners=True)
            .squeeze((2, 3))
            .transpose(1, 2)
        )


class PreconditionedGrid(torch.nn.Module):
    """The Network ValueWrapper interface used by Neu-PiG (T=1)."""

    def __init__(self, args, channels, normals=False):
        super().__init__()
        config = args.method_args["grid"]
        dtype = compute_dtype(
            args.method_args.get("performance", {}).get("grid_precision", "fp32")
        )
        count = 1 if normals else config["n_level"]
        base = 4 if normals else config["base_res"]
        if count < 1:
            raise ValueError("At least one grid level is required")
        self.levels = torch.nn.ModuleList(
            [
                GridLevel(
                    base + config["increase_per_level"] * index,
                    channels,
                    config["smoothness"] * config["local_lambda_adapt"] ** index,
                    args.device,
                    dtype=dtype,
                )
                for index in range(count)
            ]
        )
        parameters = [
            {"params": [level.values], "lr": config["lr"] * config["lr_adapt"] ** index}
            for index, level in enumerate(self.levels)
        ]
        cuda = torch.device(args.device).type == "cuda"
        self.optimizer = torch.optim.Adam(
            parameters,
            fused=cuda,
            capturable=cuda
            and args.method_args.get("performance", {}).get("cuda_graph", True),
        )
        performance = args.method_args.get("performance", {})
        self.max_dense_nodes = performance.get("dense_grid_max_nodes", 4096)
        budget_mb = performance.get("dense_grid_budget_mb", 256)
        if self.max_dense_nodes < 0 or not math.isfinite(budget_mb) or budget_mb < 0:
            raise ValueError("Dense grid cache limits must be finite and nonnegative")
        self.dense_budget_bytes = budget_mb * 2**20

    def prepare(self, points):
        remaining = self.dense_budget_bytes
        for level in self.levels:
            required = points.shape[-2] * level.resolution**3 * points.element_size()
            limit = self.max_dense_nodes if required <= remaining else 0
            level.prepare(points, limit)
            if level.sampling is not None:
                remaining -= required

    def forward(self, data, limit=None):
        levels = self.levels if limit is None else self.levels[:limit]
        # Mean-before-tanh is pcgrid's original parameterization order.
        values = torch.stack([level(data["points"]) for level in levels]).mean(0)
        return {"Network": torch.tanh(values)}

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
