from __future__ import annotations

import torch
from contextlib import nullcontext
from typing import TYPE_CHECKING

from src.utilities.loss_f import Loss_f

if TYPE_CHECKING:
    from pcgrid.value_wrapper import ValueWrapper
from src.utilities.util import edgelength

from pytorch3d.structures import Meshes
from src.io.datasets import optimization_dataset
from src.network.network import Network
import os
import math
from torch.multiprocessing import Queue
from src.rotation import Rotation
from src.loss_time_smoothing import LossTimeSmoothing
from src.network.precision import compute_dtype


class Optimization(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mlp_dtype = compute_dtype(self.performance.get("mlp_precision", "bf16"))
        self.grid_dtype = compute_dtype(self.performance.get("grid_precision", "fp32"))
        needs_scaling = torch.float16 in (self.mlp_dtype, self.grid_dtype)
        self.loss_scale = float(
            self.performance.get("loss_scale", 128.0 if needs_scaling else 1.0)
        )
        if not math.isfinite(self.loss_scale) or self.loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive")
        self.loss_f = Loss_f(self.args)
        self.rotation = Rotation(self.args)
        self.loss_smoothing = LossTimeSmoothing(self.args)

        mlp_args = self.method_args.get("mlp", {})
        point_dim = mlp_args.get("point_dim", 28)
        normal_dim = mlp_args.get("normal_dim", 4)
        point_dim = point_dim + normal_dim
        layers = mlp_args.get("layers", 3)
        time_dim = mlp_args.get("time_dim", 8)
        hidden_dim = mlp_args.get("hidden", 512)

        self.mlp = Network(
            self.args,
            point_dim=point_dim,
            time_dim=time_dim,
            hidden=hidden_dim,
            layers=layers,
            time_encoding=self.method_args["time_encoding"],
            outdim=self.rotation.rotation_dim + 3,
        ).to(self.args.device)
        os.makedirs(os.path.join(self.args.io_args["out_path"], "plots"), exist_ok=True)
        os.makedirs(
            os.path.join(self.args.io_args["out_path"], "plots", "kernel"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.args.io_args["out_path"], "plots", "chamfer_distance"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                self.args.io_args["out_path"], "plots", "smoothed_chamfer_distance"
            ),
            exist_ok=True,
        )

        param_groups = [
            {"params": self.mlp.parameters(), "lr": 0.001},
        ]
        if self.args.instant_ngp:
            import tinycudann as tcnn

            encoding = {
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 2.0,
                "interpolation": "Linear",
            }

            self.encoding = tcnn.Encoding(3, encoding).to(self.args.device)
            param_groups.append({"params": self.encoding.parameters(), "lr": 0.005})
        self.optimizer = torch.optim.Adam(
            param_groups,
            fused=(
                torch.device(self.args.device).type == "cuda"
                and self.performance.get("fused_adam", True)
            ),
            capturable=(
                torch.device(self.args.device).type == "cuda"
                and self.performance.get("cuda_graph", True)
            ),
        )

    @property
    def method_args(self):
        return self.args.method_args

    @property
    def performance(self):
        return self.method_args.get("performance", {})

    @property
    def epochs(self):
        return self.method_args["optimization"]["epochs"]

    @property
    def idx(self):
        return self.method_args["keyframe_index"]

    def chamfer_distance(self, pred_points: torch.Tensor, goal: torch.Tensor):
        if pred_points.is_cuda and self.performance.get("spatial_knn", True):
            return self.loss_f.fast_chamfer(pred_points, goal)
        cd = self.loss_f.calc_chamfer(
            pred_points,
            None,
            goal,
            None,
            single_direction=False,
            reduce_mean=False,
        )
        return cd[0]

    def forward_prediction(
        self,
        grids: list[ValueWrapper],
        v: torch.Tensor,
        data: dict,
        vertex_normals: torch.Tensor,
        compute_loss: bool = True,
    ):

        time = data.get("time")
        if time is None:
            time = data["target_index"].squeeze() / data["target_index"].max()
        if self.args.instant_ngp:
            network_input = self.encoding(v)[None].expand(
                data["target_index"].shape[0], -1, -1
            )

        else:
            point_input = self._point_input
            normal_input = self._normal_input

            point_grid, normal_grid = grids
            point_values = point_grid(point_input)
            normal_values = normal_grid(normal_input)
            network_input = torch.cat(
                (point_values["Network"], normal_values["Network"]), dim=-1
            ).expand(data["target_index"].shape[0], -1, -1)

        with torch.autocast(
            device_type=v.device.type,
            dtype=self.mlp_dtype,
            enabled=v.device.type == "cuda" and self.mlp_dtype != torch.float32,
        ):
            transformation_parameters = self.mlp(
                network_input, time, time_embedding=data.get("time_embedding")
            )
        transformation_parameters = transformation_parameters.float()
        translation = torch.tanh(
            transformation_parameters[..., self.rotation.rotation_dim :] * 0.1
        )
        rotation_parameters = transformation_parameters[
            ..., : self.rotation.rotation_dim
        ]
        transform = self.rotation(rotation_parameters)

        transformed_points = (transform @ v[None, ..., None]).squeeze(
            dim=-1
        ) + translation

        cd_p = (
            self.chamfer_distance(transformed_points, data["target_xyz"])
            if compute_loss
            else None
        )
        return transformed_points, cd_p

    def regularization_loss(
        self, v: torch.Tensor, f: torch.Tensor, base_length: torch.Tensor
    ):
        if self.performance.get("cache_edges", True):
            endpoints = v[:, self._edges]
            edge_length = (endpoints[:, :, 0] - endpoints[:, :, 1]).norm(dim=-1)
        else:
            edge_length = edgelength(v, f[None].expand(v.shape[0], -1, -1))
        return torch.nn.functional.l1_loss(
            edge_length, base_length.expand_as(edge_length)
        )

    def loss_time_smoothing(
        self,
        losses: torch.Tensor,
        epoch,
    ):
        return self.loss_smoothing(losses, epoch)

    def forward(
        self,
        grids: list[ValueWrapper],
        vertices: torch.Tensor,
        faces: torch.Tensor,
        data: dict,
        q_progress: Queue,
        runtime=None,
    ):
        dataset = optimization_dataset.OptimizationDataset(self.args, data)
        if len(dataset) == 0 or self.epochs < 1:
            raise ValueError("Optimization requires at least one frame and one epoch")
        self.loss_f.target_indices.clear()
        # The target sequence is immutable throughout optimization. Batch once.
        batches = []
        for start in range(0, len(dataset), 1000):
            indices = dataset.indices[start : start + 1000]
            batch = {
                "target": dataset.points[start : start + 1000],
                "target_index": indices,
            }
            batch["target_xyz"] = batch["target"][..., :3].squeeze((1, 2)).contiguous()
            self.loss_f.prepare_target(batch["target_xyz"])
            batch["time"] = indices / indices.max().clamp_min(1)
            if not any(p.requires_grad for p in self.mlp.time_encoding.parameters()):
                with torch.no_grad():
                    batch["time_embedding"] = self.mlp.time_encoding(batch["time"])
            batches.append(batch)
        with torch.no_grad():
            template = Meshes(verts=vertices[None], faces=faces[None])
            vertex_normals = template.verts_normals_packed()
            self._edges = template.edges_packed()
            endpoints = vertices[self._edges]
            base_length = (endpoints[:, 0] - endpoints[:, 1]).norm(dim=-1)[None]
        grid_index = torch.zeros(1, device=vertices.device, dtype=torch.long)
        self._point_input = {
            "points": vertices[None, None, None],
            "grid_index": grid_index,
        }
        self._normal_input = {
            "points": vertex_normals[None, None, None],
            "grid_index": grid_index,
        }
        if not self.args.instant_ngp:
            for grid, grid_input in zip(grids, (self._point_input, self._normal_input)):
                if hasattr(grid, "prepare"):
                    grid.prepare(grid_input["points"])

        phase = runtime.phase if runtime is not None else lambda name: nullcontext()
        training_parameters = [
            p for group in self.optimizer.param_groups for p in group["params"]
        ]
        if self.loss_scale != 1.0:
            from src.network.preconditioned_grid import PreconditionedGrid

            if not all(isinstance(grid, PreconditionedGrid) for grid in grids):
                raise ValueError(
                    "Scaled half-precision training requires the fast grid backend"
                )
            training_parameters += [
                level.values for grid in grids for level in grid.levels
            ]

        def train_step(batch, epoch):
            with phase("prediction_and_chamfer"):
                transformed_points, cd_p = self.forward_prediction(
                    grids, vertices, batch, vertex_normals
                )
            with phase("time_smoothing"):
                cd_smoothed = self.loss_time_smoothing(cd_p, epoch)
            with phase("edge_regularization"):
                regularization = self.regularization_loss(
                    transformed_points, faces, base_length
                )
                regularization = (
                    regularization * self.method_args["optimization"]["edgeloss"]
                )
            with phase("backward"):
                ((cd_smoothed + regularization) * self.loss_scale).backward()
                if self.loss_scale != 1.0:
                    gradients = [
                        p.grad for p in training_parameters if p.grad is not None
                    ]
                    torch._foreach_mul_(gradients, 1.0 / self.loss_scale)
            with phase("grid_step"):
                for grid in grids:
                    grid.step()
            with phase("adam_step"):
                self.optimizer.step()
            with phase("zero_grad"):
                for grid in grids:
                    grid.zero_grad()
                self.optimizer.zero_grad(set_to_none=True)
            return cd_smoothed, regularization

        from src.network.preconditioned_grid import PreconditionedGrid

        use_graph = (
            self.performance.get("cuda_graph", True)
            and vertices.is_cuda
            and len(batches) == 1
            and all(isinstance(grid, PreconditionedGrid) for grid in grids)
            and self.performance.get("fused_adam", True)
            and self.performance.get("spatial_knn", True)
            and not self.args.instant_ngp
            and not (runtime is not None and runtime.diagnostic)
        )
        graph = None
        warmup_stream = torch.cuda.Stream(device=vertices.device) if use_graph else None
        if warmup_stream is not None:
            warmup_stream.wait_stream(torch.cuda.current_stream(vertices.device))
        epoch_tensor = torch.zeros((), device=vertices.device)
        if runtime is not None:
            runtime.start()
        for epoch in range(self.epochs):
            if use_graph and epoch >= 3:
                epoch_tensor.fill_(epoch)
                if graph is None:
                    # First three real epochs initialize optimizer/library state.
                    # Capture performs no extra optimization updates; each replay
                    # below is exactly one scheduled training epoch.
                    torch.cuda.synchronize(vertices.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warmup_stream):
                        cd_smoothed, regularization = train_step(
                            batches[0], epoch_tensor
                        )
                graph.replay()
            else:
                if warmup_stream is not None:
                    warmup_stream.wait_stream(
                        torch.cuda.current_stream(vertices.device)
                    )
                    with torch.cuda.stream(warmup_stream):
                        for batch in batches:
                            cd_smoothed, regularization = train_step(batch, epoch)
                    torch.cuda.current_stream(vertices.device).wait_stream(
                        warmup_stream
                    )
                else:
                    for batch in batches:
                        cd_smoothed, regularization = train_step(batch, epoch)
            if (
                (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs
            ) and q_progress is not None:
                cd_value = cd_smoothed.detach().cpu().item()
                regularization_value = regularization.detach().cpu().item()
                if not (
                    math.isfinite(cd_value) and math.isfinite(regularization_value)
                ):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}")
                q_progress.put(
                    f"{self.args.device};{cd_value};{regularization_value};{self.args.io_args['directory']};{epoch+1}"
                )
            if runtime is not None:
                runtime.epoch_end()
        if runtime is not None:
            runtime.stop()
        final_vertices = []
        with torch.no_grad():
            for batch in batches:
                predicted_vertices, _ = self.forward_prediction(
                    grids, vertices, batch, vertex_normals, compute_loss=False
                )
                final_vertices.append(predicted_vertices.cpu())

        final_vertices = torch.cat(final_vertices)
        final_faces = faces.detach().cpu().repeat(len(final_vertices), 1, 1)
        return Meshes(verts=final_vertices, faces=final_faces)
