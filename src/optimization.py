import torch

# from ext.m2v.im2mesh.utils import mesh
from src.utilities.loss_f import Loss_f
from torch.utils.data import DataLoader

from pcgrid.value_wrapper import ValueWrapper
from src.utilities.util import edgelength

from pytorch3d.structures import Meshes
from src.io.datasets import optimization_dataset
from pytorch3d.transforms import so3_exp_map, quaternion_to_matrix
from src.network.network import Network
import matplotlib.pyplot as plt
import os
from PIL import Image
import io
from torch.multiprocessing import Queue
from src.rotation import Rotation
from src.loss_time_smoothing import LossTimeSmoothing


class Optimization(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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
        self.optimizer = torch.optim.Adam(param_groups)

    @property
    def method_args(self):
        return self.args.method_args

    @property
    def epochs(self):
        return self.method_args["optimization"]["epochs"]

    @property
    def idx(self):
        return self.method_args["keyframe_index"]

    def chamfer_distance(self, pred_points: torch.Tensor, goal: torch.Tensor):
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
    ):

        time = data["target_index"].squeeze() / data["target_index"].max()
        if self.args.instant_ngp:
            network_input = self.encoding(v)[None].expand(
                data["target_index"].shape[0], -1, -1
            )

        else:
            point_input = {
                "points": v[None, None, None],
                "grid_index": torch.zeros(1, device=v.device, dtype=torch.long),
            }

            normal_input = {
                "points": vertex_normals[None, None, None],
                "grid_index": torch.zeros(1, device=v.device, dtype=torch.long),
            }

            point_grid, normal_grid = grids
            point_values = point_grid(point_input)
            normal_values = normal_grid(normal_input)
            network_input = torch.cat(
                (point_values["Network"], normal_values["Network"]), dim=-1
            ).expand(data["target_index"].shape[0], -1, -1)

        with torch.autocast(
            device_type=v.device.type,
            dtype=torch.bfloat16,
            enabled=v.device.type == "cuda",
        ):
            transformation_parameters = self.mlp(network_input, time)
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

        cd_p = self.chamfer_distance(
            transformed_points,
            data["target"][..., :3].squeeze(dim=(1, 2)),
        )
        return transformed_points, cd_p

    def regularization_loss(
        self, v: torch.Tensor, f: torch.Tensor, base_length: torch.Tensor
    ):
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
    ):
        dataset = optimization_dataset.OptimizationDataset(self.args, data)
        dataloader = DataLoader(dataset=dataset, batch_size=1000, shuffle=False)
        with torch.no_grad():
            vertex_normals = Meshes(
                verts=vertices[None, ...], faces=faces[None, ...]
            ).verts_normals_packed()
            base_length = edgelength(vertices, faces)

        for epoch in range(self.epochs):
            for batch in dataloader:

                transformed_points, cd_p = self.forward_prediction(
                    grids, vertices, batch, vertex_normals
                )
                cd_smoothed = self.loss_time_smoothing(cd_p, epoch)
                cd_p = cd_p.mean()

                regularization = self.regularization_loss(
                    transformed_points, faces, base_length
                )

                regularization = (
                    regularization * self.method_args["optimization"]["edgeloss"]
                )
                (cd_smoothed + regularization).backward()

                for grid in grids:
                    grid.step()
                self.optimizer.step()
                for grid in grids:
                    grid.zero_grad()
                self.optimizer.zero_grad()
            if (
                (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs
            ) and q_progress is not None:
                q_progress.put(
                    f"{self.args.device};{cd_smoothed.detach().cpu().item()};{regularization.detach().cpu().item()};{self.args.io_args['directory']};{epoch+1}"
                )
        final_vertices = []
        with torch.no_grad():
            for batch in dataloader:
                predicted_vertices, _ = self.forward_prediction(
                    grids, vertices, batch, vertex_normals
                )
                final_vertices.append(predicted_vertices.cpu())

        final_vertices = torch.cat(final_vertices)
        final_faces = faces.detach().cpu().repeat(len(final_vertices), 1, 1)
        return Meshes(verts=final_vertices, faces=final_faces)
