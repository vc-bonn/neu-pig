from pytorch3d.ops import sample_points_from_meshes, knn_gather
import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points


# Wrapper Class for DynoSurf Loss Code
class Loss_f(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.sampling_num = 10000

    def calc_chamfer(
        self,
        verts: torch.Tensor,
        faces: torch.Tensor,
        target_points: torch.Tensor,
        target_normals: torch.Tensor,
        single_direction=False,
        reduce_mean=True,
    ):
        if faces is not None:
            meshes = Meshes(verts, faces)
            pred_points, pred_normals = sample_points_from_meshes(
                meshes, self.sampling_num, return_normals=True
            )
        else:
            pred_points = verts
            pred_normals = None

        chamfer_distance_dir1 = self.calc_robust_chamfer_single_direction(
            target_points,
            target_normals,
            pred_points,
            pred_normals,
            return_normals=pred_normals is not None and target_normals is not None,
            alpha=0.3,
            reduce_mean=reduce_mean,
        )
        if single_direction:
            return chamfer_distance_dir1
        chamfer_distance_dir2 = self.calc_robust_chamfer_single_direction(
            pred_points,
            pred_normals,
            target_points,
            target_normals,
            return_normals=pred_normals is not None and target_normals is not None,
            alpha=0.3,
            reduce_mean=reduce_mean,
        )
        chamfer_distance = (
            chamfer_distance_dir1[0] + chamfer_distance_dir2[0],
            (
                chamfer_distance_dir1[1] + chamfer_distance_dir2[1]
                if pred_normals is not None and target_normals is not None
                else None
            ),
        )
        return chamfer_distance

    def welsch_weight(self, x, alpha=1.0):
        return torch.exp(-(x**2) / (2 * alpha * alpha))

    def calc_robust_chamfer_single_direction(
        self,
        x_points,
        x_normals,
        y_points,
        y_normals,
        return_normals=True,
        abs_cosine=True,
        alpha=1.0,
        reduce_mean=True,
    ):
        closest_xy = knn_points(x_points, y_points, K=1)  # (b, n, 1)
        indices_xy = closest_xy.idx
        dists_xy = torch.squeeze(closest_xy.dists, dim=-1)  # (b, n)
        robust_weight_xy = self.welsch_weight(dists_xy, alpha).detach()

        robust_dists_xy = robust_weight_xy * dists_xy

        cham_x = torch.sum(robust_dists_xy, dim=1)  # (b,)

        if return_normals:
            # Gather the normals using the indices and keep only value for k=0
            x_normals_near = knn_gather(y_normals, indices_xy)[..., 0, :]  # (b, n, 3)

            cosine_sim = torch.nn.functional.cosine_similarity(
                x_normals, x_normals_near, dim=2, eps=1e-6
            )
            # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
            cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if reduce_mean:
            cham_dist = torch.mean(cham_x)  # (1,)
            cham_normals = torch.mean(cham_norm_x) if return_normals else None
        else:
            cham_dist = cham_x  # (b,)
            cham_normals = cham_norm_x if return_normals else None

        return cham_dist, cham_normals
