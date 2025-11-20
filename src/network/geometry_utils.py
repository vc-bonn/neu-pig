import torch
import os
from pathlib import Path
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes


def compute_keyframe(points, res=512, method="ours"):
    if method == "ours":
        positions = (points[..., :3].squeeze() + 1) / 2
        positions = (positions * res).int()
        outputs = torch.stack(
            [
                torch.tensor([(torch.unique(p_, dim=0) / res).shape[0]])
                for p_ in positions
            ]
        )
        x = torch.exp(
            -0.001 * torch.arange(-points.shape[0] // 2, points.shape[0] // 2, 1) ** 2
        )[:, None]
        index = (outputs * x).argmax().item()
        return index
    elif method == "first":
        return 0
    elif method == "middle":
        return points.shape[0] // 2
    else:
        raise Exception("Unknown Keyframe Method [{}]".format(method))


def init_surf(points, normals, args: dict, method="ours"):
    if method == "ours":
        from src.io.initialization import poisson

        return poisson(points, normals)

    elif method == "tetra":
        from src.io.initialization import marching_tetras

        return marching_tetras(args, points.squeeze()[..., :3], normals)

    elif method == "diffusion":
        from src.io.initialization import diffusion

        return diffusion(args, points.squeeze()[..., :3], normals)

    else:
        raise Exception("Unknown Init Method [{}]".format(method))


def scale_and_save(args, meshes: Meshes):
    points = meshes.verts_padded().cpu().squeeze()[..., :3]
    points /= 0.95
    points *= (args.points_max.cpu() - args.points_min.cpu()).max() / 2
    # points *= (args.points_max.cpu() - args.points_min.cpu()) / 2
    points += (args.points_max.cpu() + args.points_min.cpu()) / 2
    for idx, (p, f) in enumerate(zip(points, meshes.faces_padded())):
        save_obj(
            Path(
                os.path.join(
                    args.io_args["out_path"],
                    "%04d.obj" % idx,
                )
            ),
            p,
            f,
        )

    return Meshes(verts=points, faces=meshes.faces_padded())
