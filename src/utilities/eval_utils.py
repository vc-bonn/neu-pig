import numpy as np
import torch
from pytorch3d.structures import Meshes
import trimesh
from pykdtree.kdtree import KDTree

from ext.dynosurf.evaluation.utils import (
    distance_p2p,
    get_threshold_percentage,
    get_nearest_neighbors_indices_batch,
)


def get_loc_scale(mesh: Meshes):
    # Determine bounding box
    assert mesh._N == 1, "Batch size greater than 1 not supported"
    vertices = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    bbox = mesh.bounding_box.bounds
    # Compute location and scale
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()

    return np.array(loc), np.array(scale)


def correspondance(preds_goal: np.ndarray, preds_pred: np.ndarray):
    kdtree = KDTree(preds_goal)
    _, ind = kdtree.query(preds_pred, k=1)
    pc_nn_t = preds_goal[ind]
    l2_loss = np.mean(np.linalg.norm(preds_pred - pc_nn_t, axis=-1))
    return l2_loss.mean().item()


# Disclaimer: This evaluation function was taken from DynoSurf:
# https://github.com/yaoyx689/DynoSurf
def eval_pointcloud(
    pointcloud,
    pointcloud_tgt,
    normals=None,
    normals_tgt=None,
    thresholds=np.linspace(1.0 / 1000, 1, 1000),
):
    pointcloud = np.asarray(pointcloud)
    pointcloud_tgt = np.asarray(pointcloud_tgt)
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud, normals
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2

    accuracy, accuracy_normals = distance_p2p(
        pointcloud, normals, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2

    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals
    chamferL1 = 0.5 * (completeness + accuracy)

    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    cor = correspondance(pointcloud_tgt, pointcloud)

    out_dict = {
        "completeness": completeness.mean(),
        "accuracy": accuracy.mean(),
        "normals completeness": completeness_normals.mean(),
        "normals accuracy": accuracy_normals.mean(),
        "normals consistency": normals_correctness.mean(),
        "completeness2": completeness2.mean(),
        "accuracy2": accuracy2.mean(),
        "chamfer-L2": chamferL2.mean(),
        "chamfer-L1": chamferL1.mean(),
        "f-score-5": F[4],  # threshold = 0.5%
        "f-score": F[9],  # threshold = 1.0%
        "f-score-15": F[14],  # threshold = 1.5%
        "f-score-20": F[19],  # threshold = 2.0%
        "correspondance": cor,
        "normals_per_point": normals_correctness,
        "chamfer-L2_per_point": chamferL2,
    }

    return out_dict


def eval_metrics(
    pred_meshes: Meshes, goal_meshes: Meshes, loc_scales: list[torch.Tensor]
):
    metrics = None
    for verts_goal, faces_goal, verts_pred, faces_pred, loc_scale in zip(
        goal_meshes.verts_padded(),
        goal_meshes.faces_padded(),
        pred_meshes.verts_padded(),
        pred_meshes.faces_padded(),
        loc_scales,
    ):
        loc, scale = loc_scale
        verts_goal = (verts_goal.cpu().numpy() - loc[None, :]) * (1 / scale)
        verts_pred = (verts_pred.cpu().numpy() - loc[None, :]) * (1 / scale)
        pred_mesh = trimesh.Trimesh(vertices=verts_pred, faces=faces_pred)
        goal_mesh = trimesh.Trimesh(vertices=verts_goal, faces=faces_goal)
        pred_sample, idx = pred_mesh.sample(100000, return_index=True)
        pred_normals = pred_mesh.face_normals[idx]
        goal_sample, idx = goal_mesh.sample(100000, return_index=True)
        goal_normals = goal_mesh.face_normals[idx]
        result_dict = eval_pointcloud(
            pred_sample, goal_sample, pred_normals, goal_normals
        )
        if metrics is None:
            metrics = {k: [v] for k, v in result_dict.items()}
        else:
            for k, v in result_dict.items():
                metrics[k].append(v)
    return metrics


def eval_meshes(
    pred_meshes: Meshes, goal_meshes: Meshes, loc_scales: list[torch.Tensor]
):

    metrics = eval_metrics(pred_meshes, goal_meshes, loc_scales)
    return {k: np.array(v).mean() for k, v in metrics.items()}
