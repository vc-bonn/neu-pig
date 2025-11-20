import os
import numpy as np
from tqdm import tqdm
from utils import eval_correspondences_mesh, eval_correspondences_mesh2
import trimesh

import sys

sys.path.append("/data/kaltheuner/git/preconditioned-deformation-nets/")

# pred_path='../ours/DT4D/'
# gt_path='../preprocessed-data/DT4D/'

# pred_path='../ours/DFAUST/'
# gt_path='../preprocessed-data/DFAUST/'

m2v = False
dyno = False
#
#
#
#
#
#
#
#
#
#

corr_error_set = []

OURS = False
T = 17

pred_path = "/data/kaltheuner/CVPR_2026/supplemental/grid_level/twelve/Date2025-11-17_Time08-17-44"
gt_path = "/data/kaltheuner/preprocessed-data/AMA"
seq_list = [
    o for o in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, o))
]
for seq_name in tqdm(seq_list):
    pred_file_list = []
    gt_pc_list = []
    gt_names = []

    for t in range(T):
        if seq_name[-7:] == "surface" and not dyno:
            dirs = os.listdir(os.path.join(pred_path, seq_name))
            dirs.sort()
            pred_file_list.append(
                os.path.join(pred_path, seq_name, dirs[-1], "%04d.obj" % t)
            )
        elif m2v:
            pred_file_list.append(
                os.path.join(
                    pred_path, seq_name, "0", "vis", "corr_map", "{}_pred.obj".format(t)
                )
            )
        elif dyno:
            pred_file_list.append(
                os.path.join(pred_path, seq_name, "meshes", "%04d.obj" % t)
            )
        else:
            pred_file_list.append(os.path.join(pred_path, seq_name, "%04d.obj" % t))

        if seq_name[-7:] == "surface" and not OURS:
            gt_names.append(
                os.path.join(gt_path, seq_name[:-8], "pcl_seqs/%04d.npy" % t)
            )
            gt_data = np.load(
                os.path.join(
                    os.path.join(gt_path, seq_name[:-8], "pcl_seqs/%04d.npy" % t)
                )
            )
        elif m2v:
            dirs = os.listdir(gt_path)
            p = [d for d in dirs if d.startswith("test_" + seq_name)][0]
            gt_names.append(
                os.path.join(os.path.join(gt_path, p, "pcl_seqs/%04d.npy" % t))
            )
            gt_data = np.load(
                os.path.join(os.path.join(gt_path, p, "pcl_seqs/%04d.npy" % t))
            )
        elif OURS:
            if seq_name[-7:] == "surface":
                gt_names.append(os.path.join(gt_path, seq_name[:-8], "%04d.obj" % t))
                gt_data = trimesh.load(
                    os.path.join(os.path.join(gt_path, seq_name[:-8], "%04d.obj" % t))
                )
            else:
                gt_names.append(os.path.join(gt_path, seq_name, "%04d.obj" % t))
                gt_data = trimesh.load(
                    os.path.join(os.path.join(gt_path, seq_name, "%04d.obj" % t))
                )
        else:
            gt_names.append(os.path.join(gt_path, seq_name, "pcl_seqs/%04d.npy" % t))
            gt_data = np.load(
                os.path.join(os.path.join(gt_path, seq_name, "pcl_seqs/%04d.npy" % t))
            )
        gt_pc_list.append(gt_data)

    if OURS:
        n_samples = 10000
        samples = np.random.choice(
            gt_pc_list[0].vertices.shape[0],
            size=min(gt_pc_list[0].vertices.shape[0], n_samples),
            replace=False,
        )
        gt_pc_list = [d.vertices[samples] for d in gt_pc_list]
    pred_meshes = [trimesh.load(p) for p in pred_file_list]
    corr_error = eval_correspondences_mesh(pred_file_list, gt_pc_list)

    corr_error_set.append(corr_error)

print("corr error: ", np.array(corr_error_set).mean())
