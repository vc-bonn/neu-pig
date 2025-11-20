import os
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from pytorch3d.loss import chamfer_distance

from gpytoolbox import remesh_botsch
import open3d as o3d
import numpy as np
import trimesh
from pytorch3d.ops import sample_points_from_meshes
from contextlib import contextmanager

import datetime
from pathlib import Path

import os, traceback, getpass, sys
from pathlib import Path
import importlib
import sys


@contextmanager
def temp_sys_path(path):
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.pop(0)


def mkdirs(path: str) -> None:
    mkdir_path = ""
    if path[0] == "/":
        mkdir_path += "/"
    for directory in path.split("/"):
        if directory == "":
            continue
        mkdir_path += directory + "/"
        if not os.path.isdir(mkdir_path):
            os.mkdir(mkdir_path)


def debug_path_info(path):
    try:
        p = Path(path).expanduser().resolve()
        print("DEBUG writer path:", str(p))
        print(" exists:", p.exists())
        print(" is_file:", p.is_file())
        print(" is_dir:", p.is_dir())
        print(" cwd:", Path.cwd())
        print(" user:", getpass.getuser())
        try:
            st = p.parent.stat()
            print(" parent mode:", oct(st.st_mode & 0o777), "uid:", st.st_uid)
        except Exception as e:
            print(" parent stat failed:", e)
        print(" os.access(write):", os.access(p.parent, os.W_OK))
        print(" listed:", list(p.parent.iterdir())[:20])
    except Exception:
        traceback.print_exc()


def scale_points(points):
    points = points.squeeze()[..., :3]
    if points.dim() == 2:
        points = points[None, :, :]
    # points_min = (points.min(dim=0)[0]).min(dim=0)[0]
    # points_max = (points.max(dim=0)[0]).max(dim=0)[0]
    points_min = torch.stack(
        [
            points[..., 0].min(),
            points[..., 1].min(),
            points[..., 2].min(),
        ]
    )
    points_max = torch.stack(
        [
            points[..., 0].max(),
            points[..., 1].max(),
            points[..., 2].max(),
        ]
    )

    points = scale(points, points_min, points_max)
    points = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    return points[:, None, None, :, :], points_min, points_max


def scale(points: torch.Tensor, points_min, points_max):
    points -= (points_max + points_min) / 2
    # points /= (points_max - points_min) / 2
    points /= (points_max - points_min).max() / 2
    points *= 0.95
    return points


def initialize_meshes(args, verts, faces):
    save_obj(
        os.path.join(
            args.io_args["out_path"],
            "init_geometry_{}.obj".format(verts.shape[0]),
        ),
        verts,
        faces,
    )

    verts = verts.cpu().double().numpy()
    faces = faces.cpu().to(torch.int32).numpy()

    # while True:
    # verts, faces = remesh_botsch(
    #     verts,
    #     faces,
    #     5,
    #     None,
    #     True,
    # )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces))
    mesh.compute_vertex_normals()

    mesh = mesh.simplify_quadric_decimation(
        target_number_of_triangles=min(20000, args.number_points * 4)
    )
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles)

    # mesh_1 = Meshes(
    #     torch.tensor(v).to(torch.float32)[None],
    #     torch.tensor(f).to(torch.int32)[None],
    # )
    # mesh_2 = Meshes(
    #     torch.tensor(verts).to(torch.float32)[None],
    #     torch.tensor(faces).to(torch.int32)[None],
    # )
    # p_1 = sample_points_from_meshes(mesh_1, num_samples=100000)
    # p_2 = sample_points_from_meshes(mesh_2, num_samples=100000)

    # distance, _ = chamfer_distance(p_1, p_2)
    # if distance < 5e-4:
    #     break
    # else:
    #     print(
    #         "Mesh Simplification Failed [Open3D error], repeating mesh initialization"
    #     )

    # verts, faces = remesh_botsch(
    #     v,
    #     f,
    #     5,
    #     None,
    #     True,
    # )
    verts = torch.from_numpy(v).to(torch.float32)
    faces = torch.from_numpy(f).to(torch.int64)

    save_obj(
        os.path.join(
            args.io_args["out_path"],
            "init_geometry_{}.obj".format(verts.shape[0]),
        ),
        verts,
        faces,
    )
    return verts, faces


def edgelength(v: torch.Tensor, f: torch.Tensor):
    """assumes all vertices v and faces f are from the same mesh, with v being transformed, keeping the same connectivity f"""
    if v.dim() == 2:
        v = v[None]
    if f.dim() == 2:
        f = f[None]
    assert v.dim() == 3 and f.dim() == 3 and v.shape[0] == f.shape[0]
    if v.shape[-1] == 4:
        v = (v[..., :3] / v[..., 3:])[..., :3]

    mesh = Meshes(
        verts=v,
        faces=f,
    )

    edges_packed = mesh[0].edges_packed()
    verts_padded = mesh.verts_padded()
    verts_edges = verts_padded[:, edges_packed]

    v0, v1 = verts_edges.unbind(-2)
    return (v0 - v1).norm(dim=-1, p=2)



def debug_import(module_name):
    # 1. Check if the module is installed
    try:
        import module_name
        print(f"Module '{module_name}' is installed and available.")
    except ImportError as e:
        print(f"Error importing '{module_name}': {e}")
        
        # 2. Check if the module is in the sys.path
        print("\nChecking sys.path...")
        for path in sys.path:
            print(f"Checking path: {path}")
        
        # 3. Look for the module file manually
        print(f"\nLooking for '{module_name}' in your project directory...")
        possible_paths = []
        for root, dirs, files in os.walk('.'):
            if module_name in files:
                possible_paths.append(root)
        
        if possible_paths:
            print(f"Found '{module_name}' in the following directories:")
            for path in possible_paths:
                print(f"- {path}")
        else:
            print(f"'{module_name}' was not found in your current directory.")

        # 4. Check if module is a package or submodule
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                print(f"Module '{module_name}' found in {spec.origin}")
            else:
                print(f"Module '{module_name}' is not found anywhere in the PYTHONPATH.")
        except Exception as ex:
            print(f"Error in checking module spec: {ex}")