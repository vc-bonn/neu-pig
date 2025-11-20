import torch
import os
import pathlib
import importlib.util
import tempfile
import json
import trimesh

from pytorch3d.io import load_objs_as_meshes, save_obj

OUT_DIR = pathlib.Path("/data/kaltheuner/CVPR_2026/related_work/dynosurf/")

SEQ_BASE = "/data/kaltheuner/processed_data/AMA/"
SEQ_PATHS = [os.path.join(SEQ_BASE, o) for o in os.listdir(SEQ_BASE) if os.path.isdir(os.path.join(SEQ_BASE, o))]

def write_mesh_with_normals(verts: torch.Tensor, faces: torch.Tensor, out_path: str):
    """
    Write mesh with per-vertex normals to disk.
    verts: (V,3) float tensor
    faces: (F,3) long tensor (triangle indices)
    out_path: path ending with .ply or .obj
    """
    # Ensure correct shapes & device
    assert verts.ndim == 2 and verts.size(1) == 3
    assert faces.ndim == 2 and faces.size(1) == 3

    device = verts.device
    verts = verts.float()
    faces = faces.long()

    # 1) compute face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (F,3)
    # normalize face normals (avoid div by zero)
    fnorm = face_normals.norm(dim=1, keepdim=True)
    fnorm = torch.where(fnorm == 0, torch.ones_like(fnorm), fnorm)
    face_normals = face_normals / fnorm

    # 2) accumulate face normals to vertices (area weighting could be added by using magnitude)
    vertex_normals = torch.zeros_like(verts, device=device)  # (V,3)
    # scatter_add: add each face normal to its 3 vertices
    faces_exp = faces.unsqueeze(-1).expand(-1, -1, 3)  # (F,3,3) for broadcasting
    # Add face_normals to each vertex index
    # Using scatter_add_ over flattened indices:
    # For each corner in faces, add face_normals
    # shape prep:
    face_normals_expanded = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)  # (F*3,3)
    vert_indices = faces.reshape(-1)  # (F*3,)
    vertex_normals = vertex_normals.index_add(0, vert_indices, face_normals_expanded)

    # 3) normalize vertex normals
    vnorm = vertex_normals.norm(dim=1, keepdim=True)
    vnorm = torch.where(vnorm == 0, torch.ones_like(vnorm), vnorm)
    vertex_normals = vertex_normals / vnorm

    # 4) convert to numpy and export with trimesh
    verts_np = verts.cpu().numpy()
    faces_np = faces.cpu().numpy()
    normals_np = vertex_normals.cpu().numpy()

    tm = trimesh.Trimesh(vertices=verts_np, faces=faces_np,
                         vertex_normals=normals_np, process=False)

    # Export: format chosen by extension; trimesh deduces format (.ply or .obj)
    tm.export(out_path)
    print(f"Wrote mesh to {out_path} (vertices={verts_np.shape[0]}, faces={faces_np.shape[0]})")


def prepare_objs(objs:list[pathlib.Path]):
    meshes = load_objs_as_meshes([o.as_posix() for o in objs], device="cpu")
    for o, mesh in zip(objs, meshes):
        write_mesh_with_normals(mesh.verts_padded()[0], mesh.faces_padded()[0], o.as_posix())


def prepare_files(path:pathlib.Path, input:pathlib.Path):
    """Prepare necessary files for DynoSurf from the given sequence path."""
    input_json = path / "inputs.json"
    
    objs = [o.as_posix() for o in sorted(list(pathlib.Path(input,"gt").glob("*.obj")))]
    prepare_objs([pathlib.Path(o) for o in objs])
    with open(input_json, "w") as f:
        json.dump(
            {"fitting_point_clouds": objs},
            f
        )
    return input_json

if __name__ == "__main__":
    if not os.path.exists("ext/dynosurf"):
        repo_url = "https://github.com/yaoyx689/DynoSurf.git"
        clone_dir = "ext/dynosurf"
        if importlib.util.find_spec("git") is None:
            raise ImportError("GitPython is not installed. Please install it to clone the DynoSurf repository.")
        if importlib.util.find_spec("kaolin") is None:
            raise ImportError("Kaolin is not installed. Please install all Dynosurf Modules.")
        import git
        git.Repo.clone_from(repo_url, clone_dir)

    os.makedirs(OUT_DIR, exist_ok=True)
    for seq_path in SEQ_PATHS:
        # Run DynoSurf on each sequence path
        seq_path = pathlib.Path(seq_path)
        dirs = [pathlib.Path(seq_path,o) for o in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, o))]
        os.makedirs(OUT_DIR / seq_path.name, exist_ok=True)
        for d in dirs:
            os.makedirs(OUT_DIR / seq_path.name / d.name, exist_ok=True)
            input_file = prepare_files(OUT_DIR / seq_path.name / d.name, d)

            os.system(f"python ext/dynosurf/code/run_trainer.py --conf ext/dynosurf/confs/base.conf --input_file {input_file.as_posix()} --logs_path {(OUT_DIR / seq_path.name / d.name).as_posix()}")