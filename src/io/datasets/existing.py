from pathlib import Path

import torch
from pytorch3d.io import load_objs_as_meshes, load_ply, save_ply
from pytorch3d.ops import estimate_pointcloud_normals, sample_points_from_meshes
from torch.utils.data import Dataset


def _frame_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []

    def sort_key(path: Path) -> tuple[int, int | str]:
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.stem)

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == suffix
        ),
        key=sort_key,
    )


class ExistingDataset(Dataset):
    """Load one complete point-cloud/mesh sequence."""

    def __init__(self, args) -> None:
        self.args = args
        input_directory = Path(self.io_args["input_directory"])

        if args.target == "obj":
            target_files = self._required_files(input_directory / "gt", ".obj")
            target_meshes = load_objs_as_meshes(
                [str(path) for path in target_files], load_textures=False
            )
            self.points, self.normals = sample_points_from_meshes(
                target_meshes, args.number_points, return_normals=True
            )
            gt_files = _frame_files(input_directory, ".obj")
            gt_meshes = (
                load_objs_as_meshes(
                    [str(path) for path in gt_files], load_textures=False
                )
                if gt_files
                else target_meshes
            )
        elif args.target == "ply":
            point_files = self._required_files(
                input_directory / "point_clouds", ".ply"
            )
            point_data = [load_ply(str(path))[0].float() for path in point_files]
            self.points = torch.stack(point_data)
            self.normals = estimate_pointcloud_normals(
                self.points, neighborhood_size=16
            )
            gt_files = _frame_files(input_directory, ".obj")
            if not gt_files:
                gt_files = self._required_files(input_directory / "gt", ".obj")
            gt_meshes = load_objs_as_meshes(
                [str(path) for path in gt_files], load_textures=False
            )
        else:
            raise ValueError(f"Unsupported target type: {args.target}")

        self.points = self.points.float()
        self.normals = self.normals.float()
        self.gt_points = gt_meshes.verts_padded().float()
        self.gt_normals = gt_meshes.verts_normals_padded().float()
        self.gt_faces = gt_meshes.faces_padded()

        self._add_noise(args.noise / 100.0)
        self._save_inputs()

    @property
    def io_args(self) -> dict:
        return self.args.io_args

    @staticmethod
    def _required_files(directory: Path, suffix: str) -> list[Path]:
        files = _frame_files(directory, suffix)
        if not files:
            raise FileNotFoundError(f"No {suffix} files found in {directory}")
        return files

    def _add_noise(self, noise_fraction: float) -> None:
        if noise_fraction <= 0:
            return

        xyz = self.points[..., :3]
        extent = xyz.amax(dim=(0, 1)) - xyz.amin(dim=(0, 1))
        xyz.add_(extent.norm() * torch.randn_like(xyz) * noise_fraction)

        noise_directory = Path(self.io_args["out_path"]) / "input_meshes_noisy"
        noise_directory.mkdir(parents=True, exist_ok=True)
        for index, points in enumerate(self.points):
            save_ply(
                str(noise_directory / f"{index:04d}.ply"),
                verts=points[..., :3],
                faces=None,
            )

    def _save_inputs(self) -> None:
        output_directory = Path(self.io_args["out_path"]) / "input_meshes"
        output_directory.mkdir(parents=True, exist_ok=True)
        for index, (points, normals) in enumerate(zip(self.points, self.normals)):
            save_ply(
                str(output_directory / f"{index:04d}.ply"),
                verts=points,
                verts_normals=normals,
                faces=None,
            )

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the full sequence without DataLoader collation/copies."""
        return {
            "points": self.points,
            "normals": self.normals,
            "gt_points": self.gt_points,
            "gt_normals": self.gt_normals,
            "gt_faces": self.gt_faces,
        }

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.as_dict().items()}


# Backward-compatible name for older imports.
existingDataset = ExistingDataset
