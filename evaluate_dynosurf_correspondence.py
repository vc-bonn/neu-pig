"""Re-evaluate optimized Neu-PiG meshes with DynoSurf correspondence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from ext.dynosurf.evaluation.utils import eval_correspondences_mesh


DEFAULT_RUN_ROOT = Path("/data/kaltheuner/CVPR_2026/ablations/steps/120/Date2025-10-30_Time09-04-35")
DEFAULT_GT_ROOT = Path("/data/kaltheuner/processed_data/AMA/120")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Neu-PiG output directory with DynoSurf's original "
            "temporal correspondence metric."
        )
    )
    parser.add_argument(
        "run_root",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=f"Run directory containing sequence/meshes folders (default: {DEFAULT_RUN_ROOT})",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_ROOT,
        help=f"Dataset root containing sequence/pcl_seqs folders (default: {DEFAULT_GT_ROOT})",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        dest="sequences",
        help="Evaluate only this sequence; repeat the option for multiple sequences",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output JSON path (default: RUN_ROOT/DynoSurfCorrespondence.json, "
            "with a reference-mode suffix when not using 'first')"
        ),
    )
    parser.add_argument(
        "--reference-frame",
        choices=("first", "middle"),
        default="first",
        help=(
            "Frame used to establish correspondence; 'middle' uses index T//2 "
            "and evaluates every other frame (default: first)"
        ),
    )
    parser.add_argument(
        "--update-metrics",
        action="store_true",
        help="Replace 'correspondance' in each sequence's Metrics.json",
    )
    return parser.parse_args()


def numeric_frame_files(directory: Path, suffix: str) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == suffix and path.stem.isdigit()
        ),
        key=lambda path: int(path.stem),
    )


def mesh_directory(sequence_dir: Path) -> Path | None:
    nested_directory = sequence_dir / "meshes"
    if nested_directory.is_dir():
        return nested_directory
    if sequence_dir.is_dir() and numeric_frame_files(sequence_dir, ".obj"):
        return sequence_dir
    return None


def sequence_directories(run_root: Path, names: list[str] | None) -> list[Path]:
    if names:
        directories = [run_root / name for name in names]
    else:
        directories = sorted(
            path
            for path in run_root.iterdir()
            if path.is_dir() and mesh_directory(path) is not None
        )

    missing = [str(path) for path in directories if mesh_directory(path) is None]
    if missing:
        raise FileNotFoundError(
            "No numeric OBJ frames found in these sequence directories: "
            + ", ".join(missing)
        )
    if not directories:
        raise FileNotFoundError(
            f"No sequences with numeric OBJ frames found in {run_root}"
        )
    return directories


def load_gt_pointclouds(
    sequence_dir: Path, gt_root: Path, mesh_files: list[Path]
) -> tuple[list[np.ndarray], Path]:
    gt_sequence_dir = gt_root / sequence_dir.name
    pcl_directory = gt_sequence_dir / "pcl_seqs"
    if pcl_directory.is_dir():
        gt_pointclouds = []
        for mesh_file in mesh_files:
            gt_file = pcl_directory / f"{mesh_file.stem}.npy"
            if not gt_file.is_file():
                raise FileNotFoundError(
                    f"No GT point cloud for {sequence_dir.name} frame "
                    f"{mesh_file.stem}: {gt_file}"
                )
            pointcloud = np.asarray(np.load(gt_file), dtype=np.float32)
            if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
                raise ValueError(
                    f"Expected {gt_file} to have shape (N, 3), "
                    f"got {pointcloud.shape}"
                )
            gt_pointclouds.append(pointcloud)
        return gt_pointclouds, pcl_directory

    mesh_gt_directory = gt_sequence_dir / "gt"
    if not mesh_gt_directory.is_dir():
        mesh_gt_directory = sequence_dir / "gt"
    if not mesh_gt_directory.is_dir():
        raise FileNotFoundError(
            f"No GT pcl_seqs or gt mesh directory found for {sequence_dir.name}; "
            f"checked {pcl_directory}, {gt_sequence_dir / 'gt'}, and "
            f"{sequence_dir / 'gt'}"
        )

    gt_pointclouds = []
    num_gt_points = None
    for mesh_file in mesh_files:
        gt_file = mesh_gt_directory / f"{mesh_file.stem}.obj"
        fallback_gt_file = gt_sequence_dir / f"{mesh_file.stem}.obj"
        if (not gt_file.is_file() or gt_file.stat().st_size == 0) and (
            fallback_gt_file.is_file() and fallback_gt_file.stat().st_size > 0
        ):
            gt_file = fallback_gt_file
        if not gt_file.is_file() or gt_file.stat().st_size == 0:
            raise FileNotFoundError(
                f"No non-empty GT mesh for {sequence_dir.name} frame "
                f"{mesh_file.stem}; checked {mesh_gt_directory} and "
                f"{gt_sequence_dir}"
            )
        gt_mesh = trimesh.load_mesh(gt_file, process=False)
        pointcloud = np.asarray(gt_mesh.vertices, dtype=np.float32)
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            raise ValueError(
                f"Expected vertices from {gt_file} to have shape (N, 3), "
                f"got {pointcloud.shape}"
            )
        if num_gt_points is None:
            num_gt_points = len(pointcloud)
        elif len(pointcloud) != num_gt_points:
            raise ValueError(
                f"GT vertex count changes at {gt_file}; temporal identities "
                "cannot be tracked"
            )
        gt_pointclouds.append(pointcloud)
    return gt_pointclouds, mesh_gt_directory


def evaluate_sequence(
    sequence_dir: Path, gt_root: Path, reference_frame: str
) -> tuple[float, int, str, Path]:
    mesh_dir = mesh_directory(sequence_dir)
    if mesh_dir is None:
        raise FileNotFoundError(
            f"No numeric OBJ frames found for sequence {sequence_dir.name}"
        )
    mesh_files = numeric_frame_files(mesh_dir, ".obj")
    if len(mesh_files) < 2:
        raise ValueError(
            f"{sequence_dir.name} requires at least two numeric OBJ mesh frames"
        )

    gt_pointclouds, gt_source = load_gt_pointclouds(
        sequence_dir, gt_root, mesh_files
    )

    reference_index = 0 if reference_frame == "first" else len(mesh_files) // 2
    evaluation_order = [reference_index] + [
        index for index in range(len(mesh_files)) if index != reference_index
    ]
    ordered_mesh_files = [mesh_files[index] for index in evaluation_order]
    ordered_gt_pointclouds = [gt_pointclouds[index] for index in evaluation_order]

    value = eval_correspondences_mesh(
        [str(mesh_file) for mesh_file in ordered_mesh_files], ordered_gt_pointclouds
    )
    if not np.isfinite(value):
        raise ValueError(f"DynoSurf returned a non-finite value for {sequence_dir.name}")
    return float(value), len(mesh_files), mesh_files[reference_index].stem, gt_source


def update_metrics(sequence_dir: Path, value: float) -> None:
    metrics_path = sequence_dir / "Metrics.json"
    metrics = {}
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)
    metrics["correspondance"] = value
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    gt_root = args.gt_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_root}")
    if not gt_root.is_dir():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    results = {}
    sequence_dirs = sequence_directories(run_root, args.sequences)
    for index, sequence_dir in enumerate(sequence_dirs, start=1):
        value, num_frames, reference_frame, gt_source = evaluate_sequence(
            sequence_dir, gt_root, args.reference_frame
        )
        results[sequence_dir.name] = {
            "correspondance": value,
            "frames": num_frames,
            "reference_frame": reference_frame,
            "gt_source": str(gt_source),
        }
        if args.update_metrics:
            update_metrics(sequence_dir, value)
        print(
            f"[{index:>2}/{len(sequence_dirs)}] {sequence_dir.name}: "
            f"{value:.10f} ({num_frames} frames, reference {reference_frame})"
        )

    mean_value = float(
        np.mean([result["correspondance"] for result in results.values()])
    )
    output = {
        "metric": "DynoSurf eval_correspondences_mesh",
        "run_root": str(run_root),
        "gt_root": str(gt_root),
        "reference_mode": args.reference_frame,
        "correspondance": mean_value,
        "sequences": results,
    }
    default_output_name = "DynoSurfCorrespondence.json"
    if args.reference_frame != "first":
        default_output_name = f"DynoSurfCorrespondence_{args.reference_frame}.json"
    output_path = (args.output or run_root / default_output_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)
        output_file.write("\n")

    print(f"Mean DynoSurf correspondence: {mean_value:.10f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
