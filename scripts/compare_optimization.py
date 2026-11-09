"""Paired optimization benchmark with cached inputs and reconstruction evaluation."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from pytorch3d.structures import Meshes

from Main import build_parser, _seed_everything
from scripts.benchmark_optimization import InputDataset, Runtime, ProgressSink
from src.network.geometry_utils import compute_keyframe, init_surf
from src.utilities.util import initialize_meshes, scale_points


def freeze_reference(root, revision=None):
    """Freeze current source or a Git revision, adding timing hooks if needed."""
    destination = root / "reference" / "baseline_src"
    if destination.exists():
        raise FileExistsError(
            f"Reference already exists: {destination}; choose another --root"
        )
    repository = Path(__file__).resolve().parents[1]
    if revision is None:
        shutil.copytree(
            repository / "src",
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.so"),
        )
    else:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=repository,
            text=True,
        ).strip()
        files = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", commit, "--", "src"],
            cwd=repository,
            text=True,
        ).splitlines()
        for name in files:
            target = destination / Path(name).relative_to("src")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(
                    ["git", "show", f"{commit}:{name}"], cwd=repository
                )
            )
        (destination.parent / "revision.txt").write_text(commit + "\n")
    optimizer = destination / "optimization.py"
    source = optimizer.read_text()
    if "runtime=None" not in source:
        # cvpr-state predates the observer API. Add timing only; leave every
        # computation and optimizer update intact.
        replacements = (
            (
                "        q_progress: Queue,\n",
                "        q_progress: Queue,\n        runtime=None,\n",
            ),
            (
                "        for epoch in range(self.epochs):",
                "        if runtime is not None:\n            runtime.start()\n        for epoch in range(self.epochs):",
            ),
            (
                "        final_vertices = []",
                "            if runtime is not None:\n                runtime.epoch_end()\n        if runtime is not None:\n            runtime.stop()\n        final_vertices = []",
            ),
        )
        for before, after in replacements:
            if source.count(before) != 1:
                raise ValueError("Reference source has an unsupported timing layout")
            source = source.replace(before, after)
        optimizer.write_text(source)
    for path in destination.rglob("*.py"):
        path.write_text(
            path.read_text()
            .replace("from src.", "from baseline_src.")
            .replace("import src.", "import baseline_src.")
        )


def prepare(sequence, root, config):
    cache = root / "inputs" / f"{sequence.name}.pt"
    if cache.exists():
        # Locally generated caches contain argparse namespaces and are trusted.
        return torch.load(cache, weights_only=False)
    _seed_everything(0)
    args = build_parser().parse_args([])
    args.device = "cuda:0"
    args.method_args = copy.deepcopy(config)
    args.io_args = {
        "input_directory": str(sequence),
        "directory": sequence.name,
        "out_path": str(root / "inputs" / sequence.name),
        "noise": 0.0,
    }
    Path(args.io_args["out_path"]).mkdir(parents=True, exist_ok=True)
    data = InputDataset(args).as_dict()
    args.T = len(data["points"])
    points, args.points_min, args.points_max = scale_points(data["points"])
    keyframe = compute_keyframe(points.squeeze(), method=args.keyframe)
    args.method_args["keyframe_index"] = keyframe
    vertices, faces = init_surf(
        points[keyframe].squeeze()[..., :3], data["normals"][keyframe], args
    )
    vertices, faces = initialize_meshes(args, vertices, faces)
    data["points"] = points
    payload = dict(
        args=args,
        data=data,
        vertices=vertices,
        faces=faces,
        rng_state=torch.get_rng_state(),
        cuda_rng_state=torch.cuda.get_rng_state(),
    )
    torch.save(payload, cache)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path.home() / "AMA")
    parser.add_argument(
        "--root", type=Path, default=Path("results/optimization_comparison")
    )
    parser.add_argument(
        "--input-cache-root",
        type=Path,
        help="Reuse a different experiment's sampled inputs and initial meshes",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--freeze-reference",
        action="store_true",
        help="Snapshot source as reference (never overwrites an existing snapshot)",
    )
    parser.add_argument(
        "--reference-revision",
        help="Git revision to snapshot, e.g. cvpr-state; requires --freeze-reference",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--performance", type=Path)
    opts = parser.parse_args()
    if opts.limit < 1 or (opts.epochs is not None and opts.epochs < 1):
        parser.error("limit and epochs must be positive")
    if opts.reference_revision and not opts.freeze_reference:
        parser.error("--reference-revision requires --freeze-reference")
    if Path(opts.label).name != opts.label or opts.label in (".", ".."):
        parser.error("label must be a single directory name")
    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA")
    opts.dataset = opts.dataset.expanduser()
    if not opts.dataset.is_dir():
        parser.error(f"Dataset directory does not exist: {opts.dataset}")
    if (opts.root / opts.label / "results.json").exists():
        parser.error("Results already exist for this label; use a new --label")
    torch.set_num_threads(8)
    torch.cuda.set_device(0)
    opts.root.mkdir(parents=True, exist_ok=True)
    if opts.freeze_reference:
        freeze_reference(opts.root, opts.reference_revision)
    if opts.reference and not (opts.root / "reference" / "baseline_src").is_dir():
        parser.error(
            "No frozen baseline exists. Use --freeze-reference before editing source code."
        )
    if opts.reference:
        sys.path.insert(0, str(opts.root / "reference"))
        Optimization = importlib.import_module("baseline_src.optimization").Optimization
        make_grid = importlib.import_module("baseline_src.opt_run")._make_grid
    else:
        from src.optimization import Optimization
        from src.opt_run import _make_grid as make_grid
    config = json.loads(Path("configs/method/fit_.json").read_text())
    performance = json.loads(opts.performance.read_text()) if opts.performance else {}
    sequences = sorted(p for p in opts.dataset.iterdir() if p.is_dir())[: opts.limit]
    if not sequences:
        parser.error("Dataset contains no sequence directories")
    destination = opts.root / opts.label
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(
        label=opts.label,
        reference=opts.reference,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cpu_threads=8,
        config=config,
        performance=performance,
        evaluation_samples_per_mesh=100000,
        runs=[],
    )
    for index, sequence in enumerate(sequences):
        print(f"[{index+1}/{len(sequences)}] {sequence.name}", flush=True)
        payload = prepare(sequence, opts.input_cache_root or opts.root, config)
        args, data = copy.deepcopy(payload["args"]), payload["data"]
        if opts.epochs is not None:
            # Short experiments retain the full-run smoothing schedule.
            class ShortOptimization(Optimization):
                @property
                def epochs(self):
                    return opts.epochs

            optimization_class = ShortOptimization
        else:
            optimization_class = Optimization
        args.method_args["performance"] = performance
        args.io_args["out_path"] = str(destination / sequence.name)
        torch.set_rng_state(payload["rng_state"])
        torch.cuda.set_rng_state(payload["cuda_rng_state"])
        started = time.perf_counter()
        mlp = args.method_args["mlp"]
        grids = [
            make_grid(args, mlp["point_dim"]),
            make_grid(args, mlp["normal_dim"], normals=True),
        ]
        optimizer = optimization_class(args)
        runtime = Runtime(args.device, opts.diagnostic)
        torch.cuda.synchronize()
        setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        mesh = optimizer(
            grids,
            payload["vertices"].cuda(),
            payload["faces"].cuda(),
            data,
            ProgressSink(),
            runtime=runtime,
        )
        torch.cuda.synchronize()
        call_seconds = time.perf_counter() - started
        predicted = mesh.verts_padded()
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("Nonfinite reconstructed vertices")
        torch.save(
            dict(vertices=predicted, faces=mesh.faces_padded()),
            destination / f"{sequence.name}.pt",
        )
        row = dict(
            sequence=sequence.name,
            frames=args.T,
            vertices=len(payload["vertices"]),
            epochs=optimizer.epochs,
            optimization_seconds=runtime.seconds,
            setup_seconds=setup_seconds,
            call_seconds=call_seconds,
            epoch_ms=runtime.epoch_ms,
            peak_allocated_mib=runtime.peak_memory_mib,
        )
        row["network_optimizer_steps"] = sorted(
            {
                int(state["step"].item())
                for state in optimizer.optimizer.state.values()
                if "step" in state
            }
        )
        if row["network_optimizer_steps"] != [optimizer.epochs]:
            raise RuntimeError(
                f"Unexpected optimizer update count: {row['network_optimizer_steps']}"
            )
        if not opts.reference:
            optimizers = [optimizer.optimizer] + [
                g.optimizer for g in grids if hasattr(g, "optimizer")
            ]
            parameters = [
                p
                for opt in optimizers
                for group in opt.param_groups
                for p in group["params"]
            ]
            states = [
                value
                for opt in optimizers
                for state in opt.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ]
            row["parameter_dtypes"] = sorted({str(p.dtype) for p in parameters})
            row["optimizer_state_dtypes"] = sorted(
                {str(value.dtype) for value in states}
            )
            row["parameters_finite"] = all(
                bool(torch.isfinite(p).all()) for p in parameters
            )
            row["optimizer_states_finite"] = all(
                bool(torch.isfinite(value).all()) for value in states
            )
            row["loss_scale"] = optimizer.loss_scale
            if not row["parameters_finite"] or not row["optimizer_states_finite"]:
                raise FloatingPointError("Nonfinite parameters or optimizer state")
        if opts.diagnostic:
            row["phase_seconds"] = dict(runtime.phases)
        print(
            f"  optimization {runtime.seconds:.3f}s; setup {setup_seconds:.3f}s; call {call_seconds:.3f}s",
            flush=True,
        )
        if not opts.skip_evaluation:
            from src.utilities.eval_utils import eval_meshes, get_loc_scale

            # Match the production output normalization without writing OBJ files.
            world = predicted / 0.95 * (args.points_max - args.points_min).max() / 2
            world = world + (args.points_max + args.points_min) / 2
            world_mesh = Meshes(verts=world, faces=mesh.faces_padded())
            ground_truth = Meshes(
                verts=data["gt_points"],
                faces=data["gt_faces"],
                verts_normals=data["gt_normals"],
            )
            loc_scales = [get_loc_scale(frame) for frame in ground_truth]
            np.random.seed(271828 + index)
            started = time.perf_counter()
            metrics = eval_meshes(world_mesh, ground_truth, loc_scales)
            row["metrics"] = {
                key: float(metrics[key])
                for key in ["chamfer-L2", "normals consistency", "f-score-5"]
            }
            row["evaluation_seconds"] = time.perf_counter() - started
            print(
                f"  metrics {row['metrics']} ({row['evaluation_seconds']:.2f}s outside timer)",
                flush=True,
            )
            del ground_truth, world_mesh, metrics
        report["runs"].append(row)
        (destination / "results.json").write_text(json.dumps(report, indent=2))
        del mesh, predicted, optimizer, grids, payload, data
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Saved {destination / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Native pcgrid loads cholespy, which can crash during interpreter teardown.
    # Use normal teardown for the default separable backend.
    if any(name == "pcgrid" or name.startswith("pcgrid.") for name in sys.modules):
        os._exit(0)
