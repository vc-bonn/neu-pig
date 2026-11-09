"""Time the real optimization loop, without initialization, export or evaluation.

Run from the repository root using the project's Python environment:
  python scripts/benchmark_optimization.py --dataset ~/AMA --limit 10
Baseline epochs use CUDA events without per-epoch synchronization. A separate
short run synchronizes phase boundaries for a diagnostic wall-time breakdown.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from Main import build_parser, _seed_everything
from src.io.datasets.existing import ExistingDataset
from src.network.geometry_utils import compute_keyframe, init_surf
from src.opt_run import _make_grid
from src.optimization import Optimization
from src.utilities.util import initialize_meshes, scale_points


class InputDataset(ExistingDataset):
    def _save_inputs(self):
        # Input export is irrelevant to optimization; avoid thousands of files.
        pass


class Runtime:
    def __init__(self, device, diagnostic=False):
        self.device = device
        self.diagnostic = diagnostic
        self.events = []
        self.phases = defaultdict(float)

    def event(self):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.events.append(event)

    def start(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()
        self.event()

    def epoch_end(self):
        self.event()
        if (len(self.events) - 1) % 100 == 0:
            print(f"  epoch {len(self.events) - 1}", flush=True)

    def stop(self):
        torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.started
        self.epoch_ms = [
            a.elapsed_time(b) for a, b in zip(self.events, self.events[1:])
        ]
        self.peak_memory_mib = torch.cuda.max_memory_allocated(self.device) / 2**20

    def phase(self, name):
        return self._phase(name) if self.diagnostic else nullcontext()

    @contextmanager
    def _phase(self, name):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.profiler.record_function(name):
            yield
        torch.cuda.synchronize(self.device)
        self.phases[name] += time.perf_counter() - started


class ProgressSink:
    def put(self, message):
        # Retain the production loop's periodic scalar transfers, without UI/IPC.
        pass


def run_optimizer(
    args,
    data,
    vertices,
    faces,
    diagnostic=False,
    trace_path=None,
    optimization_class=Optimization,
):
    grids = [
        _make_grid(args, args.method_args["mlp"].get("point_dim", 28)),
        _make_grid(args, args.method_args["mlp"].get("normal_dim", 4), normals=True),
    ]
    optimizer = optimization_class(args)
    vertices, faces = vertices.to(args.device), faces.to(args.device)
    runtime = Runtime(args.device, diagnostic)
    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        if trace_path
        else nullcontext()
    )
    with profiler as prof:
        mesh = optimizer(grids, vertices, faces, data, ProgressSink(), runtime=runtime)
    if trace_path:
        prof.export_chrome_trace(str(trace_path))
        trace_path.with_suffix(".txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        )
    result = {
        "optimization_seconds": runtime.seconds,
        "epoch_mean_ms": statistics.mean(runtime.epoch_ms),
        "epoch_median_ms": statistics.median(runtime.epoch_ms),
        "first_epoch_ms": runtime.epoch_ms[0],
        "steady_epoch_mean_ms": statistics.mean(
            runtime.epoch_ms[min(10, len(runtime.epoch_ms) - 1) :]
        ),
        "peak_allocated_mib": runtime.peak_memory_mib,
        "epoch_ms": runtime.epoch_ms,
    }
    if diagnostic:
        result["phase_seconds"] = dict(runtime.phases)
    del mesh, optimizer, grids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--config", type=Path, default=Path("configs/method/fit_.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("results/runtime_ama"))
    parser.add_argument("--profile-epochs", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    options = parser.parse_args()
    if options.limit < 1 or options.profile_epochs < 0 or options.threads < 1:
        parser.error(
            "limit and threads must be positive; profile-epochs must be nonnegative"
        )
    if not torch.cuda.is_available() or torch.device(options.device).type != "cuda":
        parser.error("This benchmark requires a CUDA device")
    if not options.dataset.expanduser().is_dir():
        parser.error(f"Dataset directory does not exist: {options.dataset}")
    torch.set_num_threads(options.threads)
    torch.cuda.set_device(options.device)
    sequences = sorted(p for p in options.dataset.expanduser().iterdir() if p.is_dir())[
        : options.limit
    ]
    if not sequences:
        parser.error("No sequence directories found")
    options.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(options.config.read_text())
    report = {
        "dataset": str(options.dataset.expanduser().resolve()),
        "selection": "first sequence directories in lexicographic order",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_threads": options.threads,
        "seed": 0,
        "points_per_frame": 5000,
        "config": config,
        "timing_scope": "training loop only; includes batching, forward/loss/backward/grid/Adam updates and periodic scalar transfers; excludes loading, surface/grid/network setup, final prediction, export and evaluation",
        "runs": [],
    }
    report_path = options.output / "runtime.json"
    first_inputs = None
    for index, sequence in enumerate(sequences):
        _seed_everything(0)
        args = build_parser().parse_args([])
        args.device = options.device
        args.method_args = copy.deepcopy(config)
        args.io_args = {
            "input_directory": str(sequence),
            "directory": sequence.name,
            "out_path": str(options.output / sequence.name),
            "noise": 0.0,
        }
        Path(args.io_args["out_path"]).mkdir(parents=True, exist_ok=True)
        print(f"[{index+1}/{len(sequences)}] Preparing {sequence.name}", flush=True)
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
        print(
            f"  {args.T} frames, {len(vertices)} vertices, {config['optimization']['epochs']} epochs",
            flush=True,
        )
        result = run_optimizer(args, data, vertices, faces)
        result.update(
            sequence=sequence.name,
            frames=args.T,
            vertices=len(vertices),
            faces=len(faces),
            keyframe=keyframe,
        )
        report["runs"].append(result)
        report_path.write_text(json.dumps(report, indent=2))
        print(
            f"  optimization: {result['optimization_seconds']:.3f}s ({result['epoch_median_ms']:.3f} ms/epoch)",
            flush=True,
        )
        if index == 0 and options.profile_epochs:
            first_inputs = (copy.deepcopy(args), data, vertices, faces)
    if first_inputs:
        print("Separate diagnostic profile of first sequence", flush=True)
        args, data, vertices, faces = first_inputs

        # Preserve the production 1,000-epoch smoothing schedule during profiling.
        class ProfileOptimization(Optimization):
            @property
            def epochs(self):
                return options.profile_epochs

        _seed_everything(0)
        report["diagnostic"] = run_optimizer(
            args,
            data,
            vertices,
            faces,
            True,
            options.output / "trace.json",
            ProfileOptimization,
        )
        report["diagnostic"]["epochs"] = options.profile_epochs
        report["diagnostic"]["sequence"] = sequences[0].name
        report["diagnostic"][
            "note"
        ] = "Separate fresh model; synchronization and profiler overhead included; not baseline timing."
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
    # See opt_run.py: cholespy 2.1.0 can crash during interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    if any(name == "pcgrid" or name.startswith("pcgrid.") for name in sys.modules):
        os._exit(0)
