from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import traceback
from pathlib import Path
from queue import Empty, Queue as LocalQueue
from typing import Callable

import torch
import torch.multiprocessing as mp
from pytorch3d.structures import Meshes
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from src.io.datasets.existing import ExistingDataset
from src.network.geometry_utils import compute_keyframe, init_surf, scale_and_save
from src.utilities.eval_utils import eval_meshes, get_loc_scale
from src.utilities.util import initialize_meshes, scale_points

_DONE = None


class OptRun:
    def __init__(self, args) -> None:
        self.args = args

    def run(self) -> None:
        jobs = self._build_jobs()
        if not jobs:
            raise ValueError(
                f"No sequence directories found in {self.args.io_args['directory_path']}"
            )

        if self.args.debug:
            self._run_debug(jobs)
        else:
            self._run_parallel(jobs)

    def _build_jobs(self) -> list:
        input_root = Path(self.args.io_args["directory_path"])
        if not input_root.is_dir():
            raise NotADirectoryError(f"Dataset directory not found: {input_root}")

        directories = sorted(path for path in input_root.iterdir() if path.is_dir())
        run_directory = Path(
            self.args.io_args["base_out_path"]
        ) / dt.datetime.now().strftime("Date%Y-%m-%d_Time%H-%M-%S")
        run_directory.mkdir(parents=True, exist_ok=True)
        self.args.io_args["base_out_path"] = str(run_directory)
        self.args.io_args["out_path"] = str(run_directory)

        jobs = []
        for directory in directories:
            args = copy.deepcopy(self.args)
            args.io_args["input_directory"] = str(directory)
            args.io_args["directory"] = directory.name
            args.io_args["out_path"] = str(run_directory / directory.name)
            Path(args.io_args["out_path"]).mkdir(parents=True, exist_ok=True)
            jobs.append(args)
        return jobs

    def _run_debug(self, jobs: list) -> None:
        queues = [LocalQueue() for _ in range(5)]
        q_data, q_init, q_optimized, q_output, q_progress = queues

        prepare_data(jobs, q_data, q_progress)
        initialize_surface(q_data, q_init, q_progress, worker_count=1)
        optimize(q_init, q_optimized, q_progress, self.args.devices[0])
        write_output(q_optimized, q_output, q_progress, worker_count=1)
        evaluate_results(q_output, q_progress)

    def _run_parallel(self, jobs: list) -> None:
        worker_count = min(len(self.args.devices), len(jobs))
        if worker_count == 0:
            raise ValueError("At least one device is required")

        context = mp.get_context("spawn")
        queues = [context.Queue() for _ in range(5)]
        q_data, q_init, q_optimized, q_output, q_progress = queues
        data_consumed = context.Event()
        surface_consumed = context.Event()
        optimization_consumed = context.Event()
        output_consumed = context.Event()
        optimization_barrier = context.Barrier(worker_count)

        worker_specs = [
            (
                "data",
                prepare_data,
                (jobs, q_data, q_progress, data_consumed),
            ),
            (
                "surface initialization",
                initialize_surface,
                (
                    q_data,
                    q_init,
                    q_progress,
                    worker_count,
                    data_consumed,
                    surface_consumed,
                ),
            ),
            *[
                (
                    f"optimization ({device})",
                    optimize,
                    (
                        q_init,
                        q_optimized,
                        q_progress,
                        device,
                        optimization_barrier,
                        surface_consumed,
                        optimization_consumed,
                    ),
                )
                for device in self.args.devices[:worker_count]
            ],
            (
                "output",
                write_output,
                (
                    q_optimized,
                    q_output,
                    q_progress,
                    worker_count,
                    optimization_consumed,
                    output_consumed,
                ),
            ),
            (
                "evaluation",
                evaluate_results,
                (q_output, q_progress, output_consumed),
            ),
        ]
        processes = [
            context.Process(
                target=_guarded_worker,
                args=(name, function, q_progress, *arguments),
                name=name,
            )
            for name, function, arguments in worker_specs
        ]

        try:
            for process in processes:
                process.start()
            self._show_progress(q_progress, processes, len(jobs), worker_count)
            for process in processes:
                process.join()
            failed = [process for process in processes if process.exitcode]
            if failed:
                names = ", ".join(process.name for process in failed)
                raise RuntimeError(f"Worker process failed: {names}")
        except BaseException:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            raise
        finally:
            for process in processes:
                if process.pid is not None:
                    process.join()
            for process_queue in queues:
                process_queue.close()

    def _show_progress(
        self, q_progress, processes, job_count: int, worker_count: int
    ) -> None:
        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            disable=not self.args.verbose,
        ) as progress:
            tasks = {
                "data": progress.add_task("Data loading", total=job_count, start=False),
                "init": progress.add_task(
                    "Surface initialization", total=job_count, start=False
                ),
                "opt": progress.add_task("Optimization", total=job_count, start=False),
                "output": progress.add_task("Output", total=job_count, start=False),
                "eval": progress.add_task("Evaluation", total=job_count, start=False),
            }
            device_tasks = {
                device: progress.add_task(
                    "          -> ",
                    total=self.args.method_args["optimization"]["epochs"],
                    start=False,
                )
                for device in self.args.devices[:worker_count]
            }

            evaluations = 0
            while evaluations < job_count:
                try:
                    message = q_progress.get(timeout=0.5)
                except Empty:
                    failed = [
                        process
                        for process in processes
                        if process.exitcode is not None and process.exitcode != 0
                    ]
                    if failed:
                        names = ", ".join(process.name for process in failed)
                        raise RuntimeError(f"Worker process failed: {names}")
                    continue

                if isinstance(message, tuple) and message[0] == "error":
                    _, worker, details = message
                    raise RuntimeError(f"{worker} failed:\n{details}")

                if message in {
                    "data_start",
                    "init_start",
                    "opt_start",
                    "output_start",
                    "eval_start",
                }:
                    progress.start_task(tasks[message.removesuffix("_start")])
                elif message == "prepare_data":
                    progress.advance(tasks["data"])
                elif message == "init_surface":
                    progress.advance(tasks["init"])
                elif message == "opt":
                    progress.advance(tasks["opt"])
                elif message == "output":
                    progress.advance(tasks["output"])
                elif message == "eval":
                    progress.advance(tasks["eval"])
                    evaluations += 1
                elif isinstance(message, str) and message.startswith("value;"):
                    progress.print(
                        f"[bold green] Current Mean Vertices: {message.split(';', 1)[1]}"
                    )
                elif isinstance(message, str):
                    parts = message.split(";")
                    if len(parts) == 5 and parts[0] in device_tasks:
                        device, cd_loss, reg_loss, sequence, epoch = parts
                        task = device_tasks[device]
                        if not progress.tasks[task].started:
                            progress.start_task(task)
                        progress.update(
                            task,
                            description=(
                                f"    -> {sequence} - loss: {float(cd_loss):.4f}"
                                f" - reg: {float(reg_loss):.4f}"
                            ),
                            completed=int(epoch),
                        )


# Keep process failures visible to the parent instead of waiting forever on a queue.
def _guarded_worker(
    name: str,
    function: Callable,
    q_progress,
    *arguments,
) -> None:
    try:
        function(*arguments)
    except BaseException:
        q_progress.put(("error", name, traceback.format_exc()))
        raise


def _configure_worker() -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.captureWarnings(False)


def prepare_data(args_list: list, out_q, q_progress, output_consumed=None) -> None:
    _configure_worker()
    q_progress.put("data_start")
    for args in args_list:
        data = ExistingDataset(args).as_dict()
        out_q.put((args, data))
        q_progress.put("prepare_data")
    out_q.put(_DONE)
    if output_consumed is not None:
        output_consumed.wait()


def initialize_surface(
    in_q,
    out_q,
    q_progress,
    worker_count: int,
    input_consumed=None,
    output_consumed=None,
) -> None:
    _configure_worker()
    q_progress.put("init_start")

    while True:
        item = in_q.get()
        if item is _DONE:
            if input_consumed is not None:
                input_consumed.set()
            for _ in range(worker_count):
                out_q.put(_DONE)
            if output_consumed is not None:
                output_consumed.wait()
            return

        args, data = item
        args.T = data["points"].shape[0]
        target_points, args.points_min, args.points_max = scale_points(data["points"])
        keyframe_index = compute_keyframe(target_points.squeeze(), method=args.keyframe)
        args.method_args["keyframe_index"] = keyframe_index

        verts, faces = init_surf(
            target_points[keyframe_index].squeeze()[..., :3],
            data["normals"][keyframe_index],
            args=args,
        )
        verts, faces = initialize_meshes(args, verts, faces)
        data["points"] = target_points
        out_q.put((args, data, verts, faces))
        q_progress.put("init_surface")


def _make_grid(args, grid_values: int, *, normals: bool = False) -> ValueWrapper:
    # Import pcgrid only in optimization workers. Importing it also loads
    # cholespy, whose native module crashes during shutdown in processes which
    # never use the solver (notably evaluation and the parent process).
    from pcgrid.value_wrapper import ValueWrapper

    grid_args = {
        **args.method_args["grid"],
        "T": 1,
        "T_lambda_dampening": -1,
    }
    if normals:
        grid_args.update(n_level=1, base_res=4)

    return ValueWrapper(
        {
            "device": args.device,
            "wrapper_args": {
                "parameterization": {
                    "Network": {
                        "grid_values": grid_values,
                        "method": "tanh",
                    }
                },
                "grids": {
                    "grid_0": {
                        "parameters": ["Network"],
                        "grid_args": grid_args,
                    }
                },
                "defaults": [],
            },
        }
    )


def optimize(
    in_q,
    out_q,
    q_progress,
    device: str,
    input_barrier=None,
    input_consumed=None,
    output_consumed=None,
) -> None:
    from src.optimization import Optimization

    _configure_worker()
    q_progress.put("opt_start")

    while True:
        item = in_q.get()
        if item is _DONE:
            if input_barrier is not None and input_barrier.wait() == 0:
                input_consumed.set()
            out_q.put(_DONE)
            if output_consumed is not None:
                output_consumed.wait()
                # cholespy 2.1.0 segfaults in its native module destructor on
                # Python 3.12. At this point the downstream worker has consumed
                # every result, so exit successfully without interpreter cleanup.
                import os

                os._exit(0)
            return

        args, data, verts, faces = item
        args.device = device
        mlp_args = args.method_args.get("mlp", {})
        grids = [
            _make_grid(args, mlp_args.get("point_dim", 28)),
            _make_grid(args, mlp_args.get("normal_dim", 4), normals=True),
        ]

        optimizer = Optimization(args)
        mesh = optimizer(
            grids,
            verts.to(device),
            faces.to(device),
            data,
            q_progress,
        )
        out_q.put((args, data, mesh))
        q_progress.put("opt")


def write_output(
    in_q,
    out_q,
    q_progress,
    worker_count: int,
    input_consumed=None,
    output_consumed=None,
) -> None:
    _configure_worker()
    q_progress.put("output_start")
    completed_workers = 0

    while completed_workers < worker_count:
        item = in_q.get()
        if item is _DONE:
            completed_workers += 1
            continue

        args, data, mesh = item
        gt_meshes = Meshes(
            verts=data["gt_points"],
            faces=data["gt_faces"],
            verts_normals=data["gt_normals"],
        )
        loc_scales = [get_loc_scale(mesh) for mesh in gt_meshes]
        mesh = scale_and_save(args, mesh)

        out_q.put((args, data, mesh, loc_scales))
        q_progress.put("output")

    if input_consumed is not None:
        input_consumed.set()
    out_q.put(_DONE)
    if output_consumed is not None:
        output_consumed.wait()


def evaluate_results(in_q, q_progress, input_consumed=None) -> None:
    _configure_worker()
    q_progress.put("eval_start")
    metrics: dict[str, list[float]] = {}
    base_path = None

    while True:
        item = in_q.get()
        if item is _DONE:
            if input_consumed is not None:
                input_consumed.set()
            if base_path is not None:
                averages = {
                    key: sum(values) / len(values) for key, values in metrics.items()
                }
                _write_json(Path(base_path) / "Metrics.json", averages)
            return

        args, data, mesh, loc_scales = item
        base_path = args.io_args["base_out_path"]
        goal_meshes = Meshes(
            verts=data["gt_points"][..., :3],
            faces=data["gt_faces"],
            verts_normals=data["gt_normals"][..., :3],
        )
        result = {
            key: float(value)
            for key, value in eval_meshes(mesh, goal_meshes, loc_scales).items()
            if key
            in ["chamfer-L2", "normals consistency", "f-score-5"]
        }
        for key, value in result.items():
            metrics.setdefault(key, []).append(value)
        log_metrics(result, args)
        q_progress.put("eval")


@torch.no_grad()
def log_metrics(data: dict, args) -> None:
    output_path = Path(args.io_args["out_path"])
    _write_json(output_path / "method_args.json", args.method_args)
    _write_json(output_path / "Metrics.json", data)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file)


# Backward compatibility for callers using the original class name.
Opt_Run = OptRun
