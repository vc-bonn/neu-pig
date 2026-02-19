import os
import torch
from torch.utils.data import DataLoader
from src.io.datasets.existing import existingDataset
from src.optimization import Optimization
from src.utilities.util import scale_points, initialize_meshes
import logging
import copy
from pathlib import Path
from pcgrid.value_wrapper import ValueWrapper
from pytorch3d.structures import Meshes
from src.network.geometry_utils import compute_keyframe, init_surf, scale_and_save
from src.utilities.eval_utils import eval_meshes, get_loc_scale
import json
import datetime
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
)
from torch.multiprocessing import Queue
import torch.multiprocessing as mp


class Opt_Run:
    def __init__(self, args: dict):
        self.args = args

    def run(self) -> None:
        if self.args.debug:
            self._run_directory_debug()
        else:
            self._run_directory()

    def _run_directory(self) -> None:
        if self.args.io_args["directory_path"][-1] != "/":
            self.args.io_args["directory_path"] = (
                self.args.io_args["directory_path"] + "/"
            )
        directories = os.listdir(self.args.io_args["directory_path"])
        path = self.args.io_args["base_out_path"]
        now = datetime.datetime.now()
        self.args.io_args["out_path"] = os.path.join(
            path, now.strftime("Date%Y-%m-%d_Time%H-%M-%S")
        )
        self.args.io_args["base_out_path"] = os.path.join(
            path, now.strftime("Date%Y-%m-%d_Time%H-%M-%S")
        )
        Path.mkdir(Path(self.args.io_args["out_path"]), parents=True, exist_ok=True)

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            if self.args.verbose:
                t1 = progress.add_task(
                    "Data-Loading", total=len(directories), start=False
                )
                t2 = progress.add_task(
                    "Surface-Init", total=len(directories), start=False
                )
                t3 = progress.add_task(
                    "Optimization", total=len(directories), start=False
                )
                opt_subtasks = [
                    progress.add_task(
                        "          ↳ ",
                        total=self.args.method_args["optimization"]["epochs"],
                        start=False,
                    )
                    for _ in range(min(len(self.args.devices), len(directories)))
                ]
                t4 = progress.add_task("Output", total=len(directories), start=False)
                t5 = progress.add_task(
                    "Evaluation", total=len(directories), start=False
                )

            args_list = []
            for directory in directories:
                args = copy.deepcopy(self.args)
                args.io_args["input_directory"] = (
                    args.io_args["directory_path"] + directory + "/"
                )
                args.io_args["directory"] = directory
                args.io_args["out_path"] = os.path.join(
                    args.io_args["base_out_path"], directory
                )

                Path.mkdir(Path(args.io_args["out_path"]), parents=True, exist_ok=True)
                args.state = {
                    "data": False,
                    "init": False,
                    "opt": False,
                    "eval": False,
                    "device": None,
                }
                args_list.append(args)

            ctx = mp.Manager()
            q_data = ctx.Queue()
            q_init = ctx.Queue()
            q_opts = ctx.Queue()
            q_outs = ctx.Queue()
            q_progress = ctx.Queue()

            threads = [
                mp.Process(target=prepare_data, args=(args_list, q_data, q_progress)),
                mp.Process(target=init_surface, args=(q_data, q_init, q_progress)),
            ]
            for i, device in enumerate(self.args.devices):
                if i == len(directories):
                    break
                threads.append(
                    mp.Process(target=opt, args=(q_init, q_opts, q_progress, device))
                )
            threads.append(mp.Process(target=output, args=(q_opts, q_outs, q_progress)))
            threads.append(mp.Process(target=eval, args=(q_outs, q_progress)))

            for thread in threads:
                thread.start()

            opt_dones = 0
            eval_done = 0
            while True:
                try:
                    msg = q_progress.get(timeout=1)
                except:
                    continue
                if msg == "prepare_data" and self.args.verbose:
                    progress.update(t1, advance=1)
                elif msg == "init_surface" and self.args.verbose:
                    progress.update(t2, advance=1)
                elif msg == "opt" and self.args.verbose:
                    progress.update(t3, advance=1)
                elif msg.startswith("cuda:"):
                    # msg = "device;loss;obj;epoch"
                    # example: "cuda:0;0.1234;object_01;5"
                    device_id = int(msg.split(":")[1].split(";")[0])
                    device_id = self.args.devices.index("cuda:{}".format(device_id))
                    cd_loss = float(msg.split(";")[1])
                    reg_loss = float(msg.split(";")[2])
                    obj = msg.split(";")[3]
                    epoch = int(msg.split(";")[4])
                    if epoch == 10 and self.args.verbose:
                        progress.reset(opt_subtasks[device_id])
                        progress.start_task(opt_subtasks[device_id])
                    if self.args.verbose:
                        progress.update(
                            opt_subtasks[device_id],
                            description=f"    ↳ {obj} — loss: {cd_loss:.4f} — reg: {reg_loss:.4f}",
                            completed=epoch,
                        )
                elif msg.startswith("value;") and self.args.verbose:
                    value = msg.split(";")[1]
                    progress.print(
                        "[bold green] Current Mean Vertices: {}".format(value)
                    )
                elif msg == "output" and self.args.verbose:
                    progress.update(t4, advance=1)
                elif msg == "eval" and self.args.verbose:
                    progress.update(t5, advance=1)
                    eval_done += 1
                elif msg == "data_start" and self.args.verbose:
                    if progress.tasks[t1].completed == 0:
                        progress.start_task(t1)
                elif msg == "init_start" and self.args.verbose:
                    if progress.tasks[t2].completed == 0:
                        progress.start_task(t2)
                elif msg == "opt_start" and self.args.verbose:
                    if progress.tasks[t3].completed == 0:
                        progress.start_task(t3)
                elif msg == "output_start" and self.args.verbose:
                    if progress.tasks[t4].completed == 0:
                        progress.start_task(t4)
                elif msg == "eval_start" and self.args.verbose:
                    if progress.tasks[t5].completed == 0:
                        progress.start_task(t5)
                elif msg == "opt done":
                    opt_dones += 1
                    if opt_dones == min(len(self.args.devices), len(directories)):
                        q_opts.put(["done", None, None])

                if msg == "eval" and eval_done == len(directories):
                    break
            try:
                for q in [q_data, q_init, q_opts, q_outs, q_progress]:
                    q.close()
            except:
                pass
            for i, thread in enumerate(threads):
                thread.join()

    def _run_directory_debug(self) -> None:
        if self.args.io_args["directory_path"][-1] != "/":
            self.args.io_args["directory_path"] = (
                self.args.io_args["directory_path"] + "/"
            )
        directories = os.listdir(self.args.io_args["directory_path"])
        path = self.args.io_args["base_out_path"]
        now = datetime.datetime.now()
        self.args.io_args["out_path"] = os.path.join(
            path, now.strftime("Date%Y-%m-%d_Time%H-%M-%S")
        )
        self.args.io_args["base_out_path"] = os.path.join(
            path, now.strftime("Date%Y-%m-%d_Time%H-%M-%S")
        )
        Path.mkdir(Path(self.args.io_args["out_path"]), parents=True, exist_ok=True)

        args_list = []
        for directory in directories:

            args = copy.deepcopy(self.args)
            args.io_args["input_directory"] = (
                args.io_args["directory_path"] + directory + "/"
            )
            args.io_args["directory"] = directory
            args.io_args["out_path"] = os.path.join(
                args.io_args["base_out_path"], directory
            )

            Path.mkdir(Path(args.io_args["out_path"]), parents=True, exist_ok=True)
            args.state = {
                "data": False,
                "init": False,
                "opt": False,
                "eval": False,
                "device": None,
            }
            args_list.append(args)

        ctx = mp.Manager()
        q_data = ctx.Queue()
        q_init = ctx.Queue()
        q_opts = ctx.Queue()
        q_outs = ctx.Queue()
        q_progress = ctx.Queue()

        prepare_data(args_list, q_data, q_progress)
        init_surface(q_data, q_init, q_progress)
        opt(q_init, q_opts, q_progress, self.args.devices[0])
        output(q_opts, q_outs, q_progress)
        eval(q_outs, q_progress)


def prepare_data(args_list: list[dict], out_q: Queue, q_progress: Queue):

    logging.basicConfig(level=logging.ERROR)
    logging.captureWarnings(False)

    q_progress.put("data_start")
    for args in args_list:
        input_dataset = existingDataset(args)
        dataloader = DataLoader(
            input_dataset,
            batch_size=input_dataset.__len__(),
            num_workers=0,
        )

        for _, data in enumerate(dataloader):
            for key in data.keys():
                if isinstance(data[key], dict):
                    for subkey in data[key].keys():
                        if isinstance(data[key][subkey], torch.Tensor):
                            if data[key][subkey].shape[0] == 1:
                                data[key][subkey] = data[key][subkey][0]
                            if data[key][subkey].dtype == torch.float64:
                                data[key][subkey] = data[key][subkey].to(torch.float32)
                elif isinstance(data[key], torch.Tensor):
                    if data[key].shape[0] == 1:
                        data[key] = data[key][0]
                    if data[key].dtype == torch.float64:
                        data[key] = data[key].to(torch.float32)
        try:
            out_q.put([args, data])
            q_progress.put("prepare_data")
        except:
            continue
    out_q.put(["done", None])


def init_surface(in_q: Queue, out_q: Queue, q_progress: Queue):
    logging.basicConfig(level=logging.ERROR)
    logging.captureWarnings(False)
    started = False
    while True:
        try:
            args, data = in_q.get(timeout=5)
            if not started:
                q_progress.put("init_start")
                started = True
        except:
            continue
        if args == "done":
            out_q.put(["done", None, None, None])
            break

        args.T = data["points"].shape[0]

        #####
        # Preprocess Target Points
        #####
        target_points, args.points_min, args.points_max = scale_points(data["points"])
        #####
        # Keyframe Selection
        #####
        args.method_args["keyframe_index"] = compute_keyframe(
            target_points.squeeze(), method=args.keyframe
        )

        #####
        # Initial Mesh
        #####
        verts, faces = init_surf(
            target_points[args.method_args["keyframe_index"]].squeeze()[:, :3],
            data["normals"][args.method_args["keyframe_index"]],
            args=args,
        )
        verts, faces = initialize_meshes(args, verts, faces)
        data["points"] = target_points
        try:
            out_q.put([args, data, verts, faces])
            q_progress.put("init_surface")
        except:
            return


def opt(in_q: Queue, out_q: Queue, q_progress: Queue, device: str):
    logging.basicConfig(level=logging.ERROR)
    logging.captureWarnings(False)
    started = False
    while True:
        try:
            args, data, verts, faces = in_q.get(timeout=5)
            if not started:
                q_progress.put("opt_start")
                started = True
        except:
            continue
        if args == "done":
            q_progress.put("opt done")
            out_q.put(["done", None, None])  # Tell the other processes to also stop
            break
        args.device = device

        mlp_args = args.method_args.get("mlp", {})
        point_dim = mlp_args.get("point_dim", 28)
        normal_dim = mlp_args.get("normal_dim", 4)
        wrapper_args = {
            "device": args.device,  # define the device to use
            "wrapper_args": {  # define all arguments for the wrapper
                "parameterization": {  # define the parameterizations to use via preconditioned grids
                    "Network": {  # Class name of the parameterization (see src/optimization_values.py)
                        "grid_values": point_dim,  # Number of values at each grid cell
                        "method": "tanh",  # How to transform the grid outputs (see src/optimization_values.py)
                    },
                },
                "grids": {  # Dict containing all grids to use
                    "grid_0": {  # Each grid definition supports unique hyperparameters
                        "parameters": [
                            "Network",
                        ],  # Which parameters to bind to this grid
                        "grid_args": args.method_args["grid"]
                        | {
                            "T": 1,
                            "T_lambda_dampening": -1,
                        },  # Grid hyperparameters
                    },
                },
                "defaults": [],  # Default parameterizations to use if not specified in the grid
            },
        }
        wrapper_args_2 = {
            "device": args.device,  # define the device to use
            "wrapper_args": {  # define all arguments for the wrapper
                "parameterization": {  # define the parameterizations to use via preconditioned grids
                    "Network": {  # Class name of the parameterization (see src/optimization_values.py)
                        "grid_values": normal_dim,  # Number of values at each grid cell
                        "method": "tanh",  # How to transform the grid outputs (see src/optimization_values.py)
                    },
                },
                "grids": {  # Dict containing all grids to use
                    "grid_0": {  # Each grid definition supports unique hyperparameters
                        "parameters": [
                            "Network",
                        ],  # Which parameters to bind to this grid
                        "grid_args": args.method_args["grid"]
                        | {
                            "T": 1,
                            "T_lambda_dampening": -1,
                        },  # Grid hyperparameters
                    },
                },
                "defaults": [],  # Default parameterizations to use if not specified in the grid
            },
        }
        opt = Optimization(args)
        point_grid = ValueWrapper(wrapper_args)
        wrapper_args_2["wrapper_args"]["grids"]["grid_0"]["grid_args"]["n_level"] = 1
        wrapper_args_2["wrapper_args"]["grids"]["grid_0"]["grid_args"]["base_res"] = 4
        normal_grid = ValueWrapper(wrapper_args_2)
        grids = [point_grid, normal_grid]

        meshes = opt(
            grids, verts.to(args.device), faces.to(args.device), data, q_progress
        )
        try:
            out_q.put([args, data, meshes])
            q_progress.put("opt")
        except:
            out_q.put(["done", None, None])  # Tell the other processes to also stop
            break


def output(in_q: Queue, out_q: Queue, q_progress: Queue):
    logging.basicConfig(level=logging.ERROR)
    logging.captureWarnings(False)
    started = False
    while True:
        try:
            args, data, meshes = in_q.get(timeout=5)
            if not started:
                q_progress.put("output_start")
                started = True
        except:
            continue
        if args == "done":
            out_q.put(["done", None, None, None])
            break

        gt_meshes = Meshes(
            verts=data["gt_points"],
            faces=data["gt_faces"],
            verts_normals=data["gt_normals"],
        )
        loc_scales = [get_loc_scale(m) for m in gt_meshes]
        meshes = scale_and_save(args, meshes)

        try:
            out_q.put([args, data, meshes, loc_scales])
            q_progress.put("output")
        except:
            continue


def eval(in_q: Queue, q_progress: Queue) -> None:
    logging.basicConfig(level=logging.ERROR)
    logging.captureWarnings(False)
    started = False
    metrics = None

    base_path = None
    while True:
        try:
            args, data, meshes, loc_scales = in_q.get(timeout=5)
            if not started:
                q_progress.put("eval_start")
                started = True
                base_path = args.io_args["base_out_path"]
        except:
            continue
        if args == "done":
            for k in metrics.keys():
                metrics[k] = sum(metrics[k]) / len(metrics[k])
            with open(
                os.path.join(
                    base_path,
                    "Metrics.json",
                ),
                "w",
            ) as json_file:
                json.dump(
                    metrics,
                    json_file,
                )
            break
        goal_meshes = Meshes(
            verts=data["gt_points"].squeeze()[..., :3],
            faces=data["gt_faces"],
            verts_normals=data["gt_normals"].squeeze()[..., :3],
        )
        eval_dict = eval_meshes(meshes, goal_meshes, loc_scales)

        if metrics is None:
            metrics = {k: [v] for k, v in eval_dict.items()}
        else:
            for k, v in eval_dict.items():
                metrics[k].append(v)
        log_metrics(eval_dict, args)
        try:
            q_progress.put("eval")
        except:
            continue


@torch.no_grad()
def log_metrics(data: dict, args: dict) -> None:
    with open(
        os.path.join(
            args.io_args["out_path"],
            "method_args.json",
        ),
        "w",
    ) as json_file:
        json.dump(
            args.method_args,
            json_file,
        )
    with open(
        os.path.join(
            args.io_args["out_path"],
            "Metrics.json",
        ),
        "w",
    ) as json_file:
        json.dump(
            data,
            json_file,
        )
