import argparse
from rich.console import Console
import torch
import numpy as np
import random
import json
import os

parser = argparse.ArgumentParser(description="nvdiffrecmc")
parser.add_argument(
    "-m",
    "--methodConfig",
    type=str,
    default="configs/method/fit_.json",
    help="Method config file",
)
parser.add_argument("-se", "--seed", type=int, default=0)
parser.add_argument("-d", "--devices", type=str, nargs="+", default=[0])
parser.add_argument("-t", "--target", type=str, default="obj")
parser.add_argument("-np", "--number_points", type=int, default=5000)
parser.add_argument("-o", "--out_path", type=str, default="test")
parser.add_argument("-i", "--init", type=str, default="ours")
parser.add_argument("-k", "--keyframe", type=str, default="ours")
parser.add_argument("-ngp", "--instant_ngp", action="store_true")
parser.add_argument("-seq", "--sequence", action="store_true")
parser.add_argument("-ns", "--noise", type=float, default=0)
parser.add_argument(
    "-dp",
    "--directory_path",
    type=str,
    default="/data/plack/neupig/teaser_objects/",
)
parser.add_argument("--debug", action="store_true", help="Debug Mode")
parser.add_argument("--verbose", action="store_false", help="Verbose Mode")


args = parser.parse_args()


os.environ["PYTHONWARNINGS"] = "ignore"

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


if __name__ == "__main__":
    args.devices = ["cuda:{}".format(device) for device in args.devices]
    torch.multiprocessing.set_start_method("spawn")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    console = Console(record=True)
    args.instant_ngp = args.instant_ngp > 0

    if args.verbose:
        console.log("Arguments:\n {} \n".format(args))
    args.io_args = {
        "base_out_path": args.out_path,
        "directory_path": args.directory_path,
        "noise": 0.0,
    }

    if os.path.isfile(args.methodConfig):
        with open(args.methodConfig) as json_file:
            args.method_args = json.load(json_file)
    else:
        raise Exception("Method Config File is not a File [Path not Found]")

    if args.io_args["noise"] > 0.0:
        raise Exception("Noise Ablation Study is currently not re-enabled.")

    args.io_args["base_out_path"] = os.path.abspath(args.io_args["base_out_path"])
    args.io_args["directory_path"] = os.path.abspath(args.io_args["directory_path"])

    if args.verbose:
        console.log("IO Arguments:\n {} \n".format(args.io_args))
        console.log("Method Arguments:\n {} \n".format(args.method_args))
    from src.opt_run import Opt_Run

    o = Opt_Run(args)
    o.run()
