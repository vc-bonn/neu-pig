import argparse
import json
import os
import random
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Neu-PIG optimization")
    parser.add_argument(
        "-m",
        "--methodConfig",
        dest="method_config",
        type=Path,
        default=Path("configs/method/fit_.json"),
        help="Method config file",
    )
    parser.add_argument("-se", "--seed", type=int, default=0)
    parser.add_argument("-d", "--devices", nargs="+", default=["0"])
    parser.add_argument("-t", "--target", choices=("obj", "ply"), default="obj")
    parser.add_argument("-np", "--number_points", type=int, default=5000)
    parser.add_argument("-o", "--out_path", type=Path, default=Path("test"))
    parser.add_argument("-i", "--init", default="ours")
    parser.add_argument("-k", "--keyframe", default="ours")
    parser.add_argument("-ngp", "--instant_ngp", action="store_true")
    parser.add_argument("-ns", "--noise", type=float, default=0.0)
    parser.add_argument(
        "-dp",
        "--directory_path",
        type=Path,
        default=Path("/data/kaltheuner/preprocessed-data/AMA"),
    )
    parser.add_argument("--debug", action="store_true", help="Run synchronously")
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=True)
    return parser


def _device_name(device: str) -> str:
    device = str(device)
    return device if device == "cpu" or device.startswith("cuda:") else f"cuda:{device}"


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = build_parser().parse_args()

    import torch

    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    torch.multiprocessing.set_start_method("spawn", force=True)
    _seed_everything(args.seed)

    args.devices = [_device_name(device) for device in args.devices]
    args.io_args = {
        "base_out_path": str(args.out_path.resolve()),
        "directory_path": str(args.directory_path.resolve()),
        "noise": args.noise,
    }

    if not args.method_config.is_file():
        raise FileNotFoundError(f"Method config not found: {args.method_config}")
    with args.method_config.open(encoding="utf-8") as config_file:
        args.method_args = json.load(config_file)

    if args.noise > 0.0:
        raise NotImplementedError("Noise ablation is currently disabled")

    if args.verbose:
        from rich.console import Console

        console = Console(record=True)
        console.log(f"Arguments:\n {args}\n")
        console.log(f"IO Arguments:\n {args.io_args}\n")
        console.log(f"Method Arguments:\n {args.method_args}\n")

    from src.opt_run import OptRun

    OptRun(args).run()


if __name__ == "__main__":
    main()
