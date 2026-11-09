"""Reproduce independent MLP/grid precision screening and paired AMA evaluation.

Run in the project environment; the input-cache root must contain the previously
prepared AMA inputs to ensure all variants share the same initialization.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("screen", "trial", "ten"),
        nargs="+",
        default=["screen", "trial", "ten"],
    )
    parser.add_argument(
        "--root", type=Path, default=Path("results/precision_comparison")
    )
    parser.add_argument(
        "--input-cache-root", type=Path, default=Path("results/optimization_comparison")
    )
    parser.add_argument("--dataset", type=Path, default=Path.home() / "AMA")
    options = parser.parse_args()
    config_root = options.root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    all_variants = [
        f"{mlp}_{grid}"
        for mlp in ("bf16", "fp16", "fp32")
        for grid in ("fp32", "fp16", "bf16")
    ]
    for variant in all_variants:
        mlp, grid = variant.split("_")
        (config_root / f"{variant}.json").write_text(
            json.dumps({"mlp_precision": mlp, "grid_precision": grid}, indent=2)
        )
    stages = {
        "screen": (all_variants, ["--epochs", "200", "--skip-evaluation"], 1),
        "trial": (
            [
                "bf16_fp32",
                "fp16_fp32",
                "bf16_fp16",
                "bf16_bf16",
                "fp16_fp16",
                "fp16_bf16",
                "fp32_fp32",
            ],
            [],
            1,
        ),
        "ten": (["bf16_fp32", "fp16_fp16", "bf16_fp16"], [], 10),
    }
    runner = Path(__file__).resolve().with_name("compare_optimization.py")
    for stage in options.stage:
        variants, arguments, limit = stages[stage]
        outcomes = {}
        for variant in variants:
            label = f"{stage}_{variant}"
            command = [
                sys.executable,
                str(runner),
                "--root",
                str(options.root),
                "--dataset",
                str(options.dataset),
                "--input-cache-root",
                str(options.input_cache_root),
                "--label",
                label,
                "--limit",
                str(limit),
                "--performance",
                str(config_root / f"{variant}.json"),
                *arguments,
            ]
            print(label, flush=True)
            result = subprocess.run(command)
            outcomes[variant] = {"exit_code": result.returncode}
            (options.root / f"{stage}_outcomes.json").write_text(
                json.dumps(outcomes, indent=2)
            )
            if result.returncode and stage == "ten":
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
