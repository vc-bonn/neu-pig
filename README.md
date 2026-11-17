<h1 align="center" id="heading">Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences</h1>

<p align="center">
    <p align="center">
		<b><a href="https://cg.cs.uni-bonn.de/person/m-sc-julian-kaltheuner">Julian Kaltheuner</a></b>
        &nbsp;·&nbsp;
		<b><a href="https://cg.cs.uni-bonn.de/person/dr-hannah-droege">Hannah Droege</a></b>
        &nbsp;·&nbsp;
		<b><a href="https://cg.cs.uni-bonn.de/person/dr-markus-plack">Markus Plack</a></b>
        &nbsp;·&nbsp;
		<b><a href="https://cg.cs.uni-bonn.de/person/dr-patrick-stotko">Patrick Stotko</a></b>
        &nbsp;·&nbsp;
        <b><a href="https://cg.cs.uni-bonn.de/person/prof-dr-reinhard-klein">Reinhard Klein</a></b>
    </p>
    <p align="center">
       University of Bonn &nbsp
    </p>
    <h3 align="center">CVPR 2026</h3>
    <h3 align="center">
        <a href="https://arxiv.org/abs/2602.22212">Paper</a>
		</h3>
    <div align="center"></div>
</p>

![](assets/teaser.png)

**Updated code:** `main` includes faster grid solves, caching, fused CUDA kernels,
CUDA graphs, and MLP compilation. The original implementation is available on
[`cvpr-state`](https://github.com/vc-bonn/neu-pig/tree/cvpr-state).

## Install

Benchmark environment: Linux, Python 3.12, PyTorch 2.9.0+cu130, CUDA toolkit 13.1,
and RTX 5090. Adjust `TORCH_CUDA_ARCH_LIST=12.0` for your GPU.

```bash
git clone --recursive https://github.com/vc-bonn/neu-pig.git
cd neu-pig

conda create -n neupig python=3.12
conda activate neupig

conda install -y -c nvidia -c conda-forge cuda-toolkit=13.1 openblas "cmake<4" ninja
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install "scikit-build-core>=0.4.3" "nanobind>=2,<3"
pip install --no-build-isolation ext/pcgrid
pip install tqdm tensorboard scikit-learn charonload gpytoolbox imageio matplotlib ninja open3d opencv-python pykdtree trimesh pymeshlab openmesh tensorboardx gitpython rich
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@cbcae096a0b9b04f7c515d11bb4285a82e96b8d7"
```

## Data

Download preprocessed AMA data from [DynoSurf](https://github.com/yaoyx689/DynoSurf)
and arrange each sequence as:

```text
AMA/
├── crane_0010/
│   ├── gt/
│   ├── pcl_seqs/
│   └── points_clouds/
└── crane_0027/
    └── ...
```

## Run

With the `neupig` environment active, run from the repository root:

```bash
OMP_NUM_THREADS=8 python -u Main.py \
    --methodConfig configs/method/fit_.json \
    --directory_path ~/AMA \
    --out_path results/ama \
    --devices 0
```

This processes all sequences, including initialization, optimization, and evaluation.
Outputs go to a timestamped directory under `results/ama`. Defaults: 1,000 epochs,
5,000 points per frame, seed 0. Edit `optimization.epochs` in the method config to
change the update count.

First use builds CUDA extensions and compiles the MLP, adding startup time.
Set `performance.compile_mlp` to `false` in the method config to disable compilation.
Use `--devices 0 1` for multiple GPUs and `python Main.py --help` for other options.

Additional configs are in [configs/method](configs/method), including
[ablations](configs/method/ablations) and [supplemental experiments](configs/method/supplemental).
Use `python run.py --help` for experiment batches. Extended AMA preprocessing is in
[src/io/datasets/process_ama_dataset.py](src/io/datasets/process_ama_dataset.py).

## Optimization performance

Same model, losses, learning rates, point count, and epoch count as the CVPR code.
Measured on the first ten AMA sequences (nine `crane` sequences and `march1_0010`),
with 1,000 epochs, 5,000 points per frame, seed 0, eight CPU threads, and RTX 5090.

| Measure | Original (`cvpr-state`) | Current defaults (compiled MLP) |
|---|---:|---:|
| Total optimization time ↓ | 178.508 s | 44.013 s |
| Mean time per sequence ↓ | 17.851 s | 4.401 s |
| Optimization time reduction | — | 75.3% |
| Optimization speedup | — | 4.06× |
| Mean Chamfer-L2 ↓ | 5.24397 × 10⁻⁶ | 5.09189 × 10⁻⁶ |
| Mean normal consistency ↑ | 0.940222 | 0.940487 |
| Mean F-score (0.005) ↑ | 0.982574 | 0.984190 |
| Peak allocated tensor memory ↓ | 998 MiB | 950 MiB |

Optimization-loop timings exclude loading, initialization, compilation, and evaluation.
These are separate single-run measurements on the same machine with shared cached
inputs and initialization, not repeated trials; small quality differences are not
statistically established.

## Benchmarks and tests

```bash
python scripts/benchmark_optimization.py --dataset ~/AMA --limit 10 --profile-epochs 0
python -m unittest discover -s tests -v
```

See [Performance and benchmarking](docs/performance.md) for the CVPR comparison
commands, configuration switches, precision options, and measurement details.
