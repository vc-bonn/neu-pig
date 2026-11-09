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

**Code update:** `main` includes an optimized implementation with substantially
faster optimization and slightly improved average reconstruction metrics in our
AMA benchmark. The original CVPR implementation is preserved on the
[`cvpr-state`](https://github.com/vc-bonn/neu-pig/tree/cvpr-state) branch.

## Install

The runtime measurements and kernel tests use Python 3.12, PyTorch 2.9.0+cu130,
and CUDA toolkit 13.1 on an RTX 5090. The commands below use those versions.
`TORCH_CUDA_ARCH_LIST=12.0` targets that GPU; adjust it for your GPU when building
CUDA extensions.

```bash
git clone https://github.com/vc-bonn/neu-pig.git neu-pig
cd neu-pig

git submodule update --init

conda create -n neupig python=3.12
conda activate neupig

conda install -y -c nvidia -c conda-forge cuda-toolkit=13.1 openblas "cmake<4" ninja
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install "scikit-build-core>=0.4.3" "nanobind>=2,<3"
pip install --no-build-isolation ext/pcgrid
pip install tqdm tensorboard scikit-learn charonload gpytoolbox imageio matplotlib ninja open3d opencv-python pykdtree trimesh pymeshlab openmesh tensorboardx gitpython rich
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@cbcae096a0b9b04f7c515d11bb4285a82e96b8d7"
```

## Run

Activate the environment and run from the repository root with preprocessed AMA:

```bash
OMP_NUM_THREADS=8 python -u Main.py \
    --methodConfig configs/method/fit_.json \
    --directory_path ~/AMA \
    --out_path results/ama \
    --devices 0 \
    --target obj \
    --number_points 5000 \
    --seed 0
```

This processes all sequence directories under `~/AMA`, including initialization,
optimization, output, and evaluation. Results are written to a timestamped directory
under `results/ama`. Set the update count with `optimization.epochs` in the method
config (1,000 by default). MLP compilation is enabled by default and reports its first preparation
per worker in the terminal; the first use of a configuration can take longer.
To save terminal output, append `2>&1 | tee run.log` to the command.

The default dataset path is `~/AMA`. Use `python Main.py --help` for all options.
`--devices` accepts multiple GPU IDs;
`--target` accepts `obj` or `ply`. `--debug` runs the pipeline synchronously, and
`--quiet` disables progress displays. Nonzero `--noise` is currently unsupported.

## PATH-STRUCTURE
To download the preprocessed data, see the data section of [Dynosurf](https://github.com/yaoyx689/DynoSurf?tab=readme-ov-file).
``` 
--AMA
|
|--crane_0010
|-|
|-|--gt
|-|--pcl_seqs
|-|--points_clouds
|
|--crane_0027
|-|
|-|--gt
|-|--pcl_seqs
|-|--points_clouds
```

## RUNS
All run configs are predefined in configs/method/runs. For the ablation study see the configs/method/ablations path. Alter the path arguments accordingly [-o / -dp]. To create the extended AMA dataset, with sequences in the range of 40-120 frames, see `src/io/datasets/process_ama_dataset.py`.

For the chained execution we supply the root-level `run.py` script. See the --help arguments for all the run/supplemental run options. We support multi-gpu runs, setting multiple gpu ids via --devices.

To run the initialization methods from DynoSurf or Motion2VecSets, make sure to install their required packages. For Motion2VecSets, we recommend creating a separate conda environment with the appropriate Python version to avoid version conflicts. For the execution of DynoSurf, please refer to their [repository](https://github.com/yaoyx689/DynoSurf), or execute ```src/related_work/run_dynosurf.py```.
Paths need to be altered according to your setup.

## Optimization performance

The updated implementation uses separable grid solves, cached sampling operators
and training inputs, exact spatial nearest neighbors, foreach Adam, fused geometry
and edge-loss kernels, and CUDA graph replay. The model, losses, learning rates,
point count, and epoch count are preserved. MLP compilation is enabled by default.

Measured on the first ten AMA sequences in lexicographic order (nine `crane`
sequences and `march1_0010`, rather than ten distinct subjects), with 1,000 epochs,
5,000 points per frame, seed 0, eight CPU threads, and an NVIDIA RTX 5090
(PyTorch 2.9.0+cu130, CUDA toolkit 13.1). This paired comparison uses the previously
optimized `main` revision `9394e83` as its baseline:

| Measure | Previous main (`9394e83`) | Without compilation | Current defaults (compiled MLP) |
|---|---:|---:|---:|
| Total optimization time ↓ | 66.893 s | 52.362 s | 44.013 s |
| Mean time per sequence ↓ | 6.689 s | 5.236 s | 4.401 s |
| Optimization time reduction | — | 21.7% | 34.2% |
| Mean Chamfer-L2 ↓ | 5.23195 × 10⁻⁶ | 5.13538 × 10⁻⁶ | 5.09189 × 10⁻⁶ |
| Mean normal consistency ↑ | 0.940279 | 0.940384 | 0.940487 |
| Mean F-score (0.005) ↑ | 0.982628 | 0.983448 | 0.984190 |
| Peak allocated tensor memory ↓ | 1,239 MiB | 1,253 MiB | 950 MiB |

Timing excludes loading, initialization, operator precomputation, MLP compilation,
final prediction, and evaluation; it includes loss computation, backward, optimizer
updates, and CUDA graph warmup/capture. Every variant performs 1,000 updates.
Evaluation uses the same cached inputs and initialization, with 100,000 surface
samples per frame. MLP preparation took 4.006 s across these ten compiled runs
with compiler caches already populated. An earlier cold compilation trial required
roughly 31 s outside the optimization loop; first-use startup can exceed the loop
savings. Full-call timings also depend on CUDA extension build caches.

Average metrics improved slightly, but these single-seed results do not establish
statistical significance. Chamfer-L2 regressed on four sequences without
compilation (largest increase 4.83%) and five with compilation (largest increase 6.97%).
Equivalent formulas and different GPU arithmetic can change training trajectories;
compilation also changes BF16 rounding near activation boundaries.

For historical context, the previous update reduced optimization time from
178.508 s on `cvpr-state` to 66.940 s in a separate ten-sequence measurement (2.67×).
The table above measures the additional improvements against a fresh baseline.

### Configuration

The spatial and optimization CUDA extensions compile on first use and require `ninja`, a C++ compiler,
and a CUDA toolkit compatible with PyTorch (`CUDA_HOME` if needed). The environment
above was used for benchmarking and testing; a fresh environment installation
has not been independently repeated. BF16 requires a compatible GPU.

Optional top-level `performance` settings in a method config control the fast paths.
To use native pcgrid and PyTorch3D KNN, set `separable_grid`, `spatial_knn`, and
`cuda_graph` to `false`. Non-CG solver configurations also retain native pcgrid.
Adam uses the `foreach` backend by default for both the MLP and fast grids;
`performance.adam_backend` also accepts `fused` or `single`. The legacy
`fused_adam` flag remains supported when `adam_backend` is absent.
Dense sampling caches default to at most 4,096 nodes per level and a 256 MiB budget
per grid wrapper (`dense_grid_max_nodes` and `dense_grid_budget_mb`).

`mlp_precision` and `grid_precision` independently accept `fp32`, `fp16`, or `bf16`.
Defaults are BF16 for the MLP and FP32 for grids. Master parameters, Adam state,
geometry, interpolation, and losses stay FP32. The optional
`configs/method/precision/fp16.json` enables FP16 for both components, which saved
2.36% optimization time and 7.38% peak allocated tensor memory in an earlier paired
ten-sequence run before the latest runtime changes. FP16 uses fixed loss scaling
(default `loss_scale: 128`), requires the fast grid backend, and does not skip
updates on overflow.

The default CUDA path also fuses quaternion vertex transformation and edge loss,
precomputes fixed interpolation stencils for uncached grid levels, and reuses the
fixed target's query ordering. These preserve the mathematical model and exact
nearest-neighbor search. Disable individual paths with `fused_geometry`,
`fused_edge_loss`, `fixed_grid_sampling`, or `reuse_knn_query_order` under `performance`.
The specialized kernels support first derivatives. Other rotation modes use the
general geometry path; grid coordinates requiring gradients use general sampling.

Selective MLP compilation is enabled by default with `max-autotune`. The terminal
(and captured stdout logs) reports the first preparation per worker and its elapsed
time when the MLP is ready. Later preparations are reported only if they take at
least one second. Compatible cached kernels are reused. To disable compilation,
set `performance.compile_mlp` to `false`. The default settings can also be written
explicitly in a method config:

```json
{
    "performance": {
        "compile_mlp": true,
        "compile_mlp_mode": "max-autotune"
    }
}
```

`compile_mlp_mode` accepts `default` or `max-autotune`. Compilation buckets vertex
counts internally (default `compile_vertex_bucket: 128`), then removes padded
outputs before geometry and losses. It warms up forward/backward without optimizer
updates before timing and CUDA graph capture. Compilation and warmup costs are
reported separately as `mlp_preparation_seconds` and remain included in the full
optimizer call time; first-use costs can outweigh training savings on short jobs.

### Benchmarks and tests

Run from the repository root with preprocessed AMA data:

```bash
# Optimization only; --profile-epochs 0 skips the separate diagnostic trace.
python scripts/benchmark_optimization.py --dataset ~/AMA --limit 10

# Paired runtime and evaluation comparison, explicitly using 1,000 updates.
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label baseline --reference --freeze-reference --reference-revision 9394e83 \
    --limit 10 --epochs 1000
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label compiled --limit 10 --epochs 1000

# Optional comparison without compilation. --performance accepts the performance
# object alone, rather than a full method config.
printf '%s\n' '{"compile_mlp": false}' > results/comparison/eager.json
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label eager --limit 10 --epochs 1000 --performance results/comparison/eager.json
# Use origin/cvpr-state instead of 9394e83 for the original CVPR implementation.

# Independent MLP/grid precision experiments and kernel tests.
python scripts/precision_sweep.py --dataset ~/AMA --input-cache-root results/comparison
python -m unittest discover -s tests -v
```

The paired runner adds timing hooks to the original source and shares cached inputs,
templates, and RNG states between variants. Evaluation runs outside the optimization
timer; `--skip-evaluation` disables it. Results and caches are written under `results/`
and excluded from Git. Use a new root for each experiment and a new label for each
run; snapshots and existing results are not overwritten. Load only locally generated
caches. Run GPU benchmarks sequentially to avoid contention.
