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
```
git clone https://github.com/vc-bonn/neu-pig.git neu-pig
cd neu-pig

git submodule update --init

conda create -n neupig python=3.12
conda activate neupig


conda install -y -c nvidia -c conda-forge cuda-toolkit=12.8 openblas "cmake<4" ninja
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install "scikit-build-core>=0.4.3" "nanobind>=2,<3"
pip install --no-build-isolation ext/pcgrid
pip install tqdm tensorboard scikit-learn charonload gpytoolbox imageio matplotlib ninja open3d opencv-python pykdtree trimesh pymeshlab openmesh tensorboardx gitpython rich
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=12.0 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@cbcae096a0b9b04f7c515d11bb4285a82e96b8d7"
```

## Main.py Arguments
```
python Main.py --...
[-m / --methodConfig] - Path to the method config file (str)
[-se / --seed] - random seed (int)
[-d / --device] - cuda device number (int)
[-t / --target] - which target type to use. If pcl is chosen, make sure that the directory path is as configured in the AMA dataset. ("pcl"/"obj")
[-np / --number_points] - if [-t == "obj"] set the number of target points sampled per obj file (int)
[-o / --out_path] - output directory (str)
[-dp / --directory_path] - path to the input objects, see PATH-STRUCTURE section (str)
[-s / --skip] - how many objects to skip (int)
[-ns / --noise] - standard deviation of gaussian noise added to target points (float)
```

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
All run configs are predefined in configs/method/runs. For the ablation study see the configs/method/ablations path. Alter the path arguments accordingly [-o / -dp]. To create the extended AMA dataset, with sequences in the range of 40-120 frames, see src/io/dataset/process_ama_dataset.py.

For the chained execution we supply a run.py script, see src/run.py. See the --help arguments for all the run/supplemental run options. We support multi-gpu runs, setting multiple gpu ids via --devices.

To run the initialization methods from DynoSurf or Motion2VecSets, make sure to install their required packages. For Motion2VecSets, we recommend creating a separate conda environment with the appropriate Python version to avoid version conflicts. For the execution of DynoSurf, please refer to their [repository](https://github.com/yaoyx689/DynoSurf), or execute ```src/related_work/run_dynosurf.py```.
Paths need to be altered according to your setup.

## Optimization performance

The updated implementation uses separable grid solves, cached sampling operators
and training inputs, exact spatial nearest neighbors, fused Adam, and CUDA graph
replay. The model, losses, learning rates, point count, and epoch count are preserved.

Measured on the first ten AMA sequences in lexicographic order, with 1,000 epochs,
5,000 points per frame, seed 0, eight CPU threads, and an NVIDIA RTX 5090
(PyTorch 2.9.0+cu130, CUDA toolkit 13.1):

| Measure | Original (`cvpr-state`) | Updated |
|---|---:|---:|
| Total optimization time ↓ | 178.508 s | 66.940 s |
| Mean time per sequence ↓ | 17.851 s | 6.694 s |
| Mean Chamfer-L2 ↓ | 5.24397 × 10⁻⁶ | 5.15854 × 10⁻⁶ |
| Mean normal consistency ↑ | 0.940222 | 0.940355 |
| Mean F-score (0.005) ↑ | 0.982574 | 0.983526 |

Optimization is **2.67× faster (62.5% less time)** in this comparison. Timing excludes
loading, initialization, operator precomputation, final prediction, and evaluation;
it includes loss computation, backward, updates, and CUDA graph warmup/capture.
Evaluation uses the same cached inputs and initialization, with 100,000 surface
samples per frame. Average metrics improved slightly, but these single-seed results
do not establish statistical significance: Chamfer-L2 improved on eight sequences
and regressed on two. Direct solves and GPU arithmetic can change training trajectories.

### Configuration

The spatial CUDA extension compiles on first use and requires `ninja`, a C++ compiler,
and a CUDA toolkit compatible with PyTorch (`CUDA_HOME` if needed). The environment
above was used for benchmarking; the older installation example has not been
revalidated for the updated code. BF16 requires a compatible GPU.

Optional top-level `performance` settings in a method config control the fast paths.
To use native pcgrid and PyTorch3D KNN, set `separable_grid`, `spatial_knn`, and
`cuda_graph` to `false`. Non-CG solver configurations also retain native pcgrid.
`fused_adam` controls the network optimizer; the fast grids use fused Adam on CUDA.
Dense sampling caches default to at most 4,096 nodes per level and a 256 MiB budget
per grid wrapper (`dense_grid_max_nodes` and `dense_grid_budget_mb`).

`mlp_precision` and `grid_precision` independently accept `fp32`, `fp16`, or `bf16`.
Defaults are BF16 for the MLP and FP32 for grids. Master parameters, Adam state,
geometry, interpolation, and losses stay FP32. The optional
`configs/method/precision/fp16.json` enables FP16 for both components, saving a further
2.36% optimization time and 7.38% peak allocated tensor memory in a separate paired
ten-sequence run. FP16 uses fixed loss scaling (default `loss_scale: 128`), requires
the fast grid backend, and does not skip updates on overflow.

### Benchmarks and tests

Run from the repository root with preprocessed AMA data:

```bash
# Optimization only; --profile-epochs 0 skips the separate diagnostic trace.
python scripts/benchmark_optimization.py --dataset ~/AMA --limit 10

# Paired runtime and evaluation comparison with the preserved original revision.
git fetch origin
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label baseline --reference --freeze-reference --reference-revision origin/cvpr-state
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label optimized

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
