# Performance and benchmarking

Run commands from the repository root. See the [README](../README.md) for installation and data setup.

## Results

The updated implementation uses separable grid solves, cached sampling operators
and training inputs, exact spatial nearest neighbors, foreach Adam, fused geometry
and edge-loss kernels, and CUDA graph replay. The model, losses, learning rates,
point count, and epoch count are preserved. MLP compilation is enabled by default.

Measured on the first ten AMA sequences in lexicographic order (nine `crane`
sequences and `march1_0010`, rather than ten distinct subjects), with 1,000 epochs,
5,000 points per frame, seed 0, eight CPU threads, and an NVIDIA RTX 5090
(PyTorch 2.9.0+cu130, CUDA toolkit 13.1). The baseline is the original CVPR
implementation preserved on `cvpr-state` (`97e9da0`):

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

The CVPR baseline was measured in an earlier run; the current implementation was
measured separately on the same machine using the same cached inputs and
initialization. These are single-run timings rather than interleaved repeated
measurements.

Timing excludes loading, initialization, operator precomputation, MLP compilation,
final prediction, and evaluation; it includes loss computation, backward, optimizer
updates, and CUDA graph warmup/capture. Every variant performs 1,000 updates.
Evaluation uses the same cached inputs and initialization, with 100,000 surface
samples per frame. MLP preparation took 4.006 s across these ten compiled runs
with compiler caches already populated. An earlier cold compilation trial required
roughly 31 s outside the optimization loop; first-use startup can exceed the loop
savings. Full-call timings also depend on CUDA extension build caches.

Average metrics improved slightly, but these single-seed results do not establish
statistical significance. Compared with `cvpr-state`, Chamfer-L2 improved on nine
sequences and regressed on one (`crane_0129`, a 5.03% increase).
Equivalent formulas and different GPU arithmetic can change training trajectories;
compilation also changes BF16 rounding near activation boundaries.

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

# Runtime and evaluation comparison against the original CVPR implementation.
git fetch origin
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label baseline --reference --freeze-reference --reference-revision origin/cvpr-state \
    --limit 10 --epochs 1000
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label compiled --limit 10 --epochs 1000

# Optional comparison without compilation. --performance accepts the performance
# object alone, rather than a full method config.
printf '%s\n' '{"compile_mlp": false}' > results/comparison/eager.json
python scripts/compare_optimization.py --dataset ~/AMA --root results/comparison \
    --label eager --limit 10 --epochs 1000 --performance results/comparison/eager.json

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
