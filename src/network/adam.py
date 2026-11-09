"""Adam backend selection shared by the MLP and spatial grids."""

import torch


def make_adam(parameters, device, performance):
    backend = performance.get("adam_backend")
    if backend is None:
        legacy = performance.get("fused_adam")
        backend = "foreach" if legacy is None else ("fused" if legacy else "single")
    if backend not in ("foreach", "fused", "single"):
        raise ValueError("adam_backend must be foreach, fused, or single")
    cuda = torch.device(device).type == "cuda"
    return torch.optim.Adam(
        parameters,
        foreach=backend == "foreach",
        fused=backend == "fused" and cuda,
        capturable=cuda and performance.get("cuda_graph", True),
    )
