import torch


def compute_dtype(name):
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if name not in dtypes:
        raise ValueError(f"Unknown precision {name!r}; choose fp32, fp16, or bf16")
    return dtypes[name]
