"""Small utilities for reproducibility and tensor conversion."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute() or base_dir is None:
        return path.expanduser().resolve()
    return (Path(base_dir).expanduser().resolve() / path).resolve()


def numpy_to_tensor_safe(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    arr = np.ascontiguousarray(arr)
    try:
        tensor = torch.from_numpy(arr)
        return tensor.to(dtype) if dtype is not None else tensor
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            return torch.tensor(arr.tolist(), dtype=dtype)
        raise


def tensor_to_numpy_safe(tensor: torch.Tensor, dtype=None) -> np.ndarray:
    cpu_tensor = tensor.detach().cpu().contiguous()
    try:
        out = cpu_tensor.numpy()
        return out.astype(dtype, copy=False) if dtype is not None else out
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            return np.array(cpu_tensor.tolist(), dtype=dtype if dtype is not None else None)
        raise
