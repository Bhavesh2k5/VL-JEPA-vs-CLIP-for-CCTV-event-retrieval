from typing import Union

import numpy as np
import torch
import torch.nn.functional as F


def l2_normalize_tensor(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalizes PyTorch tensor along final dimension."""
    return F.normalize(x, p=2, dim=-1, eps=eps)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalizes NumPy array along final dimension."""
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def as_float32(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Converts input tensor or array to contiguous float32 NumPy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype=np.float32)
