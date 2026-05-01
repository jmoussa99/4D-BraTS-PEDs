from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def choose_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_patch_size(value: str) -> tuple[int, int, int]:
    parts = [int(part) for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("patch size must have three integers, for example 96,160,160")
    return tuple(parts)  # type: ignore[return-value]


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
