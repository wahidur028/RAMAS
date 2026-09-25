from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOL = 1e-12


class ContractError(RuntimeError):
    """A causal, mathematical, data, or reproducibility contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, float_format="%.17g")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def simplex(values: np.ndarray, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or len(vector) == 0 or not np.isfinite(vector).all():
        raise ContractError(f"{label} must be a finite non-empty vector")
    if (vector < -TOL).any() or not np.isclose(vector.sum(), 1.0, atol=1e-10):
        raise ContractError(f"{label} must lie on the probability simplex")
    clipped = np.clip(vector, 0.0, 1.0)
    return clipped / clipped.sum()


def row_stochastic(values: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0 or not np.isfinite(matrix).all():
        raise ContractError(f"{label} must be a finite non-empty matrix")
    if (matrix < -TOL).any() or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ContractError(f"Every row of {label} must lie on the probability simplex")
    clipped = np.clip(matrix, 0.0, 1.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def exposure_grid(step: float) -> np.ndarray:
    if not np.isfinite(step) or not 0.0 < step <= 1.0:
        raise ContractError("Exposure grid step must lie inside (0,1]")
    count = int(np.floor(1.0 / step + TOL))
    grid = np.arange(count + 1, dtype=float) * step
    grid = grid[grid <= 1.0 + TOL]
    return np.unique(np.r_[np.clip(grid, 0.0, 1.0), 1.0])


def exact_upper_tail_mean(values: np.ndarray, tail_probability: float) -> float:
    data = np.sort(np.asarray(values, dtype=float))[::-1]
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ContractError("Tail mean requires a finite non-empty vector")
    if not 0.0 < tail_probability <= 1.0:
        raise ContractError("Tail probability must lie inside (0,1]")
    mass = tail_probability * len(data)
    full = int(np.floor(mass))
    fraction = mass - full
    total = float(np.sum(data[:full]))
    if fraction > 0.0:
        total += fraction * float(data[full])
    return total / mass

