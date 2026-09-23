from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ContractError


@dataclass(frozen=True)
class BlockBootstrapResult:
    observed_mean_log_advantage: float
    confidence_interval_low: float
    confidence_interval_high: float
    one_sided_p_value: float


def circular_block_mean_test(
    paired_log_advantage: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
) -> BlockBootstrapResult:
    values = np.asarray(paired_log_advantage, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ContractError("Block bootstrap requires a finite vector with at least two rows")
    if not 1 <= block_length <= len(values) or resamples < 999:
        raise ContractError("Invalid block-bootstrap contract")

    observed = float(np.mean(values))
    centered = values - observed
    blocks_needed = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length, dtype=int)
    rng = np.random.default_rng(seed)
    null_means = np.empty(resamples, dtype=float)
    sample_means = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        starts = rng.integers(0, len(values), size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % len(values)).reshape(-1)[: len(values)]
        null_means[draw] = float(np.mean(centered[indices]))
        sample_means[draw] = float(np.mean(values[indices]))

    p_value = float((1 + np.sum(null_means >= observed)) / (resamples + 1))
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return BlockBootstrapResult(observed, float(low), float(high), p_value)
