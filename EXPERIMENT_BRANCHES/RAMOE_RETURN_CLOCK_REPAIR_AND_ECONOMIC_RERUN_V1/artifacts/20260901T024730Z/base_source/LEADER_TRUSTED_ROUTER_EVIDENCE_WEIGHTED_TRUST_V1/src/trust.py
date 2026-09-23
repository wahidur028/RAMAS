from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ContractError, TOL, row_stochastic
from .router import masked_trust_matrix, validate_admission_mask


@dataclass(frozen=True)
class TrustUpdateResult:
    trust_matrix: np.ndarray
    score_matrix: np.ndarray
    growth_component: np.ndarray
    tail_component: np.ndarray
    turnover_component: np.ndarray
    support_mask: np.ndarray
    posterior_mass: np.ndarray
    effective_sample_size: np.ndarray


def _exact_tail_signal(losses: np.ndarray, tail_probability: float) -> np.ndarray:
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ContractError("Tail signal requires a finite vector")
    if not 0.0 < tail_probability <= 1.0:
        raise ContractError("Tail probability must lie inside (0,1]")
    order = np.argsort(-values, kind="stable")
    mass = tail_probability * len(values)
    full = int(np.floor(mass))
    fraction = mass - full
    weights = np.zeros(len(values), dtype=float)
    if full:
        weights[order[:full]] = 1.0
    if fraction > 0.0:
        weights[order[full]] = fraction
    # The unweighted mean of this signal equals the exact upper-tail mean.
    return values * weights * len(values) / mass


def exact_weighted_upper_tail_mean(
    values: np.ndarray,
    weights: np.ndarray,
    tail_probability: float,
) -> float:
    """Return the exact weighted mean of the largest tail_probability mass.

    The posterior weights define the regime-conditional empirical distribution.
    Fractional boundary mass is included, so the result is deterministic and is
    always bounded by the observed minimum and maximum values.
    """
    observations = np.asarray(values, dtype=float)
    posterior_weights = np.asarray(weights, dtype=float)
    if (
        observations.ndim != 1
        or posterior_weights.shape != observations.shape
        or len(observations) == 0
        or not np.isfinite(observations).all()
        or not np.isfinite(posterior_weights).all()
    ):
        raise ContractError("Weighted tail mean requires aligned finite vectors")
    if (posterior_weights < 0.0).any() or not 0.0 < tail_probability <= 1.0:
        raise ContractError("Weighted tail mean received invalid weights or tail probability")
    total_mass = float(np.sum(posterior_weights))
    if total_mass <= TOL:
        raise ContractError("Weighted tail mean requires positive posterior mass")

    required_mass = tail_probability * total_mass
    remaining = required_mass
    numerator = 0.0
    for index in np.argsort(-observations, kind="stable"):
        take = min(float(posterior_weights[index]), remaining)
        numerator += float(observations[index]) * take
        remaining -= take
        if remaining <= TOL:
            break
    if remaining > max(TOL, required_mass * 1e-12):
        raise ContractError("Weighted tail mean could not allocate the requested mass")
    result = numerator / required_mass
    if result < float(np.min(observations)) - TOL or result > float(np.max(observations)) + TOL:
        raise ContractError("Weighted tail mean escaped the observed range")
    return float(result)


def effective_sample_size(weights: np.ndarray) -> float:
    posterior_weights = np.asarray(weights, dtype=float)
    if posterior_weights.ndim != 1 or not np.isfinite(posterior_weights).all():
        raise ContractError("Effective sample size requires a finite vector")
    if (posterior_weights < 0.0).any():
        raise ContractError("Effective sample size weights cannot be negative")
    squared_mass = float(np.dot(posterior_weights, posterior_weights))
    if squared_mass <= TOL:
        return 0.0
    return float(np.sum(posterior_weights) ** 2 / squared_mass)


def risk_aware_trust_update(
    trust_matrix: np.ndarray,
    regime_probabilities: np.ndarray,
    expert_net_returns: np.ndarray,
    expert_turnover: np.ndarray,
    admission_mask: np.ndarray,
    *,
    always_admitted_indices: tuple[int, ...],
    eta: float,
    cvar_alpha: float,
    tail_penalty: float,
    turnover_penalty: float,
    minimum_posterior_mass: float,
    minimum_effective_sample_size: float,
) -> TrustUpdateResult:
    trust = np.asarray(trust_matrix, dtype=float)
    q = np.asarray(regime_probabilities, dtype=float)
    returns = np.asarray(expert_net_returns, dtype=float)
    turnover = np.asarray(expert_turnover, dtype=float)
    if q.ndim != 2 or returns.ndim != 2 or turnover.shape != returns.shape:
        raise ContractError("Trust-update arrays have incompatible shapes")
    if q.shape[0] != returns.shape[0] or trust.shape != (q.shape[1], returns.shape[1]):
        raise ContractError("Trust-update dimensions do not match")
    if len(q) < 2 or not np.isfinite(q).all() or not np.isfinite(returns).all() or not np.isfinite(turnover).all():
        raise ContractError("Trust evidence must be finite and contain at least two rows")
    if (q < -TOL).any() or not np.allclose(q.sum(axis=1), 1.0, atol=1e-10):
        raise ContractError("Router evidence must be row-stochastic")
    if (returns <= -1.0).any() or (turnover < 0.0).any():
        raise ContractError("Invalid expert return or turnover evidence")
    if not np.isfinite(
        [
            eta,
            cvar_alpha,
            tail_penalty,
            turnover_penalty,
            minimum_posterior_mass,
            minimum_effective_sample_size,
        ]
    ).all():
        raise ContractError("Trust parameters must be finite")
    if (
        eta < 0.0
        or not 0.0 < cvar_alpha < 1.0
        or tail_penalty < 0.0
        or turnover_penalty < 0.0
        or minimum_posterior_mass <= 0.0
        or minimum_effective_sample_size <= 0.0
    ):
        raise ContractError("Invalid trust parameter")

    mask = validate_admission_mask(
        admission_mask,
        expert_count=returns.shape[1],
        always_admitted_indices=always_admitted_indices,
    )
    current = masked_trust_matrix(
        trust,
        mask,
        always_admitted_indices=always_admitted_indices,
    )
    posterior_mass = q.sum(axis=0)
    effective_samples = np.asarray(
        [effective_sample_size(q[:, regime]) for regime in range(q.shape[1])],
        dtype=float,
    )
    support_mask = (
        (posterior_mass >= minimum_posterior_mass - TOL)
        & (effective_samples >= minimum_effective_sample_size - TOL)
    )
    safe_mass = np.maximum(posterior_mass[:, None], TOL)
    log_growth = np.log1p(returns)
    losses = np.maximum(-returns, 0.0)
    growth = q.T @ log_growth / safe_mass
    tail = np.zeros_like(growth)
    for regime in range(q.shape[1]):
        if posterior_mass[regime] <= TOL:
            continue
        for expert in range(returns.shape[1]):
            tail[regime, expert] = exact_weighted_upper_tail_mean(
                losses[:, expert],
                q[:, regime],
                1.0 - cvar_alpha,
            )
    churn = q.T @ turnover / safe_mass
    score = growth - tail_penalty * tail - turnover_penalty * churn

    updated = current.copy()
    for regime in np.flatnonzero(support_mask):
        logits = np.log(np.clip(current[regime, mask], 1e-300, None)) + eta * score[regime, mask]
        logits -= np.max(logits)
        updated[regime, mask] = np.exp(logits)
        updated[regime, mask] /= np.sum(updated[regime, mask])
        updated[regime, ~mask] = 0.0
    updated = row_stochastic(updated, "updated trust matrix")
    if not np.array_equal(updated[:, ~mask], np.zeros((updated.shape[0], np.sum(~mask)))):
        raise ContractError("Rejected expert was resurrected by trust learning")
    return TrustUpdateResult(
        updated,
        score,
        growth,
        tail,
        churn,
        support_mask,
        posterior_mass,
        effective_samples,
    )
