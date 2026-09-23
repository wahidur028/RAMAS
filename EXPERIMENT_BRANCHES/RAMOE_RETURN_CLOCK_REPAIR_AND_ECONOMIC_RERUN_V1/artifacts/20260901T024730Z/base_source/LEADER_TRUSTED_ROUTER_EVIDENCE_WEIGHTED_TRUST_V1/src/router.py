from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ContractError, TOL, row_stochastic, simplex


@dataclass(frozen=True)
class RouterDecision:
    expert_weights: np.ndarray
    desired_exposure: float
    fallback_used: bool


def validate_admission_mask(
    admission_mask: np.ndarray,
    *,
    expert_count: int,
    always_admitted_indices: tuple[int, ...],
) -> np.ndarray:
    mask = np.asarray(admission_mask)
    if mask.dtype != np.bool_ or mask.shape != (expert_count,):
        raise ContractError("Admission mask must be a boolean vector matching the experts")
    if not always_admitted_indices:
        raise ContractError("At least one fail-safe expert must always remain admitted")
    for index in always_admitted_indices:
        if index < 0 or index >= expert_count or not mask[index]:
            raise ContractError("Every fail-safe expert must remain admitted")
    if not mask.any():
        raise ContractError("Admission mask removed every expert")
    return mask.copy()


def masked_trust_matrix(
    trust_matrix: np.ndarray,
    admission_mask: np.ndarray,
    *,
    always_admitted_indices: tuple[int, ...],
) -> np.ndarray:
    trust = row_stochastic(trust_matrix, "trust matrix")
    mask = validate_admission_mask(
        admission_mask,
        expert_count=trust.shape[1],
        always_admitted_indices=always_admitted_indices,
    )
    gated = trust * mask[None, :]
    mass = gated.sum(axis=1, keepdims=True)
    if (mass <= TOL).any():
        raise ContractError("Admission mask removed all trust mass from a regime")
    gated /= mass
    gated[:, ~mask] = 0.0
    return row_stochastic(gated, "masked trust matrix")


def route_experts(
    regime_posterior: np.ndarray,
    trust_matrix: np.ndarray,
    admission_mask: np.ndarray,
    *,
    always_admitted_indices: tuple[int, ...],
) -> np.ndarray:
    posterior = simplex(regime_posterior, "regime posterior")
    trust = masked_trust_matrix(
        trust_matrix,
        admission_mask,
        always_admitted_indices=always_admitted_indices,
    )
    if len(posterior) != trust.shape[0]:
        raise ContractError("Router posterior and trust regimes do not match")
    weights = simplex(posterior @ trust, "expert weights")
    mask = np.asarray(admission_mask, dtype=bool)
    if not np.array_equal(weights[~mask], np.zeros(np.sum(~mask))):
        raise ContractError("Rejected expert received nonzero weight")
    return weights


def aggregate_exposure(expert_weights: np.ndarray, expert_exposures: np.ndarray) -> float:
    weights = simplex(expert_weights, "expert weights")
    exposures = np.asarray(expert_exposures, dtype=float)
    if exposures.shape != weights.shape or not np.isfinite(exposures).all():
        raise ContractError("Expert exposures must be finite and match expert weights")
    if ((exposures < -TOL) | (exposures > 1.0 + TOL)).any():
        raise ContractError("Long-only expert exposures must lie inside [0,1]")
    return float(np.clip(weights @ np.clip(exposures, 0.0, 1.0), 0.0, 1.0))


def route_or_fallback(
    regime_posterior: np.ndarray | None,
    trust_matrix: np.ndarray,
    admission_mask: np.ndarray,
    expert_exposures: np.ndarray,
    fallback_weights: np.ndarray,
    *,
    always_admitted_indices: tuple[int, ...],
) -> RouterDecision:
    fallback = simplex(fallback_weights, "fallback weights")
    mask = validate_admission_mask(
        admission_mask,
        expert_count=len(fallback),
        always_admitted_indices=always_admitted_indices,
    )
    if (fallback[~mask] > TOL).any():
        raise ContractError("Fallback may not allocate to rejected experts")
    used_fallback = False
    try:
        if regime_posterior is None:
            raise ContractError("Router posterior is unavailable")
        weights = route_experts(
            regime_posterior,
            trust_matrix,
            mask,
            always_admitted_indices=always_admitted_indices,
        )
    except ContractError:
        weights = fallback
        used_fallback = True
    desired = aggregate_exposure(weights, expert_exposures)
    return RouterDecision(weights, desired, used_fallback)


def l1_allocation_change(
    posterior_a: np.ndarray,
    posterior_b: np.ndarray,
    trust_matrix: np.ndarray,
    admission_mask: np.ndarray,
    *,
    always_admitted_indices: tuple[int, ...],
) -> tuple[float, float]:
    qa = simplex(posterior_a, "posterior a")
    qb = simplex(posterior_b, "posterior b")
    wa = route_experts(
        qa, trust_matrix, admission_mask,
        always_admitted_indices=always_admitted_indices,
    )
    wb = route_experts(
        qb, trust_matrix, admission_mask,
        always_admitted_indices=always_admitted_indices,
    )
    return float(np.sum(np.abs(qa - qb))), float(np.sum(np.abs(wa - wb)))

