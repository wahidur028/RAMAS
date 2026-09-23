from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from .contracts import EXPERTS, OPERATORS, REGIMES, ContractError, MarketState


TrustMatrix = dict[str, dict[str, float]]


def validate_matrix(matrix: TrustMatrix) -> TrustMatrix:
    if set(matrix) != set(REGIMES):
        raise ContractError("trust-matrix regime rows changed")
    output: TrustMatrix = {}
    for regime in REGIMES:
        row = {str(k): float(v) for k, v in matrix[regime].items()}
        if set(row) != set(EXPERTS):
            raise ContractError("trust-matrix expert columns changed")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in row.values()):
            raise ContractError("trust weights must be finite and in [0,1]")
        if abs(sum(row.values()) - 1.0) > 1e-9:
            raise ContractError("every trust row must sum to one")
        output[regime] = row
    return output


def row_l1(first: dict[str, float], second: dict[str, float]) -> float:
    return sum(abs(float(first[name]) - float(second[name])) for name in EXPERTS)


def _mix(current: dict[str, float], target: dict[str, float], strength: float) -> dict[str, float]:
    row = {name: (1.0 - strength) * current[name] + strength * target[name] for name in EXPERTS}
    total = sum(row.values())
    return {name: max(0.0, row[name] / total) for name in EXPERTS}


def _softmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if set(scores) != set(EXPERTS):
        raise ContractError("risk scores must cover every expert")
    numbers = {name: float(scores[name]) for name in EXPERTS}
    maximum = max(numbers.values())
    exponentials = {name: math.exp(numbers[name] - maximum) for name in EXPERTS}
    denominator = sum(exponentials.values())
    return {name: exponentials[name] / denominator for name in EXPERTS}


def apply_operator(
    current: TrustMatrix,
    frozen: TrustMatrix,
    *,
    operator: str,
    target_regime: str,
    strength: float,
    expert_risk_scores: dict[str, float],
) -> TrustMatrix:
    current = validate_matrix(current)
    frozen = validate_matrix(frozen)
    if operator not in OPERATORS or target_regime not in REGIMES:
        raise ContractError("invalid trust operator or target row")
    candidate = deepcopy(current)
    if operator == "KEEP":
        return candidate
    if operator == "DEFENSIVE_SHRINK":
        target = {"cash": 1.0, "buy_hold": 0.0, "atp": 0.0}
    elif operator == "RISK_AWARE_REWEIGHT":
        target = _softmax_scores(expert_risk_scores)
    elif operator == "ROLLBACK":
        target = frozen[target_regime]
    else:  # pragma: no cover - guarded above
        raise ContractError("unsupported trust operator")
    candidate[target_regime] = _mix(current[target_regime], target, strength)
    return validate_matrix(candidate)


def allocation(probabilities: dict[str, float], matrix: TrustMatrix) -> dict[str, float]:
    matrix = validate_matrix(matrix)
    return {
        expert: sum(float(probabilities[regime]) * matrix[regime][expert] for regime in REGIMES)
        for expert in EXPERTS
    }


def btc_exposure(state: MarketState, matrix: TrustMatrix) -> float:
    weights = allocation(state.probabilities, matrix)
    return weights["buy_hold"] + weights["atp"] * state.atp_signal

