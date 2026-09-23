from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REGIMES = ("bear", "bull", "mix")
EXPERTS = ("cash", "buy_hold", "atp")
OPERATORS = ("KEEP", "DEFENSIVE_SHRINK", "RISK_AWARE_REWEIGHT", "ROLLBACK")
ALLOWED_STRENGTHS = (0.0, 0.05, 0.10)
ALLOWED_HORIZONS = (3, 7, 14)


class ContractError(ValueError):
    pass


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class MarketState:
    decision_time_utc: str
    probabilities: dict[str, float]
    hard_regime: str
    transition_day: bool
    router_confidence: float
    router_entropy: float
    expert_exposures: dict[str, float]
    realized_volatility_z: float
    drawdown_90: float
    atp_signal: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MarketState":
        required = {
            "decision_time_utc",
            "probabilities",
            "hard_regime",
            "transition_day",
            "router_confidence",
            "router_entropy",
            "expert_exposures",
            "realized_volatility_z",
            "drawdown_90",
            "atp_signal",
        }
        if set(value) != required:
            raise ContractError(f"market-state keys changed: {sorted(set(value) ^ required)}")
        regime = str(value["hard_regime"])
        if regime not in REGIMES:
            raise ContractError("hard_regime is invalid")
        probabilities = {str(k): _finite(v, f"probability.{k}") for k, v in value["probabilities"].items()}
        if set(probabilities) != set(REGIMES):
            raise ContractError("probability regimes changed")
        if any(item < 0.0 or item > 1.0 for item in probabilities.values()):
            raise ContractError("probabilities must be in [0,1]")
        if abs(sum(probabilities.values()) - 1.0) > 1e-9:
            raise ContractError("probabilities must sum to one")
        exposures = {str(k): _finite(v, f"expert_exposure.{k}") for k, v in value["expert_exposures"].items()}
        if set(exposures) != set(EXPERTS):
            raise ContractError("expert exposure keys changed")
        if any(item < 0.0 or item > 1.0 for item in exposures.values()):
            raise ContractError("expert exposures must be in [0,1]")
        confidence = _finite(value["router_confidence"], "router_confidence")
        entropy = _finite(value["router_entropy"], "router_entropy")
        atp_signal = _finite(value["atp_signal"], "atp_signal")
        if not (0.0 <= confidence <= 1.0 and 0.0 <= entropy <= 1.0):
            raise ContractError("router confidence and entropy must be in [0,1]")
        if atp_signal not in (0.0, 1.0):
            raise ContractError("atp_signal must be 0 or 1")
        return cls(
            decision_time_utc=str(value["decision_time_utc"]),
            probabilities=probabilities,
            hard_regime=regime,
            transition_day=bool(value["transition_day"]),
            router_confidence=confidence,
            router_entropy=entropy,
            expert_exposures=exposures,
            realized_volatility_z=_finite(value["realized_volatility_z"], "realized_volatility_z"),
            drawdown_90=_finite(value["drawdown_90"], "drawdown_90"),
            atp_signal=atp_signal,
        )


@dataclass(frozen=True)
class AgentDecision:
    operator: str
    target_regime: str
    strength: float
    horizon_days: int
    confidence: float
    hypothesis: str
    invalidation_condition: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentDecision":
        required = {
            "operator",
            "target_regime",
            "strength",
            "horizon_days",
            "confidence",
            "hypothesis",
            "invalidation_condition",
        }
        if set(value) != required:
            raise ContractError(f"agent-decision keys changed: {sorted(set(value) ^ required)}")
        operator = str(value["operator"])
        regime = str(value["target_regime"])
        strength = _finite(value["strength"], "strength")
        horizon = int(value["horizon_days"])
        confidence = _finite(value["confidence"], "confidence")
        hypothesis = str(value["hypothesis"]).strip()
        invalidation = str(value["invalidation_condition"]).strip()
        if operator not in OPERATORS:
            raise ContractError("operator is invalid")
        if regime not in REGIMES:
            raise ContractError("target_regime is invalid")
        if strength not in ALLOWED_STRENGTHS:
            raise ContractError("strength is off the frozen grid")
        if horizon not in ALLOWED_HORIZONS:
            raise ContractError("horizon_days is off the frozen grid")
        if not 0.0 <= confidence <= 1.0:
            raise ContractError("confidence must be in [0,1]")
        if operator == "KEEP":
            if strength != 0.0:
                raise ContractError("KEEP requires zero strength")
        else:
            if strength == 0.0:
                raise ContractError("a trust intervention requires nonzero strength")
            if not hypothesis or not invalidation:
                raise ContractError("an intervention requires a hypothesis and invalidation condition")
        if len(hypothesis) > 400 or len(invalidation) > 300:
            raise ContractError("agent text exceeded the frozen length limit")
        return cls(operator, regime, strength, horizon, confidence, hypothesis, invalidation)


def validate_plan(value: dict[str, Any], allowed_tools: tuple[str, ...]) -> tuple[str, ...]:
    if set(value) != {"tool_names", "planning_reason"}:
        raise ContractError("plan keys changed")
    tools = tuple(str(item) for item in value["tool_names"])
    if not tools or len(tools) > len(allowed_tools):
        raise ContractError("tool plan size is invalid")
    if len(set(tools)) != len(tools):
        raise ContractError("tool plan contains duplicates")
    if any(item not in allowed_tools for item in tools):
        raise ContractError("tool plan contains an unknown tool")
    if not str(value["planning_reason"]).strip():
        raise ContractError("planning_reason is empty")
    return tools

