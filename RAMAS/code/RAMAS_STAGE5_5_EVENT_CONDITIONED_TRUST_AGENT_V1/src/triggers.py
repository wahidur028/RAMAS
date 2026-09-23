from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import MarketState
from .events import EventRecord


@dataclass(frozen=True)
class TriggerResult:
    triggered: bool
    reasons: tuple[str, ...]
    expert_disagreement: float
    maximum_event_severity: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "expert_disagreement": self.expert_disagreement,
            "maximum_event_severity": self.maximum_event_severity,
        }


def evaluate_trigger(state: MarketState, events: list[EventRecord], config: dict[str, Any]) -> TriggerResult:
    reasons: list[str] = []
    exposures = list(state.expert_exposures.values())
    disagreement = max(exposures) - min(exposures)
    maximum_severity = max((event.severity for event in events), default=0)
    if state.transition_day:
        reasons.append("ROUTER_TRANSITION")
    if state.router_confidence < float(config["router_confidence_strictly_below"]):
        reasons.append("LOW_ROUTER_CONFIDENCE")
    if state.router_entropy >= float(config["router_entropy_at_least"]):
        reasons.append("HIGH_ROUTER_ENTROPY")
    if disagreement >= float(config["expert_disagreement_at_least"]):
        reasons.append("EXPERT_DISAGREEMENT")
    if state.realized_volatility_z >= float(config["realized_volatility_z_at_least"]):
        reasons.append("VOLATILITY_STRESS")
    if state.drawdown_90 <= float(config["drawdown_90_at_most"]):
        reasons.append("DRAWDOWN_STRESS")
    if maximum_severity >= int(config["verified_event_severity_at_least"]):
        reasons.append("HIGH_SEVERITY_EVENT")
    return TriggerResult(bool(reasons), tuple(reasons), disagreement, maximum_severity)

