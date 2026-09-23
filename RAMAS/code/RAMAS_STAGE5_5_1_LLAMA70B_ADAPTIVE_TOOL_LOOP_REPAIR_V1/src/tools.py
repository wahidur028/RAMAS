from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .contracts import EXPERTS, MarketState
from .events import EventRecord, parse_utc
from .memory import MemoryStore, make_context_key
from .trust import TrustMatrix, apply_operator, btc_exposure, row_l1


TOOL_NAMES = (
    "inspect_current_state",
    "get_verified_events",
    "inspect_expert_risk",
    "retrieve_similar_episodes",
    "simulate_trust_candidates",
    "inspect_active_intervention",
)

REQUIRED_TOOLS = {
    "KEEP": frozenset({"inspect_current_state"}),
    "DEFENSIVE_SHRINK": frozenset(
        {"inspect_current_state", "get_verified_events", "inspect_expert_risk", "simulate_trust_candidates"}
    ),
    "RISK_AWARE_REWEIGHT": frozenset(
        {
            "inspect_current_state",
            "get_verified_events",
            "inspect_expert_risk",
            "retrieve_similar_episodes",
            "simulate_trust_candidates",
        }
    ),
    "ROLLBACK": frozenset(
        {"inspect_current_state", "inspect_active_intervention", "inspect_expert_risk", "simulate_trust_candidates"}
    ),
}


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    tool_result_id: str
    result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_result_id": self.tool_result_id,
            "result": self.result,
        }


class ToolRuntime:
    def __init__(
        self,
        *,
        state: MarketState,
        events: list[EventRecord],
        trigger_reasons: list[str],
        current_trust: TrustMatrix,
        frozen_trust: TrustMatrix,
        expert_risk_scores: dict[str, float],
        memory: MemoryStore,
        active_intervention: dict[str, Any] | None,
    ) -> None:
        self.state = state
        self.events = events
        self.trigger_reasons = trigger_reasons
        self.current_trust = current_trust
        self.frozen_trust = frozen_trust
        self.expert_risk_scores = expert_risk_scores
        self.memory = memory
        self.active_intervention = active_intervention

    def execute(self, tool_name: str) -> ToolResult:
        if tool_name not in TOOL_NAMES:
            raise ValueError(f"unknown tool: {tool_name}")
        result = getattr(self, f"_{tool_name}")()
        identifier = f"{tool_name}:{canonical_hash(result)[:20]}"
        return ToolResult(tool_name, identifier, result)

    def _inspect_current_state(self) -> dict[str, Any]:
        exposures = list(self.state.expert_exposures.values())
        return {
            "decision_time_utc": self.state.decision_time_utc,
            "probabilities": self.state.probabilities,
            "hard_regime": self.state.hard_regime,
            "router_confidence": self.state.router_confidence,
            "router_entropy": self.state.router_entropy,
            "transition_day": self.state.transition_day,
            "trigger_reasons": self.trigger_reasons,
            "expert_exposure_disagreement": max(exposures) - min(exposures),
            "current_btc_exposure": btc_exposure(self.state, self.current_trust),
            "realized_volatility_z": self.state.realized_volatility_z,
            "drawdown_90": self.state.drawdown_90,
        }

    def _get_verified_events(self) -> dict[str, Any]:
        decision_time = parse_utc(self.state.decision_time_utc)
        return {
            "events": [event.public_dict() for event in self.events],
            "event_count": len(self.events),
            "maximum_severity": max((event.severity for event in self.events), default=0),
            "all_available_by_decision_time": all(
                event.available_at <= decision_time for event in self.events
            ),
        }

    def _inspect_expert_risk(self) -> dict[str, Any]:
        return {
            "risk_adjusted_scores": {name: float(self.expert_risk_scores[name]) for name in EXPERTS},
            "interpretation": "higher_is_better_past_only_score",
        }

    def _retrieve_similar_episodes(self) -> dict[str, Any]:
        key = make_context_key(
            self.state.hard_regime,
            self.trigger_reasons,
            [event.category for event in self.events],
        )
        return {
            "context_key": key,
            "episodes": self.memory.retrieve(key, 20),
            "lesson": self.memory.lessons.get(key),
        }

    def _simulate_trust_candidates(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for operator in ("KEEP", "DEFENSIVE_SHRINK", "RISK_AWARE_REWEIGHT", "ROLLBACK"):
            for strength in ((0.0,) if operator == "KEEP" else (0.05, 0.10)):
                candidate = apply_operator(
                    self.current_trust,
                    self.frozen_trust,
                    operator=operator,
                    target_regime=self.state.hard_regime,
                    strength=strength,
                    expert_risk_scores=self.expert_risk_scores,
                )
                rows.append(
                    {
                        "operator": operator,
                        "target_regime": self.state.hard_regime,
                        "strength": strength,
                        "proposed_btc_exposure": btc_exposure(self.state, candidate),
                        "target_row_l1_change": row_l1(
                            self.current_trust[self.state.hard_regime],
                            candidate[self.state.hard_regime],
                        ),
                    }
                )
        return {"candidates": rows, "future_return_used": False}

    def _inspect_active_intervention(self) -> dict[str, Any]:
        return {"active_intervention": self.active_intervention}


def evidence_complete(operator: str, executed_tools: list[str]) -> bool:
    return REQUIRED_TOOLS[operator].issubset(set(executed_tools))


def missing_required_tools(operator: str, executed_tools: list[str]) -> tuple[str, ...]:
    executed = set(executed_tools)
    required = REQUIRED_TOOLS[operator]
    return tuple(tool for tool in TOOL_NAMES if tool in required and tool not in executed)
