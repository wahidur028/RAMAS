from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import Any, Protocol

from .contracts import AgentDecision, ContractError, MarketState, validate_plan
from .events import EventRecord, events_available_asof
from .memory import MemoryStore
from .shield import evaluate_shield
from .tools import TOOL_NAMES, ToolRuntime, evidence_complete
from .trust import TrustMatrix, apply_operator
from .triggers import evaluate_trigger


class Provider(Protocol):
    kind: str

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ControlledProvider:
    kind = "controlled_non_economic"

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"tool_names": list(TOOL_NAMES), "planning_reason": "mechanical contract exercise"}

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload["market_state"]
        return {
            "operator": "KEEP",
            "target_regime": state["hard_regime"],
            "strength": 0.0,
            "horizon_days": 7,
            "confidence": 0.5,
            "hypothesis": "Controlled provider makes no economic decision.",
            "invalidation_condition": "",
        }


class OllamaProvider:
    kind = "ollama"

    def __init__(self, config: dict[str, Any]) -> None:
        self.endpoint = str(config["endpoint"])
        self.model = str(config["model"])
        self.temperature = float(config["temperature"])
        self.seed = int(config["seed"])
        self.timeout = int(config["timeout_seconds"])

    def _call(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, sort_keys=True)},
            ],
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        encoded = json.dumps(request).encode("utf-8")
        http = urllib.request.Request(
            self.endpoint,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(http, timeout=self.timeout) as response:
            outer = json.loads(response.read().decode("utf-8"))
        if "error" in outer:
            raise RuntimeError(f"Ollama error: {outer['error']}")
        return json.loads(outer["message"]["content"])

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are a bounded evidence-gathering financial agent. Choose an ordered subset of the visible "
            "tools. Return exactly JSON with keys tool_names and planning_reason. You cannot trade or choose "
            "BTC exposure directly. For a possible trust intervention, inspect state, verified events, risk, "
            "similar episodes, and candidate simulation."
        )
        return self._call(system, payload)

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are a bounded trust-control agent. Return exactly one JSON object with keys operator, "
            "target_regime, strength, horizon_days, confidence, hypothesis, invalidation_condition. "
            "operator is KEEP, DEFENSIVE_SHRINK, RISK_AWARE_REWEIGHT, or ROLLBACK. strength is 0, 0.05, or "
            "0.10; KEEP requires 0. horizon_days is 3, 7, or 14. Never output BTC exposure or arbitrary weights. "
            "Use only the supplied tool results. Prefer KEEP when evidence is missing or conflicting."
        )
        return self._call(system, payload)


def _fallback_decision(regime: str) -> AgentDecision:
    return AgentDecision(
        operator="KEEP",
        target_regime=regime,
        strength=0.0,
        horizon_days=7,
        confidence=0.0,
        hypothesis="Provider or interface failure; trust remains unchanged.",
        invalidation_condition="",
    )


def run_agent_cycle(
    *,
    state: MarketState,
    visible_events: list[EventRecord],
    current_trust: TrustMatrix,
    frozen_trust: TrustMatrix,
    expert_risk_scores: dict[str, float],
    active_intervention: dict[str, Any] | None,
    memory: MemoryStore,
    provider: Provider,
    config: dict[str, Any],
) -> dict[str, Any]:
    causal_events = events_available_asof(visible_events, state.decision_time_utc)
    future_events_hidden = len(visible_events) - len(causal_events)
    trigger = evaluate_trigger(state, causal_events, config["trigger"])
    if not trigger.triggered:
        return {
            "status": "NOT_TRIGGERED",
            "provider": provider.kind,
            "trigger": trigger.as_dict(),
            "plan_valid": None,
            "decision_valid": None,
            "decision": asdict(_fallback_decision(state.hard_regime)),
            "executed_tools": [],
            "automatic_evidence_ids": [],
            "future_events_hidden": future_events_hidden,
            "shield": None,
            "accepted": False,
            "proposed_trust": current_trust,
        }
    runtime = ToolRuntime(
        state=state,
        events=causal_events,
        trigger_reasons=list(trigger.reasons),
        current_trust=current_trust,
        frozen_trust=frozen_trust,
        expert_risk_scores=expert_risk_scores,
        memory=memory,
        active_intervention=active_intervention,
    )
    planning_payload = {
        "market_state": asdict(state),
        "trigger": trigger.as_dict(),
        "available_tools": list(TOOL_NAMES),
    }
    try:
        plan_value = provider.plan(planning_payload)
        planned_tools = validate_plan(plan_value, TOOL_NAMES)
        plan_valid = True
        plan_error = None
    except Exception as exc:
        planned_tools = ("inspect_current_state",)
        plan_valid = False
        plan_error = str(exc)
    results = [runtime.execute(tool) for tool in planned_tools]
    decision_payload = {
        "market_state": asdict(state),
        "trigger": trigger.as_dict(),
        "tool_results": [result.as_dict() for result in results],
        "allowed_contract": {
            "operators": ["KEEP", "DEFENSIVE_SHRINK", "RISK_AWARE_REWEIGHT", "ROLLBACK"],
            "strengths": [0.0, 0.05, 0.10],
            "horizons": [3, 7, 14],
        },
    }
    try:
        decision = AgentDecision.from_dict(provider.decide(decision_payload))
        decision_valid = True
        decision_error = None
    except Exception as exc:
        decision = _fallback_decision(state.hard_regime)
        decision_valid = False
        decision_error = str(exc)
    complete = evidence_complete(decision.operator, list(planned_tools))
    proposed = apply_operator(
        current_trust,
        frozen_trust,
        operator=decision.operator,
        target_regime=decision.target_regime,
        strength=decision.strength,
        expert_risk_scores=expert_risk_scores,
    )
    shield = evaluate_shield(
        state,
        current_trust,
        proposed,
        target_regime=decision.target_regime,
        config=config["shield"],
        active_intervention=active_intervention is not None,
        evidence_complete=complete,
    )
    accepted = bool(decision.operator != "KEEP" and plan_valid and decision_valid and shield.accepted)
    if not accepted:
        proposed = current_trust
    status = "INTERVENTION_ACCEPTED" if accepted else "KEEP_OR_FAIL_CLOSED"
    return {
        "status": status,
        "provider": provider.kind,
        "trigger": trigger.as_dict(),
        "plan_valid": plan_valid,
        "plan_error": plan_error,
        "decision_valid": decision_valid,
        "decision_error": decision_error,
        "decision": asdict(decision),
        "executed_tools": list(planned_tools),
        "automatic_evidence_ids": [result.tool_result_id for result in results],
        "future_events_hidden": future_events_hidden,
        "tool_results": [result.as_dict() for result in results],
        "operator_evidence_complete": complete,
        "shield": shield.as_dict(),
        "accepted": accepted,
        "proposed_trust": proposed,
    }
