from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import Any, Protocol

from .contracts import AgentDecision, MarketState, OPERATORS, validate_plan
from .events import EventRecord, events_available_asof
from .memory import MemoryStore
from .shield import ShieldResult, evaluate_shield
from .tools import (
    TOOL_NAMES,
    ToolResult,
    ToolRuntime,
    evidence_complete,
    missing_required_tools,
)
from .trust import TrustMatrix, apply_operator
from .triggers import evaluate_trigger


LLAMA70B_MODEL = "llama3.3:70b"


class Provider(Protocol):
    kind: str

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class ControlledProvider:
    """Deterministic interface exerciser; never evidence of economic value."""

    kind = "controlled_non_economic"

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"tool_names": list(TOOL_NAMES), "planning_reason": "mechanical contract exercise"}

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = payload["market_state"]
        by_name = {item["tool_name"]: item["result"] for item in payload["tool_results"]}
        event_severity = by_name.get("get_verified_events", {}).get("maximum_severity", 0)
        volatility = by_name.get("inspect_current_state", {}).get("realized_volatility_z", 0.0)
        if event_severity >= 5 and volatility >= 2.0:
            return {
                "operator": "DEFENSIVE_SHRINK",
                "target_regime": state["hard_regime"],
                "strength": 0.05,
                "horizon_days": 3,
                "confidence": 0.7,
                "hypothesis": "Mechanical positive-path case has severe verified stress.",
                "invalidation_condition": "Verified stress or volatility falls below the test threshold.",
            }
        return {
            "operator": "KEEP",
            "target_regime": state["hard_regime"],
            "strength": 0.0,
            "horizon_days": 7,
            "confidence": 0.5,
            "hypothesis": "Controlled provider keeps trust unchanged outside positive-path cases.",
            "invalidation_condition": "",
        }


class OllamaProvider:
    kind = "ollama"

    def __init__(self, config: dict[str, Any]) -> None:
        self.endpoint = str(config["endpoint"])
        self.model = str(config["model"])
        if self.model != LLAMA70B_MODEL:
            raise ValueError(f"this experiment is locked to {LLAMA70B_MODEL}")
        self.temperature = float(config["temperature"])
        self.seed = int(config["seed"])
        self.timeout = int(config["timeout_seconds"])
        self.num_ctx = int(config["num_ctx"])
        self.num_predict = int(config["num_predict"])
        self.keep_alive = str(config["keep_alive"])

    def _call(self, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, sort_keys=True)},
            ],
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
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
            "You are the evidence-planning part of a bounded financial research agent. Return exactly one "
            "JSON object with only tool_names and planning_reason. tool_names is an ordered, duplicate-free "
            "subset of available_tools. You cannot trade, predict BTC exposure, or create arbitrary weights. "
            "Any possible non-KEEP trust action requires state, verified events, expert risk, and candidate "
            "simulation. Risk-aware reweighting also requires similar episodes; rollback requires active-"
            "intervention inspection. When uncertain, collect the evidence rather than guessing."
        )
        return self._call(system, payload)

    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are the decision part of a bounded trust-control research agent. Return exactly one JSON "
            "object with only operator, target_regime, strength, horizon_days, confidence, hypothesis, and "
            "invalidation_condition. operator must be KEEP, DEFENSIVE_SHRINK, RISK_AWARE_REWEIGHT, or "
            "ROLLBACK. strength must be 0, 0.05, or 0.10; KEEP requires 0. horizon_days must be 3, 7, or 14. "
            "Never output BTC exposure or weights. Use only tool_results available by the decision time. "
            "If revision_feedback is present, treat it as a mandatory runtime correction: do not repeat an "
            "unsafe proposal, and do not choose an operator whose required evidence is absent. Choose KEEP "
            "when support is missing or conflicting."
        )
        return self._call(system, payload)


def _fallback_decision(regime: str, reason: str = "Provider or interface failure") -> AgentDecision:
    return AgentDecision(
        operator="KEEP",
        target_regime=regime,
        strength=0.0,
        horizon_days=7,
        confidence=0.0,
        hypothesis=f"{reason}; trust remains unchanged.",
        invalidation_condition="",
    )


def _execute_unique(
    runtime: ToolRuntime,
    tool_names: tuple[str, ...] | list[str],
    results: list[ToolResult],
) -> list[str]:
    already = {item.tool_name for item in results}
    added: list[str] = []
    for tool_name in tool_names:
        if tool_name in already:
            continue
        results.append(runtime.execute(tool_name))
        already.add(tool_name)
        added.append(tool_name)
    return added


def _preview(
    *,
    decision: AgentDecision,
    state: MarketState,
    current_trust: TrustMatrix,
    frozen_trust: TrustMatrix,
    expert_risk_scores: dict[str, float],
    active_intervention: dict[str, Any] | None,
    executed_tools: list[str],
    config: dict[str, Any],
) -> tuple[TrustMatrix, bool, ShieldResult]:
    complete = evidence_complete(decision.operator, executed_tools)
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
        operator=decision.operator,
        target_regime=decision.target_regime,
        config=config["shield"],
        active_intervention=active_intervention is not None,
        evidence_complete=complete,
    )
    return proposed, complete, shield


def _decision_payload(
    *,
    state: MarketState,
    trigger: dict[str, Any],
    results: list[ToolResult],
    round_number: int,
    revision_feedback: dict[str, Any] | None,
) -> dict[str, Any]:
    executed = [result.tool_name for result in results]
    return {
        "decision_round": round_number,
        "market_state": asdict(state),
        "trigger": trigger,
        "tool_results": [result.as_dict() for result in results],
        "executed_tool_names": executed,
        "operators_with_complete_evidence": [
            operator for operator in OPERATORS if evidence_complete(operator, executed)
        ],
        "allowed_contract": {
            "operators": list(OPERATORS),
            "strengths": [0.0, 0.05, 0.10],
            "horizons": [3, 7, 14],
        },
        "revision_feedback": revision_feedback,
    }


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
    fallback = _fallback_decision(state.hard_regime)
    if not trigger.triggered:
        return {
            "status": "NOT_TRIGGERED",
            "provider": provider.kind,
            "trigger": trigger.as_dict(),
            "plan_valid": None,
            "decision_valid": None,
            "decision": asdict(fallback),
            "effective_decision": asdict(fallback),
            "executed_tools": [],
            "adaptive_tools_executed": [],
            "automatic_evidence_ids": [],
            "future_events_hidden": future_events_hidden,
            "operator_evidence_complete": None,
            "shield": None,
            "decision_round_count": 0,
            "decision_rounds": [],
            "safe_final_action": True,
            "intervention_executed": False,
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
        "required_evidence_by_operator": config["tool_loop"]["required_evidence_by_operator"],
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

    results: list[ToolResult] = []
    _execute_unique(runtime, planned_tools, results)
    trigger_dict = trigger.as_dict()
    decision_rounds: list[dict[str, Any]] = []
    adaptive_tools: list[str] = []

    first_payload = _decision_payload(
        state=state,
        trigger=trigger_dict,
        results=results,
        round_number=1,
        revision_feedback=None,
    )
    try:
        first_decision = AgentDecision.from_dict(provider.decide(first_payload))
        first_valid = True
        first_error = None
    except Exception as exc:
        first_decision = _fallback_decision(state.hard_regime, "First decision round failed")
        first_valid = False
        first_error = str(exc)
    first_tools = [item.tool_name for item in results]
    _, first_complete, first_shield = _preview(
        decision=first_decision,
        state=state,
        current_trust=current_trust,
        frozen_trust=frozen_trust,
        expert_risk_scores=expert_risk_scores,
        active_intervention=active_intervention,
        executed_tools=first_tools,
        config=config,
    )
    decision_rounds.append(
        {
            "round": 1,
            "valid": first_valid,
            "error": first_error,
            "decision": asdict(first_decision),
            "evidence_complete_at_decision": first_complete,
            "shield_preview": first_shield.as_dict(),
        }
    )

    final_decision = first_decision
    final_valid = first_valid
    final_error = first_error
    revision_requested = bool(
        first_valid
        and int(config["tool_loop"]["maximum_decision_rounds"]) >= 2
        and (not first_complete or not first_shield.accepted)
    )
    if revision_requested:
        missing = missing_required_tools(first_decision.operator, first_tools)
        adaptive_tools = _execute_unique(runtime, missing, results)
        tools_after = [item.tool_name for item in results]
        _, complete_after, shield_after = _preview(
            decision=first_decision,
            state=state,
            current_trust=current_trust,
            frozen_trust=frozen_trust,
            expert_risk_scores=expert_risk_scores,
            active_intervention=active_intervention,
            executed_tools=tools_after,
            config=config,
        )
        feedback = {
            "draft_decision": asdict(first_decision),
            "draft_evidence_complete_before_continuation": first_complete,
            "missing_tools_identified": list(missing),
            "adaptive_tools_executed": adaptive_tools,
            "draft_evidence_complete_after_continuation": complete_after,
            "draft_shield_after_continuation": shield_after.as_dict(),
            "mandatory_instruction": (
                "Re-evaluate from all tool results. Use only an operator listed in "
                "operators_with_complete_evidence and a simulated candidate that passes every shield limit. "
                "Otherwise choose KEEP."
            ),
        }
        second_payload = _decision_payload(
            state=state,
            trigger=trigger_dict,
            results=results,
            round_number=2,
            revision_feedback=feedback,
        )
        try:
            final_decision = AgentDecision.from_dict(provider.decide(second_payload))
            final_valid = True
            final_error = None
        except Exception as exc:
            final_decision = _fallback_decision(state.hard_regime, "Correction decision round failed")
            final_valid = False
            final_error = str(exc)
        second_tools = [item.tool_name for item in results]
        _, second_complete, second_shield = _preview(
            decision=final_decision,
            state=state,
            current_trust=current_trust,
            frozen_trust=frozen_trust,
            expert_risk_scores=expert_risk_scores,
            active_intervention=active_intervention,
            executed_tools=second_tools,
            config=config,
        )
        decision_rounds.append(
            {
                "round": 2,
                "valid": final_valid,
                "error": final_error,
                "decision": asdict(final_decision),
                "evidence_complete_at_decision": second_complete,
                "shield_preview": second_shield.as_dict(),
                "revision_feedback": feedback,
            }
        )

    executed_tools = [item.tool_name for item in results]
    final_proposed, final_complete, final_shield = _preview(
        decision=final_decision,
        state=state,
        current_trust=current_trust,
        frozen_trust=frozen_trust,
        expert_risk_scores=expert_risk_scores,
        active_intervention=active_intervention,
        executed_tools=executed_tools,
        config=config,
    )
    safe_final = bool(plan_valid and final_valid and final_complete and final_shield.accepted)
    intervention_executed = bool(safe_final and final_decision.operator != "KEEP")
    effective_decision = final_decision if safe_final else _fallback_decision(
        state.hard_regime, "Final proposal failed the bounded tool-loop contract"
    )
    effective_trust = final_proposed if intervention_executed else current_trust
    if intervention_executed:
        status = "INTERVENTION_ACCEPTED"
    elif safe_final:
        status = "KEEP_ACCEPTED"
    else:
        status = "FAIL_CLOSED_TO_KEEP"
    return {
        "status": status,
        "provider": provider.kind,
        "trigger": trigger_dict,
        "plan_valid": plan_valid,
        "plan_error": plan_error,
        "planned_tools": list(planned_tools),
        "decision_valid": final_valid,
        "decision_error": final_error,
        "decision": asdict(final_decision),
        "effective_decision": asdict(effective_decision),
        "executed_tools": executed_tools,
        "adaptive_tools_executed": adaptive_tools,
        "automatic_evidence_ids": [result.tool_result_id for result in results],
        "future_events_hidden": future_events_hidden,
        "tool_results": [result.as_dict() for result in results],
        "initial_operator_evidence_complete": first_complete,
        "operator_evidence_complete": final_complete,
        "revision_requested": revision_requested,
        "decision_round_count": len(decision_rounds),
        "decision_rounds": decision_rounds,
        "shield": final_shield.as_dict(),
        "safe_final_action": safe_final,
        "intervention_executed": intervention_executed,
        "accepted": intervention_executed,
        "proposed_trust": effective_trust,
    }
