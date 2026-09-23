#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.agent import LLAMA70B_MODEL, ControlledProvider, OllamaProvider, run_agent_cycle
from src.contracts import MarketState
from src.events import (
    EventRecord,
    content_sha256,
    events_available_asof,
    load_event_ledger,
    parse_utc,
    validate_manifest,
)
from src.memory import MemoryStore
from src.tools import REQUIRED_TOOLS


PACKAGE = Path(__file__).resolve().parent


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "MECHANISM_AND_INTERFACE_DEVELOPMENT_ONLY":
        raise ValueError("Stage 5.5 cannot be relabeled as economic confirmation")
    frozen = config.get("frozen_claims")
    expected = {
        "router_changed": False,
        "expert_models_changed": False,
        "daily_execution_clock_changed": False,
        "risk_layer_bypass_permitted": False,
        "llm_direct_exposure_prediction_permitted": False,
        "llm_arbitrary_weight_generation_permitted": False,
        "multi_agent_enabled": False,
        "reused_oos_opened": False,
        "economic_value_established": False,
    }
    if frozen != expected:
        raise ValueError("the frozen Stage 5.5 scientific boundary changed")
    if config["shield"]["maximum_active_interventions"] != 1:
        raise ValueError("more than one active intervention is prohibited")
    if config["economic_development_gate"]["primary_comparator"] != (
        "deterministic_event_triggered_trust_controller"
    ):
        raise ValueError("the primary comparator changed")
    if config["provider"]["model"] != LLAMA70B_MODEL:
        raise ValueError(f"the model lock must remain {LLAMA70B_MODEL}")
    if int(config["tool_loop"]["maximum_decision_rounds"]) != 2:
        raise ValueError("the adaptive tool loop must remain bounded to two decision rounds")
    configured_required = {
        operator: set(tools)
        for operator, tools in config["tool_loop"]["required_evidence_by_operator"].items()
    }
    expected_required = {operator: set(tools) for operator, tools in REQUIRED_TOOLS.items()}
    if configured_required != expected_required:
        raise ValueError("required tool evidence changed from the executable contract")


def uniform_trust() -> dict[str, dict[str, float]]:
    return {
        regime: {"cash": 1.0 / 3.0, "buy_hold": 1.0 / 3.0, "atp": 1.0 / 3.0}
        for regime in ("bear", "bull", "mix")
    }


def synthetic_case(
    index: int, positive_path_indices: set[int]
) -> tuple[MarketState, list[EventRecord], bool]:
    regimes = ("bear", "bull", "mix")
    categories = (
        "macro",
        "regulation",
        "exchange",
        "stablecoin",
        "institutional_flow",
        "onchain_stress",
        "derivatives_stress",
        "security",
        "macro",
        "exchange",
        "regulation",
        "stablecoin",
    )
    regime = regimes[index % 3]
    probabilities = {
        "bear": {"bear": 0.48, "bull": 0.22, "mix": 0.30},
        "bull": {"bear": 0.18, "bull": 0.52, "mix": 0.30},
        "mix": {"bear": 0.25, "bull": 0.25, "mix": 0.50},
    }[regime]
    decision = datetime(2023, 1, 10, 23, 59, tzinfo=timezone.utc) + timedelta(days=index * 20)
    event_time = decision - timedelta(hours=4)
    positive_path = index in positive_path_indices
    if positive_path:
        headline = f"Verified severe point-in-time {categories[index]} stress event {index + 1}"
        body = (
            "Contemporaneous interface evidence reports severe operational stress while market volatility "
            "and drawdown are elevated. No future return is supplied."
        )
    else:
        headline = f"Synthetic point-in-time {categories[index]} event {index + 1}"
        body = "Ambiguous interface-only evidence with no future return or expected action label."
    event = EventRecord(
        event_id=f"synthetic-{index + 1:03d}",
        published_at=event_time,
        available_at=event_time + timedelta(minutes=2),
        source="mechanical-preflight",
        source_url="",
        category=categories[index],
        severity=5 if positive_path else 3 + index % 2,
        headline=headline,
        body=body,
        content_sha256=content_sha256(headline, body),
    )
    state = MarketState.from_dict(
        {
            "decision_time_utc": decision.isoformat(),
            "probabilities": probabilities,
            "hard_regime": regime,
            "transition_day": index % 4 == 0,
            "router_confidence": max(probabilities.values()),
            "router_entropy": 0.78,
            "expert_exposures": {"cash": 0.0, "buy_hold": 1.0, "atp": float(index % 2)},
            "realized_volatility_z": 3.0 if positive_path else 1.2 + 0.2 * (index % 4),
            "drawdown_90": -0.20 if positive_path else -0.08 - 0.02 * (index % 4),
            "atp_signal": float(index % 2),
        }
    )
    return state, [event], positive_path


def run_preflight(config: dict[str, Any], provider_kind: str, model: str | None) -> dict[str, Any]:
    provider_config = dict(config["provider"])
    if model:
        provider_config["model"] = model
    if provider_kind == "ollama" and provider_config["model"] != LLAMA70B_MODEL:
        raise ValueError(f"this experiment permits only {LLAMA70B_MODEL}")
    provider = ControlledProvider() if provider_kind == "controlled" else OllamaProvider(provider_config)
    current = uniform_trust()
    memory = MemoryStore(config["memory"])
    traces: list[dict[str, Any]] = []
    count = int(config["interface_preflight"]["synthetic_cases"])
    positive_path_indices = set(config["interface_preflight"]["positive_path_case_indices"])
    for index in range(count):
        state, events, positive_path = synthetic_case(index, positive_path_indices)
        trace = run_agent_cycle(
            state=state,
            visible_events=events,
            current_trust=current,
            frozen_trust=uniform_trust(),
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
            active_intervention=None,
            memory=memory,
            provider=provider,
            config=config,
        )
        trace["case_index"] = index
        trace["positive_path_case"] = positive_path
        traces.append(trace)
        print(
            f"STAGE551_PREFLIGHT={index + 1}/{count} "
            f"PLAN_VALID={trace['plan_valid']} DECISION_VALID={trace['decision_valid']} "
            f"ROUNDS={trace['decision_round_count']} EVIDENCE={trace['operator_evidence_complete']} "
            f"PROPOSED={trace['decision']['operator']} EFFECTIVE={trace['effective_decision']['operator']} "
            f"SAFE={trace['safe_final_action']} INTERVENTION={trace['intervention_executed']}",
            flush=True,
        )
    valid_plans = sum(item["plan_valid"] is True for item in traces)
    valid_decisions = sum(item["decision_valid"] is True for item in traces)
    evidence_complete_count = sum(item["operator_evidence_complete"] is True for item in traces)
    safe_final_actions = sum(item["safe_final_action"] is True for item in traces)
    positive_path_interventions = sum(
        item["positive_path_case"] is True and item["intervention_executed"] is True
        for item in traces
    )
    passed = (
        valid_plans >= int(config["interface_preflight"]["minimum_valid_plans"])
        and valid_decisions >= int(config["interface_preflight"]["minimum_valid_decisions"])
        and evidence_complete_count
        >= int(config["interface_preflight"]["minimum_operator_evidence_complete"])
        and safe_final_actions >= int(config["interface_preflight"]["minimum_safe_final_actions"])
        and positive_path_interventions
        >= int(config["interface_preflight"]["minimum_positive_path_interventions"])
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "provider": provider.kind,
        "model": provider_config["model"] if provider_kind == "ollama" else None,
        "cases": count,
        "valid_plans": valid_plans,
        "valid_decisions": valid_decisions,
        "operator_evidence_complete": evidence_complete_count,
        "safe_final_actions": safe_final_actions,
        "correction_cycles_used": sum(item["decision_round_count"] == 2 for item in traces),
        "accepted_interventions": sum(item["intervention_executed"] is True for item in traces),
        "positive_path_interventions": positive_path_interventions,
        "economic_claim_permitted": False,
        "traces": traces,
    }


def evaluate_data_gate(event_csv: Path | None, manifest: Path | None) -> dict[str, Any]:
    if event_csv is None and manifest is None:
        return {
            "status": "BLOCKED",
            "reason": "POINT_IN_TIME_EVENT_LEDGER_NOT_PROVIDED",
            "economic_replay_permitted": False,
        }
    if event_csv is None or manifest is None:
        return {
            "status": "FAIL",
            "reason": "EVENT_LEDGER_AND_MANIFEST_MUST_BE_PROVIDED_TOGETHER",
            "economic_replay_permitted": False,
        }
    try:
        manifest_value = validate_manifest(manifest, event_csv)
        events = load_event_ledger(event_csv)
        first = events[0].available_at.isoformat()
        last = events[-1].available_at.isoformat()
        coverage_start = parse_utc(manifest_value["coverage_start_utc"])
        coverage_end = parse_utc(manifest_value["coverage_end_utc"])
        if coverage_start > coverage_end:
            raise ValueError("manifest coverage start follows coverage end")
        if events[0].available_at < coverage_start or events[-1].available_at > coverage_end:
            raise ValueError("event availability lies outside declared manifest coverage")
        if len(events_available_asof(events, last)) != len(events):
            raise ValueError("as-of filtering failed")
        return {
            "status": "PASS",
            "reason": "POINT_IN_TIME_EVENT_LEDGER_VALID",
            "dataset_id": manifest_value["dataset_id"],
            "events": len(events),
            "first_available_at_utc": first,
            "last_available_at_utc": last,
            "economic_replay_permitted": False,
            "next_authorized_step": "BUILD_CAUSAL_DEVELOPMENT_ADAPTER",
        }
    except Exception as exc:
        return {"status": "FAIL", "reason": str(exc), "economic_replay_permitted": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--provider", choices=("controlled", "ollama"), default="controlled")
    parser.add_argument("--model")
    parser.add_argument("--event-csv", type=Path)
    parser.add_argument("--event-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(PACKAGE / "config.json")
    validate_config(config)
    args.output.mkdir(parents=True, exist_ok=False)
    preflight = run_preflight(config, args.provider, args.model)
    data_gate = evaluate_data_gate(args.event_csv, args.event_manifest)
    if preflight["status"] != "PASS":
        decision = "REVISE_LLAMA70B_ADAPTIVE_TOOL_LOOP_FAILED"
    elif data_gate["status"] == "BLOCKED":
        decision = "MECHANISM_PASS_AWAIT_POINT_IN_TIME_EVENT_DATA"
    elif data_gate["status"] == "FAIL":
        decision = "REVISE_EVENT_DATA_PROVENANCE_GATE_FAILED"
    else:
        decision = "READY_TO_BUILD_CAUSAL_DEVELOPMENT_ADAPTER"
    final = {
        "decision": decision,
        "mechanism_tests_separate_from_economic_value": True,
        "interface_preflight_passed": preflight["status"] == "PASS",
        "point_in_time_event_data_gate_passed": data_gate["status"] == "PASS",
        "economic_value_established": False,
        "economic_replay_run": False,
        "reused_oos_opened": False,
        "router_changed": False,
        "risk_layer_bypass_permitted": False,
    }
    contract = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "provider": args.provider,
        "model": args.model or config["provider"]["model"],
        "project_root": str(args.project_root.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_claims": config["frozen_claims"],
    }
    report = (
        "# RAMAS Stage 5.5.1 Llama-70B adaptive tool-loop result\n\n"
        f"Decision: `{decision}`\n\n"
        "The repaired bounded event-conditioned trust-agent mechanism and the economic experiment are separate gates. "
        "The interface gate now requires evidence-complete and shield-safe final actions, plus at least one "
        "accepted intervention in a severe synthetic positive-path case. "
        "Passing interface tests does not show higher profit or lower risk. A causal development replay remains "
        "blocked until a hashed point-in-time event ledger and its provenance manifest pass validation.\n"
    )
    atomic_json(args.output / "00_CONTRACT.json", contract)
    atomic_json(args.output / "01_INTERFACE_PREFLIGHT.json", preflight)
    atomic_json(args.output / "02_EVENT_DATA_GATE.json", data_gate)
    atomic_json(args.output / "03_FINAL_DECISION.json", final)
    (args.output / "04_PLAIN_ENGLISH_REPORT.md").write_text(report, encoding="utf-8")
    atomic_json(args.output / "RUN_COMPLETE.json", {"status": "PASS", "decision": decision})
    print("RAMAS_STAGE5_5_1_STATUS=PASS")
    print(f"DECISION={decision}")
    print(f"INTERFACE_PREFLIGHT_PASSED={str(final['interface_preflight_passed']).upper()}")
    print(f"EVENT_DATA_GATE_PASSED={str(final['point_in_time_event_data_gate_passed']).upper()}")
    print("ECONOMIC_VALUE_ESTABLISHED=FALSE")
    print("REUSED_OOS_OPENED=FALSE")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()
