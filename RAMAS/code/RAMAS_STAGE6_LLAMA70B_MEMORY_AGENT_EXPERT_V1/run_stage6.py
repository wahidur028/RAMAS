#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage6lib.agent import ControlledProvider, OllamaProvider, SYSTEM_PROMPT
from stage6lib.contracts import ContractError, validate_decision
from stage6lib.memory import EpisodicMemory, state_vector
from stage6lib.trust import RegimeTrust, blend_exposure
from stage6lib import source_adapter as source


PACKAGE = Path(__file__).resolve().parent
TOL = 1e-12


class Stage6Error(RuntimeError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1":
        raise Stage6Error("unexpected experiment_id")
    if value.get("provider", {}).get("model") != "llama3.3:70b":
        raise Stage6Error("Stage 6 is locked to llama3.3:70b")
    claims = value.get("frozen_claims", {})
    if claims.get("qwen_used") is not False:
        raise Stage6Error("Qwen must remain disabled")
    if claims.get("router_changed") is not False:
        raise Stage6Error("router must remain frozen")
    if claims.get("existing_ramoe_trust_changed") is not False:
        raise Stage6Error("existing RAMAS trust must remain frozen")
    if claims.get("risk_layer_changed") is not False:
        raise Stage6Error("risk layer must remain frozen")
    if not 0.0 <= float(value["trust"]["initial_beta"]) <= float(value["trust"]["maximum_beta"]) <= 0.20:
        raise Stage6Error("Llama trust bounds are invalid")
    return value


def provider_from(config: dict[str, Any], kind: str) -> Any:
    if kind == "controlled":
        return ControlledProvider()
    return OllamaProvider(config["provider"], config["agent"])


def semantic_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(expected: str, bear: float, bull: float, mix: float, entropy: float,
            vol: float, drawdown: float, return_30: float, suffix: str) -> None:
        cases.append(
            {
                "case_id": f"{expected.lower()}-{suffix}",
                "expected_action": expected,
                "state": {
                    "decision_date": "2021-06-30",
                    "target_return_date": "2021-07-01",
                    "prob_bear": bear,
                    "prob_bull": bull,
                    "prob_mix": mix,
                    "hard_regime": max(
                        [(bear, "bear"), (bull, "bull"), (mix, "mix")]
                    )[1],
                    "router_entropy": entropy,
                    "router_confidence": max(bear, bull, mix),
                    "router_transition_l1": 0.10,
                    "return_1": return_30 / 10.0,
                    "return_7": return_30 / 3.0,
                    "return_30": return_30,
                    "return_90": return_30 * 2.0,
                    "realized_vol_30": vol,
                    "drawdown_90": drawdown,
                    "base_ramoe_desired_exposure": 0.50,
                    "llama_trust_beta": 0.05,
                },
            }
        )

    add("BTC", 0.08, 0.82, 0.10, 0.42, 0.45, -0.03, 0.12, "a")
    add("BTC", 0.05, 0.90, 0.05, 0.25, 0.60, -0.01, 0.08, "b")
    add("BTC", 0.10, 0.78, 0.12, 0.48, 0.70, -0.06, 0.05, "c")
    add("BTC", 0.07, 0.85, 0.08, 0.34, 0.35, -0.02, 0.15, "d")
    add("CASH", 0.80, 0.10, 0.10, 0.43, 0.70, -0.12, -0.10, "a")
    add("CASH", 0.76, 0.12, 0.12, 0.50, 1.10, -0.18, -0.16, "b")
    add("CASH", 0.35, 0.55, 0.10, 0.72, 1.00, -0.31, -0.22, "c")
    add("CASH", 0.84, 0.08, 0.08, 0.35, 0.50, -0.08, -0.06, "d")
    add("ABSTAIN", 0.34, 0.33, 0.33, 0.99, 0.70, -0.10, 0.01, "a")
    add("ABSTAIN", 0.42, 0.40, 0.18, 0.93, 0.90, -0.15, -0.02, "b")
    add("ABSTAIN", 0.20, 0.45, 0.35, 0.91, 0.60, -0.08, 0.04, "c")
    add("ABSTAIN", 0.48, 0.30, 0.22, 0.92, 1.20, -0.20, -0.05, "d")
    return cases


def run_semantic_preflight(provider: Any, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for index, case in enumerate(semantic_cases(), start=1):
        payload = {
            "task_type": "semantic_contract_test",
            "case_id": case["case_id"],
            "state": case["state"],
            "memory": {"similar_completed_episodes": [], "summary": "NO_COMPLETED_SIMILAR_EPISODES"},
            "safe_exposure_previews": {
                "BTC": 0.525,
                "CASH": 0.475,
                "ABSTAIN": 0.500,
            },
        }
        raw = ""
        error = ""
        latency = 0.0
        outer: dict[str, Any] = {}
        valid = False
        expected = False
        decision: dict[str, Any] = {
            "action": "ABSTAIN",
            "confidence": 0.0,
            "reason_codes": ["INSUFFICIENT_EVIDENCE"],
            "cited_memory_ids": [],
        }
        try:
            raw, outer, latency = provider.complete(payload)
            decision = validate_decision(raw, config["agent"], set())
            valid = True
            expected = decision["action"] == case["expected_action"]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records.append(
            {
                "case_id": case["case_id"],
                "expected_action": case["expected_action"],
                "decision": decision,
                "valid": valid,
                "expected_action_matched": expected,
                "request": payload,
                "raw_response": raw,
                "provider_response": outer,
                "latency_seconds": latency,
                "error": error,
            }
        )
        print(
            f"STAGE6_PREFLIGHT={index}/{len(semantic_cases())} "
            f"EXPECTED={case['expected_action']} ACTION={decision['action']} "
            f"VALID={valid} MATCH={expected}",
            flush=True,
        )
    valid_rate = float(np.mean([item["valid"] for item in records]))
    expected_rate = float(np.mean([item["expected_action_matched"] for item in records]))
    actions = sorted({item["decision"]["action"] for item in records if item["valid"]})
    spec = config["semantic_preflight"]
    checks = {
        "valid_rate": valid_rate + TOL >= float(spec["minimum_valid_rate"]),
        "expected_action_rate": expected_rate + TOL >= float(spec["minimum_expected_action_rate"]),
        "all_required_actions_observed": set(spec["required_actions"]).issubset(actions),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": (
            "CONTINUE_TO_2021_DEVELOPMENT_PILOT"
            if all(checks.values())
            else "STOP_LLAMA70B_SEMANTIC_CONTRACT_FAILED"
        ),
        "provider": provider.kind,
        "model": config["provider"]["model"] if provider.kind == "ollama" else "CONTROLLED",
        "system_prompt_sha256": source.sha256_text(SYSTEM_PROMPT),
        "config_sha256": source.sha256_file(PACKAGE / "config.json"),
        "valid_rate": valid_rate,
        "expected_action_rate": expected_rate,
        "observed_actions": actions,
        "checks": checks,
    }
    return summary, records


def write_preflight(output: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if output.exists():
        raise Stage6Error(f"refusing to overwrite output={output}")
    output.mkdir(parents=True)
    source.atomic_json(output / "00_PREFLIGHT.json", summary)
    source.atomic_jsonl(output / "01_PREFLIGHT_CALLS.jsonl", records)
    hashes = {
        path.name: source.sha256_file(path)
        for path in sorted(output.iterdir()) if path.is_file()
    }
    source.atomic_json(
        output / "RUN_COMPLETE.json",
        {"status": summary["status"], "decision": summary["decision"], "artifact_sha256": hashes},
    )


def verify_preflight(path: Path, config: dict[str, Any], provider: Any) -> dict[str, Any]:
    value = source.load_json(path)
    if value.get("status") != "PASS":
        raise Stage6Error("semantic preflight did not pass")
    if value.get("config_sha256") != source.sha256_file(PACKAGE / "config.json"):
        raise Stage6Error("preflight used another config")
    if value.get("system_prompt_sha256") != source.sha256_text(SYSTEM_PROMPT):
        raise Stage6Error("preflight used another system prompt")
    if provider.kind == "ollama" and value.get("model") != "llama3.3:70b":
        raise Stage6Error("preflight used another model")
    return value


def load_raw_through_2021(
    explicit: Path | None,
    corrected_run: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(Path(item).expanduser() for item in config["raw_dataset_candidates"])
    failures: list[dict[str, str]] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            if not path.is_file():
                raise Stage6Error("file does not exist")
            digest = source.sha256_file(path)
            if digest != config["expected_raw_sha256"]:
                raise Stage6Error(f"unexpected sha256={digest}")
            header = pd.read_csv(path, nrows=0).columns.tolist()
            date_col = "Date" if "Date" in header else "date"
            close_col = "Close" if "Close" in header else "close"
            raw_rows = int(config["development_window"]["raw_market_rows_through_cutoff"])
            frame = pd.read_csv(path, usecols=[date_col, close_col], nrows=raw_rows)
            frame = frame.rename(columns={date_col: "date", close_col: "close"})
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            frame["close"] = pd.to_numeric(frame["close"], errors="raise")
            if len(frame) != raw_rows or str(frame["date"].max().date()) != "2021-12-31":
                raise Stage6Error("raw development cutoff does not end on 2021-12-31")
            if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
                raise Stage6Error("raw development dates are invalid")
            return frame, {
                "path": str(path),
                "sha256": digest,
                "rows_parsed": len(frame),
                "date_min": str(frame["date"].min().date()),
                "date_max": str(frame["date"].max().date()),
                "post2021_market_values_parsed": False,
                "full_file_read_only_for_integrity_hash": True,
            }
        except Exception as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    raise Stage6Error(f"no raw dataset passed: {failures}")


def load_2021_period(
    corrected_run: Path,
    raw: pd.DataFrame,
    base_config: dict[str, Any],
    config: dict[str, Any],
) -> source.PeriodInputs:
    period_dir = corrected_run / "results" / config["source_period"]["name"]
    input_path, trace_path = source.verify_child_artifact(period_dir)
    rows = int(config["development_window"]["rows"])
    frame = pd.read_csv(input_path, nrows=rows, low_memory=False)
    trace = pd.read_csv(trace_path, nrows=rows, low_memory=False)
    required = {
        "decision_date", "return_date", "prob_bear", "prob_bull", "prob_mix",
        "canonical_asset_simple_return", "exposure_volatility_target",
    }
    if required - set(frame):
        raise Stage6Error(f"corrected inputs missing columns={sorted(required - set(frame))}")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame["return_date"] = pd.to_datetime(frame["return_date"], errors="raise").dt.normalize()
    if len(frame) != rows or len(trace) != rows:
        raise Stage6Error("2021 development row count changed")
    if str(frame["decision_date"].min().date()) != "2021-01-01":
        raise Stage6Error("2021 development start changed")
    if str(frame["return_date"].max().date()) != "2021-12-31":
        raise Stage6Error("2021 development end changed")
    decisions = pd.DatetimeIndex(frame["decision_date"])
    returns = pd.DatetimeIndex(frame["return_date"])
    if not np.all((returns - decisions).days == 1):
        raise Stage6Error("return clock is not next-day")
    q = frame[["prob_bear", "prob_bull", "prob_mix"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if (q < -TOL).any() or not np.allclose(q.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise Stage6Error("router probabilities are invalid")
    asset_returns = pd.to_numeric(frame["canonical_asset_simple_return"], errors="raise").to_numpy(float)
    if {"decision_date", "return_date"}.issubset(trace.columns):
        trace_decisions = pd.to_datetime(trace["decision_date"], errors="raise").dt.normalize()
        trace_returns = pd.to_datetime(trace["return_date"], errors="raise").dt.normalize()
        if not np.array_equal(trace_decisions.to_numpy(), decisions.to_numpy()):
            raise Stage6Error("base trace decision dates are not aligned")
        if not np.array_equal(trace_returns.to_numpy(), returns.to_numpy()):
            raise Stage6Error("base trace return dates are not aligned")
    if len(trace) != len(frame) or not np.allclose(
        pd.to_numeric(trace["asset_simple_return"], errors="raise").to_numpy(float),
        asset_returns,
        atol=1e-15,
        rtol=0.0,
    ):
        raise Stage6Error("base trace is not aligned to corrected returns")
    scenarios, risk_audit = source.build_causal_scenarios(raw, decisions, base_config)
    features, feature_audit = source.build_causal_market_features(raw, decisions, q)
    return source.PeriodInputs(
        "development_2021",
        "DEVELOPMENT_PILOT_ONLY_NOT_CONFIRMATION",
        frame,
        trace,
        decisions,
        returns,
        q,
        asset_returns,
        scenarios,
        features,
        {
            "corrected_inputs_path": str(input_path),
            "corrected_inputs_sha256": source.sha256_file(input_path),
            "base_trace_path": str(trace_path),
            "base_trace_sha256": source.sha256_file(trace_path),
            "rows_parsed": rows,
            "first_decision_date": str(decisions.min().date()),
            "last_return_date": str(returns.max().date()),
            "post2021_portfolio_rows_parsed": False,
            "risk_scenarios": risk_audit,
            "market_features": feature_audit,
        },
    )


def state_for(period: source.PeriodInputs, index: int, pretrade: float, base_desired: float) -> dict[str, Any]:
    q = period.router_probabilities[index]
    hard_regime = ("bear", "bull", "mix")[int(np.argmax(q))]
    feature = period.features.iloc[index]
    state = {
        "decision_date": str(period.decision_dates[index].date()),
        "target_return_date": str(period.return_dates[index].date()),
        "prob_bear": float(q[0]),
        "prob_bull": float(q[1]),
        "prob_mix": float(q[2]),
        "hard_regime": hard_regime,
        "pretrade_exposure": float(pretrade),
        "base_ramoe_desired_exposure": float(base_desired),
    }
    state.update({key: float(value) for key, value in feature.to_dict().items()})
    return state


def deterministic_action(state: dict[str, Any]) -> str:
    if state["prob_bear"] >= 0.60 or state["drawdown_90"] <= -0.25 or state["realized_vol_30"] >= 1.50:
        return "CASH"
    if (
        state["prob_bull"] >= 0.70
        and state["return_30"] > 0.0
        and state["realized_vol_30"] <= 1.10
    ):
        return "BTC"
    return "ABSTAIN"


def project_action(
    action: str,
    beta: float,
    base_desired: float,
    pretrade: float,
    scenarios: np.ndarray,
    base_config: dict[str, Any],
    risk: Any,
) -> Any:
    desired = blend_exposure(base_desired, action, beta)
    return risk.project_exposure(
        desired_exposure=desired,
        drifted_pretrade_exposure=pretrade,
        scenario_log_returns=scenarios,
        **source.risk_kwargs(base_config),
    )


def run_development(
    period: source.PeriodInputs,
    provider: Any,
    config: dict[str, Any],
    base_config: dict[str, Any],
    accounting: Any,
    risk: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]], EpisodicMemory, RegimeTrust, RegimeTrust]:
    memory = EpisodicMemory()
    llama_trust = RegimeTrust(config["trust"])
    deterministic_trust = RegimeTrust(config["trust"])
    deterministic_completed: list[dict[str, Any]] = []
    pretrade = {
        "full": 0.0,
        "frozen": 0.0,
        "deterministic": 0.0,
        "shadow": 0.0,
        "deterministic_shadow": 0.0,
    }
    cost_rate = float(base_config["transaction_cost_bps"]) / 10000.0
    fixed_beta = float(config["trust"]["initial_beta"])
    trace_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for index in range(len(period.frame)):
        decision_date = period.decision_dates[index].date()
        base_row = period.current_trace.iloc[index]
        base_desired = float(base_row["desired_exposure"])
        state = state_for(period, index, pretrade["full"], base_desired)
        llama_trust.maybe_update(decision_date, memory.episodes)
        deterministic_trust.maybe_update(decision_date, deterministic_completed)
        regime = state["hard_regime"]
        beta = llama_trust.value(regime)
        state["llama_trust_beta"] = beta
        memory_evidence = memory.evidence_summary(
            state, int(config["agent"]["maximum_similar_episodes"])
        )
        previews: dict[str, Any] = {}
        for action in ("BTC", "CASH", "ABSTAIN"):
            projected = project_action(
                action, beta, base_desired, pretrade["full"],
                period.scenarios[index], base_config, risk,
            )
            previews[action] = {
                "blended_desired_exposure": blend_exposure(base_desired, action, beta),
                "risk_limited_exposure": float(projected.exposure),
                "turnover": float(projected.turnover),
                "ambiguity_cvar": float(projected.ambiguity_cvar),
            }
        payload = {
            "task_type": "historical_development_decision",
            "objective": "add incremental risk-adjusted value over unchanged RAMAS after costs",
            "state": state,
            "memory": memory_evidence,
            "safe_exposure_previews": previews,
            "constraints": {
                "allowed_actions": ["BTC", "CASH", "ABSTAIN"],
                "shorting": False,
                "leverage": False,
                "invalid_or_failed_response": "ABSTAIN_TO_BASE_RAMAS",
            },
        }
        request_text = json.dumps(payload, sort_keys=True, allow_nan=False)
        if "asset_simple_return" in request_text or "canonical_asset_simple_return" in request_text:
            raise Stage6Error("future return leaked into Llama request")
        raw = ""
        outer: dict[str, Any] = {}
        latency = 0.0
        valid = False
        error = ""
        decision = {
            "action": "ABSTAIN",
            "confidence": 0.0,
            "reason_codes": ["INSUFFICIENT_EVIDENCE"],
            "cited_memory_ids": [],
        }
        try:
            raw, outer, latency = provider.complete(payload)
            visible = {item["episode_id"] for item in memory_evidence["similar_completed_episodes"]}
            decision = validate_decision(raw, config["agent"], visible)
            valid = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        action = decision["action"]
        full_project = project_action(
            action, beta, base_desired, pretrade["full"],
            period.scenarios[index], base_config, risk,
        )
        frozen_project = project_action(
            action, fixed_beta, base_desired, pretrade["frozen"],
            period.scenarios[index], base_config, risk,
        )
        det_action = deterministic_action(state)
        det_beta = deterministic_trust.value(regime)
        det_project = project_action(
            det_action, det_beta, base_desired, pretrade["deterministic"],
            period.scenarios[index], base_config, risk,
        )
        shadow_beta = 1.0 if action != "ABSTAIN" else 0.0
        shadow_project = project_action(
            action, shadow_beta, base_desired, pretrade["shadow"],
            period.scenarios[index], base_config, risk,
        )
        asset_return = float(period.asset_returns[index])

        def net(projected: Any, key: str) -> float:
            return float(
                accounting.net_return(
                    float(projected.exposure), pretrade[key], asset_return, cost_rate
                )
            )

        full_return = net(full_project, "full")
        frozen_return = net(frozen_project, "frozen")
        det_return = net(det_project, "deterministic")
        shadow_return = net(shadow_project, "shadow")
        base_return = float(base_row["portfolio_net_return"])
        shadow_advantage = float(math.log1p(shadow_return) - math.log1p(base_return))
        episode = {
            "episode_id": f"episode-{index + 1:06d}",
            "decision_date": state["decision_date"],
            "return_date": state["target_return_date"],
            "hard_regime": regime,
            "state_vector": state_vector(state).tolist(),
            "action": action,
            "confidence": float(decision["confidence"]),
            "shadow_log_advantage_vs_ramoe": shadow_advantage,
            "asset_return": asset_return,
        }
        memory.add_completed(episode)
        det_shadow_beta = 1.0 if det_action != "ABSTAIN" else 0.0
        det_shadow_project = project_action(
            det_action, det_shadow_beta, base_desired, pretrade["deterministic_shadow"],
            period.scenarios[index], base_config, risk,
        )
        det_shadow_return = float(
            accounting.net_return(
                float(det_shadow_project.exposure),
                pretrade["deterministic_shadow"],
                asset_return,
                cost_rate,
            )
        )
        deterministic_completed.append(
            {
                "hard_regime": regime,
                "action": det_action,
                "shadow_log_advantage_vs_ramoe": float(
                    math.log1p(det_shadow_return) - math.log1p(base_return)
                ),
            }
        )
        calls.append(
            {
                "request_id": f"development_2021-{index + 1:06d}",
                "request_sha256": source.sha256_text(SYSTEM_PROMPT + "\n" + request_text),
                "request": payload,
                "raw_response": raw,
                "response_sha256": source.sha256_text(raw),
                "provider_response": outer,
                "latency_seconds": latency,
                "valid": valid,
                "error": error,
            }
        )
        trace_rows.append(
            {
                "decision_date": state["decision_date"],
                "return_date": state["target_return_date"],
                "prob_bear": state["prob_bear"],
                "prob_bull": state["prob_bull"],
                "prob_mix": state["prob_mix"],
                "hard_regime": regime,
                "router_entropy": state["router_entropy"],
                "router_transition_l1": state["router_transition_l1"],
                "return_7": state["return_7"],
                "return_30": state["return_30"],
                "realized_vol_30": state["realized_vol_30"],
                "drawdown_90": state["drawdown_90"],
                "action": action,
                "confidence": decision["confidence"],
                "reason_codes": "|".join(decision["reason_codes"]),
                "cited_memory_ids": "|".join(decision["cited_memory_ids"]),
                "valid_action": valid,
                "memory_episodes_seen": len(memory_evidence["similar_completed_episodes"]),
                "llama_beta": beta,
                "base_desired_exposure": base_desired,
                "base_final_exposure": float(base_row["final_exposure"]),
                "base_net_return": base_return,
                "full_final_exposure": float(full_project.exposure),
                "full_turnover": float(full_project.turnover),
                "full_net_return": full_return,
                "frozen_beta_final_exposure": float(frozen_project.exposure),
                "frozen_beta_turnover": float(frozen_project.turnover),
                "frozen_beta_net_return": frozen_return,
                "deterministic_action": det_action,
                "deterministic_beta": det_beta,
                "deterministic_final_exposure": float(det_project.exposure),
                "deterministic_turnover": float(det_project.turnover),
                "deterministic_net_return": det_return,
                "shadow_final_exposure": float(shadow_project.exposure),
                "shadow_turnover": float(shadow_project.turnover),
                "shadow_net_return": shadow_return,
                "shadow_log_advantage_vs_ramoe": shadow_advantage,
                "asset_simple_return": asset_return,
            }
        )
        for key, projected in (
            ("full", full_project),
            ("frozen", frozen_project),
            ("deterministic", det_project),
            ("shadow", shadow_project),
            ("deterministic_shadow", det_shadow_project),
        ):
            pretrade[key] = float(accounting.drifted_exposure(projected.exposure, asset_return))
        completed = index + 1
        if completed == 1 or completed % 25 == 0 or completed == len(period.frame):
            valid_count = sum(bool(item["valid_action"]) for item in trace_rows)
            non_abstain = sum(item["action"] != "ABSTAIN" for item in trace_rows)
            print(
                f"STAGE6_PROGRESS={completed}/{len(period.frame)} "
                f"VALID={valid_count} NON_ABSTAIN={non_abstain} "
                f"BETA_{regime.upper()}={beta:.3f}",
                flush=True,
            )
    return pd.DataFrame(trace_rows), calls, memory, llama_trust, deterministic_trust


@dataclass(frozen=True)
class Stream:
    returns: np.ndarray
    exposure: np.ndarray
    turnover: np.ndarray


def metric_row(name: str, stream: Stream) -> dict[str, Any]:
    returns = np.asarray(stream.returns, dtype=float)
    wealth = np.cumprod(1.0 + returns)
    wealth_with_initial = np.concatenate(([1.0], wealth))
    peak = np.maximum.accumulate(wealth_with_initial)
    drawdown = wealth_with_initial / peak - 1.0
    std = float(np.std(returns, ddof=1))
    return {
        "variant": name,
        "terminal_growth": float(wealth[-1] - 1.0),
        "sharpe_zero_cash_rate": float(np.mean(returns) / std * math.sqrt(365.25)) if std > 0 else 0.0,
        "maximum_drawdown_loss": float(-np.min(drawdown)),
        "daily_loss_cvar_95": fractional_expected_shortfall_loss(returns, 0.05),
        "mean_exposure": float(np.mean(stream.exposure)),
        "mean_daily_turnover": float(np.mean(stream.turnover)),
    }


def fractional_expected_shortfall_loss(returns: np.ndarray, tail_fraction: float) -> float:
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be within (0,1]")
    losses = np.sort(-np.asarray(returns, dtype=float))[::-1]
    if len(losses) == 0 or not np.isfinite(losses).all():
        raise ValueError("returns must be non-empty and finite")
    mass = len(losses) * tail_fraction
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = float(np.sum(losses[:whole]))
    if fraction > 0.0:
        total += fraction * float(losses[whole])
    return float(total / mass)


def circular_block_test(values: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    block = int(config["block_length_days"])
    resamples = int(config["resamples"])
    rng = np.random.default_rng(int(config["seed"]))
    blocks_needed = int(math.ceil(n / block))
    means = np.empty(resamples, dtype=float)
    offsets = np.arange(block)
    for index in range(resamples):
        starts = rng.integers(0, n, size=blocks_needed)
        sample_index = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        means[index] = float(np.mean(values[sample_index]))
    alpha = float(config["one_sided_alpha"])
    return {
        "observed_mean_log_advantage": float(np.mean(values)),
        "lower_one_sided_confidence_bound": float(np.quantile(means, alpha)),
        "one_sided_p_value": float((1 + np.sum(means <= 0.0)) / (resamples + 1)),
        "block_length_days": block,
        "resamples": resamples,
        "seed": int(config["seed"]),
    }


def evaluate(
    trace: pd.DataFrame,
    period: source.PeriodInputs,
    config: dict[str, Any],
    base_config: dict[str, Any],
    adapter: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n = len(trace)
    asset = trace["asset_simple_return"].to_numpy(float)
    cost = float(base_config["transaction_cost_bps"]) / 10000.0
    bh_return, bh_turnover = adapter.constant_exposure_returns(asset, 1.0, cost)
    cash_return, cash_turnover = adapter.constant_exposure_returns(asset, 0.0, cost)
    streams = {
        "ramas_base": Stream(
            trace["base_net_return"].to_numpy(float),
            trace["base_final_exposure"].to_numpy(float),
            period.current_trace["turnover"].to_numpy(float),
        ),
        "ramas_llama70b_adaptive_memory": Stream(
            trace["full_net_return"].to_numpy(float),
            trace["full_final_exposure"].to_numpy(float),
            trace["full_turnover"].to_numpy(float),
        ),
        "ramas_llama70b_frozen_trust": Stream(
            trace["frozen_beta_net_return"].to_numpy(float),
            trace["frozen_beta_final_exposure"].to_numpy(float),
            trace["frozen_beta_turnover"].to_numpy(float),
        ),
        "deterministic_same_information": Stream(
            trace["deterministic_net_return"].to_numpy(float),
            trace["deterministic_final_exposure"].to_numpy(float),
            trace["deterministic_turnover"].to_numpy(float),
        ),
        "llama70b_shadow_only": Stream(
            trace["shadow_net_return"].to_numpy(float),
            trace["shadow_final_exposure"].to_numpy(float),
            trace["shadow_turnover"].to_numpy(float),
        ),
        "buy_and_hold": Stream(np.asarray(bh_return), np.ones(n), np.asarray(bh_turnover)),
        "cash": Stream(np.asarray(cash_return), np.zeros(n), np.asarray(cash_turnover)),
    }
    metrics = pd.DataFrame([metric_row(name, stream) for name, stream in streams.items()])
    yearly = metrics.copy()
    yearly.insert(0, "year", 2021)
    full = metrics.loc[metrics["variant"] == "ramas_llama70b_adaptive_memory"].iloc[0]
    base = metrics.loc[metrics["variant"] == "ramas_base"].iloc[0]
    deterministic = metrics.loc[metrics["variant"] == "deterministic_same_information"].iloc[0]
    log_advantage = np.log1p(streams["ramas_llama70b_adaptive_memory"].returns) - np.log1p(
        streams["ramas_base"].returns
    )
    inference = circular_block_test(log_advantage, config["bootstrap"])
    spec = config["development_gate"]
    valid_rate = float(trace["valid_action"].mean())
    non_abstain = int((trace["action"] != "ABSTAIN").sum())
    distinct = int(trace["action"].nunique())
    checks = {
        "minimum_valid_action_rate": valid_rate + TOL >= float(spec["minimum_valid_action_rate"]),
        "minimum_non_abstain_actions": non_abstain >= int(spec["minimum_non_abstain_actions"]),
        "minimum_distinct_actions": distinct >= int(spec["minimum_distinct_actions"]),
        "terminal_growth_above_ramoe": float(full["terminal_growth"]) > float(base["terminal_growth"]),
        "sharpe_improvement_vs_ramoe": float(full["sharpe_zero_cash_rate"]) >= (
            float(base["sharpe_zero_cash_rate"]) + float(spec["minimum_sharpe_improvement_vs_ramoe"])
        ),
        "maximum_drawdown_not_worse": float(full["maximum_drawdown_loss"]) <= float(base["maximum_drawdown_loss"]) + TOL,
        "daily_cvar_not_worse": float(full["daily_loss_cvar_95"]) <= float(base["daily_loss_cvar_95"]) + TOL,
        "positive_bootstrap_lower_bound": inference["lower_one_sided_confidence_bound"] > 0.0,
        "growth_above_deterministic_controller": float(full["terminal_growth"]) > float(deterministic["terminal_growth"]),
    }
    gate = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "valid_action_rate": valid_rate,
        "non_abstain_actions": non_abstain,
        "distinct_actions": sorted(trace["action"].unique().tolist()),
        "inference_vs_ramoe": inference,
        "scientific_scope": "2021_DEVELOPMENT_PILOT_ONLY",
    }
    return metrics, yearly, gate


def write_development(
    output: Path,
    config: dict[str, Any],
    provider: Any,
    corrected_run: Path,
    base_dir: Path,
    raw_audit: dict[str, Any],
    period: source.PeriodInputs,
    preflight: dict[str, Any],
    trace: pd.DataFrame,
    calls: list[dict[str, Any]],
    memory: EpisodicMemory,
    llama_trust: RegimeTrust,
    deterministic_trust: RegimeTrust,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    gate: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise Stage6Error(f"refusing to overwrite output={output}")
    output.mkdir(parents=True)
    decision = (
        "CONTINUE_TO_STAGE6_LOCK_AND_YEARLY_DIAGNOSTIC"
        if gate["passed"]
        else "REVISE_STAGE6_2021_DEVELOPMENT_GATE_FAILED_NO_POST2021"
    )
    source.atomic_json(
        output / "00_CONTRACT.json",
        {
            "experiment_id": config["experiment_id"],
            "phase": config["phase"],
            "provider": provider.kind,
            "model": config["provider"]["model"] if provider.kind == "ollama" else "CONTROLLED",
            "corrected_source_run": str(corrected_run),
            "frozen_base_source": str(base_dir),
            "config_sha256": source.sha256_file(PACKAGE / "config.json"),
            "system_prompt_sha256": source.sha256_text(SYSTEM_PROMPT),
            "frozen_claims": config["frozen_claims"],
            "timeline": {
                "2015_2017": "router_warmup_history_only",
                "2018_2020": "requires_historical_RAMAS_base_trace_before_agent_training_claim",
                "2021": "current_development_pilot",
                "2022_2025": "not_used_by_this_package",
            },
        },
    )
    source.atomic_json(output / "01_SOURCE_AUDIT.json", {"raw_market": raw_audit, "period": period.source_audit})
    source.atomic_json(output / "02_SEMANTIC_PREFLIGHT.json", preflight)
    source.atomic_csv(output / "03_DAILY_TRACE.csv", trace)
    source.atomic_csv(output / "04_VARIANT_METRICS.csv", metrics)
    source.atomic_csv(output / "05_YEARLY_METRICS.csv", yearly)
    source.atomic_json(output / "06_DEVELOPMENT_GATE.json", gate)
    source.atomic_jsonl(output / "07_LLM_CALLS.jsonl", calls)
    source.atomic_jsonl(output / "08_MEMORY_EPISODES.jsonl", memory.episodes)
    source.atomic_json(
        output / "09_TRUST_STATE.json",
        {
            "llama_beta_final": llama_trust.beta,
            "llama_monthly_events": llama_trust.events,
            "deterministic_beta_final": deterministic_trust.beta,
            "deterministic_monthly_events": deterministic_trust.events,
        },
    )
    final = {
        "decision": decision,
        "development_gate_passed": gate["passed"],
        "economic_value_established": False,
        "post2021_confirmation_run": False,
        "post2021_rows_used": False,
        "router_changed": False,
        "existing_ramoe_trust_changed": False,
        "risk_layer_changed": False,
        "qwen_used": False,
        "next_step": (
            "freeze Stage 6 and run 2022, 2023, 2024, and 2025 one year at a time"
            if gate["passed"]
            else "inspect failed checks and revise only on development data"
        ),
    }
    source.atomic_json(output / "10_FINAL_DECISION.json", final)
    report = f"""# RAMAS Stage 6 result

Decision: **{decision}**

Llama 70B was used as a separate, low-trust BTC/CASH/ABSTAIN expert. Its signal was blended with the unchanged RAMAS desired exposure. The unchanged risk layer remained the final authority. Memory contained only completed earlier episodes, and Llama trust changed only at month boundaries from completed prior outcomes.

This package evaluated only the 2021 development pilot. It did not use 2022-2025 portfolio rows or market values. Passing this pilot would permit a later frozen, year-by-year diagnostic; it would not by itself establish economic value.
"""
    (output / "11_PLAIN_ENGLISH_REPORT.md").write_text(report, encoding="utf-8")
    hashes = {
        path.name: source.sha256_file(path)
        for path in sorted(output.iterdir()) if path.is_file()
    }
    source.atomic_json(
        output / "RUN_COMPLETE.json",
        {
            "status": "PASS",
            "decision": decision,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": hashes,
        },
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["ollama", "controlled"], default="ollama")
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    parser.add_argument("--corrected-run", type=Path)
    parser.add_argument("--full-dataset", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    provider = provider_from(config, args.provider)
    output = args.output.expanduser().resolve()
    if args.preflight_only:
        summary, records = run_semantic_preflight(provider, config)
        write_preflight(output, summary, records)
        print("RAMAS_STAGE6_SEMANTIC_PREFLIGHT_STATUS=PASS" if summary["status"] == "PASS" else "RAMAS_STAGE6_SEMANTIC_PREFLIGHT_STATUS=FAIL", flush=True)
        print(f"DECISION={summary['decision']}", flush=True)
        print(f"OUTPUT={output}", flush=True)
        if summary["status"] != "PASS":
            raise SystemExit(4)
        return
    if args.preflight_evidence is None:
        raise Stage6Error("full development requires --preflight-evidence")
    preflight = verify_preflight(args.preflight_evidence.resolve(), config, provider)
    project_root = args.project_root.expanduser().resolve()
    corrected_run = source.resolve_corrected_run(project_root, args.corrected_run, config)
    base_dir = source.resolve_base_dir(corrected_run)
    adapter, accounting, risk, _statistics = source.import_base(base_dir)
    base_config = adapter.load_config(base_dir / "config.json")
    raw, raw_audit = load_raw_through_2021(args.full_dataset, corrected_run, config)
    period = load_2021_period(corrected_run, raw, base_config, config)
    trace, calls, memory, llama_trust, deterministic_trust = run_development(
        period, provider, config, base_config, accounting, risk
    )
    metrics, yearly, gate = evaluate(trace, period, config, base_config, adapter)
    final = write_development(
        output, config, provider, corrected_run, base_dir, raw_audit, period,
        preflight, trace, calls, memory, llama_trust, deterministic_trust,
        metrics, yearly, gate,
    )
    print("RAMAS_STAGE6_STATUS=PASS", flush=True)
    print(f"DECISION={final['decision']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={str(final['development_gate_passed']).upper()}", flush=True)
    print("POST2021_ROWS_USED=FALSE", flush=True)
    print("QWEN_USED=FALSE", flush=True)
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
