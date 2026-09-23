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


class Stage61Error(RuntimeError):
    pass


# Retained only for the unused Stage 6 compatibility loaders copied into this
# derivative package.  New Stage 6.1 paths raise Stage61Error directly.
Stage6Error = Stage61Error


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != "RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1":
        raise Stage61Error("unexpected experiment_id")
    if value.get("provider", {}).get("model") != "llama3.3:70b":
        raise Stage61Error("Stage 6.1 is locked to llama3.3:70b")
    claims = value.get("frozen_claims", {})
    if claims.get("qwen_used") is not False:
        raise Stage61Error("Qwen must remain disabled")
    if claims.get("router_changed") is not False:
        raise Stage61Error("router must remain frozen")
    if claims.get("existing_ramoe_trust_changed") is not False:
        raise Stage61Error("existing RAMoE trust must remain frozen")
    if claims.get("risk_layer_changed") is not False:
        raise Stage61Error("risk layer must remain frozen")
    if not 0.0 <= float(value["trust"]["initial_beta"]) <= float(value["trust"]["maximum_beta"]) <= 0.20:
        raise Stage61Error("Llama trust bounds are invalid")
    timeline = value.get("continuous_timeline", {})
    if timeline.get("year_end_action") != "REPORT_AND_CONTINUE_WITHOUT_RESET":
        raise Stage61Error("year-end continuation contract changed")
    if value.get("research_contract", {}).get("no_yearly_economic_hard_stop") is not True:
        raise Stage61Error("an economic hard stop was introduced")
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
            "CONTINUE_TO_CONTINUOUS_PREQUENTIAL_REPLAY"
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
        raise Stage61Error(f"refusing to overwrite output={output}")
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
        raise Stage61Error("semantic preflight did not pass")
    if value.get("config_sha256") != source.sha256_file(PACKAGE / "config.json"):
        raise Stage61Error("preflight used another config")
    if value.get("system_prompt_sha256") != source.sha256_text(SYSTEM_PROMPT):
        raise Stage61Error("preflight used another system prompt")
    if provider.kind == "ollama" and value.get("model") != "llama3.3:70b":
        raise Stage61Error("preflight used another model")
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


def load_raw_continuous(
    explicit: Path | None,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the frozen market file through the final required date.

    The full-file hash is checked before any value is parsed.  Later prices are
    available only to construct each day's causal state and to score a decision
    after its next-day return has completed; they are never placed in a prompt.
    """
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(Path(item).expanduser() for item in config["raw_dataset_candidates"])
    failures: list[dict[str, str]] = []
    seen: set[Path] = set()
    cutoff = pd.Timestamp(config["continuous_timeline"]["last_return_date"])
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            if not path.is_file():
                raise Stage61Error("file does not exist")
            digest = source.sha256_file(path)
            if digest != config["expected_raw_sha256"]:
                raise Stage61Error(f"unexpected sha256={digest}")
            header = pd.read_csv(path, nrows=0).columns.tolist()
            date_col = "Date" if "Date" in header else "date"
            close_col = "Close" if "Close" in header else "close"
            frame = pd.read_csv(path, usecols=[date_col, close_col])
            frame = frame.rename(columns={date_col: "date", close_col: "close"})
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            frame["close"] = pd.to_numeric(frame["close"], errors="raise")
            if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
                raise Stage61Error("raw dates are duplicated or out of order")
            if cutoff not in set(frame["date"]):
                raise Stage61Error(f"raw data does not contain cutoff={cutoff.date()}")
            used = frame.loc[frame["date"] <= cutoff].reset_index(drop=True)
            if not np.isfinite(used["close"].to_numpy(float)).all() or (used["close"] <= 0).any():
                raise Stage61Error("raw close prices are invalid")
            return used, {
                "path": str(path),
                "sha256": digest,
                "full_file_rows": int(len(frame)),
                "rows_parsed_for_replay": int(len(used)),
                "date_min": str(used["date"].min().date()),
                "date_max": str(used["date"].max().date()),
                "post2021_market_values_parsed": True,
                "future_value_sent_to_llama": False,
            }
        except Exception as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    raise Stage61Error(f"no raw dataset passed: {failures}")


def load_continuous_period(
    corrected_run: Path,
    raw: pd.DataFrame,
    base_config: dict[str, Any],
    config: dict[str, Any],
) -> source.PeriodInputs:
    frames: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    period_audits: list[dict[str, Any]] = []
    for spec in config["source_periods"]:
        period_dir = corrected_run / "results" / spec["name"]
        input_path, trace_path = source.verify_child_artifact(period_dir)
        frame = pd.read_csv(input_path, low_memory=False)
        trace = pd.read_csv(trace_path, low_memory=False)
        expected_rows = int(spec["rows"])
        if len(frame) != expected_rows or len(trace) != expected_rows:
            raise Stage61Error(
                f"source row count changed for {spec['name']}: "
                f"inputs={len(frame)} trace={len(trace)} expected={expected_rows}"
            )
        frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
        frame["return_date"] = pd.to_datetime(frame["return_date"], errors="raise").dt.normalize()
        first_decision = str(frame["decision_date"].min().date())
        last_return = str(frame["return_date"].max().date())
        if first_decision != spec["first_decision_date"] or last_return != spec["last_return_date"]:
            raise Stage61Error(f"source date contract changed for {spec['name']}")
        frame["source_period"] = spec["name"]
        frames.append(frame)
        traces.append(trace)
        period_audits.append(
            {
                "name": spec["name"],
                "scientific_label": spec["scientific_label"],
                "rows": expected_rows,
                "first_decision_date": first_decision,
                "last_return_date": last_return,
                "corrected_inputs_path": str(input_path),
                "corrected_inputs_sha256": source.sha256_file(input_path),
                "base_trace_path": str(trace_path),
                "base_trace_sha256": source.sha256_file(trace_path),
            }
        )
    frame = pd.concat(frames, ignore_index=True)
    trace = pd.concat(traces, ignore_index=True)
    required = {
        "decision_date", "return_date", "prob_bear", "prob_bull", "prob_mix",
        "canonical_asset_simple_return", "exposure_volatility_target",
    }
    if required - set(frame):
        raise Stage61Error(f"corrected inputs missing columns={sorted(required - set(frame))}")
    decisions = pd.DatetimeIndex(frame["decision_date"])
    returns = pd.DatetimeIndex(frame["return_date"])
    expected_total = int(config["continuous_timeline"]["total_rows"])
    if len(frame) != expected_total or len(trace) != expected_total:
        raise Stage61Error("continuous row count changed")
    if decisions.duplicated().any() or returns.duplicated().any():
        raise Stage61Error("continuous dates overlap")
    if not decisions.is_monotonic_increasing or not returns.is_monotonic_increasing:
        raise Stage61Error("continuous dates are out of order")
    if not np.all((returns - decisions).days == 1):
        raise Stage61Error("return clock is not next-day")
    if not np.all(np.diff(returns.values).astype("timedelta64[D]").astype(int) == 1):
        raise Stage61Error("continuous return calendar has a gap")
    q = frame[["prob_bear", "prob_bull", "prob_mix"]].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    if (q < -TOL).any() or not np.allclose(q.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise Stage61Error("router probabilities are invalid")
    asset_returns = pd.to_numeric(
        frame["canonical_asset_simple_return"], errors="raise"
    ).to_numpy(float)
    if {"decision_date", "return_date"}.issubset(trace.columns):
        trace_decisions = pd.to_datetime(trace["decision_date"], errors="raise").dt.normalize()
        trace_returns = pd.to_datetime(trace["return_date"], errors="raise").dt.normalize()
        if not np.array_equal(trace_decisions.to_numpy(), decisions.to_numpy()):
            raise Stage61Error("base trace decision dates are not aligned")
        if not np.array_equal(trace_returns.to_numpy(), returns.to_numpy()):
            raise Stage61Error("base trace return dates are not aligned")
    trace_asset = pd.to_numeric(trace["asset_simple_return"], errors="raise").to_numpy(float)
    if not np.allclose(trace_asset, asset_returns, atol=1e-15, rtol=0.0):
        raise Stage61Error("base trace is not aligned to corrected returns")
    scenarios, risk_audit = source.build_causal_scenarios(raw, decisions, base_config)
    features, feature_audit = source.build_causal_market_features(raw, decisions, q)
    return source.PeriodInputs(
        "continuous_2021_2025",
        "2021_BURN_IN_THEN_REUSED_OOS_PREQUENTIAL_DIAGNOSTIC",
        frame,
        trace,
        decisions,
        returns,
        q,
        asset_returns,
        scenarios,
        features,
        {
            "periods": period_audits,
            "rows": int(len(frame)),
            "first_decision_date": str(decisions.min().date()),
            "last_return_date": str(returns.max().date()),
            "strictly_continuous_calendar": True,
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Stage61Error(f"JSONL row {number} is not an object: {path}")
            output.append(value)
    return output


def verify_result_directory(directory: Path, required: set[str]) -> dict[str, Any]:
    complete = source.load_json(directory / "RUN_COMPLETE.json")
    if complete.get("status") != "PASS":
        raise Stage61Error(f"result is not complete: {directory}")
    hashes = complete.get("artifact_sha256", {})
    for name in required:
        path = directory / name
        if not path.is_file() or hashes.get(name) != source.sha256_file(path):
            raise Stage61Error(f"result hash mismatch: {path}")
    return complete


def load_stage6_seed(
    directory: Path,
    period: source.PeriodInputs,
    config: dict[str, Any],
    base_config: dict[str, Any],
    accounting: Any,
    risk: Any,
) -> dict[str, Any]:
    """Restore the completed 2021 state without rerunning 364 Llama calls."""
    required = {
        "00_CONTRACT.json", "03_DAILY_TRACE.csv", "07_LLM_CALLS.jsonl",
        "08_MEMORY_EPISODES.jsonl", "09_TRUST_STATE.json", "10_FINAL_DECISION.json",
    }
    verify_result_directory(directory, required)
    contract = source.load_json(directory / "00_CONTRACT.json")
    final = source.load_json(directory / "10_FINAL_DECISION.json")
    spec = config["stage6_2021_seed"]
    if contract.get("experiment_id") != spec["experiment_id"]:
        raise Stage61Error("2021 seed used another experiment")
    if contract.get("config_sha256") != spec["accepted_config_sha256"]:
        raise Stage61Error("2021 seed used another Stage 6 config")
    if contract.get("system_prompt_sha256") != spec["accepted_system_prompt_sha256"]:
        raise Stage61Error("2021 seed used another Llama prompt")
    if contract.get("model") != "llama3.3:70b" or final.get("qwen_used") is not False:
        raise Stage61Error("2021 seed violated the Llama-70B-only contract")
    trace = pd.read_csv(directory / "03_DAILY_TRACE.csv", low_memory=False)
    expected = int(spec["expected_rows"])
    if len(trace) != expected:
        raise Stage61Error(f"2021 seed rows changed: {len(trace)} != {expected}")
    seed_decisions = pd.to_datetime(trace["decision_date"], errors="raise").dt.normalize()
    seed_returns = pd.to_datetime(trace["return_date"], errors="raise").dt.normalize()
    if str(seed_decisions.min().date()) != spec["first_decision_date"]:
        raise Stage61Error("2021 seed start date changed")
    if str(seed_returns.max().date()) != spec["last_return_date"]:
        raise Stage61Error("2021 seed end date changed")
    if not np.array_equal(seed_decisions.to_numpy(), period.decision_dates[:expected].to_numpy()):
        raise Stage61Error("2021 seed decisions do not match the continuous source")
    if not np.array_equal(seed_returns.to_numpy(), period.return_dates[:expected].to_numpy()):
        raise Stage61Error("2021 seed returns do not match the continuous source")
    if not np.allclose(
        trace["asset_simple_return"].to_numpy(float),
        period.asset_returns[:expected], atol=1e-15, rtol=0.0,
    ):
        raise Stage61Error("2021 seed asset returns do not match the continuous source")
    memory = EpisodicMemory(load_jsonl(directory / "08_MEMORY_EPISODES.jsonl"))
    if len(memory.episodes) != expected:
        raise Stage61Error("2021 seed memory is incomplete")
    trust_state = source.load_json(directory / "09_TRUST_STATE.json")
    llama_trust = RegimeTrust(config["trust"])
    llama_trust.beta = {
        key: float(value) for key, value in trust_state["llama_beta_final"].items()
    }
    llama_trust.events = list(trust_state["llama_monthly_events"])
    llama_trust.last_month = (2021, 12)
    deterministic_trust = RegimeTrust(config["trust"])
    deterministic_trust.beta = {
        key: float(value) for key, value in trust_state["deterministic_beta_final"].items()
    }
    deterministic_trust.events = list(trust_state["deterministic_monthly_events"])
    deterministic_trust.last_month = (2021, 12)
    last_asset = float(period.asset_returns[expected - 1])
    pretrade = {
        "full": float(accounting.drifted_exposure(float(trace.iloc[-1]["full_final_exposure"]), last_asset)),
        "frozen": float(accounting.drifted_exposure(float(trace.iloc[-1]["frozen_beta_final_exposure"]), last_asset)),
        "deterministic": float(accounting.drifted_exposure(float(trace.iloc[-1]["deterministic_final_exposure"]), last_asset)),
        "shadow": float(accounting.drifted_exposure(float(trace.iloc[-1]["shadow_final_exposure"]), last_asset)),
        "deterministic_shadow": 0.0,
    }
    deterministic_completed: list[dict[str, Any]] = []
    det_shadow_pretrade = 0.0
    cost_rate = float(base_config["transaction_cost_bps"]) / 10000.0
    for index in range(expected):
        action = str(trace.iloc[index]["deterministic_action"])
        base_desired = float(trace.iloc[index]["base_desired_exposure"])
        projected = project_action(
            action, 1.0, base_desired, det_shadow_pretrade,
            period.scenarios[index], base_config, risk,
        )
        asset_return = float(period.asset_returns[index])
        shadow_return = float(
            accounting.net_return(
                float(projected.exposure), det_shadow_pretrade, asset_return, cost_rate
            )
        )
        deterministic_completed.append(
            {
                "hard_regime": str(trace.iloc[index]["hard_regime"]),
                "action": action,
                "shadow_log_advantage_vs_ramoe": float(
                    math.log1p(shadow_return) - math.log1p(float(trace.iloc[index]["base_net_return"]))
                ),
            }
        )
        det_shadow_pretrade = float(accounting.drifted_exposure(projected.exposure, asset_return))
    pretrade["deterministic_shadow"] = det_shadow_pretrade
    trace = trace.copy()
    trace["return_year"] = seed_returns.dt.year.to_numpy(int)
    trace["scientific_segment"] = "BURN_IN_2021"
    trace["memory_total_before"] = np.arange(expected, dtype=int)
    trace["effective_intervention"] = (
        np.abs(trace["full_final_exposure"].to_numpy(float) - trace["base_final_exposure"].to_numpy(float))
        > TOL
    )
    return {
        "start_index": expected,
        "trace_rows": trace.to_dict(orient="records"),
        "calls": load_jsonl(directory / "07_LLM_CALLS.jsonl"),
        "memory": memory,
        "llama_trust": llama_trust,
        "deterministic_trust": deterministic_trust,
        "deterministic_completed": deterministic_completed,
        "pretrade": pretrade,
        "audit": {
            "mode": "RESUMED_FROM_VERIFIED_STAGE6_2021",
            "path": str(directory),
            "rows_reused": expected,
            "llama_calls_avoided": expected,
            "state_boundary": "2021-12-31 completed return -> 2022-01-01 return decision",
        },
    }


def run_continuous(
    period: source.PeriodInputs,
    provider: Any,
    config: dict[str, Any],
    base_config: dict[str, Any],
    accounting: Any,
    risk: Any,
    seed: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], EpisodicMemory, RegimeTrust, RegimeTrust]:
    if seed is None:
        start_index = 0
        memory = EpisodicMemory()
        llama_trust = RegimeTrust(config["trust"])
        deterministic_trust = RegimeTrust(config["trust"])
        deterministic_completed: list[dict[str, Any]] = []
        pretrade = {
            "full": 0.0, "frozen": 0.0, "deterministic": 0.0,
            "shadow": 0.0, "deterministic_shadow": 0.0,
        }
        trace_rows: list[dict[str, Any]] = []
        calls: list[dict[str, Any]] = []
    else:
        start_index = int(seed["start_index"])
        memory = seed["memory"]
        llama_trust = seed["llama_trust"]
        deterministic_trust = seed["deterministic_trust"]
        deterministic_completed = seed["deterministic_completed"]
        pretrade = dict(seed["pretrade"])
        trace_rows = list(seed["trace_rows"])
        calls = list(seed["calls"])
    cost_rate = float(base_config["transaction_cost_bps"]) / 10000.0
    fixed_beta = float(config["trust"]["initial_beta"])
    for index in range(start_index, len(period.frame)):
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
            "objective": "add incremental risk-adjusted value over the pre-agent RAMoE control after costs",
            "state": state,
            "memory": memory_evidence,
            "safe_exposure_previews": previews,
            "constraints": {
                "allowed_actions": ["BTC", "CASH", "ABSTAIN"],
                "shorting": False,
                "leverage": False,
                "invalid_or_failed_response": "ABSTAIN_TO_PRE_AGENT_RAMOE",
            },
        }
        request_text = json.dumps(payload, sort_keys=True, allow_nan=False)
        if "asset_simple_return" in request_text or "canonical_asset_simple_return" in request_text:
            raise Stage61Error("future return leaked into Llama request")
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
                "request_id": f"continuous-{period.return_dates[index].date()}",
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
                "return_year": int(period.return_dates[index].year),
                "scientific_segment": (
                    "BURN_IN_2021" if period.return_dates[index].year == 2021
                    else "REUSED_OOS_PREQUENTIAL_DIAGNOSTIC"
                ),
                "memory_total_before": len(memory.episodes) - 1,
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
                "effective_intervention": bool(
                    abs(float(full_project.exposure) - float(base_row["final_exposure"])) > TOL
                ),
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
        if index == start_index or completed % 25 == 0 or completed == len(period.frame):
            valid_count = sum(bool(item["valid_action"]) for item in trace_rows)
            non_abstain = sum(item["action"] != "ABSTAIN" for item in trace_rows)
            print(
                f"STAGE61_PROGRESS={completed}/{len(period.frame)} "
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


def build_continuous_streams(
    trace: pd.DataFrame,
    period: source.PeriodInputs,
    base_config: dict[str, Any],
    adapter: Any,
) -> dict[str, Stream]:
    asset = trace["asset_simple_return"].to_numpy(float)
    cost = float(base_config["transaction_cost_bps"]) / 10000.0
    bh_return, bh_turnover = adapter.constant_exposure_returns(asset, 1.0, cost)
    cash_return, cash_turnover = adapter.constant_exposure_returns(asset, 0.0, cost)
    return {
        "ramas_full_continuous_llama70b_memory_trust": Stream(
            trace["full_net_return"].to_numpy(float),
            trace["full_final_exposure"].to_numpy(float),
            trace["full_turnover"].to_numpy(float),
        ),
        "pre_agent_ramoe_internal_control": Stream(
            trace["base_net_return"].to_numpy(float),
            trace["base_final_exposure"].to_numpy(float),
            period.current_trace["turnover"].to_numpy(float),
        ),
        "llama70b_frozen_agent_trust_ablation": Stream(
            trace["frozen_beta_net_return"].to_numpy(float),
            trace["frozen_beta_final_exposure"].to_numpy(float),
            trace["frozen_beta_turnover"].to_numpy(float),
        ),
        "deterministic_same_information_ablation": Stream(
            trace["deterministic_net_return"].to_numpy(float),
            trace["deterministic_final_exposure"].to_numpy(float),
            trace["deterministic_turnover"].to_numpy(float),
        ),
        "llama70b_unblended_shadow_diagnostic": Stream(
            trace["shadow_net_return"].to_numpy(float),
            trace["shadow_final_exposure"].to_numpy(float),
            trace["shadow_turnover"].to_numpy(float),
        ),
        "buy_and_hold_external_reference": Stream(
            np.asarray(bh_return), np.ones(len(trace)), np.asarray(bh_turnover)
        ),
        "cash_only_external_reference": Stream(
            np.asarray(cash_return), np.zeros(len(trace)), np.asarray(cash_turnover)
        ),
    }


def classify_metric_pattern(proposed: pd.Series, control: pd.Series) -> str:
    profit_better = float(proposed["terminal_growth"]) > float(control["terminal_growth"])
    sharpe_better = float(proposed["sharpe_zero_cash_rate"]) > float(control["sharpe_zero_cash_rate"])
    drawdown_better = float(proposed["maximum_drawdown_loss"]) < float(control["maximum_drawdown_loss"])
    cvar_better = float(proposed["daily_loss_cvar_95"]) < float(control["daily_loss_cvar_95"])
    count = sum((profit_better, sharpe_better, drawdown_better, cvar_better))
    if count == 4:
        return "BETTER_ON_ALL_REPORTED_PROFIT_AND_RISK_METRICS"
    if count == 0:
        return "WORSE_ON_ALL_REPORTED_PROFIT_AND_RISK_METRICS"
    return "MIXED_METRIC_FINDING_DO_NOT_REJECT_ARCHITECTURE"


def evaluate_continuous(
    trace: pd.DataFrame,
    period: source.PeriodInputs,
    config: dict[str, Any],
    base_config: dict[str, Any],
    adapter: Any,
    memory: EpisodicMemory,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    streams = build_continuous_streams(trace, period, base_config, adapter)
    full_metrics = pd.DataFrame(
        [metric_row(name, stream) for name, stream in streams.items()]
    )
    yearly_rows: list[dict[str, Any]] = []
    years = trace["return_year"].to_numpy(int)
    for year in sorted(set(years.tolist())):
        mask = years == year
        for name, stream in streams.items():
            row = metric_row(
                name,
                Stream(stream.returns[mask], stream.exposure[mask], stream.turnover[mask]),
            )
            row["return_year"] = int(year)
            row["rows"] = int(np.sum(mask))
            yearly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)
    yearly = yearly[["return_year", "rows"] + [c for c in yearly.columns if c not in {"return_year", "rows"}]]
    proposed_name = "ramas_full_continuous_llama70b_memory_trust"
    control_name = "pre_agent_ramoe_internal_control"
    finding_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, np.ndarray]] = [("FULL_2021_2025", np.ones(len(trace), dtype=bool))]
    scopes.extend((str(year), years == year) for year in sorted(set(years.tolist())))
    for scope_name, mask in scopes:
        proposed = metric_row(
            proposed_name,
            Stream(
                streams[proposed_name].returns[mask],
                streams[proposed_name].exposure[mask],
                streams[proposed_name].turnover[mask],
            ),
        )
        control = metric_row(
            control_name,
            Stream(
                streams[control_name].returns[mask],
                streams[control_name].exposure[mask],
                streams[control_name].turnover[mask],
            ),
        )
        log_advantage = np.log1p(streams[proposed_name].returns[mask]) - np.log1p(
            streams[control_name].returns[mask]
        )
        inference = circular_block_test(log_advantage, config["bootstrap"])
        finding_rows.append(
            {
                "scope": scope_name,
                "rows": int(np.sum(mask)),
                "growth_delta_vs_pre_agent_ramoe": float(
                    proposed["terminal_growth"] - control["terminal_growth"]
                ),
                "sharpe_delta_vs_pre_agent_ramoe": float(
                    proposed["sharpe_zero_cash_rate"] - control["sharpe_zero_cash_rate"]
                ),
                "maximum_drawdown_loss_delta_vs_pre_agent_ramoe": float(
                    proposed["maximum_drawdown_loss"] - control["maximum_drawdown_loss"]
                ),
                "daily_cvar_95_delta_vs_pre_agent_ramoe": float(
                    proposed["daily_loss_cvar_95"] - control["daily_loss_cvar_95"]
                ),
                "mean_daily_log_advantage": inference["observed_mean_log_advantage"],
                "lower_one_sided_95_bound": inference["lower_one_sided_confidence_bound"],
                "one_sided_p_value": inference["one_sided_p_value"],
                "metric_pattern": classify_metric_pattern(pd.Series(proposed), pd.Series(control)),
                "economic_value_established": bool(
                    inference["lower_one_sided_confidence_bound"] > 0.0
                    and float(proposed["terminal_growth"]) > float(control["terminal_growth"])
                    and float(proposed["sharpe_zero_cash_rate"]) >= float(control["sharpe_zero_cash_rate"])
                    and float(proposed["maximum_drawdown_loss"]) <= float(control["maximum_drawdown_loss"])
                    and float(proposed["daily_loss_cvar_95"]) <= float(control["daily_loss_cvar_95"])
                ),
            }
        )
    findings = pd.DataFrame(finding_rows)
    effective = trace["effective_intervention"].astype(bool)
    valid = trace["valid_action"].astype(bool)
    valid_rate = float(valid.mean())
    invalid_fail_closed = bool(
        (trace.loc[~valid, "action"] == config["runtime_validity"]["invalid_response_action"]).all()
    )
    expected_counts = {2021: 364, 2022: 365, 2023: 365, 2024: 366, 2025: 148}
    observed_counts = {
        int(year): int(count) for year, count in trace.groupby("return_year").size().items()
    }
    checks = {
        "all_expected_rows_present": len(trace) == int(config["continuous_timeline"]["total_rows"]),
        "expected_year_counts": observed_counts == expected_counts,
        "invalid_responses_fail_closed": invalid_fail_closed,
        "minimum_valid_action_rate": (
            valid_rate + TOL >= float(config["runtime_validity"]["minimum_valid_action_rate"])
        ),
        "memory_has_one_completed_episode_per_row": len(memory.episodes) == len(trace),
        "next_day_clock": bool(
            np.all(
                (pd.to_datetime(trace["return_date"]) - pd.to_datetime(trace["decision_date"])).dt.days
                == 1
            )
        ),
        "qwen_not_used": config["provider"]["model"] == "llama3.3:70b",
        "no_year_boundary_reset_declared": all(
            config["frozen_claims"][key] is False
            for key in (
                "year_boundary_resets_memory",
                "year_boundary_resets_agent_trust",
                "year_boundary_resets_portfolio_state",
            )
        ),
    }
    diagnostics = {
        "technical_status": "PASS" if all(checks.values()) else "FAIL",
        "technical_checks": checks,
        "observed_return_year_counts": observed_counts,
        "valid_action_rate": valid_rate,
        "non_abstain_actions": int((trace["action"] != "ABSTAIN").sum()),
        "effective_interventions": int(effective.sum()),
        "effective_intervention_rate": float(effective.mean()),
        "completed_memory_episodes": len(memory.episodes),
        "interpretation_rule": (
            "Annual and metric-specific wins/losses are research findings. "
            "Economic underperformance does not stop or erase the continuous architecture."
        ),
        "untouched_oos_confirmation": False,
        "novelty_established_by_this_run": False,
    }
    return full_metrics, yearly, findings, diagnostics


def write_continuous(
    output: Path,
    config: dict[str, Any],
    provider: Any,
    corrected_run: Path,
    base_dir: Path,
    raw_audit: dict[str, Any],
    period: source.PeriodInputs,
    preflight: dict[str, Any],
    seed_audit: dict[str, Any],
    trace: pd.DataFrame,
    calls: list[dict[str, Any]],
    memory: EpisodicMemory,
    llama_trust: RegimeTrust,
    deterministic_trust: RegimeTrust,
    metrics: pd.DataFrame,
    yearly: pd.DataFrame,
    findings: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        raise Stage61Error(f"refusing to overwrite output={output}")
    output.mkdir(parents=True)
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
            "research_contract": config["research_contract"],
            "frozen_claims": config["frozen_claims"],
            "timeline": {
                "2015_2020": "market_and_router_history_only_no_agent_episode_trace_available",
                "2021": "burn_in_state_carried_forward",
                "2022_2023": "continuous_walk_forward_development_evidence",
                "2024_2025": "continuous_reused_oos_diagnostic_not_untouched_confirmation",
            },
        },
    )
    source.atomic_json(
        output / "01_SOURCE_AND_SEED_AUDIT.json",
        {"raw_market": raw_audit, "period": period.source_audit, "seed": seed_audit},
    )
    source.atomic_json(output / "02_SEMANTIC_PREFLIGHT.json", preflight)
    source.atomic_csv(output / "03_DAILY_TRACE.csv", trace)
    source.atomic_csv(output / "04_FULL_HORIZON_METRICS.csv", metrics)
    source.atomic_csv(output / "05_YEARLY_METRICS.csv", yearly)
    source.atomic_csv(output / "06_PRE_AGENT_RAMOE_COMPARATIVE_FINDINGS.csv", findings)
    source.atomic_json(output / "07_TECHNICAL_DIAGNOSTICS.json", diagnostics)
    source.atomic_jsonl(output / "08_LLM_CALLS.jsonl", calls)
    source.atomic_jsonl(output / "09_MEMORY_EPISODES.jsonl", memory.episodes)
    source.atomic_json(
        output / "10_TRUST_STATE.json",
        {
            "llama_beta_final": llama_trust.beta,
            "llama_monthly_events": llama_trust.events,
            "deterministic_beta_final": deterministic_trust.beta,
            "deterministic_monthly_events": deterministic_trust.events,
        },
    )
    technical_pass = diagnostics["technical_status"] == "PASS"
    full_finding = findings.loc[findings["scope"] == "FULL_2021_2025"].iloc[0]
    final = {
        "decision": (
            "CONTINUOUS_PIPELINE_COMPLETE_REQUIRES_SCIENTIFIC_INTERPRETATION"
            if technical_pass else "STOP_TECHNICAL_VALIDITY_FAILURE"
        ),
        "technical_pipeline_valid": technical_pass,
        "full_horizon_economic_value_vs_pre_agent_ramoe_established": bool(
            full_finding["economic_value_established"]
        ),
        "overall_economic_value_against_external_baselines_established": False,
        "economic_result_is_not_a_software_acceptance_gate": True,
        "mixed_or_negative_years_retained_as_findings": True,
        "external_baseline_phase_completed": False,
        "novelty_established": False,
        "untouched_oos_confirmation": False,
        "post2021_rows_used": True,
        "post2021_scientific_label": "REUSED_OOS_DIAGNOSTIC",
        "qwen_used": False,
        "router_changed": False,
        "existing_ramoe_trust_changed": False,
        "risk_layer_changed": False,
        "next_step": (
            "interpret yearly and full-horizon technical/economic findings, then freeze the method "
            "for external baseline and later untouched confirmation experiments"
        ),
    }
    source.atomic_json(output / "11_FINAL_RESEARCH_STATUS.json", final)
    report = f"""# RAMAS Stage 6.1 continuous research result

Technical status: **{diagnostics['technical_status']}**

This is one uninterrupted prequential process. The verified 2021 Llama-70B memory,
regime-specific agent trust, and portfolio state are carried into 2022. The same
state then continues through 2023, 2024, and 2025. Year boundaries create reports;
they do not reset learning and do not impose an economic stop.

The proposed method is `ramas_full_continuous_llama70b_memory_trust`. The pre-agent
RAMoE stream is an internal mechanism control, not the headline external baseline.
Buy-and-hold and cash-only are included only as descriptive references in this
pipeline-validation run. ML/RL and full-LLM comparisons remain a later baseline phase.

The 2024-2025 rows have already been accessed by the broader project, so their result
is explicitly reused-OOS diagnostic evidence, not untouched confirmation. This run
cannot by itself establish novelty; novelty requires a separate literature audit.
"""
    (output / "12_PLAIN_ENGLISH_REPORT.md").write_text(report, encoding="utf-8")
    hashes = {
        path.name: source.sha256_file(path)
        for path in sorted(output.iterdir()) if path.is_file()
    }
    source.atomic_json(
        output / "RUN_COMPLETE.json",
        {
            "status": "PASS" if technical_pass else "FAIL",
            "decision": final["decision"],
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
    parser.add_argument(
        "--resume-stage6-output",
        type=Path,
        help="Verified Stage 6 development_2021 result used as the continuous initial state",
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    provider = provider_from(config, args.provider)
    output = args.output.expanduser().resolve()
    if args.preflight_only:
        summary, records = run_semantic_preflight(provider, config)
        write_preflight(output, summary, records)
        print("RAMAS_STAGE6_1_SEMANTIC_PREFLIGHT_STATUS=PASS" if summary["status"] == "PASS" else "RAMAS_STAGE6_1_SEMANTIC_PREFLIGHT_STATUS=FAIL", flush=True)
        print(f"DECISION={summary['decision']}", flush=True)
        print(f"OUTPUT={output}", flush=True)
        if summary["status"] != "PASS":
            raise SystemExit(4)
        return
    if args.preflight_evidence is None:
        raise Stage61Error("continuous replay requires --preflight-evidence")
    preflight = verify_preflight(args.preflight_evidence.resolve(), config, provider)
    project_root = args.project_root.expanduser().resolve()
    corrected_run = source.resolve_corrected_run(project_root, args.corrected_run, config)
    base_dir = source.resolve_base_dir(corrected_run)
    adapter, accounting, risk, _statistics = source.import_base(base_dir)
    base_config = adapter.load_config(base_dir / "config.json")
    raw, raw_audit = load_raw_continuous(args.full_dataset, config)
    period = load_continuous_period(corrected_run, raw, base_config, config)
    seed = None
    seed_audit: dict[str, Any] = {
        "mode": "FRESH_REPLAY_FROM_2021",
        "rows_reused": 0,
        "llama_calls_avoided": 0,
    }
    if args.resume_stage6_output is not None:
        seed = load_stage6_seed(
            args.resume_stage6_output.expanduser().resolve(),
            period, config, base_config, accounting, risk,
        )
        seed_audit = seed["audit"]
    trace, calls, memory, llama_trust, deterministic_trust = run_continuous(
        period, provider, config, base_config, accounting, risk, seed
    )
    metrics, yearly, findings, diagnostics = evaluate_continuous(
        trace, period, config, base_config, adapter, memory
    )
    final = write_continuous(
        output, config, provider, corrected_run, base_dir, raw_audit, period,
        preflight, seed_audit, trace, calls, memory, llama_trust,
        deterministic_trust, metrics, yearly, findings, diagnostics,
    )
    print("RAMAS_STAGE6_1_STATUS=PASS" if final["technical_pipeline_valid"] else "RAMAS_STAGE6_1_STATUS=FAIL", flush=True)
    print(f"DECISION={final['decision']}", flush=True)
    print(f"TECHNICAL_PIPELINE_VALID={str(final['technical_pipeline_valid']).upper()}", flush=True)
    print("POST2021_ROWS_USED=TRUE", flush=True)
    print("POST2021_LABEL=REUSED_OOS_DIAGNOSTIC", flush=True)
    print("QWEN_USED=FALSE", flush=True)
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
