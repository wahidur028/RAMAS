"""Bounded scientific audit before inference, with unresolved clocks made explicit.

Integrity and matched inputs are testable here. Forecast fit chronology and the
actual close/inference/fill sequence are not established by a historical CSV.
This module never repairs the legacy control reference or supplies old decisions
to a new arm. The returned adaptive rows are reporting-only reference evidence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage63lib.inputs import (
    EXOGENOUS_STATE_KEYS, FEATURE_KEYS, PORTFOLIO_STATE_KEYS,
    InputAuditError, _close, _hash, _inside, _required,
)
from stage63lib.legacy import SYSTEM_PROMPT, legacy, source


DIAGNOSTIC_SCOPE = (
    "LEGACY_MECHANISM_DIAGNOSTIC_NOT_REALISTIC_EXECUTION_OR_FRESH_OOS_CONFIRMATION"
)
RECORDED_ROUTER_CLOCK = (
    "router decision d-1 -> router target/baseline decision d -> baseline target return d+1"
)
SEAM_RETURN_DATE = "2024-01-01"
TOLERANCE = 1e-10


def _clock_and_numeric_audit(period: Any) -> dict[str, Any]:
    decisions = pd.DatetimeIndex(pd.to_datetime(period.decision_dates, errors="raise"))
    returns = pd.DatetimeIndex(pd.to_datetime(period.return_dates, errors="raise"))
    count = len(period.frame)
    if not count or len(decisions) != count or len(returns) != count:
        raise InputAuditError("Decision/return clock row counts are inconsistent")
    if decisions.hasnans or returns.hasnans:
        raise InputAuditError("Missing decision or return date")
    if not decisions.is_unique or not returns.is_unique:
        raise InputAuditError("Duplicate date in decision/return clock")
    if not decisions.is_monotonic_increasing or not returns.is_monotonic_increasing:
        raise InputAuditError("Decision/return clock is not increasing")
    day = pd.Timedelta(days=1)
    if not ((returns - decisions) == day).all():
        raise InputAuditError("Holding return must be exactly one day after its decision")
    if count > 1 and not ((decisions[1:] - decisions[:-1]) == day).all():
        raise InputAuditError("Missing calendar day in continuous decision clock")
    q = np.asarray(period.router_probabilities, dtype=float)
    if q.shape != (count, 3) or not np.isfinite(q).all():
        raise InputAuditError("Regime probabilities must be a finite N x 3 array")
    if (q < 0).any() or (q > 1).any():
        raise InputAuditError("Regime probability outside [0, 1]")
    simplex_error = _close(q.sum(axis=1), np.ones(count), "Regime probability simplex", 1e-12)
    asset = np.asarray(period.asset_returns, dtype=float)
    if asset.shape != (count,) or not np.isfinite(asset).all() or (asset <= -1).any():
        raise InputAuditError("Asset returns must be finite and greater than -1")
    if set(period.features.columns) != set(FEATURE_KEYS) or len(period.features) != count:
        raise InputAuditError("Unexpected feature schema or count; future fields are not allowed")
    if not np.isfinite(period.features.to_numpy(float)).all():
        raise InputAuditError("Non-finite decision features")
    scenarios = np.asarray(period.scenarios, dtype=float)
    if len(scenarios) != count or not np.isfinite(scenarios).all():
        raise InputAuditError("Risk scenario count or finite-value check failed")
    desired = np.asarray(period.current_trace["desired_exposure"], dtype=float)
    if desired.shape != (count,) or not np.isfinite(desired).all() or ((desired < 0) | (desired > 1)).any():
        raise InputAuditError("Core desired exposures must be finite in [0, 1]")
    return {
        "rows": count,
        "first_decision_date": decisions[0].date().isoformat(),
        "last_return_date": returns[-1].date().isoformat(),
        "consecutive_one_day_holding_intervals": True,
        "probability_order": ["bear", "bull", "mix"],
        "probability_simplex_maximum_error": simplex_error,
        "feature_allowlist": list(FEATURE_KEYS),
        "numeric_inputs_finite": True,
        "date_alignment_proves_forecast_fit_chronology": False,
    }


def _reference_accounting_audit(
    period: Any, cost: float, accounting: Any, *, require_known_seam: bool,
) -> dict[str, Any]:
    """Detect the pinned reset using continuous drift; do not correct the data."""
    trace = period.current_trace
    required = {"final_exposure", "pretrade_exposure", "portfolio_net_return"}
    if not required.issubset(trace.columns) or len(trace) != len(period.frame):
        raise InputAuditError("Legacy reference lacks required exposure/accounting fields")
    values = trace[list(required)].to_numpy(float)
    if not np.isfinite(values).all():
        raise InputAuditError("Legacy reference accounting contains non-finite values")
    continuous_pretrade = 0.0
    discrepancies = []
    recorded_net_error = 0.0
    for i in range(len(trace)):
        row = trace.iloc[i]
        x, p = float(row["final_exposure"]), float(row["pretrade_exposure"])
        if not 0 <= x <= 1 or not 0 <= p <= 1:
            raise InputAuditError(f"Legacy reference exposure outside [0, 1] at row {i}")
        r, observed = float(period.asset_returns[i]), float(row["portfolio_net_return"])
        recorded_net = float(accounting.net_return(x, p, r, cost))
        recorded_net_error = max(recorded_net_error, _close(
            recorded_net, observed, f"Legacy recorded-pretrade accounting[{i}]", TOLERANCE,
        ))
        expected_net = float(accounting.net_return(x, continuous_pretrade, r, cost))
        if abs(p - continuous_pretrade) > TOLERANCE or abs(expected_net - observed) > TOLERANCE:
            return_date = pd.Timestamp(period.return_dates[i]).date().isoformat()
            if return_date != SEAM_RETURN_DATE or abs(p) > TOLERANCE:
                raise InputAuditError(f"Unexpected legacy reference state discontinuity at {return_date}")
            discrepancies.append({
                "index_zero_based": i,
                "decision_date": pd.Timestamp(period.decision_dates[i]).date().isoformat(),
                "return_date": return_date,
                "recorded_pretrade_exposure": p,
                "continuous_pretrade_exposure": continuous_pretrade,
                "recorded_final_exposure": x,
                "recorded_reference_net_return": observed,
                "same_final_exposure_continuous_pretrade_net_return": expected_net,
                "recorded_minus_conditional_continuous_net_return": observed - expected_net,
            })
        continuous_pretrade = float(accounting.drifted_exposure(x, r))
    if require_known_seam and len(discrepancies) != 1:
        raise InputAuditError("Pinned legacy reference must retain exactly the known 2024 accounting seam")
    return {
        "known_reference_discontinuities": discrepancies,
        "known_reference_discontinuity_count": len(discrepancies),
        "maximum_recorded_accounting_error": recorded_net_error,
        "reference_repaired": False,
        "reference_return_used_by_agent_outcome_memory": "EXACT_UNCHANGED_LEGACY_REFERENCE",
        "conditional_calculation_is_repaired_policy_rerun": False,
        "limitation": (
            "The conditional comparison holds the recorded final exposure fixed. A complete repair "
            "can also change projected exposure, memory and subsequent decisions. A shared error "
            "does not establish an unbiased estimate for a corrected trading system."
        ),
    }


def _verify_adaptive_archive(result: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    _hash(result / "RUN_COMPLETE.json", config["expected_adaptive_complete_sha256"], "Stage 6.2 completion")
    complete = source.load_json(result / "RUN_COMPLETE.json")
    if complete.get("status") != "PASS":
        raise InputAuditError("Selected adaptive reference did not complete")
    hashes = complete.get("artifact_sha256")
    required = {"00_CONTRACT.json", "01_SOURCE_AUDIT.json", "03_DAILY_TRACE.csv", "10_FINAL_STATUS.json"}
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        raise InputAuditError("Adaptive completion lacks required artifact identities")
    verified = {name: _hash(_inside(result, name), digest, "Stage 6.2 artifact") for name, digest in hashes.items()}
    if verified["03_DAILY_TRACE.csv"] != config["expected_adaptive_trace_sha256"]:
        raise InputAuditError("Adaptive daily trace is not the frozen reference")
    return complete, verified


def _matched_contract(config: dict[str, Any], contract: dict[str, Any], complete: dict[str, Any]) -> dict[str, Any]:
    old = contract["config"]
    common_keys = (
        "agent", "trust", "analysis", "cost_rate", "arms", "initialization",
        "continue_across_years", "no_yearly_economic_hard_stop", "no_memory_disables",
        "legacy_control_seam_preserved_for_matched_replication", "source_run_id",
        "source_result_relative", "corrected_run_relative", "expected_source_complete_sha256",
        "expected_source_trace_sha256", "expected_vendor_prompt_sha256", "expected_rows",
        "minimum_valid_action_rate", "maximum_consecutive_invalid", "preflight", "qwen_used",
    )
    for key in common_keys:
        if config.get(key) != old.get(key):
            raise InputAuditError(f"Matched experiment setting changed unexpectedly: {key}")
    # A service address is operational; the model, budgets and failure rules are scientific settings.
    for key in ("kind", "model", "temperature", "seed", "timeout_seconds", "maximum_retries", "num_ctx", "num_predict"):
        if config["provider"].get(key) != old["provider"].get(key):
            raise InputAuditError(f"Matched provider setting changed unexpectedly: {key}")
    identity = contract["identity"]
    if identity != complete["identity"]:
        raise InputAuditError("Adaptive contract and completion identities differ")
    if identity["system_prompt_sha256"] != source.sha256_text(SYSTEM_PROMPT):
        raise InputAuditError("System prompt is not the matched adaptive prompt")
    provider = identity["provider_identity"]
    if provider["model"] != "llama3.3:70b" or provider["model_digest"] != config["expected_model_digest"]:
        raise InputAuditError("Adaptive model identity is not the pinned Llama 70B")
    if config.get("agent_trust_mode") != "FIXED_POSITIVE" or config.get("fixed_beta") != 0.05:
        raise InputAuditError("This experiment requires fixed beta 0.05")
    if config.get("common_legacy_trust_rule") is not False or old.get("common_legacy_trust_rule") is not True:
        raise InputAuditError("Expected fixed-versus-adaptive trust intervention is not declared")
    if config.get("scientific_scope") != DIAGNOSTIC_SCOPE:
        raise InputAuditError("Known clock limitations require the explicit legacy diagnostic scope")
    return {"common_config_keys_verified": list(common_keys), "adaptive_reference_identity": identity}


def audit_scientific_inputs(
    project_root: Path, config: dict[str, Any], period: Any,
    base_config: dict[str, Any], accounting: Any,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Run after inputs.load_inputs and before model metadata or inference calls."""
    root = project_root.expanduser().resolve()
    result = _required(_inside(root, config["adaptive_reference_relative"]), directory=True)
    complete, artifact_hashes = _verify_adaptive_archive(result, config)
    contract = source.load_json(result / "00_CONTRACT.json")
    match = _matched_contract(config, contract, complete)
    old_audit = source.load_json(result / "01_SOURCE_AUDIT.json")
    base_dir = source.resolve_base_dir(_inside(root, config["corrected_run_relative"]))
    base_files = old_audit["base_manifest_files"]
    for name, digest in base_files.items():
        _hash(_inside(base_dir, name), digest, "Matched deterministic numerical source")
    if base_config != source.load_json(base_dir / "config.json"):
        raise InputAuditError("Loaded numerical configuration differs from the matched source")
    if base_config.get("router", {}).get("execution_clock") != RECORDED_ROUTER_CLOCK:
        raise InputAuditError("Frozen upstream router clock contract has changed")
    with (result / "03_DAILY_TRACE.csv").open(newline="", encoding="utf-8") as handle:
        adaptive_rows = list(csv.DictReader(handle))
    clock = _clock_and_numeric_audit(period)
    if len(adaptive_rows) != clock["rows"] or clock["rows"] != int(config["expected_rows"]):
        raise InputAuditError("Adaptive and current input row counts differ")
    maximum_state_error = 0.0
    for i, row in enumerate(adaptive_rows):
        state = legacy.state_for(period, i, 0.0, float(period.current_trace.iloc[i]["desired_exposure"]))
        if row["decision_date"] != state["decision_date"] or row["return_date"] != state["target_return_date"]:
            raise InputAuditError(f"Adaptive daily clock does not match current inputs at row {i}")
        _close(float(row["asset_simple_return"]), period.asset_returns[i], f"Adaptive asset return[{i}]", 1e-12)
        if row["hard_regime"] != state["hard_regime"]:
            raise InputAuditError(f"Adaptive regime argmax differs at row {i}")
        for arm in ("memory", "no_memory"):
            name = f"calls/{i + 1:06d}-{arm}.json"
            if name not in artifact_hashes:
                raise InputAuditError(f"Adaptive source call is not in the pinned manifest: {name}")
            archived = source.load_json(result / name)["request"]["state"]
            if set(archived) != set(EXOGENOUS_STATE_KEYS + PORTFOLIO_STATE_KEYS):
                raise InputAuditError(f"Unexpected adaptive state fields: {name}")
            for key in EXOGENOUS_STATE_KEYS:
                if isinstance(state[key], str):
                    if state[key] != archived[key]:
                        raise InputAuditError(f"Adaptive exogenous state mismatch: {name}: {key}")
                else:
                    maximum_state_error = max(maximum_state_error, _close(
                        state[key], archived[key], f"Adaptive state {name}: {key}", TOLERANCE,
                    ))
    reference = _reference_accounting_audit(period, float(config["cost_rate"]), accounting, require_known_seam=True)
    audit = {
        "status": "DIAGNOSTIC_WITH_OPEN_TIMING",
        "integrity_checks_passed": True,
        "scientific_validation_established": False,
        "permitted_scope": DIAGNOSTIC_SCOPE,
        "model_calls_during_audit": 0,
        "adaptive_reference_result": str(result),
        "adaptive_complete_sha256": config["expected_adaptive_complete_sha256"],
        "adaptive_trace_sha256": config["expected_adaptive_trace_sha256"],
        "adaptive_artifacts_verified": len(artifact_hashes),
        "matched_numerical_source_hashes": base_files,
        **match,
        "input_clock_and_values": clock,
        "matched_exogenous_states_verified": 2 * len(adaptive_rows),
        "maximum_matched_state_error": maximum_state_error,
        "adaptive_rows_are_reporting_only_not_agent_seed": True,
        "legacy_reference_accounting": reference,
        "forecast_clock": {
            "recorded_source_contract": RECORDED_ROUTER_CLOCK,
            "recorded_model_name": base_config["router"]["preferred_model"],
            "interpretation": "Recorded prior-day posterior targets allocation day, not the subsequent holding day.",
            "holding_period_target_alignment_verified": False,
            "original_prediction_fit_label_timestamps_independently_verified": False,
            "report_label": "RECORDED_REGIME_POSTERIOR_NOT_VERIFIED_NEXT_HOLDING_PERIOD_FORECAST",
        },
        "execution_clock": {
            "return_arithmetic": "Close[d+1]/Close[d]-1; verified by inputs.load_inputs against raw prices",
            "features_and_risk_cutoff": "Through allocation-day close d, reconstructed by inputs.load_inputs",
            "fill_assumption": "Idealized fill at close d despite features requiring that close",
            "realistic_inference_and_execution_latency_verified": False,
        },
        "unresolved_scientific_limits": [
            "Original router training/feature/label timestamps require upstream evidence; date matching alone is insufficient.",
            "Recorded posterior target differs from the intended next holding-period regime target.",
            "Close-derived information plus same-close fill is idealized and does not prove executable returns.",
            "Legacy outcome-memory reference contains the explicitly measured 2024 restart seam.",
            "Post-2021 observations were previously inspected; this is reused-OOS diagnostic evidence.",
            "A retrospectively used pretrained model may know historical events even with supplied past-only features.",
            "Forecast fit/fill corrections change the candidate method and must be evaluated as separate matched runs.",
        ],
    }
    return audit, adaptive_rows


def audit_controlled_fixture(
    period: Any, base_config: dict[str, Any], accounting: Any,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Explicit synthetic branch. It cannot be accepted as market-source evidence."""
    if period.source_audit.get("synthetic") is not True or period.source_audit.get("economic_evidence") is not False:
        raise InputAuditError("Controlled audit requires an explicitly synthetic, non-economic fixture")
    return {
        "status": "CONTROLLED_FIXTURE_ONLY_NOT_SCIENTIFIC_EVIDENCE",
        "integrity_checks_passed": True,
        "scientific_validation_established": False,
        "economic_evidence": False,
        "adaptive_reference_identity": None,
        "model_calls_during_audit": 0,
        "input_clock_and_values": _clock_and_numeric_audit(period),
        "market_source_audit_performed": False,
    }, []
