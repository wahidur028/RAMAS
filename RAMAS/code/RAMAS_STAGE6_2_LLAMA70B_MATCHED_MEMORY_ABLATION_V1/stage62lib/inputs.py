"""Reconstruct and audit the exact Stage 6.1 exogenous input stream.

Archived agent decisions are checked solely as provenance/replication evidence.
They are never returned to the new agents or used to seed either arm.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage62lib.legacy import SYSTEM_PROMPT, legacy, source


class InputAuditError(RuntimeError):
    pass


FEATURE_KEYS = (
    "return_1", "return_7", "return_30", "return_90", "realized_vol_30",
    "drawdown_90", "router_entropy", "router_confidence", "router_transition_l1",
)
EXOGENOUS_STATE_KEYS = (
    "decision_date", "target_return_date", "prob_bear", "prob_bull", "prob_mix",
    "hard_regime", "base_ramoe_desired_exposure", *FEATURE_KEYS,
)
PORTFOLIO_STATE_KEYS = ("pretrade_exposure", "llama_trust_beta")
PREVIEW_KEYS = (
    "blended_desired_exposure", "risk_limited_exposure", "turnover", "ambiguity_cvar",
)


def _inside(directory: Path, relative: str) -> Path:
    """Resolve a relative artifact name without permitting external targets."""
    directory = directory.resolve()
    if Path(relative).is_absolute():
        raise InputAuditError(f"expected relative path, received: {relative}")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise InputAuditError(f"path escapes its source directory: {relative}")
    return path


def _required(path: Path, directory: bool = False) -> Path:
    okay = path.is_dir() if directory else path.is_file()
    if not okay:
        raise InputAuditError(
            f"Required frozen {'directory' if directory else 'file'} is missing: {path}. "
            "Restore that exact historical source; this experiment does not select a newer run."
        )
    return path


def _hash(path: Path, expected: str, label: str) -> str:
    _required(path)
    digest = source.sha256_file(path)
    if digest != expected:
        raise InputAuditError(f"{label} checksum differs: {path}; expected={expected}; actual={digest}")
    return digest


def _close(actual: Any, expected: Any, label: str, tolerance: float = 1e-10) -> float:
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise InputAuditError(f"{label}: different shapes or non-finite numeric values")
    error = float(np.max(np.abs(left - right))) if left.size else 0.0
    if error > tolerance:
        raise InputAuditError(f"{label}: maximum absolute difference={error}, allowed={tolerance}")
    return error


def _verify_archive(result: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    complete_path = _required(result / "RUN_COMPLETE.json")
    _hash(complete_path, config["expected_source_complete_sha256"], "Stage 6.1 completion record")
    complete = source.load_json(complete_path)
    if complete.get("status") != "PASS":
        raise InputAuditError("Stage 6.1 source did not finish successfully")
    artifacts = complete.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise InputAuditError("Stage 6.1 source is missing its artifact hash map")
    required = {
        "00_CONTRACT.json", "01_SOURCE_AND_SEED_AUDIT.json", "03_DAILY_TRACE.csv", "08_LLM_CALLS.jsonl",
    }
    if not required.issubset(artifacts):
        raise InputAuditError("Stage 6.1 completion record does not identify every required artifact")
    verified = {}
    for relative, expected in artifacts.items():
        verified[relative] = _hash(_inside(result, relative), expected, "Stage 6.1 artifact")
    if verified["03_DAILY_TRACE.csv"] != config["expected_source_trace_sha256"]:
        raise InputAuditError("Stage 6.1 daily trace is not the selected historical trace")
    return complete, verified


def _audit_archived_requests(
    result: Path, period: Any, archived: pd.DataFrame,
    base_config: dict[str, Any], accounting: Any, risk: Any,
) -> dict[str, Any]:
    """Reconstruct features/previews, retaining no old responses in returned data."""
    calls = legacy.load_jsonl(result / "08_LLM_CALLS.jsonl")
    if len(calls) != len(period.frame):
        raise InputAuditError("Archived request count does not equal the matched daily input count")
    if set(period.features.columns) != set(FEATURE_KEYS):
        raise InputAuditError("Feature schema changed; review it before extending the agent input allowlist")
    pretrade = 0.0
    max_state_error = 0.0
    max_preview_error = 0.0
    max_net_error = 0.0
    for index, call in enumerate(calls):
        old_row = archived.iloc[index]
        payload = call["request"]
        request_text = json.dumps(payload, sort_keys=True, allow_nan=False)
        if source.sha256_text(SYSTEM_PROMPT + "\n" + request_text) != call["request_sha256"]:
            raise InputAuditError(f"Archived request digest mismatch at row {index}")
        if source.sha256_text(call["raw_response"]) != call["response_sha256"]:
            raise InputAuditError(f"Archived response digest mismatch at row {index}")
        old_state = payload["state"]
        if set(old_state) != set(EXOGENOUS_STATE_KEYS + PORTFOLIO_STATE_KEYS):
            raise InputAuditError(f"Unexpected archived state fields at row {index}; do not copy them to new agents")
        base_desired = float(period.current_trace.iloc[index]["desired_exposure"])
        reconstructed = legacy.state_for(period, index, pretrade, base_desired)
        for key in EXOGENOUS_STATE_KEYS:
            if isinstance(reconstructed[key], str):
                if reconstructed[key] != old_state[key]:
                    raise InputAuditError(f"Archived state field {key} differs at row {index}")
            else:
                max_state_error = max(max_state_error, _close(
                    reconstructed[key], old_state[key], f"state[{index}].{key}",
                ))
        _close(pretrade, old_state["pretrade_exposure"], f"archived portfolio clock[{index}]")
        beta = float(old_row["llama_beta"])
        _close(beta, old_state["llama_trust_beta"], f"archived beta[{index}]")
        previews = payload["safe_exposure_previews"]
        if set(previews) != {"BTC", "CASH", "ABSTAIN"}:
            raise InputAuditError(f"Unexpected archived risk preview actions at row {index}")
        for action in ("BTC", "CASH", "ABSTAIN"):
            if set(previews[action]) != set(PREVIEW_KEYS):
                raise InputAuditError(f"Unexpected archived risk preview fields at row {index}")
            projected = legacy.project_action(
                action, beta, base_desired, pretrade, period.scenarios[index], base_config, risk,
            )
            actual_preview = {
                "blended_desired_exposure": legacy.blend_exposure(base_desired, action, beta),
                "risk_limited_exposure": float(projected.exposure),
                "turnover": float(projected.turnover),
                "ambiguity_cvar": float(projected.ambiguity_cvar),
            }
            for key in PREVIEW_KEYS:
                max_preview_error = max(max_preview_error, _close(
                    actual_preview[key], previews[action][key], f"risk preview[{index}].{action}.{key}",
                ))
        old_exposure = float(old_row["full_final_exposure"])
        old_net = accounting.net_return(
            old_exposure, pretrade, float(period.asset_returns[index]),
            float(base_config["transaction_cost_bps"]) / 10000.0,
        )
        max_net_error = max(max_net_error, _close(old_net, old_row["full_net_return"], f"archived accounting[{index}]"))
        pretrade = float(accounting.drifted_exposure(old_exposure, float(period.asset_returns[index])))
    return {
        "archived_requests_verified": len(calls),
        "archived_response_digests_verified": len(calls),
        "archived_risk_previews_reconstructed": 3 * len(calls),
        "maximum_state_absolute_difference": max_state_error,
        "maximum_risk_preview_absolute_difference": max_preview_error,
        "maximum_accounting_absolute_difference": max_net_error,
        "numeric_tolerance": 1e-10,
        "exogenous_state_allowlist": list(EXOGENOUS_STATE_KEYS),
        "archived_portfolio_state_reconstructed_for_audit_only": True,
        "archived_decisions_or_responses_returned_to_new_arms": False,
    }


def load_inputs(
    project_root: Path, config: dict[str, Any], full_dataset: Path | None = None,
) -> tuple[Any, dict[str, Any], Any, Any, dict[str, Any]]:
    """Load frozen source data, verify against all original calls, then return it.

    The period holds only the existing deterministic source stream. Agent memory,
    trust and portfolio state must be independently initialized by the runtime.
    """
    root = project_root.expanduser().resolve()
    result = _required(_inside(root, config["source_result_relative"]), directory=True)
    corrected = _required(_inside(root, config["corrected_run_relative"]), directory=True)
    _, artifact_hashes = _verify_archive(result, config)
    contract = source.load_json(result / "00_CONTRACT.json")
    prior_audit = source.load_json(result / "01_SOURCE_AND_SEED_AUDIT.json")
    vendor_config_path = Path(legacy.__file__).resolve().parent / "config.json"
    _hash(vendor_config_path, contract["config_sha256"], "Vendored Stage 6.1 configuration")
    legacy_config = legacy.load_config(vendor_config_path)
    prompt_hash = source.sha256_text(SYSTEM_PROMPT)
    if prompt_hash != config["expected_vendor_prompt_sha256"] or prompt_hash != contract["system_prompt_sha256"]:
        raise InputAuditError("The archived and vendored Llama system prompt identities differ")
    if legacy_config["expected_raw_sha256"] != prior_audit["raw_market"]["sha256"]:
        raise InputAuditError("Raw market identity differs between the original run and vendor configuration")
    if int(config["expected_rows"]) != int(prior_audit["period"]["rows"]):
        raise InputAuditError("Requested row count differs from the matched original run")

    recorded_periods = {item["name"]: item for item in prior_audit["period"]["periods"]}
    if len(recorded_periods) != len(prior_audit["period"]["periods"]):
        raise InputAuditError("Duplicate source period in original audit")
    if set(recorded_periods) != {item["name"] for item in legacy_config["source_periods"]}:
        raise InputAuditError("Corrected period identities have changed")
    pinned_corrected_hashes = {}
    for name, expected in recorded_periods.items():
        directory = _required(_inside(corrected / "results", name), directory=True)
        inputs, trace = source.verify_child_artifact(directory)
        pinned_corrected_hashes[name] = {
            "corrected_inputs_sha256": _hash(inputs, expected["corrected_inputs_sha256"], "Original corrected input"),
            "base_trace_sha256": _hash(trace, expected["base_trace_sha256"], "Original deterministic control trace"),
        }

    base_dir = source.resolve_base_dir(corrected)
    # Inspect every manifest path before invoking source.import_base().
    manifest = _required(base_dir / "MANIFEST.sha256")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _digest, relative = line.split(maxsplit=1)
            _required(_inside(base_dir, relative.strip().lstrip("*")))
    base_manifest_files = source.verify_manifest(base_dir)
    adapter, accounting, risk, _statistics = source.import_base(base_dir)
    base_config = adapter.load_config(base_dir / "config.json")
    _close(float(base_config["transaction_cost_bps"]) / 10000.0, config["cost_rate"], "Transaction cost", 1e-15)
    if full_dataset is not None:
        _hash(full_dataset.expanduser().resolve(), legacy_config["expected_raw_sha256"], "Explicit raw market file")
        # An explicitly selected invalid file must never silently fall back.
        raw_config = dict(legacy_config, raw_dataset_candidates=[])
    else:
        raw_config = legacy_config
    raw, raw_audit = legacy.load_raw_continuous(full_dataset, raw_config)
    period = legacy.load_continuous_period(corrected, raw, base_config, legacy_config)
    archived = pd.read_csv(result / "03_DAILY_TRACE.csv", low_memory=False)
    if len(period.frame) != int(config["expected_rows"]) or len(archived) != len(period.frame):
        raise InputAuditError("The source and archived daily trace have different row counts")
    for key, dates in (("decision_date", period.decision_dates), ("return_date", period.return_dates)):
        archived_dates = pd.to_datetime(archived[key], errors="raise")
        if not np.array_equal(archived_dates.to_numpy(), dates.to_numpy()):
            raise InputAuditError(f"Archived {key} is not aligned to the matched input stream")
    _close(period.router_probabilities, archived[["prob_bear", "prob_bull", "prob_mix"]].to_numpy(float), "Router probabilities", 1e-12)
    _close(period.asset_returns, archived["asset_simple_return"], "Asset return stream", 1e-12)
    for source_column, archived_column in (
        ("desired_exposure", "base_desired_exposure"),
        ("final_exposure", "base_final_exposure"),
        ("portfolio_net_return", "base_net_return"),
    ):
        _close(period.current_trace[source_column], archived[archived_column], f"Deterministic source {source_column}", 1e-12)
    close = raw.set_index("date")["close"]
    raw_returns = close.loc[period.return_dates].to_numpy(float) / close.loc[period.decision_dates].to_numpy(float) - 1.0
    raw_return_error = _close(raw_returns, period.asset_returns, "Raw close-to-close return clock", 1e-12)
    preview_audit = _audit_archived_requests(result, period, archived, base_config, accounting, risk)
    audit = {
        "status": "VERIFIED_BEFORE_NEW_LLM_CALLS",
        "project_root": str(root),
        "source_result": str(result),
        "source_run_id": config["source_run_id"],
        "source_complete_sha256": source.sha256_file(result / "RUN_COMPLETE.json"),
        "source_artifact_sha256": artifact_hashes,
        "source_audit_sha256": artifact_hashes["01_SOURCE_AND_SEED_AUDIT.json"],
        "corrected_source_run": str(corrected),
        "corrected_period_identity_checks": pinned_corrected_hashes,
        "frozen_base_source": str(base_dir),
        "base_manifest_sha256": source.sha256_file(manifest),
        "base_manifest_files": base_manifest_files,
        "base_config_sha256": source.sha256_file(base_dir / "config.json"),
        "vendor_config_sha256": source.sha256_file(vendor_config_path),
        "system_prompt_sha256": prompt_hash,
        "raw_market": raw_audit,
        "period": period.source_audit,
        "maximum_raw_return_clock_difference": raw_return_error,
        "strict_next_day_return_clock": True,
        "archived_state_reconstruction": preview_audit,
        "archived_agent_seed_reused": False,
        "new_arms_have_independent_empty_initial_memory": True,
        "old_responses_used_for_new_decisions": False,
        "legacy_control_2024_boundary_seam": "PRESERVED_AS_COMMON_REFERENCE; NOT_REPAIRED_IN_THIS_MATCHED_TEST",
        "upstream_router_training_provenance": "INHERITED_FROZEN_SOURCE; NOT_RETRAINED_OR_INDEPENDENTLY_REAUDITED",
        "base_manifest_provenance_limit": (
            "The original Stage 6.1 output did not independently pin the base manifest. "
            "Its present manifest is verified and recorded; all original numeric risk previews "
            "and portfolio returns are reconstructed before new model calls."
        ),
    }
    return period, base_config, accounting, risk, audit
