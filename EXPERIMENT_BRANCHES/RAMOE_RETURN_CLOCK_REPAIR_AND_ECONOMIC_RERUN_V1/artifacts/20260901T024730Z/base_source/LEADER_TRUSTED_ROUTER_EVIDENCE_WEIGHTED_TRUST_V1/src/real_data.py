from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core import ContractError, TOL, sha256_file


@dataclass(frozen=True)
class Stage0Evidence:
    artifact_directory: Path
    decision: dict[str, Any]
    modality_audit: pd.DataFrame
    hashes: dict[str, str]


@dataclass(frozen=True)
class RealRouterInputs:
    decision_dates: pd.DatetimeIndex
    return_dates: pd.DatetimeIndex
    router_posteriors: np.ndarray
    expert_exposures: np.ndarray
    asset_simple_returns: np.ndarray
    scenario_log_returns: np.ndarray
    initial_trust: np.ndarray
    aligned_frame: pd.DataFrame
    input_contract: dict[str, Any]
    modality_alignment: pd.DataFrame


def _inside(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ContractError(f"Path escapes project root: {path}")
    return resolved


def resolve_first_file(project_root: Path, candidates: list[str]) -> Path:
    for relative in candidates:
        path = _inside(project_root, project_root / relative)
        if path.is_file():
            return path
    raise ContractError(f"None of the configured files exists: {candidates}")


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"Duplicate JSON key={key} in {path}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object in {path}")
    return value


def load_latest_stage0_evidence(project_root: Path, config: dict[str, Any]) -> Stage0Evidence:
    stage0 = config["stage0"]
    required = [stage0["decision_file"], stage0["modality_audit_file"]]
    candidates: list[Path] = []
    for relative in stage0["artifact_roots"]:
        root = _inside(project_root, project_root / relative)
        if root.is_dir():
            candidates.extend(sorted((p for p in root.iterdir() if p.is_dir()), reverse=True))
    for directory in candidates:
        complete_path = directory / "RUN_COMPLETE.json"
        if not complete_path.is_file() or any(not (directory / name).is_file() for name in required):
            continue
        complete = _load_json(complete_path)
        if complete.get("status") != "PASS":
            continue
        hashes = complete.get("artifact_sha256")
        if not isinstance(hashes, dict):
            raise ContractError("Stage-0 completion manifest has no artifact hashes")
        for name in required:
            if hashes.get(name) != sha256_file(directory / name):
                raise ContractError(f"Stage-0 artifact hash failed for {name}")
        decision = _load_json(directory / stage0["decision_file"])
        audit = pd.read_csv(directory / stage0["modality_audit_file"], low_memory=False)
        if decision.get("scientific_decision") != stage0["expected_decision"]:
            raise ContractError("Stage-0 decision differs from the frozen adapter contract")
        passed_stage0 = decision.get("stage0_modalities_passed", [])
        passed_information = decision.get("information_modalities_passed", [])
        if len(passed_stage0) != int(stage0["expected_stage0_modalities_passed"]):
            raise ContractError("Unexpected number of Stage-0 data-contract passes")
        if len(passed_information) != int(stage0["expected_information_modalities_passed"]):
            raise ContractError("Unexpected number of incremental-information passes")
        return Stage0Evidence(directory, decision, audit, {name: hashes[name] for name in required})
    raise ContractError("No complete hash-verified Stage-0 artifact was found")


def production_admission_mask(config: dict[str, Any], stage0: Stage0Evidence) -> np.ndarray:
    if stage0.decision.get("scientific_decision") != config["stage0"]["expected_decision"]:
        raise ContractError("Stage-0 decision cannot be overridden")
    names = list(config["expert_names"])
    admission = config["production_default_admission"]
    mask = np.asarray([bool(admission[name]) for name in names], dtype=bool)
    for name in config["always_admitted_experts"]:
        if not mask[names.index(name)]:
            raise ContractError("A fail-safe expert was rejected")
    if mask[names.index("specialist_atp")]:
        raise ContractError("Specialist ATP cannot be admitted after the frozen Stage-0 rejection")
    return mask


def load_baseline(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    features = list(config["baseline"]["required_rule_features"])
    required = {"information_date", "decision_date", "target_date", "target_log_return", *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"Baseline lacks columns={missing}")
    frame = frame[["information_date", "decision_date", "target_date", "target_log_return", *features]].copy()
    for column in ["information_date", "decision_date", "target_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame = frame.sort_values("decision_date").reset_index(drop=True)
    if frame["decision_date"].duplicated().any():
        raise ContractError("Baseline contains duplicate decision dates")
    if not (frame["decision_date"] - frame["information_date"]).eq(pd.Timedelta(days=1)).all():
        raise ContractError("Baseline violates information d-1 -> decision d")
    if not (frame["target_date"] - frame["decision_date"]).eq(pd.Timedelta(days=1)).all():
        raise ContractError("Baseline violates decision d -> return d+1")
    if frame["target_date"].max() > pd.Timestamp(config["development_end"]):
        raise ContractError("Baseline contains post-development targets")
    numeric = ["target_log_return", *features]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ContractError("Baseline contains non-finite numeric values")
    if (frame["target_log_return"] <= np.log(1e-12)).any():
        raise ContractError("Baseline contains an impossible return")
    return frame


def load_daily_router(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    posterior = list(config["router"]["posterior_columns"])
    required = {"decision_date", "target_date", *posterior}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(
            f"Router lacks frozen daily posterior columns={missing}; intraday probabilities may not substitute"
        )
    if "model" in frame.columns:
        preferred = str(config["router"]["preferred_model"])
        frame = frame.loc[frame["model"].astype(str) == preferred].copy()
        if frame.empty:
            raise ContractError(f"Preferred Router model not found: {preferred}")
    frame = frame[["decision_date", "target_date", *posterior]].copy()
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise").dt.normalize()
    frame = frame.sort_values("target_date").reset_index(drop=True)
    if frame["target_date"].duplicated().any():
        raise ContractError("Router has duplicate target dates after model selection")
    if not (frame["target_date"] - frame["decision_date"]).eq(pd.Timedelta(days=1)).all():
        raise ContractError("Router violates prediction d-1 -> regime target d")
    if frame["target_date"].max() > pd.Timestamp(config["development_end"]):
        raise ContractError("Router contains post-development targets")
    frame[posterior] = frame[posterior].apply(pd.to_numeric, errors="raise")
    q = frame[posterior].to_numpy(dtype=float)
    if not np.isfinite(q).all() or (q < -TOL).any() or not np.allclose(q.sum(axis=1), 1.0, atol=1e-9):
        raise ContractError("Daily Router posterior is not a probability simplex")
    return frame.rename(
        columns={"decision_date": "router_information_date", "target_date": "router_target_date"}
    )


def align_baseline_and_router(
    baseline: pd.DataFrame,
    router: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    posterior = list(config["router"]["posterior_columns"])
    start = router["router_target_date"].min()
    end = router["router_target_date"].max()
    selected = baseline.loc[baseline["decision_date"].between(start, end)].copy()
    aligned = selected.merge(
        router,
        left_on="decision_date",
        right_on="router_target_date",
        how="left",
        validate="one_to_one",
    )
    valid = aligned[posterior].notna().all(axis=1)
    if valid.any() and not (
        aligned.loc[valid, "router_information_date"] == aligned.loc[valid, "information_date"]
    ).all():
        raise ContractError("Router posterior was not available by the baseline information date")
    partial = aligned[posterior].notna().any(axis=1) & ~valid
    if partial.any():
        raise ContractError("Router has partially missing posterior rows")
    if len(aligned) < int(config["minimum_replay_rows"]):
        raise ContractError("Real replay contains fewer than the frozen minimum rows")
    return aligned.reset_index(drop=True)


def derive_rule_exposures(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    rules = config["rule_experts"]
    trend = (
        (frame["momentum_30"].to_numpy(dtype=float) > 0.0)
        & (frame["momentum_90"].to_numpy(dtype=float) > 0.0)
    ).astype(float)
    realized = frame["realized_vol_30"].to_numpy(dtype=float)
    if (realized < 0.0).any():
        raise ContractError("Realized volatility may not be negative")
    volatility = np.divide(
        float(rules["volatility_target_daily"]),
        realized,
        out=np.ones_like(realized),
        where=realized > 0.0,
    )
    volatility = np.clip(volatility, 0.0, 1.0)
    drawdown_value = frame["drawdown_90"].to_numpy(dtype=float)
    mild = float(rules["drawdown_mild_threshold"])
    severe = float(rules["drawdown_severe_threshold"])
    if severe >= mild:
        raise ContractError("Severe drawdown threshold must be below mild threshold")
    drawdown = np.where(
        drawdown_value <= severe,
        float(rules["drawdown_severe_exposure"]),
        np.where(drawdown_value <= mild, float(rules["drawdown_mild_exposure"]), 1.0),
    )
    exposures = np.column_stack(
        [
            np.zeros(len(frame)),
            np.ones(len(frame)),
            trend,
            volatility,
            drawdown,
            np.zeros(len(frame)),
        ]
    )
    if not np.isfinite(exposures).all() or ((exposures < -TOL) | (exposures > 1.0 + TOL)).any():
        raise ContractError("Rule expert exposure contract failed")
    return exposures


def build_past_only_scenarios(
    full_baseline: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    config: dict[str, Any],
) -> np.ndarray:
    spec = config["risk_scenarios"]
    windows = [int(value) for value in spec["past_only_windows"]]
    samples = int(spec["quantile_samples"])
    minimum = int(spec["minimum_history"])
    if min(windows) < 2 or max(windows) > minimum or samples < 20:
        raise ContractError("Invalid risk-scenario configuration")
    positions = np.linspace(0.5 / samples, 1.0 - 0.5 / samples, samples)
    target_dates = full_baseline["target_date"].to_numpy(dtype="datetime64[ns]")
    log_returns = full_baseline["target_log_return"].to_numpy(dtype=float)
    output = np.empty((len(decision_dates), len(windows), samples), dtype=float)
    for row, decision_date in enumerate(decision_dates):
        cutoff = np.searchsorted(target_dates, np.datetime64(decision_date), side="right")
        if cutoff < minimum:
            raise ContractError(f"Insufficient past-only scenario history at {decision_date.date()}")
        history = log_returns[:cutoff]
        for model, window in enumerate(windows):
            output[row, model] = np.quantile(history[-window:], positions, method="linear")
    if not np.isfinite(output).all():
        raise ContractError("Risk scenarios contain non-finite values")
    return output


def audit_modality_alignment(
    project_root: Path,
    replay_dates: pd.DatetimeIndex,
    stage0: Stage0Evidence,
    config: dict[str, Any],
) -> pd.DataFrame:
    stage_status = {
        str(row["modality"]): str(row["status"])
        for _, row in stage0.modality_audit.iterrows()
        if "modality" in stage0.modality_audit.columns and "status" in stage0.modality_audit.columns
    }
    rows: list[dict[str, Any]] = []
    for modality, relative in config["modality_files"].items():
        path = _inside(project_root, project_root / relative)
        exists = path.is_file()
        overlap = 0
        availability_violations = 0
        digest: str | None = None
        if exists:
            frame = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {"decision_date", "available_at_utc"})
            if {"decision_date", "available_at_utc"} - set(frame.columns):
                raise ContractError(f"Modality timestamp columns missing for {modality}")
            frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
            frame["available_at_utc"] = pd.to_datetime(frame["available_at_utc"], utc=True, errors="raise")
            decision_open = frame["decision_date"].dt.tz_localize("UTC")
            availability_violations = int((frame["available_at_utc"] > decision_open).sum())
            if availability_violations:
                raise ContractError(f"Late modality values detected for {modality}")
            overlap = int(frame["decision_date"].isin(replay_dates).sum())
            digest = sha256_file(path)
        rows.append(
            {
                "modality": modality,
                "stage0_status": stage_status.get(modality, "UNKNOWN"),
                "file_exists": exists,
                "overlap_rows": overlap,
                "availability_violations": availability_violations,
                "sha256": digest or "",
                "used_for_expert_admission": False,
                "used_as_router_input": False,
            }
        )
    return pd.DataFrame(rows)


def build_real_router_inputs(project_root: Path, config: dict[str, Any]) -> tuple[RealRouterInputs, Stage0Evidence]:
    baseline_path = resolve_first_file(project_root, list(config["baseline"]["candidates"]))
    router_path = resolve_first_file(project_root, list(config["router"]["prediction_candidates"]))
    stage0 = load_latest_stage0_evidence(project_root, config)
    baseline = load_baseline(baseline_path, config)
    router = load_daily_router(router_path, config)
    aligned = align_baseline_and_router(baseline, router, config)
    posterior_columns = list(config["router"]["posterior_columns"])
    q = aligned[posterior_columns].to_numpy(dtype=float)
    exposures = derive_rule_exposures(aligned, config)
    scenarios = build_past_only_scenarios(
        baseline,
        pd.DatetimeIndex(aligned["decision_date"]),
        config,
    )
    simple_returns = np.expm1(aligned["target_log_return"].to_numpy(dtype=float))
    if (simple_returns <= -1.0).any() or not np.isfinite(simple_returns).all():
        raise ContractError("Asset simple-return conversion failed")
    modality_alignment = audit_modality_alignment(
        project_root,
        pd.DatetimeIndex(aligned["decision_date"]),
        stage0,
        config,
    )
    input_contract = {
        "baseline_path": str(baseline_path),
        "baseline_sha256": sha256_file(baseline_path),
        "router_path": str(router_path),
        "router_sha256": sha256_file(router_path),
        "stage0_artifact_directory": str(stage0.artifact_directory),
        "stage0_decision": stage0.decision["scientific_decision"],
        "rows": int(len(aligned)),
        "first_information_date": str(aligned["information_date"].min().date()),
        "first_decision_date": str(aligned["decision_date"].min().date()),
        "last_return_date": str(aligned["target_date"].max().date()),
        "valid_router_rows": int(np.isfinite(q).all(axis=1).sum()),
        "router_fallback_input_rows": int((~np.isfinite(q).all(axis=1)).sum()),
        "clock": config["router"]["execution_clock"],
        "router_source": "daily posterior only",
        "specialist_training_performed": False,
        "dqn_training_performed": False,
    }
    inputs = RealRouterInputs(
        decision_dates=pd.DatetimeIndex(aligned["decision_date"]),
        return_dates=pd.DatetimeIndex(aligned["target_date"]),
        router_posteriors=q,
        expert_exposures=exposures,
        asset_simple_returns=simple_returns,
        scenario_log_returns=scenarios,
        initial_trust=np.full((len(config["regime_names"]), len(config["expert_names"])), 1.0 / len(config["expert_names"])),
        aligned_frame=aligned,
        input_contract=input_contract,
        modality_alignment=modality_alignment,
    )
    return inputs, stage0
