#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parent
TOL = 1e-12


class Stage53Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise Stage53Error(f"Duplicate JSON key={key} in {path}")
            output[key] = value
        return output

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise Stage53Error(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.temporary")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.temporary")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.temporary")
    frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
    temporary.replace(path)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "EXPLORATORY_DEVELOPMENT_ONLY":
        raise Stage53Error("Stage 5.3 must remain exploratory and development-only")
    if config.get("expected_raw_sha256") != "b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e":
        raise Stage53Error("Raw dataset identity changed")
    controller = config.get("transparent_controller", {})
    if controller.get("mapping") != {"bear": 0.25, "bull": 0.5, "mix": 0.25}:
        raise Stage53Error("The transparent controller changed")
    if controller.get("allowed_residuals") != [-0.25, 0.0, 0.25]:
        raise Stage53Error("The residual action grid changed")
    if (
        float(controller.get("minimum_desired_exposure", -1.0)) != 0.0
        or float(controller.get("maximum_desired_exposure", -1.0)) != 0.75
    ):
        raise Stage53Error("The bounded desired-exposure range changed")
    required_tools = [
        "retrieve_mature_incidents",
        "compare_allowed_actions",
        "inspect_current_risk",
        "inspect_evidence_ledger",
    ]
    if config.get("agent", {}).get("required_tools_for_nonzero_residual") != required_tools:
        raise Stage53Error("The required tool contract changed")
    trigger = config.get("trigger", {})
    if (
        trigger.get("transition_day") is not True
        or abs(float(trigger.get("router_confidence_strictly_below", 0.0)) - 0.8) > TOL
    ):
        raise Stage53Error("The predeclared agent trigger changed")
    expected_claims = {
        "router_changed": False,
        "return_clock_changed": False,
        "risk_layer_changed": False,
        "transaction_cost_changed": False,
        "existing_experts_weighted": False,
        "monthly_trust_updates_enabled": False,
        "dqn_enabled": False,
        "hourly_data_used": False,
        "multimodal_data_used": False,
        "multi_agent_enabled": False,
        "reused_oos_opened": False,
        "pre2024_is_independent_confirmation": False,
    }
    if config.get("frozen_claims") != expected_claims:
        raise Stage53Error("The frozen scientific boundary changed")
    preflight = config.get("interface_preflight", {})
    if int(preflight.get("sample_triggered_days", 0)) != 30:
        raise Stage53Error("The interface preflight must sample thirty triggered days")
    if int(preflight.get("minimum_valid_plans", 0)) != 29:
        raise Stage53Error("The planning-interface threshold changed")
    if int(preflight.get("minimum_valid_decisions", 0)) != 29:
        raise Stage53Error("The decision-interface threshold changed")
    if abs(float(preflight.get("minimum_full_run_interface_rate", 0.0)) - 0.99) > TOL:
        raise Stage53Error("The full-run interface threshold changed")


def verify_manifest(directory: Path) -> dict[str, str]:
    manifest = directory / "MANIFEST.sha256"
    if not manifest.is_file():
        raise Stage53Error(f"Missing source manifest: {manifest}")
    verified: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        path = directory / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise Stage53Error(f"Source manifest mismatch: {path}")
        verified[relative] = digest
    return verified


def resolve_corrected_run(project_root: Path, explicit: Path | None, config: dict[str, Any]) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    candidates.extend(sorted(project_root.glob(str(config["source_artifact_relative_glob"])), reverse=True))
    for candidate in candidates:
        required = [
            candidate / "results" / "corrected_pre2024" / "RUN_COMPLETE.json",
            candidate / "base_source",
        ]
        if candidate.is_dir() and all(path.exists() for path in required):
            return candidate.resolve()
    raise Stage53Error("No extracted corrected-clock development source artifact was found")


def resolve_base_dir(corrected_run: Path) -> Path:
    matches = sorted(
        path.parent
        for path in (corrected_run / "base_source").rglob("config.json")
        if (path.parent / "src" / "risk.py").is_file()
        and (path.parent / "src" / "accounting.py").is_file()
    )
    if len(matches) != 1:
        raise Stage53Error(f"Expected one frozen base source, found {len(matches)}")
    verify_manifest(matches[0])
    return matches[0].resolve()


def import_base(base_dir: Path):
    sys.path.insert(0, str(base_dir))
    importlib.invalidate_caches()
    adapter = importlib.import_module("run_real_adapter")
    accounting = importlib.import_module("src.accounting")
    risk = importlib.import_module("src.risk")
    statistics = importlib.import_module("src.statistics")
    return adapter, accounting, risk, statistics


def verify_child_artifact(period_dir: Path) -> tuple[Path, Path]:
    complete = load_json(period_dir / "RUN_COMPLETE.json")
    if complete.get("status") != "PASS":
        raise Stage53Error(f"Corrected source is incomplete: {period_dir}")
    hashes = complete.get("artifact_sha256", {})
    input_path = period_dir / "02_CORRECTED_INPUTS.csv"
    trace_path = period_dir / "03_TRUSTED_ROUTER_TRACE.csv"
    for path in [input_path, trace_path]:
        if hashes.get(path.name) != sha256_file(path):
            raise Stage53Error(f"Corrected source hash mismatch: {path}")
    return input_path, trace_path


def resolve_raw_dataset(
    explicit: Path | None,
    corrected_run: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    audit_path = corrected_run / "results" / "corrected_pre2024" / "01_RETURN_CLOCK_AUDIT.json"
    if audit_path.is_file():
        source = load_json(audit_path).get("raw_dataset", {})
        if source.get("path"):
            candidates.append(Path(str(source["path"])).expanduser())
    candidates.extend(Path(item).expanduser() for item in config["raw_dataset_candidates"])
    seen: set[Path] = set()
    failures: list[dict[str, str]] = []
    for candidate in candidates:
        path = candidate.resolve()
        if path in seen:
            continue
        seen.add(path)
        try:
            if not path.is_file():
                raise Stage53Error("file does not exist")
            digest = sha256_file(path)
            if digest != str(config["expected_raw_sha256"]):
                raise Stage53Error(f"unexpected sha256={digest}")
            header = pd.read_csv(path, nrows=0).columns.tolist()
            date_column = "Date" if "Date" in header else "date" if "date" in header else None
            close_column = "Close" if "Close" in header else "close" if "close" in header else None
            if date_column is None or close_column is None:
                raise Stage53Error("Date/Close columns are missing")
            raw = pd.read_csv(path, usecols=[date_column, close_column], low_memory=False)
            raw[date_column] = pd.to_datetime(raw[date_column], errors="raise").dt.normalize()
            raw[close_column] = pd.to_numeric(raw[close_column], errors="raise")
            raw = raw.sort_values(date_column).reset_index(drop=True)
            if raw[date_column].duplicated().any() or (raw[close_column] <= 0.0).any():
                raise Stage53Error("raw dates or closing prices are invalid")
            frame = raw.rename(columns={date_column: "date", close_column: "close"})
            return frame, {
                "path": str(path),
                "sha256": digest,
                "rows": int(len(frame)),
                "date_min": str(frame["date"].min().date()),
                "date_max": str(frame["date"].max().date()),
            }
        except Exception as exc:
            failures.append({"path": str(path), "reason": str(exc)})
    raise Stage53Error(f"No raw dataset passed the frozen hash contract: {failures}")


def build_causal_scenarios(
    raw: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    base_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    dates = pd.DatetimeIndex(raw["date"])
    close = raw["close"].to_numpy(float)
    ending_log_returns = np.log(close[1:] / close[:-1])
    ending_dates = dates[1:]
    spec = base_config["risk_scenarios"]
    windows = [int(value) for value in spec["past_only_windows"]]
    samples = int(spec["quantile_samples"])
    minimum = int(spec["minimum_history"])
    probabilities = np.linspace(0.5 / samples, 1.0 - 0.5 / samples, samples)
    output = np.empty((len(decision_dates), len(windows), samples), dtype=float)
    latest_used: list[pd.Timestamp] = []
    for row, decision_date in enumerate(pd.DatetimeIndex(decision_dates)):
        cutoff = int(np.searchsorted(ending_dates.values, decision_date.to_datetime64(), side="right"))
        if cutoff < minimum:
            raise Stage53Error(f"Insufficient past history at {decision_date.date()}")
        history = ending_log_returns[:cutoff]
        latest_used.append(ending_dates[cutoff - 1])
        for model, window in enumerate(windows):
            output[row, model] = np.quantile(history[-window:], probabilities, method="linear")
    if not np.isfinite(output).all():
        raise Stage53Error("Risk scenarios contain non-finite values")
    if any(used > decision for used, decision in zip(latest_used, decision_dates)):
        raise Stage53Error("Risk scenarios used a future return")
    return output, {
        "past_only_windows": windows,
        "quantile_samples": samples,
        "first_latest_return_date_used": str(latest_used[0].date()),
        "last_latest_return_date_used": str(latest_used[-1].date()),
        "future_return_used": False,
    }


def build_causal_market_features(
    raw: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    router_probabilities: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = pd.DatetimeIndex(raw["date"])
    close = raw["close"].to_numpy(float)
    date_to_index = {value: index for index, value in enumerate(dates)}
    rows: list[dict[str, Any]] = []
    latest_used: list[pd.Timestamp] = []
    previous_q: np.ndarray | None = None
    for row_index, decision_date in enumerate(pd.DatetimeIndex(decision_dates)):
        if decision_date not in date_to_index:
            raise Stage53Error(f"Decision date missing from raw prices: {decision_date.date()}")
        index = date_to_index[decision_date]
        if index < 90:
            raise Stage53Error(f"Insufficient causal feature history: {decision_date.date()}")
        history = close[: index + 1]
        log_daily = np.diff(np.log(history))
        q = np.asarray(router_probabilities[row_index], dtype=float)
        positive_q = q[q > 0.0]
        entropy = float(-np.sum(positive_q * np.log(positive_q)) / np.log(3.0))
        transition_l1 = 0.0 if previous_q is None else float(np.sum(np.abs(q - previous_q)))
        trailing_peak = float(np.max(history[-90:]))
        rows.append(
            {
                "return_1": float(history[-1] / history[-2] - 1.0),
                "return_7": float(history[-1] / history[-8] - 1.0),
                "return_30": float(history[-1] / history[-31] - 1.0),
                "return_90": float(history[-1] / history[-91] - 1.0),
                "realized_vol_30": float(np.std(log_daily[-30:], ddof=1) * np.sqrt(365.25)),
                "drawdown_90": float(history[-1] / trailing_peak - 1.0),
                "router_entropy": entropy,
                "router_confidence": float(np.max(q)),
                "router_transition_l1": transition_l1,
            }
        )
        latest_used.append(dates[index])
        previous_q = q
    if any(used > decision for used, decision in zip(latest_used, decision_dates)):
        raise Stage53Error("Feature builder used a future price")
    return pd.DataFrame(rows), {
        "latest_price_dates_equal_decision_dates": bool(
            all(used == decision for used, decision in zip(latest_used, decision_dates))
        ),
        "future_price_used": False,
        "feature_columns": list(rows[0]),
    }


@dataclass(frozen=True)
class PeriodInputs:
    name: str
    label: str
    frame: pd.DataFrame
    current_trace: pd.DataFrame
    decision_dates: pd.DatetimeIndex
    return_dates: pd.DatetimeIndex
    router_probabilities: np.ndarray
    asset_returns: np.ndarray
    scenarios: np.ndarray
    features: pd.DataFrame
    source_audit: dict[str, Any]


def load_period_inputs(
    period_dir: Path,
    period_name: str,
    spec: dict[str, Any],
    raw: pd.DataFrame,
    base_config: dict[str, Any],
) -> PeriodInputs:
    input_path, trace_path = verify_child_artifact(period_dir)
    frame = pd.read_csv(input_path, low_memory=False)
    trace = pd.read_csv(trace_path, low_memory=False)
    required = {
        "decision_date",
        "return_date",
        "prob_bear",
        "prob_bull",
        "prob_mix",
        "canonical_asset_simple_return",
        "exposure_volatility_target",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise Stage53Error(f"Corrected inputs lack columns={missing}")
    frame["decision_date"] = pd.to_datetime(frame["decision_date"], errors="raise").dt.normalize()
    frame["return_date"] = pd.to_datetime(frame["return_date"], errors="raise").dt.normalize()
    if len(frame) != int(spec["rows"]):
        raise Stage53Error(f"{period_name} row count changed")
    if str(frame["decision_date"].min().date()) != str(spec["first_decision_date"]):
        raise Stage53Error(f"{period_name} first decision date changed")
    if str(frame["return_date"].max().date()) != str(spec["last_return_date"]):
        raise Stage53Error(f"{period_name} last return date changed")
    decisions = pd.DatetimeIndex(frame["decision_date"])
    returns = pd.DatetimeIndex(frame["return_date"])
    if not np.all((returns - decisions).days == 1):
        raise Stage53Error(f"{period_name} return clock is not next-day")
    q = frame[["prob_bear", "prob_bull", "prob_mix"]].apply(pd.to_numeric, errors="raise").to_numpy(float)
    if (q < -TOL).any() or not np.allclose(q.sum(axis=1), 1.0, atol=1e-9, rtol=0.0):
        raise Stage53Error(f"{period_name} router probabilities are invalid")
    asset_returns = pd.to_numeric(frame["canonical_asset_simple_return"], errors="raise").to_numpy(float)
    if not np.isfinite(asset_returns).all() or (asset_returns <= -1.0).any():
        raise Stage53Error(f"{period_name} returns are invalid")
    if len(trace) != len(frame):
        raise Stage53Error(f"{period_name} current trace does not align")
    if not np.allclose(trace["asset_simple_return"].to_numpy(float), asset_returns, atol=1e-15, rtol=0.0):
        raise Stage53Error(f"{period_name} trace uses another return stream")
    scenarios, risk_audit = build_causal_scenarios(raw, decisions, base_config)
    features, feature_audit = build_causal_market_features(raw, decisions, q)
    hard_regimes = np.argmax(q, axis=1)
    features["transition_day"] = np.r_[False, hard_regimes[1:] != hard_regimes[:-1]]
    feature_audit["transition_day_derived_from_frozen_router"] = True
    return PeriodInputs(
        period_name,
        str(spec["scientific_label"]),
        frame,
        trace,
        decisions,
        returns,
        q,
        asset_returns,
        scenarios,
        features,
        {
            "period": period_name,
            "corrected_inputs_path": str(input_path),
            "corrected_inputs_sha256": sha256_file(input_path),
            "current_trace_path": str(trace_path),
            "current_trace_sha256": sha256_file(trace_path),
            "rows": int(len(frame)),
            "first_decision_date": str(decisions.min().date()),
            "last_return_date": str(returns.max().date()),
            "risk_scenarios": risk_audit,
            "features": feature_audit,
        },
    )


def normalized_state_vector(state: dict[str, Any]) -> np.ndarray:
    def clip_scale(value: float, low: float, high: float) -> float:
        return float((np.clip(value, low, high) - low) / (high - low))

    return np.asarray(
        [
            state["prob_bear"],
            state["prob_bull"],
            state["prob_mix"],
            state["router_entropy"],
            state["router_confidence"],
            clip_scale(state["router_transition_l1"], 0.0, 2.0),
            clip_scale(state["return_7"], -0.5, 0.5),
            clip_scale(state["return_30"], -0.8, 0.8),
            clip_scale(state["return_90"], -1.0, 2.0),
            clip_scale(state["realized_vol_30"], 0.0, 2.0),
            clip_scale(state["drawdown_90"], -1.0, 0.0),
            state["pretrade_exposure"],
        ],
        dtype=float,
    )


