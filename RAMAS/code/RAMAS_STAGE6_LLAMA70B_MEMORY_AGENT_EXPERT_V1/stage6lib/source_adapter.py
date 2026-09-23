from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SourceError(RuntimeError):
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
                raise SourceError(f"duplicate JSON key={key} in {path}")
            output[key] = value
        return output

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise SourceError(f"expected JSON object: {path}")
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


def verify_manifest(directory: Path) -> dict[str, str]:
    manifest = directory / "MANIFEST.sha256"
    if not manifest.is_file():
        raise SourceError(f"missing frozen source manifest: {manifest}")
    verified: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        path = directory / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise SourceError(f"frozen source manifest mismatch: {path}")
        verified[relative] = digest
    return verified


def resolve_corrected_run(
    project_root: Path,
    explicit: Path | None,
    config: dict[str, Any],
) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    candidates.extend(
        sorted(project_root.glob(str(config["source_artifact_relative_glob"])), reverse=True)
    )
    for candidate in candidates:
        required = [
            candidate / "results" / "corrected_pre2024" / "RUN_COMPLETE.json",
            candidate / "results" / "corrected_reused_oos" / "RUN_COMPLETE.json",
            candidate / "base_source",
        ]
        if candidate.is_dir() and all(path.exists() for path in required):
            return candidate.resolve()
    raise SourceError("no extracted corrected-clock source artifact was found")


def resolve_base_dir(corrected_run: Path) -> Path:
    matches = sorted(
        path.parent
        for path in (corrected_run / "base_source").rglob("config.json")
        if (path.parent / "src" / "risk.py").is_file()
        and (path.parent / "src" / "accounting.py").is_file()
    )
    if len(matches) != 1:
        raise SourceError(f"expected one frozen base source, found {len(matches)}")
    verify_manifest(matches[0])
    return matches[0].resolve()


def import_base(base_dir: Path):
    # Stage 6 uses the source package's top-level `src` namespace. This package
    # deliberately uses `stage6lib` to avoid module-name collision.
    if "src" in sys.modules:
        raise SourceError("top-level src module was loaded before frozen base import")
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
        raise SourceError(f"corrected source is incomplete: {period_dir}")
    hashes = complete.get("artifact_sha256", {})
    input_path = period_dir / "02_CORRECTED_INPUTS.csv"
    trace_path = period_dir / "03_TRUSTED_ROUTER_TRACE.csv"
    for path in (input_path, trace_path):
        if hashes.get(path.name) != sha256_file(path):
            raise SourceError(f"corrected source hash mismatch: {path}")
    return input_path, trace_path


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
        cutoff = int(
            np.searchsorted(
                ending_dates.values,
                decision_date.to_datetime64(),
                side="right",
            )
        )
        if cutoff < minimum:
            raise SourceError(f"insufficient scenario history at {decision_date.date()}")
        history = ending_log_returns[:cutoff]
        latest_used.append(ending_dates[cutoff - 1])
        for model, window in enumerate(windows):
            output[row, model] = np.quantile(
                history[-window:], probabilities, method="linear"
            )
    if not np.isfinite(output).all():
        raise SourceError("risk scenarios contain non-finite values")
    if any(used > decision for used, decision in zip(latest_used, decision_dates)):
        raise SourceError("risk scenario construction used a future return")
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
            raise SourceError(f"decision date missing from raw prices: {decision_date.date()}")
        index = date_to_index[decision_date]
        if index < 90:
            raise SourceError(f"insufficient causal feature history: {decision_date.date()}")
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
        raise SourceError("feature construction used a future price")
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


def risk_kwargs(base_config: dict[str, Any]) -> dict[str, float]:
    return {
        "transaction_cost_rate": float(base_config["transaction_cost_bps"]) / 10000.0,
        "cvar_alpha": float(base_config["cvar_alpha"]),
        "cvar_limit": float(base_config["cvar_limit"]),
        "ambiguity_quantile": float(base_config["ambiguity_quantile"]),
        "maximum_turnover": float(base_config["maximum_daily_turnover"]),
        "exposure_grid_step": float(base_config["exposure_grid_step"]),
    }
