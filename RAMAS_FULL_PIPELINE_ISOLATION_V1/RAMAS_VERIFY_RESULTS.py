#!/usr/bin/env python3
"""Independently reconstruct and verify a RAMAS isolation run.

This verifier does not import the experiment runner or Stage-6.4 metric code.
It recomputes accounting, holdings, performance, paired contrasts, circular
block intervals, token/call totals, state-match transmission, and seed
distinctness directly from durable ledgers and call journals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TOL = 1e-10
PER_YEAR = 365.25
REQUIRED = {
    "decision_date", "return_date", "hard_regime", "asset_simple_return",
    "pretrade_exposure", "exposure", "net_return_after_trading_costs",
    "turnover", "cost_fraction", "wealth_before", "wealth_after", "action",
    "beta", "core_desired_exposure", "desired_exposure", "request_sha256",
}


class VerificationError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise VerificationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def drifted(exposure: np.ndarray | float, asset_return: np.ndarray | float):
    denominator = 1.0 + np.asarray(exposure) * np.asarray(asset_return)
    if np.any(denominator <= 0):
        raise VerificationError("Nonpositive holdings-drift denominator")
    return np.asarray(exposure) * (1.0 + np.asarray(asset_return)) / denominator


def exact_tail_loss(values: np.ndarray, probability: float = 0.05) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    mass = len(ordered) * probability
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = float(ordered[:whole].sum())
    if fraction:
        total += fraction * float(ordered[whole])
    return -total / mass


def metrics(frame: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    part = frame.loc[(frame.return_date >= pd.Timestamp(start)) & (frame.return_date <= pd.Timestamp(end))].copy()
    if part.empty:
        raise VerificationError("Metric interval is empty")
    r = part.net_return_after_trading_costs.to_numpy(float)
    logs = np.log1p(r)
    wealth = np.exp(logs.cumsum())
    path = np.r_[1.0, wealth]
    drawdown = 1.0 - path / np.maximum.accumulate(path)
    mean = float(r.mean())
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    return {
        "observations": len(part),
        "first_return_date": part.return_date.min().date().isoformat(),
        "last_return_date": part.return_date.max().date().isoformat(),
        "net_return_pct": float(np.expm1(logs.sum()) * 100.0),
        "mean_daily_net_log_return_bps": float(logs.mean() * 10000.0),
        "annualized_volatility_pct": float(sd * math.sqrt(PER_YEAR) * 100.0),
        "sharpe_zero_rf": None if not math.isfinite(sd) or sd <= 1e-15 else float(mean / sd * math.sqrt(PER_YEAR)),
        "sortino_zero_target": None if downside <= 1e-15 else float(mean / downside * math.sqrt(PER_YEAR)),
        "max_drawdown_pct": float(drawdown.max() * 100.0),
        "es95_daily_loss_pct": float(exact_tail_loss(r) * 100.0),
        "mean_btc_exposure_pct": float(part.exposure.mean() * 100.0),
        "total_turnover": float(part.turnover.sum()),
        "mean_daily_turnover_pct": float(part.turnover.mean() * 100.0),
        "modeled_cost_fraction_sum": float(part.cost_fraction.sum()),
        "input_tokens": int(part.get("input_token_count", pd.Series(dtype=float)).fillna(0).sum()),
        "output_tokens": int(part.get("output_token_count", pd.Series(dtype=float)).fillna(0).sum()),
    }


def verify_ledger(frame: pd.DataFrame, cost: float, expected_rows: int, label: str) -> dict[str, float]:
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise VerificationError(f"{label}: missing columns {missing}")
    if len(frame) != expected_rows:
        raise VerificationError(f"{label}: expected {expected_rows} rows, found {len(frame)}")
    df = frame.copy()
    df["decision_date"] = pd.to_datetime(df.decision_date, errors="raise").dt.normalize()
    df["return_date"] = pd.to_datetime(df.return_date, errors="raise").dt.normalize()
    df = df.sort_values(["return_date", "decision_date"]).reset_index(drop=True)
    if df.return_date.duplicated().any() or not (df.decision_date < df.return_date).all():
        raise VerificationError(f"{label}: duplicate/invalid dates")
    if len(df) > 1 and not (df.return_date.diff().dropna() == pd.Timedelta(days=1)).all():
        raise VerificationError(f"{label}: return dates are not consecutive")
    numeric = [
        "asset_simple_return", "pretrade_exposure", "exposure",
        "net_return_after_trading_costs", "turnover", "cost_fraction",
        "wealth_before", "wealth_after", "desired_exposure",
    ]
    values = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(float)).all():
        raise VerificationError(f"{label}: nonfinite accounting values")
    if not df.exposure.between(0, 1).all() or not df.pretrade_exposure.between(0, 1).all():
        raise VerificationError(f"{label}: exposure outside [0,1]")
    expected_turnover = np.abs(df.exposure.to_numpy(float) - df.pretrade_exposure.to_numpy(float))
    expected_cost = cost * expected_turnover
    expected_net = (1.0 - expected_cost) * (
        1.0 + df.exposure.to_numpy(float) * df.asset_simple_return.to_numpy(float)
    ) - 1.0
    expected_wealth_before = np.r_[1.0, np.cumprod(1.0 + expected_net)[:-1]]
    expected_wealth_after = np.cumprod(1.0 + expected_net)
    expected_pretrade = np.r_[0.0, drifted(
        df.exposure.to_numpy(float)[:-1], df.asset_simple_return.to_numpy(float)[:-1]
    )]
    errors = {
        "turnover": float(np.max(np.abs(df.turnover - expected_turnover))),
        "cost": float(np.max(np.abs(df.cost_fraction - expected_cost))),
        "net_return": float(np.max(np.abs(df.net_return_after_trading_costs - expected_net))),
        "wealth_before": float(np.max(np.abs(df.wealth_before - expected_wealth_before))),
        "wealth_after": float(np.max(np.abs(df.wealth_after - expected_wealth_after))),
        "pretrade_clock": float(np.max(np.abs(df.pretrade_exposure - expected_pretrade))),
    }
    if max(errors.values()) > TOL:
        raise VerificationError(f"{label}: accounting reconstruction failed {errors}")
    if str(df.risk_mode.iloc[0]) == "none":
        direct_error = float(np.max(np.abs(df.exposure - np.clip(df.desired_exposure, 0, 1))))
        errors["controller_off_direct_exposure"] = direct_error
        if direct_error > TOL:
            raise VerificationError(f"{label}: controller-OFF exposure is not the bounded desired target")
    return errors


def arm_directories(run: Path) -> list[Path]:
    paths = sorted({path.parent for path in run.rglob("DAILY_LEDGER.csv")})
    paths = [path for path in paths if (path / "RUN_COMPLETE.json").is_file()]
    if not paths:
        raise VerificationError(f"No completed arm ledgers found under {run}")
    return paths


def load_arm(directory: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete = load_json(directory / "RUN_COMPLETE.json")
    for relative, expected in complete.get("artifact_sha256", {}).items():
        path = directory / relative
        if not path.is_file() or sha256(path) != expected:
            raise VerificationError(f"Arm artifact differs: {path}")
    identity = complete["identity"]
    spec = complete["spec"]
    frame = pd.read_csv(directory / "DAILY_LEDGER.csv", low_memory=False)
    model = str(identity.get("model", "NUMERICAL_ONLY"))
    seed = identity.get("sampling_seed")
    frame["model"] = model
    frame["sampling_seed"] = seed if seed is not None else np.nan
    frame["logical_arm"] = spec["name"]
    frame["factor_trust"] = spec["trust"] if spec["advisor"] == "llama" else "none"
    frame["factor_memory"] = spec.get("factor_memory", "none")
    frame["factor_controller"] = spec.get("factor_controller", "on" if spec["risk"] == "standard" else "off")
    frame["arm_uid"] = (
        spec["name"] if model == "NUMERICAL_ONLY"
        else f"{model}__seed_{seed}__{spec['name']}"
    )
    return frame, complete


def call_records(directory: Path, expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calls = sorted((directory / "calls").glob("*.json"))
    if not calls:
        return [], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_seconds": 0.0}
    if len(calls) != expected_rows:
        raise VerificationError(f"Call count differs in {directory}: {len(calls)}")
    records = []
    input_tokens = output_tokens = 0
    latency = 0.0
    for index, path in enumerate(calls):
        item = load_json(path)
        original = dict(item)
        checksum = original.pop("record_sha256", None)
        if checksum != digest(original):
            raise VerificationError(f"Call record checksum failed: {path}")
        if item.get("request_sha256") != digest(item.get("request")):
            raise VerificationError(f"Call request checksum failed: {path}")
        raw = item.get("raw_response")
        if not isinstance(raw, str) or item.get("response_sha256") != digest(raw):
            raise VerificationError(f"Call response checksum failed: {path}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"Invalid response JSON in {path}") from exc
        provider = item.get("provider_response", {})
        input_tokens += int(provider.get("prompt_eval_count", 0) or 0)
        output_tokens += int(provider.get("eval_count", 0) or 0)
        latency += float(item.get("latency_seconds", 0.0))
        request = item["request"]
        records.append({
            "index": index,
            "request_sha256": item["request_sha256"],
            "state_sha256": digest(request.get("state")),
            "preview_sha256": digest(request.get("safe_exposure_previews")),
            "memory_sha256": digest(request.get("memory")),
            "response_sha256": item["response_sha256"],
            "action": parsed.get("action"),
            "confidence": parsed.get("confidence"),
            "reason_codes": sorted(parsed.get("reason_codes", [])),
            "cited_memory_ids": sorted(parsed.get("cited_memory_ids", [])),
        })
    return records, {
        "calls": len(calls), "input_tokens": input_tokens, "output_tokens": output_tokens,
        "latency_seconds": latency, "mean_latency_seconds": latency / len(calls),
    }


def projection_only(frame: pd.DataFrame, cost: float, arm_uid: str) -> pd.DataFrame:
    df = frame.sort_values("return_date").copy()
    p = 0.0
    wealth = 1.0
    rows = []
    for row in df.itertuples(index=False):
        exposure = float(np.clip(float(row.desired_exposure), 0.0, 1.0))
        asset = float(row.asset_simple_return)
        turnover = abs(exposure - p)
        cost_fraction = cost * turnover
        net = (1.0 - cost_fraction) * (1.0 + exposure * asset) - 1.0
        before = wealth
        wealth *= 1.0 + net
        rows.append({
            "arm_uid": arm_uid + "__projection_only_off",
            "source_controller_on_arm": arm_uid,
            "model": row.model, "sampling_seed": row.sampling_seed,
            "factor_trust": row.factor_trust, "factor_memory": row.factor_memory,
            "factor_controller": "projection_only_off",
            "decision_date": row.decision_date, "return_date": row.return_date,
            "hard_regime": row.hard_regime, "asset_simple_return": asset,
            "pretrade_exposure": p, "exposure": exposure,
            "desired_exposure": float(row.desired_exposure), "turnover": turnover,
            "cost_fraction": cost_fraction, "net_return_after_trading_costs": net,
            "wealth_before": before, "wealth_after": wealth,
        })
        p = float(drifted(exposure, asset))
    return pd.DataFrame(rows)


def bootstrap(delta: np.ndarray, block: int, resamples: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(delta, dtype=float)
    n = len(values)
    if n == 0 or block <= 0 or resamples <= 0:
        raise VerificationError("Invalid bootstrap input")
    rng = np.random.default_rng(seed)
    blocks = math.ceil(n / block)
    offsets = np.arange(block)
    estimates = np.empty(resamples)
    for first in range(0, resamples, 128):
        count = min(128, resamples - first)
        starts = rng.integers(0, n, size=(count, blocks))
        indices = ((starts[:, :, None] + offsets) % n).reshape(count, -1)[:, :n]
        estimates[first:first + count] = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def aligned(left: pd.DataFrame, right: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    cols = ["return_date", "net_return_after_trading_costs"]
    a = left.loc[(left.return_date >= pd.Timestamp(start)) & (left.return_date <= pd.Timestamp(end)), cols]
    b = right.loc[(right.return_date >= pd.Timestamp(start)) & (right.return_date <= pd.Timestamp(end)), cols]
    merged = a.merge(b, on="return_date", how="inner", validate="one_to_one", suffixes=("_a", "_b"))
    if len(merged) != len(a) or len(merged) != len(b):
        raise VerificationError("Paired comparison dates are not exactly matched")
    return merged


def metric_delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "net_return_pct", "mean_daily_net_log_return_bps", "annualized_volatility_pct",
        "sharpe_zero_rf", "sortino_zero_target", "max_drawdown_pct",
        "es95_daily_loss_pct", "mean_btc_exposure_pct", "total_turnover",
        "mean_daily_turnover_pct", "modeled_cost_fraction_sum",
    ]
    result = {}
    for key in keys:
        left, right = a.get(key), b.get(key)
        result["delta_" + key] = None if left is None or right is None else float(left - right)
    return result


def contrast_rows(frames: dict[tuple, pd.DataFrame], metric_map: dict[tuple, dict[str, Any]],
                  config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    boots = []
    seed0 = int(config["bootstrap"]["seed"])
    block = int(config["bootstrap"]["block_days"])
    reps = int(config["bootstrap"]["resamples"])

    def add(kind: str, key_a: tuple, key_b: tuple, label: str, bootstrap_family: int):
        if key_a not in frames or key_b not in frames:
            return
        ma, mb = metric_map[key_a], metric_map[key_b]
        record = {"contrast": label, "kind": kind, "arm_a": key_a[-1], "arm_b": key_b[-1]}
        record.update({"model": key_a[0], "sampling_seed": key_a[1]})
        record.update(metric_delta(ma, mb))
        rows.append(record)
        pair = aligned(frames[key_a], frames[key_b], config["evaluation_start"], config["evaluation_end"])
        delta = (
            np.log1p(pair.net_return_after_trading_costs_a.to_numpy(float))
            - np.log1p(pair.net_return_after_trading_costs_b.to_numpy(float))
        ) * 10000.0
        point, low, high = bootstrap(delta, block, reps, seed0 + bootstrap_family)
        boots.append({
            "contrast": label, "kind": kind, "model": key_a[0],
            "sampling_seed": key_a[1], "paired_days": len(delta),
            "mean_daily_net_log_difference_bps": point,
            "ci95_lower_bps": low, "ci95_upper_bps": high,
            "block_days": block, "resamples": reps, "bootstrap_seed": seed0 + bootstrap_family,
            "same_time_indices_across_sampling_seeds": True,
            "sampling_seeds_resampled": False,
        })

    numerical = {
        controller: ("NUMERICAL_ONLY", None, "none", "none", controller)
        for controller in ("on", "off")
    }
    cells = sorted({(key[0], key[1]) for key in frames if key[0] != "NUMERICAL_ONLY"})
    for model, seed in cells:
        for trust in ("adaptive", "fixed"):
            for controller in ("on", "off"):
                a = (model, seed, trust, "memory", controller)
                b = (model, seed, trust, "no_memory", controller)
                add("memory", a, b, f"{model}|{seed}|{trust}|{controller}|memory-minus-no-memory", 10 if controller == "on" else 11)
            for memory in ("memory", "no_memory"):
                a = (model, seed, trust, memory, "on")
                b = (model, seed, trust, memory, "off")
                add("controller_total_system", a, b, f"{model}|{seed}|{trust}|{memory}|controller-on-minus-off", 20)
        for memory in ("memory", "no_memory"):
            for controller in ("on", "off"):
                a = (model, seed, "adaptive", memory, controller)
                b = (model, seed, "fixed", memory, controller)
                add("authority", a, b, f"{model}|{seed}|{memory}|{controller}|adaptive-minus-fixed", 30)
        for trust in ("adaptive", "fixed"):
            for memory in ("memory", "no_memory"):
                for controller in ("on", "off"):
                    a = (model, seed, trust, memory, controller)
                    b = numerical[controller]
                    add("advisor_vs_numerical", a, b, f"{model}|{seed}|{trust}|{memory}|{controller}|llm-minus-numerical", 40)
            # Difference-in-differences is computed from daily log returns.
            on = frames[(model, seed, trust, "no_memory", "on")]
            off = frames[(model, seed, trust, "no_memory", "off")]
            non = frames[numerical["on"]]
            noff = frames[numerical["off"]]
            first = aligned(on, off, config["evaluation_start"], config["evaluation_end"])
            second = aligned(non, noff, config["evaluation_start"], config["evaluation_end"])
            if not np.array_equal(first.return_date.to_numpy(), second.return_date.to_numpy()):
                raise VerificationError("Difference-in-differences dates differ")
            did = (
                np.log1p(first.net_return_after_trading_costs_a) - np.log1p(first.net_return_after_trading_costs_b)
                - np.log1p(second.net_return_after_trading_costs_a) + np.log1p(second.net_return_after_trading_costs_b)
            ).to_numpy(float) * 10000.0
            point, low, high = bootstrap(did, block, reps, seed0 + 50)
            boots.append({
                "contrast": f"{model}|{seed}|{trust}|no_memory|controller-by-advisor-DiD",
                "kind": "controller_by_advisor_difference_in_differences",
                "model": model, "sampling_seed": seed, "paired_days": len(did),
                "mean_daily_net_log_difference_bps": point, "ci95_lower_bps": low,
                "ci95_upper_bps": high, "block_days": block, "resamples": reps,
                "bootstrap_seed": seed0 + 50, "same_time_indices_across_sampling_seeds": True,
                "sampling_seeds_resampled": False,
            })
    return pd.DataFrame(rows), pd.DataFrame(boots)


def pair_transmission(call_map: dict[tuple, list[dict[str, Any]]],
                      frame_map: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    cells = sorted({(key[0], key[1]) for key in call_map})
    for model, seed in cells:
        if model == "NUMERICAL_ONLY":
            continue
        for trust in ("adaptive", "fixed"):
            for controller in ("on", "off"):
                a = (model, seed, trust, "memory", controller)
                b = (model, seed, trust, "no_memory", controller)
                if a not in call_map or b not in call_map:
                    continue
                ca, cb = call_map[a], call_map[b]
                fa = frame_map[a].sort_values("return_date").reset_index(drop=True)
                fb = frame_map[b].sort_values("return_date").reset_index(drop=True)
                if len(ca) != len(cb) or len(ca) != len(fa):
                    raise VerificationError("Transmission call/ledger lengths differ")
                state_match = np.array([x["state_sha256"] == y["state_sha256"] for x, y in zip(ca, cb)])
                action_diff = np.array([x["action"] != y["action"] for x, y in zip(ca, cb)])
                reason_diff = np.array([x["reason_codes"] != y["reason_codes"] for x, y in zip(ca, cb)])
                citation_diff = np.array([x["cited_memory_ids"] != y["cited_memory_ids"] for x, y in zip(ca, cb)])
                desired_diff = np.abs(fa.desired_exposure.to_numpy(float) - fb.desired_exposure.to_numpy(float)) > 1e-12
                exposure_diff = np.abs(fa.exposure.to_numpy(float) - fb.exposure.to_numpy(float)) > 1e-12
                rows.append({
                    "model": model, "sampling_seed": seed, "trust": trust, "controller": controller,
                    "comparison": "memory-minus-no-memory", "dates": len(ca),
                    "state_matches": int(state_match.sum()),
                    "state_match_rate": float(state_match.mean()),
                    "action_different_all_dates": int(action_diff.sum()),
                    "action_different_state_matched_dates": int((action_diff & state_match).sum()),
                    "reason_codes_different_all_dates": int(reason_diff.sum()),
                    "reason_codes_different_state_matched_dates": int((reason_diff & state_match).sum()),
                    "citations_different_all_dates": int(citation_diff.sum()),
                    "citations_different_state_matched_dates": int((citation_diff & state_match).sum()),
                    "desired_exposure_different_all_dates": int(desired_diff.sum()),
                    "desired_exposure_different_state_matched_dates": int((desired_diff & state_match).sum()),
                    "executed_exposure_different_all_dates": int(exposure_diff.sum()),
                    "executed_exposure_different_state_matched_dates": int((exposure_diff & state_match).sum()),
                })
    return pd.DataFrame(rows)


def seed_distinctness(call_map: dict[tuple, list[dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    groups = defaultdict(dict)
    for key, calls in call_map.items():
        model, seed, trust, memory, controller = key
        if model == "NUMERICAL_ONLY":
            continue
        groups[(model, trust, memory, controller)][seed] = calls
    for (model, trust, memory, controller), by_seed in sorted(groups.items()):
        seeds = sorted(by_seed)
        for i, left_seed in enumerate(seeds):
            for right_seed in seeds[i + 1:]:
                left, right = by_seed[left_seed], by_seed[right_seed]
                if len(left) != len(right):
                    raise VerificationError("Seed call counts differ")
                request_equal = all(a["request_sha256"] == b["request_sha256"] for a, b in zip(left, right))
                response_equal = all(a["response_sha256"] == b["response_sha256"] for a, b in zip(left, right))
                action_equal = all(a["action"] == b["action"] for a, b in zip(left, right))
                rows.append({
                    "model": model, "trust": trust, "memory": memory, "controller": controller,
                    "seed_a": left_seed, "seed_b": right_seed, "calls": len(left),
                    "all_requests_exactly_equal": request_equal,
                    "all_responses_exactly_equal": response_equal,
                    "all_actions_exactly_equal": action_equal,
                    "independent_realization_evidence": not response_equal,
                })
    return pd.DataFrame(rows)


def aggregate_seed_metrics(metric_frame: pd.DataFrame) -> pd.DataFrame:
    subset = metric_frame.loc[metric_frame.model != "NUMERICAL_ONLY"].copy()
    numeric = [
        "net_return_pct", "mean_daily_net_log_return_bps", "annualized_volatility_pct",
        "sharpe_zero_rf", "sortino_zero_target", "max_drawdown_pct",
        "es95_daily_loss_pct", "mean_btc_exposure_pct", "total_turnover",
    ]
    rows = []
    for keys, part in subset.groupby(["model", "logical_arm"], sort=True):
        record = {"model": keys[0], "logical_arm": keys[1], "sampling_seed_count": part.sampling_seed.nunique()}
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            record[column + "_mean"] = float(values.mean()) if len(values) else None
            record[column + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) else None
        rows.append(record)
    return pd.DataFrame(rows)


def artifact_map(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"RUN_COMPLETE.json"} and not path.name.endswith(".pending")
    }


def verify(args: argparse.Namespace) -> None:
    run = args.run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_complete = load_json(run / "RUN_COMPLETE.json")
    if source_complete.get("status") != "COMPLETE":
        raise VerificationError("Source experiment is not complete")
    for relative, expected in source_complete.get("artifact_sha256", {}).items():
        path = (run / relative).resolve()
        if run not in path.parents or not path.is_file() or sha256(path) != expected:
            raise VerificationError(f"Top-level experiment artifact differs: {path}")
    config = load_json(args.config.resolve())
    cost = float(config["cost_rate"])
    expected_rows = int(config["expected_full_rows"])
    eval_rows = int(config["expected_evaluation_rows"])
    frames: dict[tuple, pd.DataFrame] = {}
    metric_map: dict[tuple, dict[str, Any]] = {}
    call_map: dict[tuple, list[dict[str, Any]]] = {}
    metric_rows = []
    integrity = []
    costs = []
    projection_frames = []

    for directory in arm_directories(run):
        frame, complete = load_arm(directory)
        spec = complete["spec"]
        model = str(frame.model.iloc[0])
        seed = None if model == "NUMERICAL_ONLY" else int(frame.sampling_seed.iloc[0])
        trust = "none" if model == "NUMERICAL_ONLY" else str(spec["trust"])
        memory = "none" if model == "NUMERICAL_ONLY" else str(spec["factor_memory"])
        controller = str(spec["factor_controller"])
        key = (model, seed, trust, memory, controller)
        if key in frames:
            raise VerificationError(f"Duplicate experimental cell: {key}")
        errors = verify_ledger(frame, cost, expected_rows, str(frame.arm_uid.iloc[0]))
        frame["decision_date"] = pd.to_datetime(frame.decision_date).dt.normalize()
        frame["return_date"] = pd.to_datetime(frame.return_date).dt.normalize()
        values = metrics(frame, config["evaluation_start"], config["evaluation_end"])
        if values["observations"] != eval_rows:
            raise VerificationError(f"Evaluation row count differs for {key}")
        values.update({
            "model": model, "sampling_seed": seed, "logical_arm": spec["name"],
            "factor_trust": trust, "factor_memory": memory, "factor_controller": controller,
        })
        frames[key] = frame
        metric_map[key] = values
        metric_rows.append(values)
        integrity.append({"arm": frame.arm_uid.iloc[0], **errors})
        calls, call_cost = call_records(directory, expected_rows)
        call_map[key] = calls
        costs.append({
            "model": model, "sampling_seed": seed, "logical_arm": spec["name"],
            **call_cost,
        })
        if controller == "on":
            projection_frames.append(projection_only(frame, cost, str(frame.arm_uid.iloc[0])))

    metric_frame = pd.DataFrame(metric_rows)
    integrity_frame = pd.DataFrame(integrity)
    cost_frame = pd.DataFrame(costs)
    projection = pd.concat(projection_frames, ignore_index=True) if projection_frames else pd.DataFrame()
    contrast_frame, bootstrap_frame = contrast_rows(frames, metric_map, config)
    transmission = pair_transmission(call_map, frames)
    distinctness = seed_distinctness(call_map)
    aggregate = aggregate_seed_metrics(metric_frame)

    atomic_csv(metric_frame, output / "01_VERIFIED_METRICS.csv")
    atomic_csv(aggregate, output / "02_SEED_AGGREGATE_METRICS.csv")
    atomic_csv(contrast_frame, output / "03_PAIRED_CONTRASTS.csv")
    atomic_csv(bootstrap_frame, output / "04_PAIRED_BLOCK_BOOTSTRAP.csv")
    atomic_csv(transmission, output / "05_TRANSMISSION.csv")
    atomic_csv(cost_frame, output / "06_CALL_COSTS.csv")
    atomic_csv(distinctness, output / "07_SEED_DISTINCTNESS.csv")
    atomic_csv(integrity_frame, output / "08_ACCOUNTING_INTEGRITY.csv")
    if not projection.empty:
        atomic_csv(projection, output / "09_PROJECTION_ONLY_DAILY.csv")

    report = {
        "status": "PASS",
        "verified_at": now(),
        "source_run": str(run),
        "source_run_complete_sha256": sha256(run / "RUN_COMPLETE.json"),
        "configuration": str(args.config.resolve()),
        "arms_verified": len(frames),
        "full_rows_per_arm": expected_rows,
        "evaluation_rows_per_arm": eval_rows,
        "maximum_accounting_error": float(integrity_frame.drop(columns=["arm"]).max().max()),
        "model_call_records_verified": int(cost_frame.calls.sum()),
        "retrieval_occurrences": int(sum(
            int(frame.get("retrieved_memory_count", pd.Series(dtype=float)).fillna(0).gt(0).sum())
            for frame in frames.values()
        )),
        "bootstrap_contract": {
            "resamples": int(config["bootstrap"]["resamples"]),
            "block_days": int(config["bootstrap"]["block_days"]),
            "base_seed": int(config["bootstrap"]["seed"]),
            "same_time_indices_across_sampling_seeds": True,
            "sampling_seeds_resampled": False,
            "scope": "temporal uncertainty conditional on frozen serving realizations",
        },
        "interpretation": {
            "closed_loop_pairs": "total-system effects because later states may diverge",
            "state_matched_transmission": "reported separately in 05_TRANSMISSION.csv",
            "projection_only": "controller-OFF direct-target replay of each controller-ON advice sequence through full burn-in",
            "seed_sd": "descriptive serving-seed sensitivity, not population inference",
            "scientific_scope": config["scientific_scope"],
        },
        "claim_boundaries": config["claim_boundaries"],
    }
    atomic_json(output / "VERIFICATION_REPORT.json", report)
    atomic_json(output / "RUN_COMPLETE.json", {
        "status": "COMPLETE", "verified_at": now(),
        "source_run_complete_sha256": report["source_run_complete_sha256"],
        "artifact_sha256": artifact_map(output),
    })
    print(json.dumps({
        "status": "PASS", "output": str(output), "arms_verified": len(frames),
        "model_call_records_verified": report["model_call_records_verified"],
        "maximum_accounting_error": report["maximum_accounting_error"],
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
