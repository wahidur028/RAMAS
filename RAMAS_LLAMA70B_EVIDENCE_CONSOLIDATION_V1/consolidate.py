#!/usr/bin/env python3
"""Consolidate every completed llama3.3:70b RAMAS arm into one reproducible report.

Reads only. Normalizes three archived file formats into one long-format table,
recomputes all metrics with the frozen Stage 6.4 metrics module, bootstraps each
memory contrast, and averages across runs ONLY where the configuration matches
and the trajectories are not republished duplicates.

    /home/infonet/anaconda3/envs/wahid_test/bin/python consolidate.py --output <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parent
STAGE64 = Path("/home/infonet/wahid/leader_router_fresh/RAMAS_STAGE6_4_COMPONENT_SUITE_V1")
sys.path.insert(0, str(STAGE64))
from suite64.metrics import (  # noqa: E402
    PER_YEAR, expected_shortfall_loss, paired_block_comparison, prepare_daily_ledger,
)

PINNED_RUNTIME = {"python": "3.11.9", "numpy": "1.26.4", "pandas": "2.2.3"}
LEDGER_COLUMNS = ["arm", "decision_date", "return_date", "hard_regime", "asset_simple_return",
                  "pretrade_exposure", "exposure", "net_return_after_trading_costs",
                  "turnover", "cost_fraction"]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_runtime() -> dict:
    actual = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    bad = {k: (v, actual[k]) for k, v in PINNED_RUNTIME.items() if actual[k] != v}
    if bad:
        raise RuntimeError(f"Runtime differs from the pinned Stage 6.x environment: {bad}. "
                           "Use /home/infonet/anaconda3/envs/wahid_test/bin/python")
    return actual


def drifted(exposure: float, asset: float) -> float:
    denominator = 1.0 + exposure * asset
    return float(np.clip(exposure * (1.0 + asset) / denominator, 0.0, 1.0))


def _finalize(frame: pd.DataFrame, arm: str, cost_rate: float) -> pd.DataFrame:
    """Fill derivable accounting columns and validate the cost identity."""
    frame = frame.copy()
    frame["arm"] = arm
    if "pretrade_exposure" not in frame or frame["pretrade_exposure"].isna().all():
        exposures = frame["exposure"].to_numpy(float)
        returns = frame["asset_simple_return"].to_numpy(float)
        pretrade = np.zeros(len(frame), dtype=float)
        for i in range(1, len(frame)):
            pretrade[i] = drifted(exposures[i - 1], returns[i - 1])
        frame["pretrade_exposure"] = pretrade
    frame["turnover"] = (frame["exposure"] - frame["pretrade_exposure"]).abs()
    frame["cost_fraction"] = cost_rate * frame["turnover"]
    expected = (1 - frame["cost_fraction"]) * (1 + frame["exposure"] * frame["asset_simple_return"]) - 1
    frame["accounting_max_abs_error"] = float(
        np.max(np.abs(frame["net_return_after_trading_costs"].to_numpy(float) - expected.to_numpy(float))))
    return frame


def load_single_trace(spec: dict, run: dict, root: Path, cost_rate: float):
    path = root / run["path"]
    raw = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "decision_date": raw["decision_date"], "return_date": raw["return_date"],
        "hard_regime": raw.get("hard_regime", "unknown"),
        "asset_simple_return": pd.to_numeric(raw["asset_simple_return"]),
        "exposure": pd.to_numeric(raw[spec["exposure"]]),
        "net_return_after_trading_costs": pd.to_numeric(raw[spec["net_return"]]),
        "action": raw.get(spec.get("action"), ""),
        "beta": pd.to_numeric(raw.get(spec.get("beta")), errors="coerce"),
    })
    return _finalize(out, spec["arm"], cost_rate), [path]


def load_paired_trace(spec: dict, run: dict, root: Path, cost_rate: float):
    path = root / run["path"]
    raw = pd.read_csv(path, low_memory=False)
    p = spec["prefix"]
    out = pd.DataFrame({
        "decision_date": raw["decision_date"], "return_date": raw["return_date"],
        "hard_regime": raw["hard_regime"],
        "asset_simple_return": pd.to_numeric(raw["asset_simple_return"]),
        "exposure": pd.to_numeric(raw[f"{p}_exposure"]),
        "pretrade_exposure": pd.to_numeric(raw[f"{p}_pretrade_exposure"]),
        "net_return_after_trading_costs": pd.to_numeric(raw[f"{p}_net_return"]),
        "action": raw[f"{p}_action"],
        "beta": pd.to_numeric(raw[f"{p}_beta"]),
    })
    return _finalize(out, spec["arm"], cost_rate), [path]


def load_arm_ledger(arm: str, directory: Path, cost_rate: float):
    path = directory / "DAILY_LEDGER.csv"
    raw = pd.read_csv(path, low_memory=False)
    keep = ["decision_date", "return_date", "hard_regime", "asset_simple_return",
            "pretrade_exposure", "exposure", "net_return_after_trading_costs", "action", "beta"]
    out = raw[[c for c in keep if c in raw.columns]].copy()
    for extra in ("retrieved_memory_count", "cited_memory_count", "desired_exposure",
                  "shadow_log_advantage_vs_ramoe", "valid"):
        if extra in raw.columns:
            out[extra] = raw[extra]
    return _finalize(out, arm, cost_rate), [path]


def arm_metrics(frame: pd.DataFrame, start: str, end: str) -> dict:
    """Descriptive metrics over the common evaluation window."""
    f = frame.copy()
    f["return_date"] = pd.to_datetime(f["return_date"])
    f["decision_date"] = pd.to_datetime(f["decision_date"])
    window = f[(f["return_date"] >= pd.Timestamp(start)) & (f["return_date"] <= pd.Timestamp(end))]
    r = window["net_return_after_trading_costs"].to_numpy(float)
    logs = np.log1p(r)
    n = len(r)
    std = float(r.std(ddof=1)) if n > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    mean = float(r.mean())
    duration = int((window["return_date"].max() - window["decision_date"].min()).days)
    path = np.r_[0.0, logs.cumsum()]
    return {
        "eval_rows": n,
        "first_return_date": str(window["return_date"].min().date()),
        "last_return_date": str(window["return_date"].max().date()),
        "net_compounded_return_pct": float(np.expm1(logs.sum()) * 100),
        "cumulative_net_log_return": float(logs.sum()),
        "mean_daily_net_log_return_bps": float(logs.mean() * 10000),
        "annualized_volatility_pct": float(std * math.sqrt(PER_YEAR) * 100),
        "annualized_sharpe_rf_zero": float(mean / std * math.sqrt(PER_YEAR)) if std > 1e-15 else None,
        "annualized_sortino_threshold_zero": float(mean / downside * math.sqrt(PER_YEAR)) if downside > 1e-15 else None,
        "net_cagr_pct": float(np.expm1(logs.sum() * PER_YEAR / duration) * 100) if duration else None,
        "maximum_drawdown_pct": float((-np.expm1(path - np.maximum.accumulate(path))).max() * 100),
        "expected_shortfall_95_daily_loss_pct": float(expected_shortfall_loss(r) * 100),
        "mean_btc_exposure_pct": float(window["exposure"].mean() * 100),
        "turnover_sum": float(window["turnover"].sum()),
        "cost_fraction_sum_pct": float(window["cost_fraction"].sum() * 100),
        "full_period_rows": len(f),
        "accounting_max_abs_error": float(frame["accounting_max_abs_error"].iloc[0]),
    }


def collect(sources: dict, root: Path):
    cost = float(sources["cost_rate"])
    frames, inventory, consumed = [], [], []
    for run in sources["runs"]:
        fmt = run["format"]
        if fmt in ("single_trace", "paired_trace"):
            loader = load_single_trace if fmt == "single_trace" else load_paired_trace
            for spec in run["arms"]:
                frame, files = loader(spec, run, root, cost)
                frames.append(frame)
                consumed += files
                inventory.append(dict(stage=run["stage"], run_id=run["run_id"], arm=spec["arm"],
                                      trust=spec["trust"], memory=spec["memory"],
                                      controller=spec["controller"], core=spec.get("core", "original"),
                                      sampling_seed=run["sampling_seed"],
                                      prompt_variant=run["prompt_variant"], source=str(files[0])))
        elif fmt in ("arm_ledgers", "arm_ledgers_nested"):
            base = root / run["path"]
            directories = []
            if fmt == "arm_ledgers":
                directories = [d for d in sorted(base.iterdir()) if d.is_dir()]
            else:
                for sub in run["subdirs"]:
                    directories += [d for d in sorted((base / sub).iterdir()) if d.is_dir()]
            for directory in directories:
                arm = directory.name
                config = run["arm_config"].get(arm)
                if config is None:
                    continue
                frame, files = load_arm_ledger(arm, directory, cost)
                frames.append(frame)
                consumed += files
                inventory.append(dict(stage=run["stage"], run_id=run["run_id"], arm=arm,
                                      trust=config["trust"], memory=config["memory"],
                                      controller=config["controller"], core=config.get("core", "original"),
                                      sampling_seed=run["sampling_seed"],
                                      prompt_variant=run["prompt_variant"],
                                      note=config.get("note", ""), source=str(files[0])))
        else:
            raise RuntimeError(f"Unknown source format: {fmt}")
    inv = pd.DataFrame(inventory)
    # The trace-format runs carry no note field; normalise so matching is exact.
    if "note" not in inv.columns:
        inv["note"] = ""
    inv["note"] = inv["note"].fillna("").astype(str)
    inv["arm_key"] = inv["stage"] + ":" + inv["arm"]
    inv["config_key"] = ("trust=" + inv["trust"] + "|memory=" + inv["memory"]
                         + "|controller=" + inv["controller"] + "|core=" + inv["core"])
    return frames, inv, consumed


def verify_duplicates(frames, inv, sources) -> list[dict]:
    """Confirm the republished Stage 6.4 arms really are the Stage 6.2/6.3 trajectories."""
    by_key = {k: f for k, f in zip(inv["arm_key"], frames)}
    checks = []
    for later, earlier in sources["duplicate_arms_excluded_from_averaging"].items():
        if later == "note" or later not in by_key or earlier not in by_key:
            continue
        a = by_key[later]["net_return_after_trading_costs"].to_numpy(float)
        b = by_key[earlier]["net_return_after_trading_costs"].to_numpy(float)
        same = a.shape == b.shape and float(np.max(np.abs(a - b))) < 1e-12
        checks.append({"republished": later, "original": earlier, "identical": bool(same),
                       "max_abs_difference": None if a.shape != b.shape else float(np.max(np.abs(a - b)))})
    return checks


def independent_replication_transmission(frames, inv) -> dict:
    """Compare two independent runs of the SAME configuration, arm by arm.

    Stage 6.3 and Stage 7 both ran fixed-trust / no-memory / controller-on with
    the same model, seed and source stream, in separate runs with their own
    1,608 model calls each. Any day where the advisor's action differs but the
    executed exposure does not is a direct measurement of decision attenuation.
    """
    by_key = {k: f for k, f in zip(inv["arm_key"], frames)}
    pairs = [("6.3:fixed_no_memory", "7:closed_loop_no_memory"),
             ("6.3:fixed_memory", "7:closed_loop_memory")]
    out = []
    for left, right in pairs:
        if left not in by_key or right not in by_key:
            continue
        a, b = by_key[left], by_key[right]
        if len(a) != len(b):
            continue
        action_differs = (a["action"].astype(str).to_numpy() != b["action"].astype(str).to_numpy())
        exposure_differs = np.abs(a["exposure"].to_numpy(float) - b["exposure"].to_numpy(float)) > 1e-12
        net_differs = np.abs(a["net_return_after_trading_costs"].to_numpy(float)
                             - b["net_return_after_trading_costs"].to_numpy(float)) > 1e-12
        flips = np.flatnonzero(action_differs)
        out.append({
            "left": left, "right": right, "days": int(len(a)),
            "action_differs_days": int(action_differs.sum()),
            "executed_exposure_differs_days": int(exposure_differs.sum()),
            "net_return_differs_days": int(net_differs.sum()),
            "action_flips_absorbed_by_execution": int((action_differs & ~exposure_differs).sum()),
            "daily_net_returns_bit_identical": bool(not net_differs.any()),
            "flip_detail": [{
                "return_date": str(a["return_date"].iloc[i]),
                "left_action": str(a["action"].iloc[i]), "right_action": str(b["action"].iloc[i]),
                "left_exposure": float(a["exposure"].iloc[i]), "right_exposure": float(b["exposure"].iloc[i]),
            } for i in flips[:10]],
        })
    return {
        "endpoint": "INDEPENDENT_REPLICATION_OF_IDENTICAL_CONFIGURATION",
        "interpretation": ("An action flip is the largest possible change in the advisor's output. "
                           "Where the executed exposure is unchanged, the blend at beta=0.05 and the "
                           "CVaR exposure grid absorbed the advice entirely."),
        "pairs": out,
    }


def memory_contrasts(frames, inv, sources) -> pd.DataFrame:
    """Every within-run memory-vs-no-memory contrast, matched on all other factors."""
    by_key = {k: f for k, f in zip(inv["arm_key"], frames)}
    start, end = sources["evaluation_start"], sources["evaluation_end"]
    duplicates = {k for k in sources["duplicate_arms_excluded_from_averaging"] if k != "note"}
    rows = []
    for _, mem in inv[inv["memory"] == "expanding"].iterrows():
        match = inv[(inv["stage"] == mem["stage"]) & (inv["memory"] == "none")
                    & (inv["trust"] == mem["trust"]) & (inv["controller"] == mem["controller"])
                    & (inv["core"] == mem["core"])]
        # Always match on the note field: Stage 7 publishes a closed-loop and a
        # state-controlled pair with otherwise identical factor settings.
        match = match[match["note"] == mem["note"]]
        if len(match) != 1:
            continue
        nom = match.iloc[0]
        a, b = by_key[mem["arm_key"]].copy(), by_key[nom["arm_key"]].copy()
        ledger = pd.concat([a, b], ignore_index=True)
        try:
            prepared = prepare_daily_ledger(ledger, float(sources["cost_rate"]))
            boot = paired_block_comparison(prepared, mem["arm"], nom["arm"],
                                           start_date=start, end_date=end,
                                           block_length=30, n_resamples=5000, seed=16062)
        except Exception as exc:
            rows.append({"stage": mem["stage"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({
            "stage": mem["stage"], "run_id": mem["run_id"],
            "memory_arm": mem["arm"], "no_memory_arm": nom["arm"],
            "trust": mem["trust"], "controller": mem["controller"], "core": mem["core"],
            "sampling_seed": mem["sampling_seed"], "prompt_variant": mem["prompt_variant"],
            "note": mem["note"],
            "is_republished_duplicate": mem["arm_key"] in duplicates,
            "paired_days": boot["paired_days"],
            "mean_daily_net_log_difference_bps": boot["mean_daily_net_log_difference_bps"],
            "ci95_lower_bps": boot["ci95_lower_bps"], "ci95_upper_bps": boot["ci95_upper_bps"],
            "net_compounded_gap_pp": boot["net_compounded_return_gap_percentage_points"],
            "interpretation": boot["interpretation"],
        })
    return pd.DataFrame(rows)


def cross_run_replication(metrics: pd.DataFrame, sources: dict) -> pd.DataFrame:
    """Average matched configurations across independent runs, duplicates removed."""
    duplicates = {k for k in sources["duplicate_arms_excluded_from_averaging"] if k != "note"}
    usable = metrics[~metrics["arm_key"].isin(duplicates)].copy()
    rows = []
    for config, group in usable.groupby("config_key"):
        values = group["net_compounded_return_pct"]
        rows.append({
            "config_key": config, "realizations": len(group),
            "stages": ",".join(sorted(group["stage"].astype(str))),
            "seeds": ",".join(sorted({str(s) for s in group["sampling_seed"]})),
            "prompt_variants": ",".join(sorted(set(group["prompt_variant"]))),
            "mean_net_compounded_return_pct": float(values.mean()),
            "min_pct": float(values.min()), "max_pct": float(values.max()),
            "spread_pp": float(values.max() - values.min()),
            "sd_pct": float(values.std(ddof=1)) if len(group) > 1 else None,
            "mean_sharpe": float(group["annualized_sharpe_rf_zero"].mean()),
            "mean_max_drawdown_pct": float(group["maximum_drawdown_pct"].mean()),
            "cross_run_average_is_matched": bool(len(set(group["prompt_variant"])) == 1),
        })
    return pd.DataFrame(rows).sort_values("config_key")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=PACKAGE / "sources.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime = assert_runtime()
    sources = json.loads(args.sources.read_text())
    root = Path(sources["project_root"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    frames, inv, consumed = collect(sources, root)
    print(f"collected {len(frames)} llama3.3:70b arms from {len(sources['runs'])} runs", flush=True)

    dup_checks = verify_duplicates(frames, inv, sources)
    if dup_checks:
        for c in dup_checks:
            print(f"  duplicate check {c['republished']} == {c['original']}: identical={c['identical']}", flush=True)
    else:
        print("  no republished duplicates collected: Stage 6.4 arms/ contains only its 14 new arms, "
              "so the Stage 6.2/6.3 pairs are read once, from their own runs", flush=True)

    metric_rows = []
    for key, frame in zip(inv["arm_key"], frames):
        row = inv[inv["arm_key"] == key].iloc[0].to_dict()
        row.update(arm_metrics(frame, sources["evaluation_start"], sources["evaluation_end"]))
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    long_rows = []
    for key, frame in zip(inv["arm_key"], frames):
        meta = inv[inv["arm_key"] == key].iloc[0]
        part = frame.copy()
        part["arm_key"] = key
        part["stage"] = meta["stage"]
        part["run_id"] = meta["run_id"]
        part["config_key"] = meta["config_key"]
        part["sampling_seed"] = meta["sampling_seed"]
        part["prompt_variant"] = meta["prompt_variant"]
        long_rows.append(part)
    long = pd.concat(long_rows, ignore_index=True)

    contrasts = memory_contrasts(frames, inv, sources)
    replication = cross_run_replication(metrics, sources)
    sys.path.insert(0, str(PACKAGE))
    from pooled import pooled_memory_effect, REPORT_HEADER
    pooled = pooled_memory_effect(contrasts)
    transmission = independent_replication_transmission(frames, inv)
    for pair in transmission["pairs"]:
        print(f"  replication {pair['left']} vs {pair['right']}: "
              f"action differs {pair['action_differs_days']}d, exposure differs "
              f"{pair['executed_exposure_differs_days']}d, net bit-identical="
              f"{pair['daily_net_returns_bit_identical']}", flush=True)

    long.to_csv(output / "ALL_ARMS_DAILY.csv", index=False, float_format="%.17g")
    metrics.to_csv(output / "ARM_METRICS.csv", index=False, float_format="%.17g")
    inv.to_csv(output / "ARM_INVENTORY.csv", index=False)
    contrasts.to_csv(output / "MEMORY_CONTRASTS.csv", index=False, float_format="%.17g")
    replication.to_csv(output / "CROSS_RUN_REPLICATION.csv", index=False, float_format="%.17g")

    manifest = {
        "status": "COMPLETE_READ_ONLY_NO_MODEL_CALLS", "created_at": now(),
        "runtime": runtime, "sources_sha256": sha256(args.sources),
        "consolidate_sha256": sha256(PACKAGE / "consolidate.py"),
        "stage64_metrics_module_sha256": sha256(STAGE64 / "suite64" / "metrics.py"),
        "model": sources["model"], "expected_model_digest": sources["expected_model_digest"],
        "evaluation_window": [sources["evaluation_start"], sources["evaluation_end"]],
        "arms_collected": int(len(inv)),
        "duplicate_verification": dup_checks,
        "input_files_sha256": {str(p.relative_to(root)): sha256(p) for p in consumed},
        "outputs_sha256": {},
    }
    (output / "REPLICATION_TRANSMISSION.json").write_text(
        json.dumps(transmission, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (output / "POOLED_MEMORY_EFFECT.json").write_text(
        json.dumps(pooled, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    def table(frame, columns, floats=4):
        f = frame[columns].copy()
        for c in f.columns:
            if pd.api.types.is_float_dtype(f[c]):
                f[c] = f[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
        return "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n" + "\n".join(
            "| " + " | ".join(str(v) for v in row) + " |" for row in f.itertuples(index=False))

    a = pooled["all_realizations"]
    report = REPORT_HEADER.format(
        created=now(), model=sources["model"], digest=sources["expected_model_digest"],
        start=sources["evaluation_start"], end=sources["evaluation_end"],
        rows=int(metrics["eval_rows"].max()),
        py=runtime["python"], np=runtime["numpy"], pd=runtime["pandas"],
        inventory_table=table(inv, ["stage", "run_id", "arm", "trust", "memory", "controller",
                                    "core", "sampling_seed", "prompt_variant"]),
        arms=len(inv), daily_rows=len(long),
        worst_acc=float(metrics["accounting_max_abs_error"].max()),
        n_real=a["realizations"],
        contrast_table=table(contrasts, ["stage", "memory_arm", "trust", "controller",
                                         "sampling_seed", "mean_daily_net_log_difference_bps",
                                         "ci95_lower_bps", "ci95_upper_bps",
                                         "net_compounded_gap_pp", "interpretation"]),
        pooled_mean=a["mean_daily_net_log_difference_bps"],
        pos=a["positive_realizations"], neg=a["negative_realizations"],
        lo=a["min_bps"], hi=a["max_bps"], mean_gap=a["mean_compounded_gap_pp"],
        pooling_caveat="> " + pooled["why"] + "\n>\n> " + pooled["multiplicity_note"],
        replication_table=table(replication[replication["realizations"] > 1],
                                ["config_key", "realizations", "stages", "seeds",
                                 "mean_net_compounded_return_pct", "min_pct", "max_pct", "spread_pp"]),
        arm_table=table(metrics.sort_values("net_compounded_return_pct", ascending=False),
                        ["stage", "arm", "net_compounded_return_pct", "annualized_sharpe_rf_zero",
                         "maximum_drawdown_pct", "mean_btc_exposure_pct", "turnover_sum"]),
    )
    (output / "CONSOLIDATED_REPORT.md").write_text(report, encoding="utf-8")

    for name in ("ALL_ARMS_DAILY.csv", "ARM_METRICS.csv", "ARM_INVENTORY.csv",
                 "MEMORY_CONTRASTS.csv", "CROSS_RUN_REPLICATION.csv",
                 "POOLED_MEMORY_EFFECT.json", "CONSOLIDATED_REPORT.md",
                 "REPLICATION_TRANSMISSION.json"):
        manifest["outputs_sha256"][name] = sha256(output / name)
    manifest["pooled_memory_effect_bps_per_day"] = a["mean_daily_net_log_difference_bps"]
    manifest["memory_realizations"] = a["realizations"]
    (output / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote {len(long)} daily rows across {len(inv)} arms to {output}")
    print(f"memory contrasts: {len(contrasts)} | matched config groups: {len(replication)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
