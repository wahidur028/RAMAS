#!/usr/bin/env python3
"""Read-only paired validation of the RAMAS LLaMA advisor pathway.

This script:
1. Reads the archived Stage 6.4 daily ledgers.
2. Uses numerical_only as the no-advisor numerical-core baseline.
3. Compares every saved llama_* arm against that baseline.
4. Replays controller-off execution from desired_exposure.
5. Computes paired 30-day circular-block bootstrap intervals.

It never calls LLaMA and never modifies the experiment archive.
"""

import argparse
import csv
import io
import json
import math
import os
import random
import statistics
import tarfile


COST_RATE = 0.001
ANNUALIZATION_DAYS = 365.25
REQUIRED_COLUMNS = {
    "arm",
    "decision_date",
    "return_date",
    "asset_simple_return",
    "pretrade_exposure",
    "exposure",
    "net_return_after_trading_costs",
    "turnover",
    "advisor",
    "beta",
    "memory_mode",
    "core_desired_exposure",
    "desired_exposure",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        default=(
            "/home/infonet/wahid/leader_router_fresh/"
            "EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/"
            "artifacts/20260909T114202240674964Z.tar.gz"
        ),
    )
    parser.add_argument(
        "--output",
        default="ramas_llama_advisor_value_bootstrap.json",
    )
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-05-28")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--block-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def to_float(row, key):
    value = str(row.get(key, "")).strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        raise ValueError(f"Missing numeric value for {key}")
    return float(value)


def read_csv_member(tar, member):
    with tar.extractfile(member) as raw:
        text = io.TextIOWrapper(
            raw,
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        )
        reader = csv.DictReader(text)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"Empty ledger: {member.name}")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise RuntimeError(
            f"{member.name} is missing columns: {sorted(missing)}"
        )

    return rows


def discover_arm_ledgers(tar):
    ledgers = {}

    for member in tar.getmembers():
        if not member.isfile():
            continue
        if "/arms/" not in member.name:
            continue
        if not member.name.endswith("/DAILY_LEDGER.csv"):
            continue

        rows = read_csv_member(tar, member)
        arm = str(rows[0]["arm"]).strip()

        if arm in ledgers:
            raise RuntimeError(f"Duplicate ledger for arm: {arm}")

        ledgers[arm] = {
            "member": member.name,
            "rows": rows,
        }

    if "numerical_only" not in ledgers:
        raise RuntimeError(
            "The archive does not contain the numerical_only arm. "
            f"Found: {sorted(ledgers)}"
        )

    return ledgers


def validate_no_advisor_arm(rows):
    advisor_values = sorted({str(r["advisor"]).strip().lower() for r in rows})
    memory_values = sorted({str(r["memory_mode"]).strip().lower() for r in rows})
    beta_values = sorted({to_float(r, "beta") for r in rows})

    if advisor_values != ["abstain"]:
        raise RuntimeError(
            f"numerical_only is not advisor-free: {advisor_values}"
        )
    if memory_values != ["none"]:
        raise RuntimeError(
            f"numerical_only has unexpected memory modes: {memory_values}"
        )
    if any(abs(x) > 1e-12 for x in beta_values):
        raise RuntimeError(f"numerical_only has nonzero beta: {beta_values}")

    maximum_difference = max(
        abs(to_float(r, "core_desired_exposure")
            - to_float(r, "desired_exposure"))
        for r in rows
    )

    if maximum_difference > 1e-12:
        raise RuntimeError(
            "numerical_only desired exposure differs from core exposure by "
            f"{maximum_difference}"
        )

    return {
        "advisor_values": advisor_values,
        "memory_values": memory_values,
        "beta_values": beta_values,
        "max_core_desired_vs_desired_difference": maximum_difference,
    }


def validate_shared_calendar(core_rows, other_rows, arm):
    if len(core_rows) != len(other_rows):
        raise RuntimeError(
            f"Date-count mismatch for {arm}: "
            f"core={len(core_rows)}, arm={len(other_rows)}"
        )

    for index, (core, other) in enumerate(zip(core_rows, other_rows)):
        for key in ("decision_date", "return_date"):
            if core[key] != other[key]:
                raise RuntimeError(
                    f"Calendar mismatch for {arm} at row {index}: "
                    f"{key}: {core[key]} != {other[key]}"
                )

        core_return = to_float(core, "asset_simple_return")
        other_return = to_float(other, "asset_simple_return")

        if not math.isclose(core_return, other_return, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"Asset-return mismatch for {arm} at row {index}: "
                f"{core_return} != {other_return}"
            )

        core_initial = to_float(core_rows[0], "pretrade_exposure")
        other_initial = to_float(other_rows[0], "pretrade_exposure")
        if not math.isclose(
            core_initial,
            other_initial,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Initial-holding mismatch for {arm}: "
                f"core={core_initial}, arm={other_initial}"
            )


def select_window(records, start_date, end_date):
    selected = [
        record
        for record in records
        if start_date <= record["return_date"] <= end_date
    ]

    if not selected:
        raise RuntimeError(
            f"No rows in requested window {start_date} through {end_date}"
        )

    return selected


def controller_on_records(rows):
    records = []

    for row in rows:
        records.append({
            "decision_date": row["decision_date"],
            "return_date": row["return_date"],
            "asset_simple_return": to_float(row, "asset_simple_return"),
            "net_return": to_float(row, "net_return_after_trading_costs"),
            "exposure": to_float(row, "exposure"),
            "turnover": to_float(row, "turnover"),
        })

    return records


def controller_off_records(rows, arm):
    pretrade_exposure = to_float(rows[0], "pretrade_exposure")
    records = []

    for row in rows:
        asset_return = to_float(row, "asset_simple_return")

        if arm == "numerical_only":
            target = to_float(row, "core_desired_exposure")
        else:
            target = to_float(row, "desired_exposure")

        if not 0.0 <= target <= 1.0:
            raise RuntimeError(
                f"Invalid target exposure {target} for {arm} on "
                f"{row['decision_date']}"
            )

        turnover = abs(target - pretrade_exposure)
        net_return = (
            (1.0 - COST_RATE * turnover)
            * (1.0 + target * asset_return)
            - 1.0
        )

        records.append({
            "decision_date": row["decision_date"],
            "return_date": row["return_date"],
            "asset_simple_return": asset_return,
            "net_return": net_return,
            "exposure": target,
            "turnover": turnover,
        })

        denominator = 1.0 + target * asset_return
        if denominator <= 0.0:
            raise RuntimeError(
                f"Invalid exposure-drift denominator for {arm} on "
                f"{row['decision_date']}"
            )

        pretrade_exposure = target * (1.0 + asset_return) / denominator

    return records


def fractional_es95(returns):
    ordered = sorted(returns)
    tail_mass = 0.05 * len(ordered)
    full_count = int(math.floor(tail_mass))
    fractional_count = tail_mass - full_count

    if full_count <= 0:
        return float("nan")

    tail_sum = sum(ordered[:full_count])
    if fractional_count > 0.0 and full_count < len(ordered):
        tail_sum += fractional_count * ordered[full_count]

    return -(tail_sum / tail_mass) * 100.0


def calculate_metrics(records):
    returns = [x["net_return"] for x in records]
    exposures = [x["exposure"] for x in records]
    turnovers = [x["turnover"] for x in records]

    wealth = 1.0
    equity = []
    for value in returns:
        wealth *= 1.0 + value
        equity.append(wealth)

    standard_deviation = statistics.stdev(returns)
    annualized_volatility = (
        standard_deviation
        * math.sqrt(ANNUALIZATION_DAYS)
        * 100.0
    )
    sharpe = (
        statistics.mean(returns)
        / standard_deviation
        * math.sqrt(ANNUALIZATION_DAYS)
    )

    peak = 1.0
    maximum_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        maximum_drawdown = max(maximum_drawdown, (peak - value) / peak)

    return {
        "date_count": len(records),
        "net_return_pct": (wealth - 1.0) * 100.0,
        "annualized_volatility_pct": annualized_volatility,
        "sharpe": sharpe,
        "mdd_pct": maximum_drawdown * 100.0,
        "es95_pct": fractional_es95(returns),
        "total_turnover": sum(turnovers),
        "mean_exposure_pct": statistics.mean(exposures) * 100.0,
    }


def percentile(values, probability):
    if not values:
        raise RuntimeError("Cannot calculate percentile of empty values")

    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return values[lower]

    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def paired_circular_bootstrap(
    baseline_records,
    treatment_records,
    bootstrap_count,
    block_days,
    seed,
):
    if len(baseline_records) != len(treatment_records):
        raise RuntimeError("Bootstrap inputs have different lengths")

    differences_bps = []

    for baseline, treatment in zip(baseline_records, treatment_records):
        if baseline["return_date"] != treatment["return_date"]:
            raise RuntimeError("Bootstrap input dates are not aligned")
        differences_bps.append(
            (treatment["net_return"] - baseline["net_return"]) * 10000.0
        )

    count = len(differences_bps)
    rng = random.Random(seed)
    bootstrap_means = []

    for _ in range(bootstrap_count):
        sampled = []

        while len(sampled) < count:
            start = rng.randrange(count)
            for offset in range(block_days):
                if len(sampled) >= count:
                    break
                sampled.append(
                    differences_bps[(start + offset) % count]
                )

        bootstrap_means.append(statistics.mean(sampled))

    bootstrap_means.sort()
    point_estimate = statistics.mean(differences_bps)

    return {
        "point_mean_daily_difference_bps": point_estimate,
        "ci95_mean_daily_difference_bps": [
            percentile(bootstrap_means, 0.025),
            percentile(bootstrap_means, 0.975),
        ],
        "bootstrap_mean_bps": statistics.mean(bootstrap_means),
        "bootstrap_positive_fraction": (
            sum(x > 0.0 for x in bootstrap_means)
            / len(bootstrap_means)
        ),
        "daily_win_fraction": (
            sum(x > 0.0 for x in differences_bps)
            / len(differences_bps)
        ),
        "bootstrap_count": bootstrap_count,
        "block_days": block_days,
        "seed": seed,
    }


def arm_classification(arm):
    if arm.startswith("llama_memory_"):
        return "advisor_on_memory_variant"
    if arm == "llama_frozen_W":
        return "advisor_on_plus_frozen_W_confound"
    if arm == "llama_no_risk":
        return "advisor_on_plus_no_risk_confound"
    if arm in {"llama_hard_allocator", "llama_uniform_allocator"}:
        return "advisor_on_plus_allocator_confound"
    return "other_llama_arm"


def main():
    args = parse_args()

    if not os.path.isfile(args.archive):
        raise FileNotFoundError(args.archive)
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    if args.block_days < 1:
        raise ValueError("block-days must be positive")

    with tarfile.open(args.archive, "r:gz") as tar:
        ledgers = discover_arm_ledgers(tar)

    core_rows = ledgers["numerical_only"]["rows"]
    core_validation = validate_no_advisor_arm(core_rows)

    llama_arms = sorted(
        arm for arm in ledgers
        if arm.startswith("llama_")
    )

    if not llama_arms:
        raise RuntimeError("No llama_* arms found in the archive")

    core_on_all = controller_on_records(core_rows)
    core_off_all = controller_off_records(core_rows, "numerical_only")

    core_on = select_window(
        core_on_all,
        args.start_date,
        args.end_date,
    )
    core_off = select_window(
        core_off_all,
        args.start_date,
        args.end_date,
    )

    core_metrics = {
        "controller_on": calculate_metrics(core_on),
        "controller_off": calculate_metrics(core_off),
    }

    comparisons = []

    for index, arm in enumerate(llama_arms):
        llama_rows = ledgers[arm]["rows"]
        validate_shared_calendar(core_rows, llama_rows, arm)

        llama_on_all = controller_on_records(llama_rows)
        llama_off_all = controller_off_records(llama_rows, arm)

        llama_on = select_window(
            llama_on_all,
            args.start_date,
            args.end_date,
        )
        llama_off = select_window(
            llama_off_all,
            args.start_date,
            args.end_date,
        )

        llama_metrics = {
            "controller_on": calculate_metrics(llama_on),
            "controller_off": calculate_metrics(llama_off),
        }

        effects = {}

        for controller_state, baseline, treatment in (
            ("controller_on", core_on, llama_on),
            ("controller_off", core_off, llama_off),
        ):
            baseline_metrics = core_metrics[controller_state]
            treatment_metrics = llama_metrics[controller_state]

            bootstrap = paired_circular_bootstrap(
                baseline,
                treatment,
                args.bootstrap,
                args.block_days,
                args.seed + index * 100 + (0 if controller_state == "controller_on" else 1),
            )

            effects[controller_state] = {
                "net_return_difference_pp": (
                    treatment_metrics["net_return_pct"]
                    - baseline_metrics["net_return_pct"]
                ),
                "sharpe_difference": (
                    treatment_metrics["sharpe"]
                    - baseline_metrics["sharpe"]
                ),
                "mdd_difference_pp": (
                    treatment_metrics["mdd_pct"]
                    - baseline_metrics["mdd_pct"]
                ),
                "es95_difference_pp": (
                    treatment_metrics["es95_pct"]
                    - baseline_metrics["es95_pct"]
                ),
                "turnover_difference": (
                    treatment_metrics["total_turnover"]
                    - baseline_metrics["total_turnover"]
                ),
                "mean_exposure_difference_pp": (
                    treatment_metrics["mean_exposure_pct"]
                    - baseline_metrics["mean_exposure_pct"]
                ),
                "paired_bootstrap": bootstrap,
            }

        comparisons.append({
            "arm": arm,
            "classification": arm_classification(arm),
            "source_member": ledgers[arm]["member"],
            "row_count": len(llama_rows),
            "arm_config_values": {
                "advisor": sorted({str(r["advisor"]).strip().lower() for r in llama_rows}),
                "beta": sorted({to_float(r, "beta") for r in llama_rows}),
                "memory_mode": sorted({str(r["memory_mode"]).strip().lower() for r in llama_rows}),
            },
            "metrics": llama_metrics,
            "advisor_effect_vs_numerical_only": effects,
        })

    result = {
        "experiment": "RAMAS_LLaMA_advisor_value_paired_bootstrap",
        "read_only": True,
        "llama_called": False,
        "archive": args.archive,
        "baseline_arm": "numerical_only",
        "cost_bps": 10,
        "evaluation_window": {
            "return_date_start": args.start_date,
            "return_date_end": args.end_date,
            "date_count": len(core_on),
        },
        "bootstrap_contract": {
            "resamples": args.bootstrap,
            "circular_block_days": args.block_days,
            "primary_effect_unit": "mean daily net-return difference in bps/day",
            "seed": args.seed,
        },
        "baseline_validation": core_validation,
        "baseline_metrics": core_metrics,
        "comparisons": comparisons,
        "interpretation_warning": (
            "llama_memory_* arms are advisor-on memory variants. "
            "llama_frozen_W, llama_no_risk, and allocator arms include "
            "additional component changes and must not be presented as pure "
            "LLaMA-only effects."
        ),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"CREATED={os.path.abspath(args.output)}")
    print(f"BASELINE_ARM=numerical_only")
    print(f"LLAMA_ARMS={','.join(llama_arms)}")
    print(f"DATE_COUNT={len(core_on)}")
    print(f"START_DATE={args.start_date}")
    print(f"END_DATE={args.end_date}")
    print("LLAMA_CALLS=0")


if __name__ == "__main__":
    main()
