#!/usr/bin/env python3
"""Matched, read-only ablation of the RAMAS LLaMA advisor.

This script decomposes the system into three paths:

    N   = numerical-only baseline
    L   = LLaMA advisor without memory
    LM  = LLaMA advisor with memory

For both the adaptive and fixed-trust traces it reports:

    L  - N    pure no-memory LLaMA advisor effect
    LM - L    incremental memory effect
    LM - N    total LLaMA-plus-memory pathway effect

Each comparison is reported with the risk/controller ON and OFF.  The trace
archives already contain paired LLaMA and no-memory rows, so this script never
calls LLaMA and never changes the archives.
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


COST_RATE = 0.001  # 10 bps per unit turnover
ANNUALIZATION_DAYS = 365.25
DATE_KEYS = ("decision_date", "return_date")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adaptive",
        default=(
            "/home/infonet/wahid/leader_router_fresh/"
            "EXPERIMENT_BRANCHES/"
            "RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/"
            "artifacts/20260908T055052711433762Z.tar.gz"
        ),
    )
    parser.add_argument(
        "--fixed",
        default=(
            "/home/infonet/wahid/leader_router_fresh/"
            "EXPERIMENT_BRANCHES/"
            "RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/"
            "artifacts/20260909T013716356200839Z.tar.gz"
        ),
    )
    parser.add_argument(
        "--numerical-core",
        default="ramas_numerical_core_controller_ablation.json",
        help="JSON produced by the numerical-only controller ablation",
    )
    parser.add_argument(
        "--output",
        default="ramas_pure_llama_advisor_ablation.json",
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
        raise RuntimeError(f"Missing numeric value for {key}")
    return float(value)


def is_true(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def read_trace_archive(archive_path):
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(archive_path)

    with tarfile.open(archive_path, "r:gz") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.isfile()
            and os.path.basename(member.name) == "03_DAILY_TRACE.csv"
        ]

        if len(members) != 1:
            raise RuntimeError(
                f"Expected exactly one 03_DAILY_TRACE.csv in {archive_path}; "
                f"found {len(members)}"
            )

        member = members[0]
        raw = tar.extractfile(member)
        if raw is None:
            raise RuntimeError(f"Cannot extract {member.name}")

        with raw:
            text = io.TextIOWrapper(
                raw,
                encoding="utf-8-sig",
                errors="replace",
                newline="",
            )
            rows = list(csv.DictReader(text))

    if not rows:
        raise RuntimeError(f"Empty trace: {archive_path}")

    required = {
        "decision_date",
        "return_date",
        "asset_simple_return",
        "memory_net_return",
        "memory_exposure",
        "memory_turnover",
        "memory_desired_exposure",
        "memory_shadow_net_return",
        "memory_shadow_pretrade_exposure",
        "memory_shadow_exposure",
        "no_memory_net_return",
        "no_memory_exposure",
        "no_memory_turnover",
        "no_memory_desired_exposure",
        "no_memory_shadow_net_return",
        "no_memory_shadow_pretrade_exposure",
        "no_memory_shadow_exposure",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(
            f"{archive_path} is missing trace columns: {missing}"
        )

    return member.name, rows


def read_numerical_core(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Numerical-core JSON not found: {path}. "
            "Copy ramas_numerical_core_controller_ablation.json to the "
            "project root or pass --numerical-core /path/to/file.json."
        )

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload.get("daily_rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("Numerical-core JSON has no daily_rows")

    required = {
        "decision_date",
        "return_date",
        "asset_simple_return",
        "core_desired_exposure",
        "controller_on_exposure",
        "controller_on_turnover",
        "controller_on_net_return",
        "controller_off_pretrade_exposure",
        "controller_off_exposure",
        "controller_off_turnover",
        "controller_off_net_return",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise RuntimeError(
            f"Numerical-core JSON is missing columns: {missing}"
        )

    return payload, rows


def unique_date_rows(rows, label):
    result = {}
    for row in rows:
        date = str(row["return_date"])
        if date in result:
            raise RuntimeError(f"Duplicate return_date in {label}: {date}")
        result[date] = row
    return result


def match_window(core_rows, trace_rows, start_date, end_date, label):
    core_map = unique_date_rows(core_rows, "numerical_only")
    trace_map = unique_date_rows(trace_rows, label)

    dates = sorted(
        date
        for date in core_map
        if start_date <= date <= end_date
    )
    trace_dates = sorted(
        date
        for date in trace_map
        if start_date <= date <= end_date
    )

    if dates != trace_dates:
        missing_in_trace = sorted(set(dates) - set(trace_dates))
        extra_in_trace = sorted(set(trace_dates) - set(dates))
        raise RuntimeError(
            f"Calendar mismatch for {label}; "
            f"missing_in_trace={missing_in_trace[:5]}, "
            f"extra_in_trace={extra_in_trace[:5]}"
        )

    matched = []
    for date in dates:
        core = core_map[date]
        trace = trace_map[date]

        if str(core["decision_date"]) != str(trace["decision_date"]):
            raise RuntimeError(
                f"Decision-date mismatch for {label} on {date}: "
                f"{core['decision_date']} != {trace['decision_date']}"
            )

        core_return = to_float(core, "asset_simple_return")
        trace_return = to_float(trace, "asset_simple_return")
        if not math.isclose(
            core_return,
            trace_return,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(
                f"Asset-return mismatch for {label} on {date}: "
                f"{core_return} != {trace_return}"
            )

        matched.append((core, trace))

    if not matched:
        raise RuntimeError(f"No rows in {label} evaluation window")

    return matched


def validate_numerical_only(payload, rows):
    validation = payload.get("validation", {})
    if validation:
        advisor_values = validation.get("advisor_values", [])
        memory_values = validation.get("memory_values", [])
        beta_values = validation.get("beta_values", [])
        if advisor_values and advisor_values != ["abstain"]:
            raise RuntimeError(f"Unexpected numerical-only advisor: {advisor_values}")
        if memory_values and memory_values != ["none"]:
            raise RuntimeError(f"Unexpected numerical-only memory: {memory_values}")
        if beta_values and any(abs(float(x)) > 1e-12 for x in beta_values):
            raise RuntimeError(f"Unexpected numerical-only beta: {beta_values}")

    return {
        "advisor": "ABSTAIN",
        "memory": "none",
        "beta": 0.0,
        "source_validation": validation,
    }


def core_records(rows):
    records = {"controller_on": [], "controller_off": []}
    for row in rows:
        common = {
            "decision_date": str(row["decision_date"]),
            "return_date": str(row["return_date"]),
            "asset_simple_return": to_float(row, "asset_simple_return"),
            "core_desired_exposure": to_float(row, "core_desired_exposure"),
        }
        records["controller_on"].append({
            **common,
            "net_return": to_float(row, "controller_on_net_return"),
            "exposure": to_float(row, "controller_on_exposure"),
            "turnover": to_float(row, "controller_on_turnover"),
        })
        records["controller_off"].append({
            **common,
            "net_return": to_float(row, "controller_off_net_return"),
            "exposure": to_float(row, "controller_off_exposure"),
            "turnover": to_float(row, "controller_off_turnover"),
        })
    return records


def trace_records(matched, prefix):
    """Build controller ON/OFF records for memory or no_memory.

    The trace's shadow fields are audited.  For controller OFF we recompute
    return from desired exposure, shadow pretrade exposure, asset return, and
    the locked 10-bps turnover cost.  This avoids trusting a summary field for
    the decisive ablation.
    """
    records = {"controller_on": [], "controller_off": []}
    audit = {
        "prefix": prefix,
        "max_shadow_exposure_vs_desired_difference": 0.0,
        "max_shadow_return_reconstruction_difference": 0.0,
        "max_shadow_pretrade_sequential_difference": 0.0,
        "valid_fraction": None,
        "error_count": 0,
    }

    valid_key = f"{prefix}_valid"
    error_key = f"{prefix}_error"
    valid_values = []

    previous_shadow_exposure = None

    for core, row in matched:
        asset_return = to_float(row, "asset_simple_return")
        desired = to_float(row, f"{prefix}_desired_exposure")
        shadow_pretrade = to_float(
            row,
            f"{prefix}_shadow_pretrade_exposure",
        )
        shadow_exposure = to_float(row, f"{prefix}_shadow_exposure")
        stored_shadow_return = to_float(
            row,
            f"{prefix}_shadow_net_return",
        )

        if not 0.0 <= desired <= 1.0:
            raise RuntimeError(
                f"Invalid {prefix} desired exposure {desired} on "
                f"{row['decision_date']}"
            )

        audit["max_shadow_exposure_vs_desired_difference"] = max(
            audit["max_shadow_exposure_vs_desired_difference"],
            abs(shadow_exposure - desired),
        )

        if previous_shadow_exposure is not None:
            audit["max_shadow_pretrade_sequential_difference"] = max(
                audit["max_shadow_pretrade_sequential_difference"],
                abs(shadow_pretrade - previous_shadow_exposure),
            )

        shadow_turnover = abs(desired - shadow_pretrade)
        reconstructed_shadow_return = (
            (1.0 - COST_RATE * shadow_turnover)
            * (1.0 + desired * asset_return)
            - 1.0
        )
        audit["max_shadow_return_reconstruction_difference"] = max(
            audit["max_shadow_return_reconstruction_difference"],
            abs(stored_shadow_return - reconstructed_shadow_return),
        )

        records["controller_on"].append({
            "decision_date": str(row["decision_date"]),
            "return_date": str(row["return_date"]),
            "asset_simple_return": asset_return,
            "core_desired_exposure": to_float(
                core,
                "core_desired_exposure",
            ),
            "desired_exposure": desired,
            "net_return": to_float(row, f"{prefix}_net_return"),
            "exposure": to_float(row, f"{prefix}_exposure"),
            "turnover": to_float(row, f"{prefix}_turnover"),
        })

        records["controller_off"].append({
            "decision_date": str(row["decision_date"]),
            "return_date": str(row["return_date"]),
            "asset_simple_return": asset_return,
            "core_desired_exposure": to_float(
                core,
                "core_desired_exposure",
            ),
            "desired_exposure": desired,
            "net_return": reconstructed_shadow_return,
            "exposure": desired,
            "turnover": shadow_turnover,
        })

        if valid_key in row and str(row[valid_key]).strip() != "":
            valid_values.append(is_true(row[valid_key]))
        if error_key in row and str(row[error_key]).strip():
            error_value = str(row[error_key]).strip().lower()
            if error_value not in {"", "none", "null", "nan", "false"}:
                audit["error_count"] += 1

        previous_shadow_exposure = desired * (1.0 + asset_return) / (
            1.0 + desired * asset_return
        )

    if valid_values:
        audit["valid_fraction"] = sum(valid_values) / len(valid_values)

    return records, audit


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
    returns = [record["net_return"] for record in records]
    exposures = [record["exposure"] for record in records]
    turnovers = [record["turnover"] for record in records]

    if len(returns) < 2:
        raise RuntimeError("At least two daily returns are required")

    wealth = 1.0
    equity = []
    for value in returns:
        wealth *= 1.0 + value
        equity.append(wealth)

    standard_deviation = statistics.stdev(returns)
    annualized_volatility = (
        standard_deviation * math.sqrt(ANNUALIZATION_DAYS) * 100.0
    )
    sharpe = (
        statistics.mean(returns)
        / standard_deviation
        * math.sqrt(ANNUALIZATION_DAYS)
        if standard_deviation > 0.0
        else float("nan")
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
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def circular_indices(count, block_days, rng):
    indices = []
    while len(indices) < count:
        start = rng.randrange(count)
        for offset in range(block_days):
            if len(indices) >= count:
                break
            indices.append((start + offset) % count)
    return indices


def bootstrap_difference(
    baseline,
    treatment,
    bootstrap_count,
    block_days,
    seed,
):
    if len(baseline) != len(treatment):
        raise RuntimeError("Bootstrap inputs have different lengths")

    for base, treat in zip(baseline, treatment):
        if base["return_date"] != treat["return_date"]:
            raise RuntimeError("Bootstrap dates are not aligned")

    point_base = calculate_metrics(baseline)
    point_treatment = calculate_metrics(treatment)
    metric_names = [
        "net_return_pct",
        "annualized_volatility_pct",
        "sharpe",
        "mdd_pct",
        "es95_pct",
        "total_turnover",
        "mean_exposure_pct",
    ]

    point_difference = {
        name: point_treatment[name] - point_base[name]
        for name in metric_names
    }

    daily_differences_bps = [
        (treat["net_return"] - base["net_return"]) * 10000.0
        for base, treat in zip(baseline, treatment)
    ]
    point_daily_bps = statistics.mean(daily_differences_bps)

    rng = random.Random(seed)
    bootstrap_metric_values = {name: [] for name in metric_names}
    bootstrap_daily_values = []

    for _ in range(bootstrap_count):
        indices = circular_indices(len(baseline), block_days, rng)
        sampled_base = [baseline[index] for index in indices]
        sampled_treatment = [treatment[index] for index in indices]
        base_metrics = calculate_metrics(sampled_base)
        treatment_metrics = calculate_metrics(sampled_treatment)

        for name in metric_names:
            bootstrap_metric_values[name].append(
                treatment_metrics[name] - base_metrics[name]
            )

        bootstrap_daily_values.append(
            statistics.mean(daily_differences_bps[index] for index in indices)
        )

    intervals = {}
    for name, values in bootstrap_metric_values.items():
        values.sort()
        intervals[name] = [
            percentile(values, 0.025),
            percentile(values, 0.975),
        ]

    bootstrap_daily_values.sort()

    return {
        "point_difference": point_difference,
        "ci95_difference": intervals,
        "point_mean_daily_net_return_difference_bps": point_daily_bps,
        "ci95_mean_daily_net_return_difference_bps": [
            percentile(bootstrap_daily_values, 0.025),
            percentile(bootstrap_daily_values, 0.975),
        ],
        "daily_win_fraction": (
            sum(value > 0.0 for value in daily_differences_bps)
            / len(daily_differences_bps)
        ),
        "bootstrap_positive_fraction": (
            sum(value > 0.0 for value in bootstrap_daily_values)
            / len(bootstrap_daily_values)
        ),
        "bootstrap_count": bootstrap_count,
        "circular_block_days": block_days,
        "seed": seed,
    }


def action_diagnostics(baseline, treatment):
    target_key = "desired_exposure"
    changes = [
        treatment_row[target_key]
        - base_row.get(target_key, base_row["core_desired_exposure"])
        for base_row, treatment_row in zip(baseline, treatment)
    ]
    changed = [value for value in changes if abs(value) > 1e-12]
    return {
        "decision_count": len(changes),
        "target_change_fraction": len(changed) / len(changes),
        "mean_absolute_target_change": statistics.mean(
            abs(value) for value in changes
        ),
        "mean_target_change": statistics.mean(changes),
        "positive_target_change_fraction": (
            sum(value > 1e-12 for value in changes) / len(changes)
        ),
        "negative_target_change_fraction": (
            sum(value < -1e-12 for value in changes) / len(changes)
        ),
    }


def make_comparison(name, baseline, treatment, seed, bootstrap, block_days):
    baseline_metrics = calculate_metrics(baseline)
    treatment_metrics = calculate_metrics(treatment)
    return {
        "comparison": name,
        "baseline_metrics": baseline_metrics,
        "treatment_metrics": treatment_metrics,
        "effect": bootstrap_difference(
            baseline,
            treatment,
            bootstrap,
            block_days,
            seed,
        ),
        "action_diagnostics": action_diagnostics(baseline, treatment),
    }


def main():
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    if args.block_days < 1:
        raise ValueError("block-days must be positive")

    numerical_path = os.path.abspath(args.numerical_core)
    payload, core_all_rows = read_numerical_core(numerical_path)
    core_validation = validate_numerical_only(payload, core_all_rows)
    core_all = core_records(core_all_rows)

    trace_sources = {
        "adaptive": os.path.abspath(args.adaptive),
        "fixed": os.path.abspath(args.fixed),
    }

    source_results = {}
    all_comparisons = []

    for source_index, (source_name, archive_path) in enumerate(
        trace_sources.items()
    ):
        member_name, trace_rows = read_trace_archive(archive_path)
        matched = match_window(
            core_all_rows,
            trace_rows,
            args.start_date,
            args.end_date,
            source_name,
        )

        core_window = {
            state: [
                record
                for record in core_all[state]
                if args.start_date <= record["return_date"] <= args.end_date
            ]
            for state in ("controller_on", "controller_off")
        }

        memory, memory_audit = trace_records(matched, "memory")
        no_memory, no_memory_audit = trace_records(matched, "no_memory")

        treatments = {
            "llama_no_memory_vs_numerical_only": (
                "pure_llama_advisor_effect",
                no_memory,
                core_window,
            ),
            "llama_memory_vs_numerical_only": (
                "total_llama_plus_memory_effect",
                memory,
                core_window,
            ),
            "llama_memory_vs_llama_no_memory": (
                "incremental_memory_effect",
                memory,
                no_memory,
            ),
        }

        source_comparisons = {}
        for comparison_key, (interpretation, treatment, baselines) in treatments.items():
            source_comparisons[comparison_key] = {
                "interpretation": interpretation,
                "controller_on": make_comparison(
                    comparison_key,
                    baselines["controller_on"],
                    treatment["controller_on"],
                    args.seed + source_index * 1000,
                    args.bootstrap,
                    args.block_days,
                ),
                "controller_off": make_comparison(
                    comparison_key,
                    baselines["controller_off"],
                    treatment["controller_off"],
                    args.seed + source_index * 1000 + 1,
                    args.bootstrap,
                    args.block_days,
                ),
            }
            all_comparisons.append(source_comparisons[comparison_key])

        core_alignment = {
            "available": False,
            "max_no_memory_core_target_difference": None,
            "max_memory_core_target_difference": None,
        }
        if "no_memory_core_desired_exposure" in trace_rows[0]:
            core_alignment["available"] = True
            no_memory_diffs = []
            memory_diffs = []
            for core_row, trace_row in matched:
                core_target = to_float(core_row, "core_desired_exposure")
                no_memory_diffs.append(
                    abs(
                        to_float(
                            trace_row,
                            "no_memory_core_desired_exposure",
                        )
                        - core_target
                    )
                )
                memory_diffs.append(
                    abs(
                        to_float(
                            trace_row,
                            "memory_core_desired_exposure",
                        )
                        - core_target
                    )
                )
            core_alignment["max_no_memory_core_target_difference"] = max(
                no_memory_diffs
            )
            core_alignment["max_memory_core_target_difference"] = max(
                memory_diffs
            )

        source_results[source_name] = {
            "archive": archive_path,
            "trace_member": member_name,
            "date_count": len(matched),
            "start_return_date": matched[0][0]["return_date"],
            "end_return_date": matched[-1][0]["return_date"],
            "trace_audits": {
                "memory": memory_audit,
                "no_memory": no_memory_audit,
            },
            "numerical_core_alignment": core_alignment,
            "comparisons": source_comparisons,
        }

    result = {
        "experiment": "RAMAS_pure_LLaMA_advisor_memory_controller_ablation",
        "read_only": True,
        "llama_calls": 0,
        "definition": {
            "N": "numerical-only: advisor=ABSTAIN, beta=0, memory=none",
            "L": "LLaMA advisor with no retrieved memory",
            "LM": "LLaMA advisor with retrieved memory",
            "pure_llama_effect": "L - N",
            "incremental_memory_effect": "LM - L",
            "total_advisor_path_effect": "LM - N",
        },
        "inputs": {
            "numerical_core_json": numerical_path,
            "adaptive_archive": trace_sources["adaptive"],
            "fixed_archive": trace_sources["fixed"],
            "cost_bps": 10,
        },
        "evaluation_window": {
            "return_date_start": args.start_date,
            "return_date_end": args.end_date,
            "date_count": source_results["adaptive"]["date_count"],
        },
        "bootstrap_contract": {
            "resamples": args.bootstrap,
            "circular_block_days": args.block_days,
            "primary_effect": "paired mean daily net-return difference in bps/day",
            "secondary_effects": [
                "net return percentage points",
                "annualized volatility percentage points",
                "Sharpe difference",
                "maximum drawdown percentage points",
                "ES95 percentage points",
                "total turnover difference",
                "mean exposure percentage points",
            ],
            "seed": args.seed,
        },
        "numerical_only_validation": core_validation,
        "sources": source_results,
        "interpretation_warning": (
            "The pure advisor effect is the no-memory LLaMA path minus "
            "numerical_only. The memory increment is memory LLaMA minus "
            "no-memory LLaMA. Do not report LLaMA value from the memory path "
            "alone. Controller-OFF returns are recomputed from desired exposure "
            "and the locked 10-bps turnover cost."
        ),
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)

    print(f"CREATED={os.path.abspath(args.output)}")
    print(f"DATE_COUNT={result['evaluation_window']['date_count']}")
    print(f"START_DATE={args.start_date}")
    print(f"END_DATE={args.end_date}")
    print("LLAMA_CALLS=0")
    for source_name, source in source_results.items():
        print(
            f"SOURCE={source_name} "
            f"TRACE_MEMBER={source['trace_member']} "
            f"ROWS={source['date_count']}"
        )


if __name__ == "__main__":
    main()
