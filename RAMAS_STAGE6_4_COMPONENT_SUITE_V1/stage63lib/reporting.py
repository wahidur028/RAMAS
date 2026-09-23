"""Paired, retrospective attribution reports; never an economic stop gate.

The block bootstrap describes uncertainty conditional on two realized sequential
paths. It neither reruns learners under alternative markets nor proves causality
or future profitability. Dollar costs of model inference are not included.
"""
from datetime import date
from pathlib import Path
import csv
import json
import math

import numpy as np


PER_YEAR = 365.25
REGIMES = ("bull", "bear", "mix")


def _returns(values):
    r = np.asarray(values, dtype=float)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError("Returns must be a nonempty finite vector strictly above -1")
    return r


def tail_loss(values, alpha=0.05):
    """Average loss over exactly alpha times n observations, fractionally weighted."""
    r = _returns(values)
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    losses = np.sort(-r)[::-1]
    mass = len(r) * alpha
    whole = int(math.floor(mass))
    weighted = float(losses[:whole].sum())
    if whole < len(losses):
        weighted += (mass - whole) * float(losses[whole])
    return weighted / mass


def daily_metrics(values):
    r = _returns(values)
    sd = (0.0 if np.ptp(r) == 0 else float(np.std(r, ddof=1))) if len(r) > 1 else None
    return {
        "days": len(r),
        "arithmetic_mean_daily": float(r.mean()),
        "geometric_mean_daily": float(np.expm1(np.log1p(r).mean())),
        "daily_volatility": sd,
        "annualized_volatility": sd * math.sqrt(PER_YEAR) if sd is not None else None,
        "sharpe_zero_cash_rate": float(r.mean() / sd * math.sqrt(PER_YEAR)) if sd else None,
        "daily_loss_cvar95": tail_loss(r),
        "positive_day_fraction": float(np.mean(r > 0)),
        "log_growth_contribution": float(np.log1p(r).sum()),
    }


def chronological_metrics(values):
    r = _returns(values)
    wealth = np.exp(np.r_[0.0, np.cumsum(np.log1p(r))])
    metrics = daily_metrics(r)
    metrics.update({
        "cumulative_return": float(wealth[-1] - 1),
        "annualized_geometric_return": float(np.expm1(np.log1p(r).mean() * PER_YEAR)),
        "maximum_drawdown_loss": float(-np.min(wealth / np.maximum.accumulate(wealth) - 1)),
    })
    return metrics


def constant_rebalanced(asset_values, weight=0.45, cost_rate=0.001):
    asset = _returns(asset_values)
    if not 0 <= weight <= 1 or not 0 <= cost_rate < 1:
        raise ValueError("Weight must be in [0,1] and cost rate in [0,1)")
    prior = 0.0
    returns = []
    for r in asset:
        returns.append((1 - cost_rate * abs(weight - prior)) * (1 + weight * r) - 1)
        prior = weight * (1 + r) / (1 + weight * r)
    return np.asarray(returns)


def baseline_streams(asset_values, cost_rate=0.001):
    asset = _returns(asset_values)
    if not 0 <= cost_rate < 1:
        raise ValueError("cost_rate must be in [0, 1)")
    bh = asset.copy()
    bh[0] = (1 - cost_rate) * (1 + bh[0]) - 1
    return {
        "CASH": np.zeros(len(asset)),
        "BTC_BUY_AND_HOLD": bh,
        "STATIC45_REBALANCED_SECONDARY": constant_rebalanced(asset, 0.45, cost_rate),
        "STATIC50_REBALANCED": constant_rebalanced(asset, 0.50, cost_rate),
    }


def paired_bootstrap(a_values, b_values, resamples=5000, block_days=30, seed=16062):
    """Two-sided paired circular-block percentile interval on mean daily log gap."""
    a, b = _returns(a_values), _returns(b_values)
    if len(a) != len(b):
        raise ValueError("Paired streams must have the same length")
    if not isinstance(resamples, int) or resamples < 1 or not isinstance(block_days, int) or block_days < 1:
        raise ValueError("Positive integer resamples and block_days required")
    if len(a) < 2:
        raise ValueError("At least two completed paired days are needed")
    rng = np.random.default_rng(seed)
    dlog = np.log1p(a) - np.log1p(b)
    means = np.empty(resamples)
    n = len(a)
    blocks = math.ceil(n / block_days)
    for i in range(resamples):
        starts = rng.integers(n, size=blocks)
        idx = ((starts[:, None] + np.arange(block_days)) % n).ravel()[:n]
        means[i] = float(dlog[idx].mean())
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    flag = "INCONCLUSIVE"
    if low > 0:
        flag = "MEMORY_DIRECTIONAL_ADVANTAGE"
    elif high < 0:
        flag = "MEMORY_DIRECTIONAL_DISADVANTAGE"
    return {
        "paired_days": n,
        "mean_daily_log_return_difference": float(dlog.mean()),
        "mean_daily_log_return_difference_bps": float(dlog.mean() * 10000),
        "two_sided_percentile95": [low, high],
        "two_sided_percentile95_bps": [low * 10000, high * 10000],
        "block_days": block_days,
        "bootstrap_resamples": resamples,
        "seed": seed,
        "directional_flag": flag,
        "interval_interpretation": "Retrospective path-conditional paired circular-block percentiles; not a learner resimulation or calibrated p-value",
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty table")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _truth(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError("Unrecognized boolean in trace: " + str(value))


def _pct(value):
    return "undefined" if value is None else f"{100 * value:+.2f}%"


def _num(value):
    return "undefined" if value is None else f"{value:.3f}"


def _ledger(trace_rows, streams, cost):
    """Retain the original continuous path; never reset at report boundaries."""
    details = []
    weights = {"CASH": 0., "BTC_BUY_AND_HOLD": 1.,
               "STATIC45_REBALANCED_SECONDARY": .45, "STATIC50_REBALANCED": .50}
    for strategy, returns in streams.items():
        wealth, prior = 1., 0.
        peak = 1.
        for row, net in zip(trace_rows, returns):
            prefix = strategy.lower()
            x = float(row[prefix + "_exposure"]) if strategy in ("MEMORY", "NO_MEMORY") else weights[strategy]
            p = float(row.get(prefix + "_pretrade_exposure", prior))
            turnover = float(row[prefix + "_turnover"]) if strategy in ("MEMORY", "NO_MEMORY") else abs(x - prior)
            if not all(math.isfinite(v) for v in (x, p, turnover)) or not (0 <= x <= 1 and 0 <= p <= 1 and 0 <= turnover <= 1):
                raise ValueError("Nonfinite or unbounded allocation/turnover in report input")
            before = wealth
            wealth *= 1 + float(net)
            peak = max(peak, wealth)
            asset = float(row["asset_simple_return"])
            details.append({"decision_date": str(row["decision_date"]),
                            "return_date": str(row["return_date"]), "strategy": strategy,
                            "forecast_regime": str(row["hard_regime"]).lower(),
                            "asset_simple_return": asset, "net_return": float(net),
                            "gross_return_before_cost": x * asset,
                            "pretrade_btc_fraction": p, "btc_fraction": x,
                            "cash_fraction": 1 - x, "turnover": turnover,
                            "cost_fraction_pretrade_wealth": cost * turnover,
                            "cost_paid_in_initial_wealth_units": before * cost * turnover,
                            "wealth_before": before, "wealth_after": wealth,
                            "running_peak_wealth": peak,
                            "drawdown_from_running_peak": 1 - wealth / peak})
            prior = x * (1 + asset) / (1 + x * asset)
    return details


def _allocation_summary(daily_rows):
    return {
        "opening_wealth_original_path": daily_rows[0]["wealth_before"],
        "closing_wealth_original_path": daily_rows[-1]["wealth_after"],
        "mean_btc_fraction": float(np.mean([r["btc_fraction"] for r in daily_rows])),
        "minimum_btc_fraction": min(r["btc_fraction"] for r in daily_rows),
        "maximum_btc_fraction": max(r["btc_fraction"] for r in daily_rows),
        "total_turnover": sum(r["turnover"] for r in daily_rows),
        "total_cost_in_initial_wealth_units": sum(r["cost_paid_in_initial_wealth_units"] for r in daily_rows),
    }


def _interaction(fixed_rows, adaptive_rows, config, controlled):
    """Four-cell log-growth interaction with joint chronological resampling.

    The adaptive cells are archived runs. This is a supplementary retrospective
    comparison; all four realized paths use the same sampled calendar indices.
    """
    result = {"status": "NOT_REQUESTED", "primary": False,
              "contrast": "(ADAPTIVE_MEMORY - ADAPTIVE_NO_MEMORY) - (FIXED_MEMORY - FIXED_NO_MEMORY)",
              "unit": "mean daily log return", "scope": "POST2021",
              "continuous_learning_demonstrated": False,
              "economic_stop_gate": False,
              "interpretation": "Supplementary four-cell comparison conditional on archived adaptive and fresh fixed-trust realized paths; not four simultaneous randomized trials or a learner resimulation."}
    if adaptive_rows is None:
        return result
    if len(fixed_rows) != len(adaptive_rows):
        raise ValueError("Adaptive reference must cover the same complete dates as the fixed-trust trace")
    for a, f in zip(adaptive_rows, fixed_rows):
        if (str(a["decision_date"]) != str(f["decision_date"]) or
                str(a["return_date"]) != str(f["return_date"]) or
                str(a["hard_regime"]).lower() != str(f["hard_regime"]).lower() or
                not math.isclose(float(a["asset_simple_return"]), float(f["asset_simple_return"]), rel_tol=0, abs_tol=1e-14)):
            raise ValueError("Adaptive reference exogenous dates, regimes or asset returns differ")
    mask = np.asarray([date.fromisoformat(str(r["return_date"])[:10]).year > 2021 for r in fixed_rows])
    result["paired_days"] = int(mask.sum())
    if mask.sum() < 2:
        result["status"] = "CONTROLLED_NON_ECONOMIC" if controlled else "INSUFFICIENT_POST2021_DAYS"
        return result
    cells = {}
    for label, rows in (("ADAPTIVE", adaptive_rows), ("FIXED", fixed_rows)):
        for arm in ("memory", "no_memory"):
            cells[label + "_" + arm.upper()] = _returns([r[arm + "_net_return"] for r in rows])[mask]
    log_cells = {k: np.log1p(v) for k, v in cells.items()}
    adaptive_gap = log_cells["ADAPTIVE_MEMORY"] - log_cells["ADAPTIVE_NO_MEMORY"]
    fixed_gap = log_cells["FIXED_MEMORY"] - log_cells["FIXED_NO_MEMORY"]
    d = adaptive_gap - fixed_gap
    analysis = config.get("analysis", {})
    block = int(analysis.get("block_days", 30))
    resamples = int(analysis.get("bootstrap_resamples", 5000))
    seed = int(analysis.get("seed", 16062))
    if block < 1 or resamples < 1:
        raise ValueError("Positive bootstrap block and resample counts required")
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples)
    for i in range(resamples):
        starts = rng.integers(len(d), size=math.ceil(len(d) / block))
        indices = ((starts[:, None] + np.arange(block)) % len(d)).ravel()[:len(d)]
        samples[i] = d[indices].mean()
    low, high = np.quantile(samples, [.025, .975])
    result.update({"status": "CONTROLLED_NON_ECONOMIC" if controlled else "SECONDARY_REUSED_OOS_DIAGNOSTIC",
                   "mean_daily_log_interaction": float(d.mean()),
                   "mean_daily_log_interaction_bps": float(d.mean() * 10000),
                   "two_sided_percentile95": [float(low), float(high)],
                   "two_sided_percentile95_bps": [float(low * 10000), float(high * 10000)],
                   "mean_daily_log_memory_gap_adaptive": float(adaptive_gap.mean()),
                   "mean_daily_log_memory_gap_fixed": float(fixed_gap.mean()),
                   "block_days": block, "bootstrap_resamples": resamples, "seed": seed,
                   "all_four_cells_share_sampled_indices": True,
                   "cell_metrics": {k: chronological_metrics(v) for k, v in cells.items()},
                   "reference_run_id": config.get("adaptive_reference_run_id"),
                   "reference_trace_sha256": config.get("expected_adaptive_trace_sha256"),
                   "interface_failures": {label + "_" + arm:
                       sum(not _truth(r[arm + "_valid"]) for r in rows)
                       for label, rows in (("adaptive", adaptive_rows), ("fixed", fixed_rows))
                       for arm in ("memory", "no_memory")}})
    return result


def _compute_diagnostics(output, rows):
    """Aggregate only each trace's committed trading-call journal, excluding preflight."""
    result = {"monetary_inference_cost_included_in_returns": False,
              "latency_caveat": "Observed per-call service time; not an exchange execution delay model.",
              "arms": {}}
    for arm in ("memory", "no_memory"):
        latencies, prompt_counts, output_counts = [], [], []
        found, missing_usage = 0, 0
        for i, row in enumerate(rows):
            path = output / "calls" / f"{i+1:06d}-{arm}.json"
            if path.exists():
                entry = json.loads(path.read_text())
                found += 1
                usage = entry.get("provider_response", {})
                for key, sink in (("prompt_eval_count", prompt_counts), ("eval_count", output_counts)):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                        sink.append(value)
                    else:
                        missing_usage += 1
            latency = row.get(arm + "_latency_seconds")
            if latency is not None and math.isfinite(float(latency)) and float(latency) >= 0:
                latencies.append(float(latency))
        result["arms"][arm] = {"expected_trading_calls": len(rows), "journal_calls_found": found,
            "prompt_usage_records": len(prompt_counts), "output_usage_records": len(output_counts),
            "prompt_tokens_observed": sum(prompt_counts) if prompt_counts else None,
            "output_tokens_observed": sum(output_counts) if output_counts else None,
            "usage_complete": found == len(rows) and missing_usage == 0,
            "latency_records": len(latencies), "latency_total_seconds": sum(latencies) if latencies else None,
            "latency_median_seconds": float(np.median(latencies)) if latencies else None,
            "latency_p95_seconds": float(np.quantile(latencies, .95)) if latencies else None}
    return result


def write_reports(output: Path, trace_rows: list[dict], config: dict, adaptive_rows=None):
    """Write scientific reports from the complete uninterrupted paired trace.

    No policy is changed or rerun here. Memory and no_memory streams are supplied
    by independent closed-loop agent paths; shared-action shadow traces are invalid
    inputs for this experiment, and the runtime must enforce that provenance.
    """
    if not trace_rows:
        raise ValueError("A completed trace is required")
    evidence_kind = config.get("evidence_kind", "RETROSPECTIVE_LLAMA70B")
    controlled = evidence_kind == "CONTROLLED_NON_ECONOMIC"
    if not controlled and config.get("expected_rows") is not None and len(trace_rows) != int(config["expected_rows"]):
        raise ValueError("The complete configured row count is required; no missing years or partial completion")
    for row in trace_rows:
        for arm in ("memory", "no_memory"):
            if not math.isclose(float(row[arm + "_beta"]), .05, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Every Stage 6.3 arm must keep fixed beta=0.05 on every day")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    returns_dates = [date.fromisoformat(str(x["return_date"])[:10]) for x in trace_rows]
    decision_dates = [date.fromisoformat(str(x["decision_date"])[:10]) for x in trace_rows]
    if any((r - d).days != 1 for d, r in zip(decision_dates, returns_dates)):
        raise ValueError("Every realized return must be exactly the next calendar day after its decision")
    if any((later - earlier).days != 1 for earlier, later in zip(returns_dates, returns_dates[1:])):
        raise ValueError("Return dates must be consecutive calendar days in increasing order")
    years = np.asarray([x.year for x in returns_dates])
    regimes = np.asarray([str(x["hard_regime"]).lower() for x in trace_rows])
    if set(regimes) - set(REGIMES):
        raise ValueError("Unknown router forecast regime")
    asset = _returns([x["asset_simple_return"] for x in trace_rows])
    cost = float(config.get("cost_rate", 0.001))
    streams = {
        "MEMORY": _returns([x["memory_net_return"] for x in trace_rows]),
        "NO_MEMORY": _returns([x["no_memory_net_return"] for x in trace_rows]),
        **baseline_streams(asset, cost),
    }
    daily = _ledger(trace_rows, streams, cost)
    by_strategy = {k: [r for r in daily if r["strategy"] == k] for k in streams}
    scopes = {"FULL": np.ones(len(trace_rows), dtype=bool)}
    if np.any(years > 2021):
        scopes["POST2021"] = years > 2021
    scopes.update({str(y): years == y for y in sorted(set(years))})
    overall, conditional, diagnostics = [], [], []
    for scope, mask in scopes.items():
        dates = np.asarray(returns_dates)[mask]
        for strategy, r in streams.items():
            overall.append({"scope": scope, "strategy": strategy,
                            "first_return_date": dates[0].isoformat(),
                            "last_return_date": dates[-1].isoformat(),
                            **chronological_metrics(r[mask]),
                            **_allocation_summary([d for d, keep in zip(by_strategy[strategy], mask) if keep])})
            for regime in REGIMES:
                selected = mask & (regimes == regime)
                if selected.any():
                    conditional.append({"scope": scope, "forecast_regime": regime,
                                        "strategy": strategy, **daily_metrics(r[selected])})
            partition = sum(np.log1p(r[mask & (regimes == g)]).sum() for g in REGIMES)
            if not np.isclose(partition, np.log1p(r[mask]).sum(), atol=1e-12, rtol=1e-12):
                raise ValueError("Regime log-growth attribution does not partition the stream")
        selections = {"ALL": mask}
        selections.update({g: mask & (regimes == g) for g in REGIMES})
        for group, selected in selections.items():
            if not selected.any():
                continue
            subset = [row for row, keep in zip(trace_rows, selected) if keep]
            diag = {"scope": scope, "forecast_regime": group, "days": len(subset),
                    "action_disagreement_days": sum(x["memory_action"] != x["no_memory_action"] for x in subset),
                    "exposure_disagreement_days": sum(abs(float(x["memory_exposure"]) - float(x["no_memory_exposure"])) > 1e-12 for x in subset),
                    "action_disagreement_same_exposure_days": sum(x["memory_action"] != x["no_memory_action"] and
                        abs(float(x["memory_exposure"]) - float(x["no_memory_exposure"])) <= 1e-12 for x in subset)}
            comparable = [r for r in subset if r.get("memory_desired_exposure") is not None and r.get("no_memory_desired_exposure") is not None]
            diag.update({
                "desired_exposure_comparable_days": len(comparable),
                "desired_exposure_disagreement_days": sum(abs(float(r["memory_desired_exposure"]) - float(r["no_memory_desired_exposure"])) > 1e-12 for r in comparable),
                "desired_disagreement_same_final_exposure_days": sum(
                    abs(float(r["memory_desired_exposure"]) - float(r["no_memory_desired_exposure"])) > 1e-12 and
                    abs(float(r["memory_exposure"]) - float(r["no_memory_exposure"])) <= 1e-12 for r in comparable),
                "net_return_disagreement_days": sum(abs(float(r["memory_net_return"]) - float(r["no_memory_net_return"])) > 1e-12 for r in subset),
            })
            for prefix in ("memory", "no_memory"):
                desired = [x for x in subset if x.get(prefix + "_desired_exposure") is not None]
                projection_deltas = [abs(float(x[prefix + "_desired_exposure"]) -
                                         float(x[prefix + "_exposure"])) for x in desired]
                diag.update({
                    prefix + "_valid_days": sum(_truth(x[prefix + "_valid"]) for x in subset),
                    prefix + "_invalid_fallback_days": sum(not _truth(x[prefix + "_valid"]) for x in subset),
                    prefix + "_mean_exposure": float(np.mean([float(x[prefix + "_exposure"]) for x in subset])),
                    prefix + "_mean_beta": float(np.mean([float(x[prefix + "_beta"]) for x in subset])),
                    prefix + "_total_turnover": float(np.sum([float(x[prefix + "_turnover"]) for x in subset])),
                    prefix + "_cited_memory_count": sum(int(x[prefix + "_cited_memory_count"]) for x in subset),
                    prefix + "_desired_exposure_recorded_days": len(desired),
                    prefix + "_projection_modified_desired_exposure_days": sum(x > 1e-12 for x in projection_deltas) if desired else None,
                    prefix + "_mean_absolute_projection_change": float(np.mean(projection_deltas)) if desired else None,
                    prefix + "_action_headroom_recorded_days": sum(x.get(prefix + "_action_headroom") is not None for x in subset),
                    prefix + "_action_headroom_days": sum(_truth(x[prefix + "_action_headroom"]) for x in subset if x.get(prefix + "_action_headroom") is not None),
                    prefix + "_chosen_differs_from_own_abstain_exposure_days": sum(
                        abs(float(x[prefix + "_exposure"]) - float(x[prefix + "_preview_abstain_exposure"])) > 1e-12
                        for x in subset if x.get(prefix + "_preview_abstain_exposure") is not None),
                    prefix + "_retrieved_cross_year_episode_links": sum(int(x.get(prefix + "_retrieved_memory_cross_year_count", 0)) for x in subset),
                    prefix + "_retrieved_cross_year_days": sum(int(x.get(prefix + "_retrieved_memory_cross_year_count", 0)) > 0 for x in subset),
                })
                for action in ("BTC", "CASH", "ABSTAIN"):
                    diag[prefix + "_" + action.lower() + "_days"] = sum(x[prefix + "_action"] == action for x in subset)
            diagnostics.append(diag)
    analysis = config.get("analysis", {})
    post = years > 2021
    invalid_counts = {prefix + "_invalid_fallback_days_full":
                      sum(not _truth(x[prefix + "_valid"]) for x in trace_rows)
                      for prefix in ("memory", "no_memory")}
    invalid_counts.update({prefix + "_invalid_fallback_days_post2021":
                           sum(not _truth(x[prefix + "_valid"]) for x, keep in zip(trace_rows, post) if keep)
                           for prefix in ("memory", "no_memory")})
    any_invalid = any(invalid_counts.values())
    primary = {"comparison": "MEMORY minus NO_MEMORY",
               "scope": "POST2021", "burn_in_year": 2021,
               "agent_trust_mode": "FIXED_POSITIVE", "fixed_beta": .05,
               "evidence_kind": evidence_kind,
               "data_label": "CONTROLLED_NON_ECONOMIC" if controlled else "REUSED_OOS_DIAGNOSTIC",
               "economic_stop_gate": False,
               "continuous_learning_demonstrated": False,
               "invalid_decisions_retained_in_return_paths": True,
               **invalid_counts}
    if post.sum() >= 2:
        primary.update(paired_bootstrap(streams["MEMORY"][post], streams["NO_MEMORY"][post],
                                       resamples=int(analysis.get("bootstrap_resamples", 5000)),
                                       block_days=int(analysis.get("block_days", 30)),
                                       seed=int(analysis.get("seed", 16062))))
        if controlled:
            primary["directional_flag"] = "CONTROLLED_NON_ECONOMIC"
        primary["evidence_flag"] = "CONTROLLED_NON_ECONOMIC" if controlled else primary["directional_flag"]
    else:
        primary["evidence_flag"] = "CONTROLLED_NON_ECONOMIC" if controlled else "NOT_EVALUATED_INSUFFICIENT_POST2021_DAYS"
    if any_invalid and not controlled:
        # Even a warm-up failure can change later state. Keep every day and the
        # raw directional contrast, but do not label it clean memory evidence.
        primary["evidence_flag"] = "INTERFACE_FAILURES_REQUIRE_INTERPRETATION"
    contract = {
        "analysis_id": "RAMAS_STAGE6_3_FIXED_TRUST_MEMORY_ABLATION_ANALYSIS_V1",
        "fixed_beta": 0.05,
        "trust_updated": False,
        "primary": "Post-2021 mean daily log return, MEMORY minus independently generated NO_MEMORY",
        "first_return_date": returns_dates[0].isoformat(),
        "last_return_date": returns_dates[-1].isoformat(),
        "total_days": len(trace_rows),
        "burn_in_days": int(np.sum(years == 2021)),
        "post2021_days": int(post.sum()),
        "evidence_kind": evidence_kind,
        "post2021_label": primary["data_label"],
        "cash_interest": 0,
        "cost_rate": cost,
        "annualization_days": PER_YEAR,
        "buy_and_hold_cost": "One entry at the original first date; no cost reset at annual, post-2021, or regime report boundaries; no terminal sale assumed",
        "static45": "Secondary 45% BTC/55% cash daily rebalanced exposure control, fixed before this matched rerun but selected from earlier Stage6.1 diagnostics; not independently discovered on untouched data",
        "static50": "50% BTC/50% cash rebalanced daily from drifted holdings, same trading costs, no annual reset",
        "regime_label": "Hard argmax of the archived router probabilities supplied to each decision; not ex-post true market condition. Forecast-target alignment remains a declared legacy limitation.",
        "regime_metrics": "Statistics on masked original daily returns; geometric daily growth and additive log growth; no concatenated regime drawdown or standalone regime-strategy cumulative return",
        "cash_sharpe": "Undefined when return volatility is zero",
        "cvar95": "Signed losses averaged over exactly 5% of observations using fractional tail mass; not clipped to zero",
        "trading_cost_convention": "Symmetric drifted-pretrade turnover, multiplicative wealth deduction; model inference and other operating costs excluded",
        "partial_years": [int(y) for y in sorted(set(years)) if
                          min(d for d in returns_dates if d.year == y) != date(int(y), 1, 1) or
                          max(d for d in returns_dates if d.year == y) != date(int(y), 12, 31)],
        "bootstrap": {"block_days": int(analysis.get("block_days", 30)),
                      "resamples": int(analysis.get("bootstrap_resamples", 5000)),
                      "seed": int(analysis.get("seed", 16062)),
                      "interval": "Two-sided 95% paired circular-block percentile, conditional on realized paths; no adaptive learner resimulation or calibrated p-value"},
        "interpretation": "A directional primary contrast is evidence about this specific memory-enabled system on this reused path. It does not prove monotonic learning, optimized decisions, causality independent of all inference randomness, or future profitability. Other metrics are descriptive and are not all required to improve.",
        "memory_ablation_scope": "NO_MEMORY disables episode retrieval and its summary. Its own portfolio and outcome ledger still continue, but both arms have fixed beta=0.05; neither updates trust. This tests episodic retrieval under fixed positive agent influence, not model-weight training.",
        "interface_failures": "All realized invalid/fallback days remain in both paths. Any such failure, including during 2021 warm-up, overrides the economic evidence flag with INTERFACE_FAILURES_REQUIRE_INTERPRETATION while preserving raw estimates and counts. Transport failures pause the runtime for resume rather than creating fabricated realized-return days.",
        "projection_diagnostics": "Action disagreement with equal final exposure can arise from trust blending or the unchanged risk projection. Per-arm desired-versus-final exposure counts quantify projection changes when those fields are present; this is not causal attribution to one specific shield constraint.",
        "fixed_trust_interpretation": "Beta=0.05 controls blending before projection; it does not force a trade or guarantee 5 percentage points of allocation influence.",
        "source_limitations": config.get("scientific_scope", "LEGACY_MECHANISM_DIAGNOSTIC_NOT_REALISTIC_EXECUTION_OR_FRESH_OOS_CONFIRMATION"),
        "daily_monthly_yearly": "Daily ledger retains original normalized wealth and running peak. Monthly/yearly return and drawdown use each chronological reporting window; portfolio and fee accounting never reset. Return dates define months and years.",
        "secondary_interaction": "Four-cell contrast of archived adaptive-beta and fresh fixed-beta memory gaps on exactly matching exogenous dates; joint calendar-block resampling; secondary and path-conditional, not independent causal confirmation.",
        "regime_partition_verified": True,
        "policy_changes_in_reporting": False,
    }
    files = ["04_FULL_AND_YEARLY_METRICS.csv", "05_FORECAST_REGIME_METRICS.csv",
             "06_PAIRED_MEMORY_CONTRAST.json", "07_ACTION_AND_STATE_DIAGNOSTICS.csv",
             "08_RESEARCH_INTERPRETATION.md", "09_ANALYSIS_CONTRACT.json",
             "11_MONTHLY_METRICS.csv", "12_DAILY_METRICS.csv",
             "13_MEMORY_TRUST_INTERACTION.json", "14_COMPUTATIONAL_DIAGNOSTICS.json"]
    months = np.asarray([str(d)[:7] for d in returns_dates])
    monthly = []
    for month in sorted(set(months)):
        selected = months == month
        for strategy, returns in streams.items():
            ledger = [d for d, keep in zip(by_strategy[strategy], selected) if keep]
            monthly.append({"month": month, "strategy": strategy,
                            "first_return_date": ledger[0]["return_date"],
                            "last_return_date": ledger[-1]["return_date"],
                            **chronological_metrics(returns[selected]), **_allocation_summary(ledger)})
    interaction = _interaction(trace_rows, adaptive_rows, config, controlled)
    _write_csv(output / files[0], overall)
    _write_csv(output / files[1], conditional)
    _write_json(output / files[2], primary)
    _write_csv(output / files[3], diagnostics)
    _write_json(output / files[5], contract)
    _write_csv(output / files[6], monthly)
    _write_csv(output / files[7], daily)
    _write_json(output / files[8], interaction)
    _write_json(output / files[9], _compute_diagnostics(output, trace_rows))
    lines = ["# RAMAS Stage 6.3: matched memory experiment", "",
             f"Evidence flag: **{primary['evidence_flag']}**.", "",
             f"The trace covers {returns_dates[0]} through {returns_dates[-1]} ({len(trace_rows)} days). "
             "2021 is a warm-up year. The primary comparison uses completed returns after 2021, with each arm's state continuing across years.", ""]
    if controlled:
        lines += ["**This is a controlled engineering fixture. These numbers provide no evidence of Llama performance or economic value.**", ""]
    else:
        lines += ["All post-2021 results are **reused historical diagnostics**, not a fresh untouched out-of-sample test. Any incomplete calendar year is reported only through the last available date.", ""]
    lines += ["Both arms keep agent trust fixed at **beta=0.05** on every day and regime. The same Llama model generates each arm's own fresh decisions. "
              "Only the memory arm receives retrieved completed experiences and their summary. Model weights do not change. "
              "Fixed trust does not bypass allocation constraints or guarantee different trades.", "",
              "The shared legacy forecast-target and simulated execution assumptions remain limitations. Correct replay and arithmetic alone do not establish realistic executable performance.", ""]
    if any_invalid:
        lines += [f"The complete trace contains **{invalid_counts['memory_invalid_fallback_days_full']} memory-arm** and "
                  f"**{invalid_counts['no_memory_invalid_fallback_days_full']} no-memory-arm** invalid/fallback decisions. "
                  "These days remain in the returns. A failure during warm-up can also change later trust and portfolio state. "
                  "A positive numerical contrast therefore cannot automatically be treated as clean evidence of a memory benefit.", ""]
    if "mean_daily_log_return_difference" in primary:
        low, high = primary["two_sided_percentile95_bps"]
        lines += [f"After warm-up, the memory arm's average daily log return minus the no-memory arm's is "
                  f"**{primary['mean_daily_log_return_difference_bps']:+.4f} basis points/day**. "
                  f"The two-sided 95% paired block interval is **[{low:+.4f}, {high:+.4f}]** basis points/day (1 basis point = 0.01%).", ""]
        if low <= 0 <= high:
            lines += ["The interval includes zero. This run does not establish a reliable directional benefit from memory under this diagnostic.", ""]
        elif not controlled and not any_invalid:
            lines += ["The interval is above zero, favoring memory on this realized path." if low > 0 else
                      "The interval is below zero, favoring the no-memory arm on this realized path.", ""]
    else:
        lines += ["There are too few post-2021 days for the primary comparison.", ""]
    lines += ["The interval resamples the two realized return paths together in calendar blocks. It does not rerun learners on alternative histories and is not a calibrated p-value. It cannot establish that more years of memory cause steadily better decisions.", "",
              "## Return and risk", "",
              "| Scope | Strategy | Days | Return | Annualized volatility | Largest drawdown | Sharpe | Worst-5%-day average loss |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in overall:
        lines.append(f"| {row['scope']} | {row['strategy']} | {row['days']} | {_pct(row['cumulative_return'])} | "
                     f"{_pct(row['annualized_volatility'])} | {_pct(row['maximum_drawdown_loss'])} | {_num(row['sharpe_zero_cash_rate'])} | {_pct(row['daily_loss_cvar95'])} |")
    lines += ["", "Cash has no assumed interest. Buy-and-hold buys once on the first original day. "
              "The static50 control rebalances daily to 50% BTC/50% cash. The 45% control remains an exploratory exposure diagnostic chosen from an earlier result. "
              "Report boundaries do not restart positions or charge new entry costs. Model-running costs are excluded.", "",
              "## Forecast market conditions", "",
              "| Scope | Forecast | Strategy | Days | Average daily growth | Daily volatility | Worst-5%-day average loss |",
              "|---|---|---|---:|---:|---:|---:|"]
    for row in conditional:
        if row["scope"] in ("FULL", "POST2021"):
            lines.append(f"| {row['scope']} | {row['forecast_regime']} | {row['strategy']} | {row['days']} | "
                         f"{row['geometric_mean_daily'] * 100:+.4f}% | {_pct(row['daily_volatility'])} | {_pct(row['daily_loss_cvar95'])} |")
    lines += ["", "The labels are the archived router forecasts supplied to the agents, not verified realized market regimes. The target-alignment limitation still applies. Average daily growth is geometric. "
              "Bull, Bear and Mix days are scattered through time; they are not separate investable strategies. "
              "Their daily returns retain actual trading costs. We therefore do not calculate a misleading drawdown after joining disconnected regime days. "
              "The CSV also contains year-by-regime breakdowns.", "",
              "## What this test can establish", "",
              "The primary contrast tests whether enabling retrieval of completed experience improves this system's realized post-warm-up log growth. "
              "Both arms generate their own actions and carry their own portfolios and outcome ledgers forward. Trust is fixed at beta=0.05 in both. "
              "The no-memory arm still owns an outcome ledger, but sees no retrieved episodes or summary. Neither arm updates trust or model weights. "
              "Consequently, this is a test of episodic retrieval's added value, not a comparison against a system with no learning of any kind. "
              "Memory citations, growing storage, and action disagreement show mechanisms or behavioral changes; they are not proof of useful learning. "
              "Return, drawdown, tail risk, and regime-specific trade-offs should all be reported. A weak metric does not erase a valid technical finding, and a positive metric does not prove general superiority.", "",
              "The diagnostics CSV counts invalid/fallback days, action disagreements that produce identical exposure, and each arm's desired-versus-final exposure changes. "
              "Identical final exposure can hide action differences because trust blending and the common risk projection limit actual trading changes. "
              "Transport interruptions are recorded by the runtime and pause the run for resume; they are not fabricated into trading observations.", "",
              "This report never stops the experiment for economic underperformance. The complete trace is retained for scientific interpretation. "
              "Additional independently controlled repetitions and a protocol fixed before new data would be needed for stronger generalization claims.", ""]
    lines += ["## Supplementary memory-by-trust comparison", "",
              "The optional archived adaptive-trust pair is used only as a secondary four-cell diagnostic. Its responses are never reused as decisions in either new fixed-trust arm.", ""]
    if "mean_daily_log_interaction_bps" in interaction:
        ci = interaction["two_sided_percentile95_bps"]
        lines += [f"Adaptive memory gap minus fixed memory gap: **{interaction['mean_daily_log_interaction_bps']:+.4f} basis points/day**, "
                  f"with a joint paired-block interval **[{ci[0]:+.4f}, {ci[1]:+.4f}]**. "
                  "All four cells use the same resampled calendar indices. This interaction does not by itself establish useful continuous learning.", ""]
    else:
        lines += ["An aligned archived adaptive reference was not supplied, or the fixture has insufficient post-2021 observations.", ""]
    lines += ["## Output detail", "",
              "11_MONTHLY_METRICS.csv reports every recorded month; 12_DAILY_METRICS.csv records each strategy's daily return, BTC/cash fractions, turnover, paid cost, original-path wealth and running drawdown. "
              "07_ACTION_AND_STATE_DIAGNOSTICS.csv tracks advice, desired exposure, final exposure, suppressed changes and earlier-year memory. "
              "14_COMPUTATIONAL_DIAGNOSTICS.json records observed input/output tokens and latency from committed trading calls, excluding preflight. Missing usage is marked unavailable, never zero-priced.", ""]
    (output / files[4]).write_text("\n".join(lines), encoding="utf-8")
    return {"evidence_flag": primary["evidence_flag"], "primary": primary,
            "analysis_files": files, "regime_partition_verified": True}
