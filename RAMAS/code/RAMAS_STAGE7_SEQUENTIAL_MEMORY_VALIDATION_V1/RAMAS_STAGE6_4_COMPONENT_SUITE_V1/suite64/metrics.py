"""Independent, explicit financial accounting and descriptive reporting.

Every unqualified portfolio return in this module is AFTER TRADING COSTS.
Cost-before-return convention: r_net=(1-cost_fraction)*(1+x*r_BTC)-1,
where cost_fraction=c*abs(x-pretrade_exposure); cash earns zero interest.

Annualized volatility and Sharpe use 365.25 days/year and sample std (ddof=1),
matching the frozen Stage 6.3 reporting convention.
Sortino uses sqrt(mean(min(r_net, 0)**2)) over ALL observations, threshold 0.
ES95 is minus the exact fractional-weight mean of the worst 5% simple returns.
These are descriptive estimates; serial dependence is not repaired by sqrt(365.25).

Regime groups are noncontiguous selections. Their log contributions and daily
conditional moments are meaningful, but annualization and stitched drawdown are
not reported. No group operation resets the underlying recorded portfolio.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NET_RETURN = "net_return_after_trading_costs"
NET_LOG_RETURN = "net_log_return_after_trading_costs"
PER_YEAR = 365.25
REQUIRED_COLUMNS = (
    "arm", "decision_date", "return_date", "hard_regime",
    "asset_simple_return", "pretrade_exposure", "exposure", NET_RETURN,
    "turnover", "cost_fraction",
)


def atomic_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Publish only a complete, verified CSV, using a same-filesystem rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        check = temporary.read_bytes()
        if len(check) != len(payload) or hashlib.sha256(check).digest() != hashlib.sha256(payload).digest():
            raise OSError("CSV temporary write verification failed: " + str(path))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _finite_number(value):
    """JSON-safe scalar, with undefined ratios represented by null, never zero."""
    value = float(value)
    return value if math.isfinite(value) else None


def expected_shortfall_loss(returns: Iterable[float], confidence: float = 0.95) -> float:
    """Signed loss ES with exact fractional tail mass; gains can give negative ES."""
    r = np.asarray(list(returns), dtype=float)
    if r.size == 0 or not np.isfinite(r).all() or not 0 < confidence < 1:
        raise ValueError("ES requires finite nonempty returns and 0 < confidence < 1")
    tail_mass = len(r) * (1.0 - confidence)
    whole = int(math.floor(tail_mass))
    fractional = tail_mass - whole
    ordered = np.sort(r)
    total = float(ordered[:whole].sum())
    if fractional > 0:
        total += fractional * float(ordered[whole])
    return -total / tail_mass


def validate_ledger(ledger: pd.DataFrame, transaction_cost_rate: float | None = 0.001) -> pd.DataFrame:
    """Validate primitive accounting independently and return a sorted copy.

    None for transaction_cost_rate allows externally specified per-row rates,
    while still verifying turnover and the exact cost-before-return equation.
    This verifies arithmetic; it does not establish feature availability or fills.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(ledger.columns))
    if missing:
        raise ValueError(f"Missing daily ledger columns: {missing}")
    if ledger.empty:
        raise ValueError("Daily ledger is empty")
    df = ledger.copy()
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required ledger fields contain missing values")
    for column in ("decision_date", "return_date"):
        df[column] = pd.to_datetime(df[column], errors="raise", utc=True).dt.tz_convert(None).dt.normalize()
    if not (df["decision_date"] < df["return_date"]).all():
        raise ValueError("Each daily return must mature after its decision date")
    if df.duplicated(["arm", "return_date"]).any():
        raise ValueError("Duplicate arm/return_date entries")
    numeric = ["asset_simple_return", "pretrade_exposure", "exposure", NET_RETURN, "turnover", "cost_fraction"]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite {column}")
    if (df[NET_RETURN] <= -1).any() or (df["asset_simple_return"] <= -1).any():
        raise ValueError("Returns must exceed -100% for finite log wealth")
    for column in ("exposure", "pretrade_exposure"):
        if not df[column].between(0, 1).all():
            raise ValueError(f"{column} must lie within [0, 1]")
    if not df["cost_fraction"].between(0, 1, inclusive="left").all():
        raise ValueError("cost_fraction must lie within [0, 1)")
    expected_turnover = abs(df["exposure"] - df["pretrade_exposure"])
    if not np.allclose(df["turnover"], expected_turnover, rtol=0, atol=1e-11):
        raise ValueError("Turnover does not equal absolute change from drifted holdings")
    if transaction_cost_rate is not None:
        if not math.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
            raise ValueError("Invalid transaction_cost_rate")
        if not np.allclose(df["cost_fraction"], transaction_cost_rate * expected_turnover, rtol=0, atol=1e-11):
            raise ValueError("Trading cost fractions do not match declared rate")
    expected_return = (1.0 - df["cost_fraction"]) * (1.0 + df["exposure"] * df["asset_simple_return"]) - 1.0
    if not np.allclose(df[NET_RETURN], expected_return, rtol=0, atol=1e-11):
        raise ValueError("Net return violates cost-before-return accounting")
    df["arm"] = df["arm"].astype(str)
    df["hard_regime"] = df["hard_regime"].astype(str).str.upper()
    return df.sort_values(["arm", "return_date"]).reset_index(drop=True)


def prepare_daily_ledger(ledger: pd.DataFrame, transaction_cost_rate: float | None = 0.001) -> pd.DataFrame:
    """Add independently reconstructed wealth, drawdown, log returns and cost amounts."""
    df = validate_ledger(ledger, transaction_cost_rate)
    df["gross_return_before_trading_costs"] = df["exposure"] * df["asset_simple_return"]
    df[NET_LOG_RETURN] = np.log1p(df[NET_RETURN])
    df["net_log_return_bps"] = df[NET_LOG_RETURN] * 10000.0
    df["net_wealth_per_initial_unit"] = np.nan
    df["drawdown_from_running_peak"] = np.nan
    df["trading_cost_per_initial_unit"] = np.nan
    for _, part in df.groupby("arm", sort=False):
        wealth = np.exp(part[NET_LOG_RETURN].to_numpy().cumsum())
        previous_wealth = np.r_[1.0, wealth[:-1]]
        peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
        df.loc[part.index, "net_wealth_per_initial_unit"] = wealth
        df.loc[part.index, "drawdown_from_running_peak"] = 1.0 - wealth / peaks
        df.loc[part.index, "trading_cost_per_initial_unit"] = previous_wealth * part["cost_fraction"].to_numpy()
    return df


def _summary(part: pd.DataFrame, *, scope: str, period: str, conditional: bool,
             min_cagr_days: int = 365) -> dict:
    r = part[NET_RETURN].to_numpy(dtype=float)
    logs = np.log1p(r)
    n = len(r)
    mean = float(r.mean())
    std = float(r.std(ddof=1)) if n > 1 else float("nan")
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2)))
    sum_log = float(logs.sum())
    duration = int((part["return_date"].max() - part["decision_date"].min()).days)
    continuous_dates = n == duration and (part["return_date"] - part["decision_date"]).dt.days.eq(1).all()
    annualization_allowed = not conditional and bool(continuous_dates) and n > 1
    cagr = None
    if annualization_allowed and duration >= min_cagr_days:
        cagr = _finite_number(np.expm1(sum_log * PER_YEAR / duration))
    mdd = None
    if not conditional:
        log_path = np.r_[0.0, logs.cumsum()]
        mdd = float((-np.expm1(log_path - np.maximum.accumulate(log_path))).max())
    out = {
        "arm": str(part["arm"].iloc[0]), "scope": scope, "period": period,
        "observations": n,
        "first_return_date": part["return_date"].min().date().isoformat(),
        "last_return_date": part["return_date"].max().date().isoformat(),
        "calendar_days": duration,
        "is_conditional_noncontiguous_selection": conditional,
        "net_compounded_return_after_trading_costs": None if conditional else _finite_number(np.expm1(sum_log)),
        "cumulative_net_log_return_after_trading_costs": sum_log,
        "conditional_log_growth_contribution": sum_log if conditional else None,
        "mean_daily_net_log_return_bps": float(logs.mean() * 10000.0),
        "mean_daily_net_simple_return_after_trading_costs": mean,
        "daily_net_return_volatility": _finite_number(std),
        "annualized_net_return_volatility": _finite_number(std * math.sqrt(PER_YEAR)) if annualization_allowed else None,
        "net_cagr_after_trading_costs": cagr,
        "annualized_sharpe_rf_zero": _finite_number(mean / std * math.sqrt(PER_YEAR)) if annualization_allowed and std > 1e-15 else None,
        "downside_rms_zero_threshold": downside,
        "annualized_sortino_threshold_zero": _finite_number(mean / downside * math.sqrt(PER_YEAR)) if annualization_allowed and downside > 1e-15 else None,
        "maximum_drawdown": mdd,
        "calmar": cagr / mdd if cagr is not None and mdd is not None and mdd > 1e-15 else None,
        "expected_shortfall_95_daily_loss": expected_shortfall_loss(r),
        "turnover_sum": float(part["turnover"].sum()),
        "mean_daily_turnover": float(part["turnover"].mean()),
        "trading_days_with_nonzero_turnover": int((part["turnover"] > 1e-12).sum()),
        "cost_fraction_sum_not_compounded_drag": float(part["cost_fraction"].sum()),
        "mean_daily_cost_fraction": float(part["cost_fraction"].mean()),
        "mean_btc_exposure": float(part["exposure"].mean()),
        "annualization_status": "NOT_APPLICABLE_CONDITIONAL_SELECTION" if conditional else ("365_25_CALENDAR_DAYS_PER_YEAR" if annualization_allowed else "NOT_APPLICABLE_MISSING_OR_SINGLE_DAILY_OBSERVATION"),
        "cagr_status": "NOT_APPLICABLE_CONDITIONAL_SELECTION" if conditional else ("REPORTED" if cagr is not None else f"NOT_APPLICABLE_REQUIRES_{min_cagr_days}_CALENDAR_DAYS_AND_CONTIGUOUS_DAILY_RECORDS"),
        "drawdown_status": "NOT_APPLICABLE_CONDITIONAL_SELECTION" if conditional else "RESET_REPORTING_HIGH_WATER_MARK_TO_GROUP_INITIAL_WEALTH_NOT_PORTFOLIO_STATE",
    }
    if "trading_cost_per_initial_unit" in part:
        out["trading_cost_sum_per_full_run_initial_unit"] = float(part["trading_cost_per_initial_unit"].sum())
    return out


def summarize_ledger(ledger: pd.DataFrame, transaction_cost_rate: float | None = 0.001,
                     min_cagr_days: int = 365) -> pd.DataFrame:
    """Return FULL, POST2021, YEAR, MONTH, REGIME, YEAR_REGIME summaries.

    Regime outputs appear separately for FULL and POST2021 in their period labels.
    All ratios/returns are decimal fractions unless the column explicitly says bps.
    Year/month drawdowns start from that reporting interval's initial wealth;
    FULL/POST2021 drawdowns retain peaks across the years in their own interval.
    """
    df = prepare_daily_ledger(ledger, transaction_cost_rate)
    rows = []
    for _, arm in df.groupby("arm", sort=True):
        post = arm.loc[arm["return_date"] >= pd.Timestamp("2022-01-01")]
        for scope, label, part in (("FULL", "FULL", arm), ("POST2021", "POST2021", post)):
            if not part.empty:
                rows.append(_summary(part, scope=scope, period=label, conditional=False, min_cagr_days=min_cagr_days))
                for regime, group in part.groupby("hard_regime", sort=True):
                    rows.append(_summary(group, scope="REGIME", period=f"{label}:{regime}", conditional=True, min_cagr_days=min_cagr_days))
        for year, part in arm.groupby(arm["return_date"].dt.year, sort=True):
            rows.append(_summary(part, scope="YEAR", period=str(year), conditional=False, min_cagr_days=min_cagr_days))
            for regime, group in part.groupby("hard_regime", sort=True):
                rows.append(_summary(group, scope="YEAR_REGIME", period=f"{year}:{regime}", conditional=True, min_cagr_days=min_cagr_days))
        for month, part in arm.groupby(arm["return_date"].dt.to_period("M"), sort=True):
            rows.append(_summary(part, scope="MONTH", period=str(month), conditional=False, min_cagr_days=min_cagr_days))
    return pd.DataFrame(rows)


def paired_block_comparison(ledger: pd.DataFrame, arm_a: str, arm_b: str, *,
                            start_date: str | None = "2022-01-01", end_date: str | None = None,
                            block_length: int = 30, n_resamples: int = 5000, seed: int = 16062) -> dict:
    """Circular moving-block CI for paired mean daily log-return difference.

    Uses recorded trajectories and paired calendar days. It does NOT rerun memory,
    decisions, trust, or market responses inside bootstrap samples. Coverage is
    unadjusted; secondary/additional pairs are exploratory multiple comparisons.
    """
    if block_length <= 0 or n_resamples <= 0:
        raise ValueError("Block length and resample count must be positive")
    selected = ledger.loc[ledger["arm"].isin([arm_a, arm_b]), ["arm", "return_date", NET_RETURN]].copy()
    selected["return_date"] = pd.to_datetime(selected["return_date"], utc=True).dt.tz_convert(None).dt.normalize()
    if selected.duplicated(["arm", "return_date"]).any():
        raise ValueError("Paired comparison contains duplicate arm/date entries")
    if start_date is not None:
        selected = selected.loc[selected["return_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        selected = selected.loc[selected["return_date"] <= pd.Timestamp(end_date)]
    pivot = selected.pivot(index="return_date", columns="arm", values=NET_RETURN).sort_index()
    if arm_a == arm_b or arm_a not in pivot or arm_b not in pivot or pivot.empty or pivot.isna().any().any():
        raise ValueError("Paired comparison requires two distinct arms with exactly matched nonempty dates")
    values = pivot[[arm_a, arm_b]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Paired returns must be finite and greater than -100%")
    if len(pivot) > 1 and not (pivot.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Circular daily blocks require consecutive daily dates")
    delta = (np.log1p(values[:, 0]) - np.log1p(values[:, 1])) * 10000.0
    n = len(delta)
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n / block_length)
    estimates = np.empty(n_resamples)
    offsets = np.arange(block_length)
    for first in range(0, n_resamples, 128):
        count = min(128, n_resamples - first)
        starts = rng.integers(0, n, size=(count, blocks_needed))
        indices = ((starts[:, :, None] + offsets) % n).reshape(count, -1)[:, :n]
        estimates[first:first+count] = delta[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    point = float(delta.mean())
    wealth_a = float(np.exp(np.log1p(values[:, 0]).sum()))
    wealth_b = float(np.exp(np.log1p(values[:, 1]).sum()))
    direction = "A_HIGHER_ON_THIS_RESAMPLING_SPECIFICATION" if low > 0 else ("B_HIGHER_ON_THIS_RESAMPLING_SPECIFICATION" if high < 0 else "INCONCLUSIVE_NOT_EQUIVALENCE")
    return {
        "arm_a": arm_a, "arm_b": arm_b, "paired_days": n,
        "first_return_date": pivot.index.min().date().isoformat(),
        "last_return_date": pivot.index.max().date().isoformat(),
        "endpoint": "MEAN_DAILY_NET_LOG_RETURN_DIFFERENCE_BPS_A_MINUS_B",
        "mean_daily_net_log_difference_bps": point,
        "ci95_lower_bps": float(low), "ci95_upper_bps": float(high),
        "net_compounded_return_gap_percentage_points": (wealth_a - wealth_b) * 100.0,
        "relative_terminal_wealth_gain_a_over_b": wealth_a / wealth_b - 1.0,
        "identical_recorded_net_returns": bool(np.array_equal(values[:, 0], values[:, 1])),
        "interpretation": direction,
        "block_length": block_length, "n_resamples": n_resamples, "seed": seed,
        "bootstrap_method": "PAIRED_CIRCULAR_MOVING_BLOCK_PERCENTILE_CI",
        "coverage_note": "UNADJUSTED_95_PERCENT_CI; MULTIPLE_SECONDARY_PAIRS_EXPLORATORY",
        "learning_resimulated": False,
        "scope_note": "RECORDED_PATH_UNCERTAINTY; NOT MODEL_SEED_UNCERTAINTY_OR_FRESH_OUT_OF_SAMPLE_CONFIRMATION",
    }


def write_reports(ledger: pd.DataFrame, output_dir: str | Path, *,
                  transaction_cost_rate: float | None = 0.001,
                  comparisons: Iterable[tuple[str, str]] = (),
                  block_length: int = 30, n_resamples: int = 5000, seed: int = 16062,
                  min_cagr_days: int = 365) -> dict:
    """Write audit-ready daily/monthly/yearly/full/regime reports and comparisons."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    daily = prepare_daily_ledger(ledger, transaction_cost_rate)
    summaries = summarize_ledger(daily, transaction_cost_rate, min_cagr_days)
    atomic_csv(daily, output / "daily_ledger.csv")
    atomic_csv(summaries, output / "all_metrics.csv")
    scopes = {"monthly_metrics": ["MONTH"], "yearly_metrics": ["YEAR"],
              "full_period_metrics": ["FULL", "POST2021"],
              "regime_metrics": ["REGIME", "YEAR_REGIME"]}
    for name, scope_values in scopes.items():
        atomic_csv(summaries.loc[summaries["scope"].isin(scope_values)], output / f"{name}.csv")
    comparisons = [paired_block_comparison(daily, a, b, block_length=block_length,
                                          n_resamples=n_resamples, seed=seed) for a, b in comparisons]
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    conventions = {
        "schema_version": "1.0.0",
        "headline_return": "NET_COMPOUNDED_RETURN_AFTER_TRADING_COSTS",
        "units": "Returns, volatility, drawdown, ES and exposure are decimal fractions; bps and percentage-point columns explicitly named.",
        "net_simple_formula": "(1 - cost_fraction) * (1 + exposure * asset_simple_return) - 1",
        "net_log_formula": "log1p(net_return_after_trading_costs)",
        "cumulative_identity": "expm1(sum(net_log_return)) = product(1 + net_return) - 1",
        "cost_rate": transaction_cost_rate, "cash_interest_rate": 0.0, "risk_free_rate": 0.0,
        "annualization_days": PER_YEAR, "sample_volatility_ddof": 1,
        "cagr_minimum_calendar_days": min_cagr_days,
        "sortino_denominator": "sqrt(mean(min(net_simple_return, 0)^2)); uses ALL days, no ddof correction",
        "es95": "Signed loss; exact fractional-weight worst 5% daily simple-return tail; can be negative if tail gains",
        "undefined_ratios": "null/blank; cash Sharpe, Sortino and Calmar are undefined, not zero",
        "regime_scope": "Conditional, noncontiguous daily selections. No CAGR, annualized volatility/Sharpe/Sortino, drawdown, Calmar or tradable compounded portfolio return.",
        "period_boundaries": "Reporting interval high-water mark includes wealth=1 before first return; portfolios, trust and memory are NOT reset.",
        "cost_sum_warning": "Sum of daily cost fractions is descriptive turnover cost, not compounded terminal-wealth drag. Cost amounts per initial unit use actual full-run wealth.",
        "bootstrap": "Recorded-path paired circular blocks; not resimulation of learning or seed robustness; unadjusted exploratory CIs.",
    }
    (output / "metric_conventions.json").write_text(json.dumps(conventions, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"daily_rows": len(daily), "summary_rows": len(summaries),
            "arms": sorted(daily["arm"].unique().tolist()),
            "comparison_count": len(comparisons), "output_dir": str(output)}
