from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MechanicalFixture:
    decision_dates: pd.DatetimeIndex
    return_dates: pd.DatetimeIndex
    router_posteriors: np.ndarray
    expert_exposures: np.ndarray
    asset_simple_returns: np.ndarray
    scenario_log_returns: np.ndarray
    initial_trust: np.ndarray


def build_fixture(*, seed: int, rows: int, scenario_models: int, scenario_samples: int) -> MechanicalFixture:
    if rows < 90 or scenario_models < 2 or scenario_samples < 20:
        raise ValueError("Mechanical fixture is too small")
    rng = np.random.default_rng(seed)
    prehistory = rng.normal(0.0003, 0.018, size=120)
    states = np.arange(rows) % 45
    bear = states < 12
    bull = (states >= 12) & (states < 30)
    mix = ~(bear | bull)
    conditional_mean = np.where(bear, -0.0040, np.where(bull, 0.0045, 0.0003))
    conditional_scale = np.where(bear, 0.028, np.where(bull, 0.018, 0.012))
    asset_returns = np.clip(
        conditional_mean + rng.normal(0.0, conditional_scale),
        -0.35,
        0.35,
    )

    q = np.empty((rows, 3), dtype=float)
    q[bear] = np.array([0.72, 0.10, 0.18])
    q[bull] = np.array([0.10, 0.72, 0.18])
    q[mix] = np.array([0.16, 0.18, 0.66])
    q = 0.96 * q + 0.04 * rng.dirichlet(np.ones(3), size=rows)
    q /= q.sum(axis=1, keepdims=True)

    complete_history = np.r_[prehistory, asset_returns]
    lagged = complete_history[119:119 + rows]
    trend = (lagged > 0.0).astype(float)
    volatility = np.empty(rows, dtype=float)
    drawdown = np.empty(rows, dtype=float)
    for index in range(rows):
        end = 120 + index
        past = complete_history[max(0, end - 30):end]
        realized = max(float(np.std(past, ddof=1)), 1e-6)
        volatility[index] = float(np.clip(0.012 / realized, 0.0, 1.0))
        wealth = np.cumprod(1.0 + past)
        running_peak = np.maximum.accumulate(wealth)
        current_drawdown = float(wealth[-1] / running_peak[-1] - 1.0)
        drawdown[index] = 0.20 if current_drawdown < -0.06 else 0.85
    atp = ((np.arange(rows) * 17 + 3) % 11 < 5).astype(float)
    expert_exposures = np.column_stack(
        [
            np.zeros(rows),
            np.ones(rows),
            trend,
            volatility,
            drawdown,
            atp,
        ]
    )

    scenarios = np.empty((rows, scenario_models, scenario_samples), dtype=float)
    quantile_positions = np.linspace(0.005, 0.995, scenario_samples)
    normal_draws = np.quantile(
        rng.normal(size=200000),
        quantile_positions,
    )
    for index in range(rows):
        end = 120 + index
        past = complete_history[max(0, end - 60):end]
        mean = float(np.mean(past))
        scale = max(float(np.std(past, ddof=1)), 0.005)
        for model in range(scenario_models):
            scale_multiplier = 0.85 + 0.12 * model
            mean_shift = (model - (scenario_models - 1) / 2.0) * scale * 0.04
            simple = np.clip(mean + mean_shift + scale * scale_multiplier * normal_draws, -0.85, 2.0)
            # Mechanical stress only: exercise the CVaR branch on stressed days
            # without forcing the risk cap to dominate the entire fixture.
            # These values are synthetic and are never market evidence.
            if bear[index] or index % 17 == 0:
                tail_count = max(1, scenario_samples // 20)
                simple[:tail_count] = np.minimum(
                    simple[:tail_count], -0.25 - 0.03 * model
                )
            scenarios[index, model] = np.log1p(simple)

    dates = pd.date_range("2020-01-01", periods=rows + 1, freq="D")
    initial_trust = np.full((3, 6), 1.0 / 6.0, dtype=float)
    return MechanicalFixture(
        decision_dates=dates[:-1],
        return_dates=dates[1:],
        router_posteriors=q,
        expert_exposures=expert_exposures,
        asset_simple_returns=asset_returns,
        scenario_log_returns=scenarios,
        initial_trust=initial_trust,
    )
