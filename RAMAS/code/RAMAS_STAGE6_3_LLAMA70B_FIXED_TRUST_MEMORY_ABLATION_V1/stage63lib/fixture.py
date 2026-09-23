"""Synthetic integration fixture; never financial or economic evidence.

The accounting equations are genuine BTC/cash arithmetic. The risk projection is
an intentionally simple bounded test double, not the research risk model.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd


class FixtureAccounting:
    @staticmethod
    def net_return(exposure, pretrade, asset_return, cost_rate):
        return (1.0 - cost_rate * abs(exposure - pretrade)) * (1.0 + exposure * asset_return) - 1.0

    @staticmethod
    def drifted_exposure(exposure, asset_return):
        return exposure * (1.0 + asset_return) / (1.0 + exposure * asset_return)


class FixtureRisk:
    """A bounded projection double, with no claim to validate actual CVaR risk."""

    @staticmethod
    def project_exposure(desired_exposure, drifted_pretrade_exposure,
                         scenario_log_returns, **kwargs):
        grid = float(kwargs['exposure_grid_step'])
        exposure = float(np.clip(round(float(desired_exposure) / grid) * grid, 0.0, 1.0))
        turnover = abs(exposure - float(drifted_pretrade_exposure))
        # A diagnostic number only; the real risk projector is imported in real runs.
        worst = max(0.0, -float(np.min(np.expm1(scenario_log_returns))))
        return SimpleNamespace(exposure=exposure, turnover=turnover,
                               ambiguity_cvar=worst * exposure)


def build_fixture():
    """Return forty synthetic decision days crossing a calendar-year boundary."""
    decisions = pd.date_range('2021-12-10', periods=40, freq='D')
    returns = decisions + pd.Timedelta(days=1)
    regime_index = [1] * 12 + [0] * 12 + [2] * 8 + [1] * 8
    q = np.full((40, 3), 0.1, dtype=float)
    for i, g in enumerate(regime_index):
        q[i, g] = 0.8
    features = pd.DataFrame({
        'return_1': np.full(40, 0.006),
        'return_7': np.full(40, 0.04),
        'return_30': np.full(40, 0.10),
        'return_90': np.full(40, 0.15),
        'realized_vol_30': np.full(40, 0.45),
        'drawdown_90': np.full(40, -0.05),
        'router_entropy': -np.sum(q * np.log(q), axis=1) / np.log(3),
        'router_confidence': np.max(q, axis=1),
        'router_transition_l1': np.r_[0., np.abs(np.diff(q, axis=0)).sum(axis=1)],
    })
    asset_returns = 0.006 + 0.002 * np.sin(np.arange(40))
    base_config = {
        'transaction_cost_bps': 10., 'cvar_alpha': 0.95,
        'cvar_limit': 0.05, 'ambiguity_quantile': 0.95,
        'maximum_daily_turnover': 1., 'exposure_grid_step': 0.05,
    }
    accounting = FixtureAccounting()
    base_pretrade = 0.
    base_returns = []
    for r in asset_returns:
        base_returns.append(accounting.net_return(0.45, base_pretrade, float(r), 0.001))
        base_pretrade = accounting.drifted_exposure(0.45, float(r))
    frame = pd.DataFrame({'decision_date': decisions, 'return_date': returns})
    period = SimpleNamespace(
        name='CONTROLLED_SYNTHETIC_NON_ECONOMIC',
        label='SYNTHETIC_FIXTURE_NOT_MARKET_EVIDENCE',
        frame=frame, decision_dates=decisions, return_dates=returns,
        router_probabilities=q, asset_returns=asset_returns,
        scenarios=np.tile(np.linspace(-0.08, 0.08, 20), (40, 2, 1)),
        features=features,
        current_trace=pd.DataFrame({'desired_exposure': np.full(40, 0.45),
                                    'portfolio_net_return': base_returns}),
        source_audit={'synthetic': True, 'economic_evidence': False},
    )
    return period, base_config, accounting, FixtureRisk()
