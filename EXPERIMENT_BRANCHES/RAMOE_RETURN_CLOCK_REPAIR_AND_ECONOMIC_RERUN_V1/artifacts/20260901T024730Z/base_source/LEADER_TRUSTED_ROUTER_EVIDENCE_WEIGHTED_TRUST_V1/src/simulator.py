from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .accounting import drifted_exposure, net_return
from .core import ContractError, TOL, exposure_grid, row_stochastic, simplex
from .risk import project_exposure
from .router import masked_trust_matrix, route_or_fallback, validate_admission_mask
from .trust import risk_aware_trust_update


@dataclass(frozen=True)
class SimulationConfig:
    always_admitted_indices: tuple[int, ...]
    transaction_cost_rate: float
    cvar_alpha: float
    cvar_limit: float
    ambiguity_quantile: float
    maximum_turnover: float
    exposure_grid_step: float
    trust_eta: float
    trust_tail_penalty: float
    trust_turnover_penalty: float
    trust_minimum_posterior_mass: float
    trust_minimum_effective_sample_size: float


@dataclass(frozen=True)
class SimulationResult:
    trace: pd.DataFrame
    terminal_trust: np.ndarray
    trust_updates: int
    trust_rows_updated: int
    trust_update_audit: pd.DataFrame


def _validate_dates(decision_dates: pd.DatetimeIndex, return_dates: pd.DatetimeIndex) -> None:
    if len(decision_dates) == 0 or len(decision_dates) != len(return_dates):
        raise ContractError("Decision and return dates must have equal nonzero length")
    if decision_dates.hasnans or return_dates.hasnans:
        raise ContractError("Dates may not be missing")
    if not decision_dates.is_monotonic_increasing or decision_dates.has_duplicates:
        raise ContractError("Decision dates must be strictly increasing")
    if not return_dates.is_monotonic_increasing or return_dates.has_duplicates:
        raise ContractError("Return dates must be strictly increasing")
    if not np.all(return_dates.values > decision_dates.values):
        raise ContractError("Every realized return must occur after its decision")


def _nearest_turnover_exposure(
    desired: float,
    pretrade: float,
    maximum_turnover: float,
    grid_step: float,
) -> float:
    grid = exposure_grid(grid_step)
    candidates = grid[np.abs(grid - pretrade) <= maximum_turnover + TOL]
    if len(candidates) == 0:
        raise ContractError("Turnover-only projection has no feasible exposure")
    distance = np.abs(candidates - desired)
    nearest = candidates[np.isclose(distance, np.min(distance), atol=1e-14)]
    return float(np.min(nearest))


def simulate(
    *,
    decision_dates: pd.DatetimeIndex,
    return_dates: pd.DatetimeIndex,
    router_posteriors: np.ndarray,
    expert_exposures: np.ndarray,
    asset_simple_returns: np.ndarray,
    scenario_log_returns: np.ndarray,
    initial_trust: np.ndarray,
    admission_mask: np.ndarray,
    fallback_weights: np.ndarray,
    config: SimulationConfig,
    enable_admission: bool = True,
    enable_trust_updates: bool = True,
    enable_risk_projection: bool = True,
) -> SimulationResult:
    decision_dates = pd.DatetimeIndex(decision_dates)
    return_dates = pd.DatetimeIndex(return_dates)
    _validate_dates(decision_dates, return_dates)
    q = np.asarray(router_posteriors, dtype=float)
    exposures = np.asarray(expert_exposures, dtype=float)
    returns = np.asarray(asset_simple_returns, dtype=float)
    scenarios = np.asarray(scenario_log_returns, dtype=float)
    trust = row_stochastic(initial_trust, "initial trust")
    rows = len(decision_dates)
    if q.ndim != 2 or exposures.ndim != 2 or len(q) != rows or len(exposures) != rows:
        raise ContractError("Router and expert arrays must align with dates")
    if returns.shape != (rows,) or scenarios.ndim != 3 or scenarios.shape[0] != rows:
        raise ContractError("Return and scenario arrays must align with dates")
    if trust.shape != (q.shape[1], exposures.shape[1]):
        raise ContractError("Initial trust dimensions do not match router and experts")
    if not np.isfinite(exposures).all() or not np.isfinite(returns).all() or not np.isfinite(scenarios).all():
        raise ContractError("Exposure, return, and scenario arrays must be finite")
    if ((exposures < -TOL) | (exposures > 1.0 + TOL)).any() or (returns <= -1.0).any():
        raise ContractError("Long-only exposure or return contract failed")

    frozen_mask = validate_admission_mask(
        admission_mask,
        expert_count=exposures.shape[1],
        always_admitted_indices=config.always_admitted_indices,
    )
    active_mask = frozen_mask if enable_admission else np.ones(exposures.shape[1], dtype=bool)
    trust = masked_trust_matrix(
        trust,
        active_mask,
        always_admitted_indices=config.always_admitted_indices,
    )
    fallback = simplex(fallback_weights, "fallback weights")
    if (fallback[~active_mask] > TOL).any():
        raise ContractError("Fallback allocates to a rejected expert")

    expert_pretrade = np.zeros(exposures.shape[1], dtype=float)
    portfolio_pretrade = 0.0
    expert_period_returns: list[np.ndarray] = []
    expert_period_turnover: list[np.ndarray] = []
    q_period: list[np.ndarray] = []
    period = return_dates[0].to_period("M")
    trust_updates = 0
    trust_rows_updated = 0
    trust_audit_rows: list[dict[str, object]] = []
    output_rows: list[dict[str, object]] = []

    def update_from_completed_period() -> None:
        nonlocal trust, trust_updates, trust_rows_updated
        if not enable_trust_updates or len(q_period) < 2:
            return
        result = risk_aware_trust_update(
            trust,
            np.vstack(q_period),
            np.vstack(expert_period_returns),
            np.vstack(expert_period_turnover),
            active_mask,
            always_admitted_indices=config.always_admitted_indices,
            eta=config.trust_eta,
            cvar_alpha=config.cvar_alpha,
            tail_penalty=config.trust_tail_penalty,
            turnover_penalty=config.trust_turnover_penalty,
            minimum_posterior_mass=config.trust_minimum_posterior_mass,
            minimum_effective_sample_size=config.trust_minimum_effective_sample_size,
        )
        for regime in range(q.shape[1]):
            for expert in range(exposures.shape[1]):
                trust_audit_rows.append(
                    {
                        "completed_return_month": str(period),
                        "regime_index": int(regime),
                        "expert_index": int(expert),
                        "posterior_mass": float(result.posterior_mass[regime]),
                        "effective_sample_size": float(result.effective_sample_size[regime]),
                        "support_passed": bool(result.support_mask[regime]),
                        "growth_component": float(result.growth_component[regime, expert]),
                        "tail_component": float(result.tail_component[regime, expert]),
                        "turnover_component": float(result.turnover_component[regime, expert]),
                        "score": float(result.score_matrix[regime, expert]),
                        "trust_before": float(trust[regime, expert]),
                        "trust_after": float(result.trust_matrix[regime, expert]),
                    }
                )
        trust = result.trust_matrix
        trust_updates += 1
        trust_rows_updated += int(np.sum(result.support_mask))

    for index in range(rows):
        current_period = return_dates[index].to_period("M")
        if current_period != period:
            update_from_completed_period()
            expert_period_returns.clear()
            expert_period_turnover.clear()
            q_period.clear()
            period = current_period

        router_valid = (
            np.isfinite(q[index]).all()
            and (q[index] >= -TOL).all()
            and np.isclose(q[index].sum(), 1.0, atol=1e-10)
        )
        decision = route_or_fallback(
            q[index] if router_valid else None,
            trust,
            active_mask,
            exposures[index],
            fallback,
            always_admitted_indices=config.always_admitted_indices,
        )
        if enable_risk_projection:
            projection = project_exposure(
                desired_exposure=decision.desired_exposure,
                drifted_pretrade_exposure=portfolio_pretrade,
                scenario_log_returns=scenarios[index],
                transaction_cost_rate=config.transaction_cost_rate,
                cvar_alpha=config.cvar_alpha,
                cvar_limit=config.cvar_limit,
                ambiguity_quantile=config.ambiguity_quantile,
                maximum_turnover=config.maximum_turnover,
                exposure_grid_step=config.exposure_grid_step,
            )
            portfolio_exposure = projection.exposure
            ambiguity_cvar = projection.ambiguity_cvar
            risk_fallback = projection.fallback_used
        else:
            portfolio_exposure = _nearest_turnover_exposure(
                decision.desired_exposure,
                portfolio_pretrade,
                config.maximum_turnover,
                config.exposure_grid_step,
            )
            ambiguity_cvar = np.nan
            risk_fallback = False

        portfolio_turnover = abs(portfolio_exposure - portfolio_pretrade)
        portfolio_return = net_return(
            portfolio_exposure,
            portfolio_pretrade,
            returns[index],
            config.transaction_cost_rate,
        )
        individual_turnover = np.abs(exposures[index] - expert_pretrade)
        individual_returns = np.asarray(
            [
                net_return(
                    exposures[index, expert],
                    expert_pretrade[expert],
                    returns[index],
                    config.transaction_cost_rate,
                )
                for expert in range(exposures.shape[1])
            ],
            dtype=float,
        )

        output: dict[str, object] = {
            "decision_date": decision_dates[index].strftime("%Y-%m-%d"),
            "return_date": return_dates[index].strftime("%Y-%m-%d"),
            "router_valid": bool(router_valid),
            "router_fallback_used": bool(decision.fallback_used),
            "risk_fallback_used": bool(risk_fallback),
            "desired_exposure": float(decision.desired_exposure),
            "final_exposure": float(portfolio_exposure),
            "pretrade_exposure": float(portfolio_pretrade),
            "turnover": float(portfolio_turnover),
            "ambiguity_cvar": float(ambiguity_cvar),
            "asset_simple_return": float(returns[index]),
            "portfolio_net_return": float(portfolio_return),
        }
        for expert, weight in enumerate(decision.expert_weights):
            output[f"weight_{expert}"] = float(weight)
        output_rows.append(output)

        if router_valid:
            q_period.append(np.asarray(q[index], dtype=float).copy())
            expert_period_returns.append(individual_returns.copy())
            expert_period_turnover.append(individual_turnover.copy())

        portfolio_pretrade = drifted_exposure(portfolio_exposure, returns[index])
        expert_pretrade = np.asarray(
            [drifted_exposure(exposures[index, expert], returns[index]) for expert in range(exposures.shape[1])],
            dtype=float,
        )

    # The last period is now completed. This terminal update is recorded but cannot
    # affect any exposure inside the evaluated trace.
    update_from_completed_period()
    trace = pd.DataFrame(output_rows)
    if not np.isfinite(trace.drop(columns=["ambiguity_cvar"]).select_dtypes(include=[np.number])).all().all():
        raise ContractError("Simulation produced non-finite output")
    if ((trace["final_exposure"] < -TOL) | (trace["final_exposure"] > 1.0 + TOL)).any():
        raise ContractError("Simulation produced invalid exposure")
    if (trace["turnover"] > config.maximum_turnover + TOL).any():
        raise ContractError("Simulation violated the turnover limit")
    return SimulationResult(
        trace,
        trust,
        trust_updates,
        trust_rows_updated,
        pd.DataFrame(trust_audit_rows),
    )
