from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import ContractError, TOL, exact_upper_tail_mean, exposure_grid


@dataclass(frozen=True)
class ProjectionResult:
    exposure: float
    ambiguity_cvar: float
    turnover: float
    fallback_used: bool


def project_exposure(
    *,
    desired_exposure: float,
    drifted_pretrade_exposure: float,
    scenario_log_returns: np.ndarray,
    transaction_cost_rate: float,
    cvar_alpha: float,
    cvar_limit: float,
    ambiguity_quantile: float,
    maximum_turnover: float,
    exposure_grid_step: float,
) -> ProjectionResult:
    scenarios = np.asarray(scenario_log_returns, dtype=float)
    if scenarios.ndim != 2 or min(scenarios.shape) < 2 or not np.isfinite(scenarios).all():
        raise ContractError("Scenarios must be a finite model-by-return matrix")
    if not np.isfinite([desired_exposure, drifted_pretrade_exposure]).all():
        raise ContractError("Exposure must be finite")
    if not 0.0 <= desired_exposure <= 1.0 or not 0.0 <= drifted_pretrade_exposure <= 1.0:
        raise ContractError("Exposure must lie inside [0,1]")
    if not 0.0 <= transaction_cost_rate < 1.0:
        raise ContractError("Invalid transaction cost")
    if not 0.0 < cvar_alpha < 1.0 or not 0.0 < ambiguity_quantile < 1.0:
        raise ContractError("Invalid risk quantile")
    if not np.isfinite([cvar_limit, maximum_turnover]).all():
        raise ContractError("Risk and turnover limits must be finite")
    if cvar_limit < 0.0 or not 0.0 <= maximum_turnover <= 1.0:
        raise ContractError("Invalid CVaR or turnover limit")

    grid = exposure_grid(exposure_grid_step)
    feasible_turnover = np.abs(grid - drifted_pretrade_exposure) <= maximum_turnover + TOL
    if not feasible_turnover.any():
        raise ContractError("Turnover set is empty")

    cvars = np.empty(len(grid), dtype=float)
    tail_probability = 1.0 - cvar_alpha
    for index, exposure in enumerate(grid):
        turnover = abs(exposure - drifted_pretrade_exposure)
        cost_multiplier = 1.0 - transaction_cost_rate * turnover
        wealth = cost_multiplier * ((1.0 - exposure) + exposure * np.exp(scenarios))
        losses = 1.0 - wealth
        model_cvars = np.asarray(
            [exact_upper_tail_mean(row, tail_probability) for row in losses],
            dtype=float,
        )
        cvars[index] = float(
            np.quantile(model_cvars, ambiguity_quantile, method="higher")
        )

    valid = feasible_turnover & (cvars <= cvar_limit + TOL)
    fallback_used = False
    candidates = np.flatnonzero(valid)
    if len(candidates) == 0:
        fallback_used = True
        feasible = np.flatnonzero(feasible_turnover)
        minimum_cvar = float(np.min(cvars[feasible]))
        candidates = feasible[np.isclose(cvars[feasible], minimum_cvar, atol=1e-14)]

    distance = np.abs(grid[candidates] - desired_exposure)
    nearest = candidates[np.isclose(distance, np.min(distance), atol=1e-14)]
    selected = int(nearest[np.argmin(grid[nearest])])
    result = ProjectionResult(
        exposure=float(grid[selected]),
        ambiguity_cvar=float(cvars[selected]),
        turnover=float(abs(grid[selected] - drifted_pretrade_exposure)),
        fallback_used=fallback_used,
    )
    if not 0.0 <= result.exposure <= 1.0 or result.turnover > maximum_turnover + TOL:
        raise ContractError("Projection violated an exposure or turnover invariant")
    return result

