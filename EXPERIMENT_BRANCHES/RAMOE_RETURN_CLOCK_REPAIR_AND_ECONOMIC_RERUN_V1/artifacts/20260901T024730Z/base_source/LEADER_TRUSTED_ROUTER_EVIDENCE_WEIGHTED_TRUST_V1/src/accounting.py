from __future__ import annotations

import numpy as np

from .core import ContractError, TOL


def net_return(
    exposure: float,
    drifted_pretrade_exposure: float,
    asset_simple_return: float,
    transaction_cost_rate: float,
) -> float:
    values = np.asarray(
        [exposure, drifted_pretrade_exposure, asset_simple_return, transaction_cost_rate],
        dtype=float,
    )
    if not np.isfinite(values).all():
        raise ContractError("Return accounting received non-finite input")
    if not 0.0 <= exposure <= 1.0 or not 0.0 <= drifted_pretrade_exposure <= 1.0:
        raise ContractError("Exposure must lie inside [0,1]")
    if not 0.0 <= transaction_cost_rate < 1.0 or asset_simple_return <= -1.0:
        raise ContractError("Invalid return or transaction cost")
    turnover = abs(exposure - drifted_pretrade_exposure)
    wealth_multiplier = (
        (1.0 - transaction_cost_rate * turnover)
        * (1.0 + exposure * asset_simple_return)
    )
    if wealth_multiplier <= TOL:
        raise ContractError("Portfolio wealth became nonpositive")
    return float(wealth_multiplier - 1.0)


def drifted_exposure(exposure: float, asset_simple_return: float) -> float:
    if not np.isfinite([exposure, asset_simple_return]).all():
        raise ContractError("Drift accounting received non-finite input")
    if not 0.0 <= exposure <= 1.0 or asset_simple_return <= -1.0:
        raise ContractError("Invalid exposure or asset return")
    denominator = 1.0 + exposure * asset_simple_return
    if denominator <= TOL:
        raise ContractError("Portfolio wealth became nonpositive")
    result = exposure * (1.0 + asset_simple_return) / denominator
    if not -TOL <= result <= 1.0 + TOL:
        raise ContractError("Drifted exposure escaped [0,1]")
    return float(np.clip(result, 0.0, 1.0))

