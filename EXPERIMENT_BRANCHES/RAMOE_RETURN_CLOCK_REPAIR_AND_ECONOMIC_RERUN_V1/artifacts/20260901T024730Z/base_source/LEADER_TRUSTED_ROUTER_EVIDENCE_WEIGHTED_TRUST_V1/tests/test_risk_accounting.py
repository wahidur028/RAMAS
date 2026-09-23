from __future__ import annotations

import unittest

import numpy as np

from src.accounting import drifted_exposure, net_return
from src.core import ContractError, exposure_grid
from src.risk import project_exposure


class RiskAndAccountingTests(unittest.TestCase):
    def test_grid_never_exceeds_one_for_nondivisor_step(self) -> None:
        grid = exposure_grid(0.3)
        self.assertLessEqual(float(grid.max()), 1.0)
        self.assertEqual(float(grid[-1]), 1.0)

    def test_transaction_cost_is_charged_once(self) -> None:
        observed = net_return(0.7, 0.2, 0.03, 0.001)
        expected = (1.0 - 0.001 * 0.5) * (1.0 + 0.7 * 0.03) - 1.0
        self.assertAlmostEqual(observed, expected)

    def test_drifted_exposure_identity(self) -> None:
        self.assertAlmostEqual(drifted_exposure(0.4, 0.10), 0.44 / 1.04)

    def test_projection_reduces_extreme_tail_risk(self) -> None:
        scenarios = np.full((10, 200), 0.01)
        scenarios[:, :20] = -0.35
        result = project_exposure(
            desired_exposure=1.0,
            drifted_pretrade_exposure=0.5,
            scenario_log_returns=scenarios,
            transaction_cost_rate=0.001,
            cvar_alpha=0.95,
            cvar_limit=0.045,
            ambiguity_quantile=0.9,
            maximum_turnover=0.5,
            exposure_grid_step=0.05,
        )
        self.assertLess(result.exposure, 1.0)
        self.assertLessEqual(result.ambiguity_cvar, 0.045 + 1e-12)

    def test_projection_respects_turnover(self) -> None:
        scenarios = np.zeros((3, 100))
        result = project_exposure(
            desired_exposure=1.0,
            drifted_pretrade_exposure=0.0,
            scenario_log_returns=scenarios,
            transaction_cost_rate=0.001,
            cvar_alpha=0.95,
            cvar_limit=1.0,
            ambiguity_quantile=0.9,
            maximum_turnover=0.2,
            exposure_grid_step=0.05,
        )
        self.assertLessEqual(result.exposure, 0.2 + 1e-12)

    def test_infeasible_cvar_falls_back_to_minimum_risk(self) -> None:
        scenarios = np.full((3, 100), -0.20)
        result = project_exposure(
            desired_exposure=1.0,
            drifted_pretrade_exposure=0.8,
            scenario_log_returns=scenarios,
            transaction_cost_rate=0.001,
            cvar_alpha=0.95,
            cvar_limit=0.0,
            ambiguity_quantile=0.9,
            maximum_turnover=0.1,
            exposure_grid_step=0.05,
        )
        self.assertTrue(result.fallback_used)
        self.assertAlmostEqual(result.exposure, 0.7)

    def test_invalid_return_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            net_return(0.5, 0.5, -1.0, 0.001)


if __name__ == "__main__":
    unittest.main()

