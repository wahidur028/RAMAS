from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.fixture import build_fixture
from src.simulator import SimulationConfig, simulate


ROOT = Path(__file__).resolve().parents[1]


class EndToEndSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        fixture_config = config["fixture"]
        self.fixture = build_fixture(
            seed=int(fixture_config["seed"]),
            rows=100,
            scenario_models=3,
            scenario_samples=60,
        )
        names = config["expert_names"]
        admission = config["mechanical_fixture_admission"]
        fallback = config["fallback_expert_weights"]
        self.mask = np.array([admission[name] for name in names], dtype=bool)
        self.fallback = np.array([fallback[name] for name in names], dtype=float)
        self.config = SimulationConfig(
            always_admitted_indices=(0, 1),
            transaction_cost_rate=0.001,
            cvar_alpha=0.95,
            cvar_limit=0.045,
            ambiguity_quantile=0.9,
            maximum_turnover=0.35,
            exposure_grid_step=0.05,
            trust_eta=0.5,
            trust_tail_penalty=1.0,
            trust_turnover_penalty=0.1,
            trust_minimum_posterior_mass=20.0,
            trust_minimum_effective_sample_size=20.0,
        )

    def run_simulation(self, **overrides):
        arguments = dict(
            decision_dates=self.fixture.decision_dates,
            return_dates=self.fixture.return_dates,
            router_posteriors=self.fixture.router_posteriors,
            expert_exposures=self.fixture.expert_exposures,
            asset_simple_returns=self.fixture.asset_simple_returns,
            scenario_log_returns=self.fixture.scenario_log_returns,
            initial_trust=self.fixture.initial_trust,
            admission_mask=self.mask,
            fallback_weights=self.fallback,
            config=self.config,
        )
        arguments.update(overrides)
        return simulate(**arguments)

    def test_deterministic_replay(self) -> None:
        first = self.run_simulation()
        second = self.run_simulation()
        self.assertTrue(first.trace.equals(second.trace))
        self.assertTrue(np.array_equal(first.terminal_trust, second.terminal_trust))

    def test_monthly_updates_occur_only_after_completed_periods(self) -> None:
        result = self.run_simulation()
        completed_months = len(set(self.fixture.return_dates.to_period("M")))
        self.assertEqual(result.trust_updates, completed_months)

    def test_trust_audit_records_every_regime_expert_pair(self) -> None:
        result = self.run_simulation()
        expected = result.trust_updates * self.fixture.router_posteriors.shape[1] * self.fixture.expert_exposures.shape[1]
        self.assertEqual(len(result.trust_update_audit), expected)
        self.assertTrue(
            {
                "posterior_mass",
                "effective_sample_size",
                "support_passed",
                "trust_before",
                "trust_after",
            }.issubset(result.trust_update_audit.columns)
        )

    def test_rejected_atp_remains_zero_for_every_day(self) -> None:
        result = self.run_simulation()
        self.assertTrue(np.array_equal(result.trace["weight_5"].to_numpy(), np.zeros(len(result.trace))))
        self.assertTrue(np.array_equal(result.terminal_trust[:, 5], np.zeros(3)))

    def test_future_changes_cannot_change_past_trace(self) -> None:
        baseline = self.run_simulation()
        cutoff = 60
        changed = self.fixture.asset_simple_returns.copy()
        changed[cutoff:] = -changed[cutoff:]
        modified = self.run_simulation(asset_simple_returns=changed)
        self.assertTrue(baseline.trace.iloc[:cutoff].equals(modified.trace.iloc[:cutoff]))

    def test_missing_router_day_falls_back_to_cash(self) -> None:
        q = self.fixture.router_posteriors.copy()
        q[10] = np.nan
        result = self.run_simulation(router_posteriors=q)
        row = result.trace.iloc[10]
        self.assertTrue(row["router_fallback_used"])
        self.assertEqual(row["desired_exposure"], 0.0)

    def test_no_admission_ablation_restores_atp_weight(self) -> None:
        result = self.run_simulation(enable_admission=False)
        self.assertTrue((result.trace["weight_5"] > 0.0).any())

    def test_no_trust_update_still_masks_rejected_state(self) -> None:
        result = self.run_simulation(enable_trust_updates=False)
        self.assertTrue(np.array_equal(result.terminal_trust[:, 5], np.zeros(3)))


if __name__ == "__main__":
    unittest.main()
