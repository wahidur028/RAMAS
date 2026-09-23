from __future__ import annotations

import json
import math
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrozenContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (ROOT / "config.json").read_text(encoding="utf-8")
        self.config = json.loads(self.text)

    def test_json_has_no_duplicate_keys(self) -> None:
        def reject(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                self.assertNotIn(key, result)
                result[key] = value
            return result

        json.loads(self.text, object_pairs_hook=reject)

    def test_exact_expert_order_is_frozen(self) -> None:
        self.assertEqual(
            self.config["expert_names"],
            ["cash", "buy_and_hold", "trend", "volatility_target", "drawdown_control", "specialist_atp"],
        )

    def test_production_default_is_fail_closed(self) -> None:
        admission = self.config["production_default_admission"]
        self.assertTrue(admission["cash"])
        self.assertTrue(admission["buy_and_hold"])
        for name in ["trend", "volatility_target", "drawdown_control", "specialist_atp"]:
            self.assertFalse(admission[name])

    def test_fixture_admission_is_not_scientific_evidence(self) -> None:
        claims = self.config["claims"]
        self.assertFalse(claims["fixture_admissions_are_scientific_evidence"])
        self.assertFalse(claims["mechanical_pass_is_economic_validation"])
        self.assertFalse(claims["final_evaluation_period_selected"])

    def test_dqn_is_closed(self) -> None:
        claims = self.config["claims"]
        self.assertTrue(claims["dqn_closed"])
        self.assertFalse(claims["dqn_training_permitted"])
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((ROOT / "src").glob("*.py"))
        ).lower()
        forbidden = [r"torch\.", r"tensorflow", r"stable_baselines", r"\.fit\(", r"backward\("]
        for pattern in forbidden:
            self.assertIsNone(re.search(pattern, source), pattern)

    def test_real_adapter_is_bound_to_daily_router_and_stage0(self) -> None:
        claims = self.config["claims"]
        self.assertTrue(claims["daily_router_posterior_required"])
        self.assertFalse(claims["intraday_increment_established"])
        self.assertFalse(claims["stage0_decision_may_be_overridden"])
        self.assertEqual(
            self.config["router"]["posterior_columns"],
            ["daily_prob_bear", "daily_prob_bull", "daily_prob_mix"],
        )
        self.assertEqual(
            self.config["stage0"]["expected_decision"],
            "FINAL_STAGE1_FAIL_CLOSE_SPECIALIST_ATP_CURRENT_INFORMATION_SET",
        )

    def test_trust_support_is_derived_from_tail_probability(self) -> None:
        required = math.ceil(1.0 / (1.0 - float(self.config["cvar_alpha"])))
        self.assertEqual(self.config["trust_minimum_posterior_mass"], required)
        self.assertEqual(self.config["trust_minimum_effective_sample_size"], required)
        self.assertEqual(
            self.config["trust_tail_estimator"],
            "exact_regime_posterior_weighted_upper_tail_mean",
        )
        self.assertFalse(self.config["claims"]["unsupported_regime_trust_may_update"])
        self.assertFalse(self.config["claims"]["trust_thresholds_selected_by_return_search"])

    def test_economic_gate_is_single_and_frozen(self) -> None:
        gate = self.config["economic_gate"]
        self.assertEqual(gate["primary_comparator"], "constant_matched_mean_exposure")
        self.assertEqual(gate["block_length_days"], 30)
        self.assertEqual(gate["bootstrap_resamples"], 10000)
        self.assertEqual(gate["one_sided_alpha"], 0.05)
        self.assertEqual(gate["minimum_positive_years"], 2)


if __name__ == "__main__":
    unittest.main()
