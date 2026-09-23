import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stage62lib.reporting import (
    baseline_streams, chronological_metrics, paired_bootstrap, tail_loss, write_reports,
)


class ReportingTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for i, (d, r, regime, asset) in enumerate([
            ("2021-12-30", "2021-12-31", "bull", 0.10),
            ("2021-12-31", "2022-01-01", "bear", -0.02),
            ("2022-01-01", "2022-01-02", "mix", 0.03),
        ]):
            row = dict(decision_date=d, return_date=r, hard_regime=regime,
                       asset_simple_return=asset)
            for prefix in ("memory", "no_memory"):
                row.update({prefix + "_net_return": asset * .45,
                            prefix + "_exposure": .45, prefix + "_turnover": .01,
                            prefix + "_action": "ABSTAIN", prefix + "_beta": .05,
                            prefix + "_valid": True, prefix + "_cited_memory_count": 0})
            rows.append(row)
        return rows

    def test_fractional_tail_and_small_samples(self):
        # 5% of 30 is 1.5: one whole worst loss and half of the next.
        r = np.array([-.10, -.04] + [0.] * 28)
        self.assertAlmostEqual(tail_loss(r), (.10 + .5 * .04) / 1.5)
        self.assertAlmostEqual(tail_loss([-.2]), .2)
        self.assertAlmostEqual(tail_loss([.01, .02]), -.01)
        self.assertAlmostEqual(tail_loss([-.1, .1], alpha=1), 0)

    def test_cash_sharpe_is_undefined_and_drawdown_has_initial_balance(self):
        cash = chronological_metrics(np.zeros(30))
        self.assertIsNone(cash["sharpe_zero_cash_rate"])
        self.assertEqual(cash["maximum_drawdown_loss"], 0)
        self.assertAlmostEqual(chronological_metrics([-.1, 0])["maximum_drawdown_loss"], .1)

    def test_bh_entry_once_and_report_boundaries_do_not_restart_cost(self):
        asset = np.array([.1, -.02, .03])
        streams = baseline_streams(asset, cost_rate=.001)
        self.assertAlmostEqual(streams["BTC_BUY_AND_HOLD"][0], .999 * 1.1 - 1)
        np.testing.assert_array_equal(streams["BTC_BUY_AND_HOLD"][1:], asset[1:])
        with tempfile.TemporaryDirectory() as td:
            write_reports(Path(td), self.fixture(), {"analysis": {"bootstrap_resamples": 20}})
            with (Path(td) / "04_FULL_AND_YEARLY_METRICS.csv").open() as f:
                annual = list(csv.DictReader(f))
            post = next(x for x in annual if x["scope"] == "POST2021" and x["strategy"] == "BTC_BUY_AND_HOLD")
            self.assertAlmostEqual(float(post["cumulative_return"]), .98 * 1.03 - 1)

    def test_regime_partition_and_no_concatenated_regime_drawdown(self):
        with tempfile.TemporaryDirectory() as td:
            result = write_reports(Path(td), self.fixture(), {"analysis": {"bootstrap_resamples": 20}})
            self.assertTrue(result["regime_partition_verified"])
            with (Path(td) / "05_FORECAST_REGIME_METRICS.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertNotIn("maximum_drawdown_loss", rows[0])
            full = [x for x in rows if x["scope"] == "FULL" and x["strategy"] == "MEMORY"]
            self.assertEqual(sum(int(x["days"]) for x in full), 3)
            total = sum(float(x["log_growth_contribution"]) for x in full)
            self.assertAlmostEqual(total, float(np.log1p(np.array([.1, -.02, .03]) * .45).sum()))

    def test_bootstrap_deterministic_paired_and_two_sided(self):
        a = np.tile([.02, -.01, .015, -.005], 10)
        b = a * .9
        self.assertEqual(paired_bootstrap(a, b, 100), paired_bootstrap(a, b, 100))
        same = paired_bootstrap(a, a, 100)
        self.assertEqual(same["two_sided_percentile95"], [0., 0.])
        self.assertEqual(same["directional_flag"], "INCONCLUSIVE")
        positive = paired_bootstrap(np.full(40, .01), np.zeros(40), 100)
        self.assertEqual(positive["directional_flag"], "MEMORY_DIRECTIONAL_ADVANTAGE")

    def test_controlled_fixture_cannot_establish_economic_learning(self):
        rows = self.fixture()
        for row in rows:
            row["memory_net_return"] = .05
            row["no_memory_net_return"] = 0
        with tempfile.TemporaryDirectory() as td:
            result = write_reports(Path(td), rows, {
                "evidence_kind": "CONTROLLED_NON_ECONOMIC", "analysis": {"bootstrap_resamples": 20}})
            self.assertEqual(result["evidence_flag"], "CONTROLLED_NON_ECONOMIC")
            primary = json.loads((Path(td) / "06_PAIRED_MEMORY_CONTRAST.json").read_text())
            self.assertFalse(primary["continuous_learning_demonstrated"])
            self.assertFalse(primary["economic_stop_gate"])

    def test_warmup_fallback_preserves_positive_raw_gap_but_blocks_clean_memory_flag(self):
        rows = self.fixture()
        for row in rows:
            row["memory_net_return"] = .05
            row["no_memory_net_return"] = 0
        rows[0]["memory_valid"] = False
        with tempfile.TemporaryDirectory() as td:
            result = write_reports(Path(td), rows, {"analysis": {"bootstrap_resamples": 20}})
            self.assertEqual(result["evidence_flag"], "INTERFACE_FAILURES_REQUIRE_INTERPRETATION")
            self.assertEqual(result["primary"]["directional_flag"], "MEMORY_DIRECTIONAL_ADVANTAGE")
            self.assertEqual(result["primary"]["memory_invalid_fallback_days_full"], 1)
            self.assertEqual(result["primary"]["memory_invalid_fallback_days_post2021"], 0)
            self.assertEqual(result["primary"]["paired_days"], 2)
            self.assertTrue(result["primary"]["invalid_decisions_retained_in_return_paths"])
            with (Path(td) / "04_FULL_AND_YEARLY_METRICS.csv").open() as f:
                full = next(x for x in csv.DictReader(f) if x["scope"] == "FULL" and x["strategy"] == "MEMORY")
            self.assertEqual(int(full["days"]), 3)

    def test_projection_diagnostics_show_suppressed_action_differences(self):
        rows = self.fixture()
        for row in rows:
            row["memory_action"] = "BTC"
            row["no_memory_action"] = "CASH"
            row["memory_desired_exposure"] = .5
            row["no_memory_desired_exposure"] = .45
        with tempfile.TemporaryDirectory() as td:
            write_reports(Path(td), rows, {"analysis": {"bootstrap_resamples": 20}})
            with (Path(td) / "07_ACTION_AND_STATE_DIAGNOSTICS.csv").open() as f:
                full = next(x for x in csv.DictReader(f) if x["scope"] == "FULL" and x["forecast_regime"] == "ALL")
            self.assertEqual(int(full["action_disagreement_same_exposure_days"]), 3)
            self.assertEqual(int(full["memory_projection_modified_desired_exposure_days"]), 3)
            self.assertEqual(int(full["no_memory_projection_modified_desired_exposure_days"]), 0)
            self.assertAlmostEqual(float(full["memory_mean_absolute_projection_change"]), .05)

    def test_clock_requires_exact_next_day(self):
        rows = self.fixture()
        rows[0]["decision_date"] = "2021-12-29"
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "exactly the next calendar day"):
                write_reports(Path(td), rows, {})


if __name__ == "__main__":
    unittest.main()
