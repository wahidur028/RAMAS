import csv
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from stage63lib.reporting import (
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
            self.assertEqual(primary["directional_flag"], "CONTROLLED_NON_ECONOMIC")

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

    def test_fixed_trust_cannot_silently_be_adaptive(self):
        rows = self.fixture()
        rows[1]["memory_beta"] = 0.
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "fixed beta=0.05"):
                write_reports(Path(td), rows, {})

    def test_static50_drifts_and_pays_only_actual_rebalance_cost(self):
        streams = baseline_streams([.10, -.02, .03], .001)
        actual = streams["STATIC50_REBALANCED"]
        self.assertAlmostEqual(actual[0], (1 - .001 * .5) * 1.05 - 1)
        drift = .5 * 1.1 / 1.05
        self.assertAlmostEqual(actual[1], (1 - .001 * abs(.5 - drift)) * .99 - 1)
        self.assertNotAlmostEqual(actual[1], (1 - .001 * .5) * .99 - 1)

    def test_monthly_daily_outputs_preserve_continuous_wealth(self):
        with tempfile.TemporaryDirectory() as td:
            result = write_reports(Path(td), self.fixture(), {"analysis": {"bootstrap_resamples": 20}})
            with (Path(td) / "11_MONTHLY_METRICS.csv").open() as f:
                months = [r for r in csv.DictReader(f) if r["strategy"] == "MEMORY"]
            self.assertEqual([r["month"] for r in months], ["2021-12", "2022-01"])
            self.assertEqual(int(months[0]["days"]), 1)
            self.assertEqual(int(months[1]["days"]), 2)
            self.assertEqual(months[0]["closing_wealth_original_path"], months[1]["opening_wealth_original_path"])
            with (Path(td) / "12_DAILY_METRICS.csv").open() as f:
                daily = list(csv.DictReader(f))
            self.assertEqual(len(daily), 3 * 6)
            self.assertTrue(all(abs(float(r["btc_fraction"]) + float(r["cash_fraction"]) - 1) < 1e-12 for r in daily))
            for name in ("11_MONTHLY_METRICS.csv", "12_DAILY_METRICS.csv", "13_MEMORY_TRUST_INTERACTION.json"):
                self.assertIn(name, result["analysis_files"])

    def test_interaction_resamples_joint_contrast_and_is_only_secondary(self):
        fixed = self.fixture()
        adaptive = copy.deepcopy(fixed)
        for r in fixed:
            r["memory_net_return"] = .01
            r["no_memory_net_return"] = 0.
        for r in adaptive:
            r["memory_net_return"] = .02
            r["no_memory_net_return"] = .005
            r["memory_beta"] = .15
            r["no_memory_beta"] = 0.
        expected = np.log1p(.02) - np.log1p(.005) - np.log1p(.01)
        with tempfile.TemporaryDirectory() as td:
            write_reports(Path(td), fixed, {"analysis": {"bootstrap_resamples": 20}}, adaptive_rows=adaptive)
            interaction = json.loads((Path(td) / "13_MEMORY_TRUST_INTERACTION.json").read_text())
            self.assertAlmostEqual(interaction["mean_daily_log_interaction"], expected)
            np.testing.assert_allclose(interaction["two_sided_percentile95"], [expected, expected])
            self.assertFalse(interaction["primary"])
            self.assertFalse(interaction["continuous_learning_demonstrated"])
            self.assertTrue(interaction["all_four_cells_share_sampled_indices"])
            self.assertEqual(interaction["paired_days"], 2)
            self.assertEqual(interaction["status"], "SECONDARY_REUSED_OOS_DIAGNOSTIC")

    def test_misaligned_adaptive_reference_cannot_be_intersected_silently(self):
        fixed = self.fixture()
        adaptive = copy.deepcopy(fixed)
        adaptive[1]["asset_simple_return"] += .01
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "exogenous"):
                write_reports(Path(td), fixed, {"analysis": {"bootstrap_resamples": 20}}, adaptive_rows=adaptive)
            with self.assertRaisesRegex(ValueError, "same complete dates"):
                write_reports(Path(td), fixed, {"analysis": {"bootstrap_resamples": 20}}, adaptive_rows=adaptive[:-1])

    def test_positive_controlled_interaction_cannot_claim_economic_learning(self):
        rows = self.fixture()
        adaptive = copy.deepcopy(rows)
        for r in adaptive:
            r["memory_net_return"] = .5
        with tempfile.TemporaryDirectory() as td:
            write_reports(Path(td), rows, {"evidence_kind": "CONTROLLED_NON_ECONOMIC",
                "analysis": {"bootstrap_resamples": 20}}, adaptive_rows=adaptive)
            interaction = json.loads((Path(td) / "13_MEMORY_TRUST_INTERACTION.json").read_text())
            self.assertGreater(interaction["mean_daily_log_interaction"], 0)
            self.assertEqual(interaction["status"], "CONTROLLED_NON_ECONOMIC")
            self.assertFalse(interaction["continuous_learning_demonstrated"])

    def test_computational_usage_excludes_preflight_and_marks_missing(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "calls").mkdir()
            entry = {"provider_response": {"prompt_eval_count": 100, "eval_count": 20}}
            (out / "calls" / "000001-memory.json").write_text(json.dumps(entry))
            (out / "calls" / "preflight-000001.json").write_text(json.dumps(entry))
            write_reports(out, self.fixture(), {"analysis": {"bootstrap_resamples": 20}})
            compute = json.loads((out / "14_COMPUTATIONAL_DIAGNOSTICS.json").read_text())
            memory = compute["arms"]["memory"]
            no_memory = compute["arms"]["no_memory"]
            self.assertEqual(memory["journal_calls_found"], 1)
            self.assertEqual(memory["prompt_tokens_observed"], 100)
            self.assertFalse(memory["usage_complete"])
            self.assertIsNone(no_memory["prompt_tokens_observed"])

    def test_configured_complete_dates_reject_partial_real_runs(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "complete configured row count"):
                write_reports(Path(td), self.fixture(), {"expected_rows": 1608})


if __name__ == "__main__":
    unittest.main()
