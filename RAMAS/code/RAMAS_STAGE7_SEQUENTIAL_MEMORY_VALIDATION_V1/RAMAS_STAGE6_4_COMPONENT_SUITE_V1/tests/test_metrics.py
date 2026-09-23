import math
import tempfile
import unittest

import numpy as np
import pandas as pd

from suite64.metrics import (NET_RETURN, expected_shortfall_loss, paired_block_comparison,
                             prepare_daily_ledger, summarize_ledger, validate_ledger, write_reports)


def make_ledger(returns, arm="test", first_date="2021-12-31", exposures=None, rate=0.0):
    rows = []
    p = 0.0
    exposures = [1.0] * len(returns) if exposures is None else exposures
    for index, (r, x) in enumerate(zip(returns, exposures)):
        date = pd.Timestamp(first_date) + pd.Timedelta(days=index)
        turnover = abs(x - p)
        cost = rate * turnover
        rows.append({"arm": arm, "decision_date": date - pd.Timedelta(days=1),
                     "return_date": date, "hard_regime": "BULL" if index % 2 else "BEAR",
                     "asset_simple_return": r, "pretrade_exposure": p, "exposure": x,
                     NET_RETURN: (1 - cost) * (1 + x * r) - 1,
                     "turnover": turnover, "cost_fraction": cost})
        p = x * (1 + r) / (1 + x * r)
    return pd.DataFrame(rows)


class MetricTests(unittest.TestCase):
    def test_cost_before_return_and_wealth_drift(self):
        df = make_ledger([.10, -.20], exposures=[.5, .5], rate=.001)
        actual = prepare_daily_ledger(df, .001)
        self.assertAlmostEqual(actual.iloc[0][NET_RETURN], .049475)
        self.assertAlmostEqual(actual.iloc[1].pretrade_exposure, .55 / 1.05)
        expected = (1 - .001 * abs(.5 - .55 / 1.05)) * .9 - 1
        self.assertAlmostEqual(actual.iloc[1][NET_RETURN], expected)
        bad = df.copy()
        bad.loc[0, NET_RETURN] = .05 - .0005
        with self.assertRaisesRegex(ValueError, "cost-before-return"):
            validate_ledger(bad, .001)

    def test_compounding_log_identity(self):
        summary = summarize_ledger(make_ledger([.10, -.20, .25]), 0)
        row = summary.loc[summary.scope == "FULL"].iloc[0]
        self.assertAlmostEqual(row.net_compounded_return_after_trading_costs, .1)
        self.assertAlmostEqual(math.expm1(row.cumulative_net_log_return_after_trading_costs), .1)
        self.assertAlmostEqual(row.maximum_drawdown, .2)

    def test_first_observation_loss_includes_initial_peak(self):
        summary = summarize_ledger(make_ledger([-.20, .10]), 0)
        row = summary.loc[summary.scope == "FULL"].iloc[0]
        self.assertAlmostEqual(row.maximum_drawdown, .2)

    def test_cross_year_drawdown_larger_than_yearly_max(self):
        summary = summarize_ledger(make_ledger([.5, -.2, -.3], first_date="2021-12-30"), 0)
        self.assertAlmostEqual(summary.loc[summary.scope == "FULL", "maximum_drawdown"].iloc[0], .44)
        yearly = summary.loc[summary.scope == "YEAR"].set_index("period")
        self.assertAlmostEqual(yearly.loc["2021", "maximum_drawdown"], .2)
        self.assertAlmostEqual(yearly.loc["2022", "maximum_drawdown"], .3)

    def test_cash_undefined_ratios_are_not_zero(self):
        summary = summarize_ledger(make_ledger([.1, -.1, .2], exposures=[0, 0, 0]), 0)
        row = summary.loc[summary.scope == "FULL"].iloc[0]
        self.assertTrue(pd.isna(row.annualized_sharpe_rf_zero))
        self.assertTrue(pd.isna(row.annualized_sortino_threshold_zero))
        self.assertTrue(pd.isna(row.calmar))
        self.assertEqual(row.maximum_drawdown, 0)
        self.assertEqual(row.net_compounded_return_after_trading_costs, 0)

    def test_fractional_expected_shortfall(self):
        returns = [-.10, -.05] + [0.01] * 28
        self.assertAlmostEqual(expected_shortfall_loss(returns), (.10 + .5 * .05) / 1.5)
        self.assertAlmostEqual(expected_shortfall_loss([.01, .02]), -.01)

    def test_conditional_groups_do_not_invent_drawdown_or_annualization(self):
        summary = summarize_ledger(make_ledger([.1, -.2, .1, .05], first_date="2022-01-01"), 0)
        regimes = summary.loc[(summary.scope == "REGIME") & summary.period.str.startswith("FULL:")]
        full = summary.loc[summary.scope == "FULL"].iloc[0]
        self.assertAlmostEqual(regimes.conditional_log_growth_contribution.sum(), full.cumulative_net_log_return_after_trading_costs)
        self.assertTrue(regimes.maximum_drawdown.isna().all())
        self.assertTrue(regimes.annualized_net_return_volatility.isna().all())
        self.assertTrue(regimes.net_compounded_return_after_trading_costs.isna().all())

    def test_annualization_and_downside_conventions(self):
        summary = summarize_ledger(make_ledger([.1, -.1], first_date="2022-01-01"), 0)
        row = summary.loc[summary.scope == "FULL"].iloc[0]
        self.assertAlmostEqual(row.daily_net_return_volatility, math.sqrt(.02))
        self.assertAlmostEqual(row.annualized_net_return_volatility, math.sqrt(.02 * 365.25))
        self.assertAlmostEqual(row.downside_rms_zero_threshold, math.sqrt(.01 / 2))
        self.assertTrue(pd.isna(row.net_cagr_after_trading_costs))

    def test_identical_paired_ci_is_not_equivalence(self):
        df = pd.concat([make_ledger([.01, -.02] * 20, "A", "2022-01-01"),
                        make_ledger([.01, -.02] * 20, "B", "2022-01-01")])
        result = paired_block_comparison(df, "A", "B", n_resamples=200)
        self.assertEqual(result["mean_daily_net_log_difference_bps"], 0)
        self.assertEqual(result["ci95_lower_bps"], 0)
        self.assertEqual(result["ci95_upper_bps"], 0)
        self.assertEqual(result["interpretation"], "INCONCLUSIVE_NOT_EQUIVALENCE")
        self.assertFalse(result["learning_resimulated"])

    def test_pairing_rejects_missing_dates(self):
        df = pd.concat([make_ledger([.1, -.1], "A", "2022-01-01"),
                        make_ledger([.1], "B", "2022-01-01")])
        with self.assertRaisesRegex(ValueError, "exactly matched"):
            paired_block_comparison(df, "A", "B")

    def test_reports_record_all_three_time_resolutions(self):
        with tempfile.TemporaryDirectory() as output:
            result = write_reports(make_ledger([.1, -.1]), output, transaction_cost_rate=0)
            self.assertEqual(result["daily_rows"], 2)
            self.assertEqual(len(pd.read_csv(output + "/monthly_metrics.csv")), 2)
            self.assertEqual(len(pd.read_csv(output + "/yearly_metrics.csv")), 2)


if __name__ == "__main__":
    unittest.main()
