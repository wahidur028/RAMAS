from __future__ import annotations

import math
import unittest

from run_agent_residual_audit import (
    STATIC_MAPPING,
    circular_block_bootstrap,
    fractional_expected_shortfall,
    next_day_clock_valid,
    replay_policy,
)


class AgentResidualAuditTests(unittest.TestCase):
    def test_static_mapping_is_frozen(self):
        self.assertEqual(STATIC_MAPPING, {"bear": 0.25, "bull": 0.5, "mix": 0.25})

    def test_next_day_clock_rejects_same_day_return(self):
        self.assertTrue(next_day_clock_valid([{"decision_date": "2023-01-01", "return_date": "2023-01-02"}]))
        self.assertFalse(next_day_clock_valid([{"decision_date": "2023-01-01", "return_date": "2023-01-01"}]))

    def test_replay_uses_drifted_pretrade_and_symmetric_cost(self):
        rows = [{"asset_simple_return": 0.10}, {"asset_simple_return": -0.05}]
        replay = replay_policy(rows, [0.5, 0.25], 0.001)
        expected_first = (1 - 0.001 * 0.5) * (1 + 0.5 * 0.10) - 1
        drifted = 0.5 * 1.10 / 1.05
        expected_second = (1 - 0.001 * abs(0.25 - drifted)) * (1 + 0.25 * -0.05) - 1
        self.assertAlmostEqual(replay[0]["net_return"], expected_first, places=15)
        self.assertAlmostEqual(replay[1]["pretrade_exposure"], drifted, places=15)
        self.assertAlmostEqual(replay[1]["net_return"], expected_second, places=15)

    def test_fractional_expected_shortfall_uses_exact_tail_mass(self):
        values = [-0.10, -0.05] + [0.0] * 18
        self.assertAlmostEqual(fractional_expected_shortfall(values, 0.05), 0.10)
        values = [-0.10, -0.05] + [0.0] * 28
        expected = -((-0.10) + 0.5 * (-0.05)) / 1.5
        self.assertAlmostEqual(fractional_expected_shortfall(values, 0.05), expected)

    def test_bootstrap_is_deterministic(self):
        values = [0.001, -0.002, 0.003, 0.0] * 20
        first = circular_block_bootstrap(values, 5, 200, 12031)
        second = circular_block_bootstrap(values, 5, 200, 12031)
        self.assertEqual(first, second)
        self.assertTrue(0.0 <= first["one_sided_p_value"] <= 1.0)

    def test_replay_rejects_invalid_exposure(self):
        with self.assertRaises(ValueError):
            replay_policy([{"asset_simple_return": 0.01}], [1.25], 0.001)


if __name__ == "__main__":
    unittest.main()
