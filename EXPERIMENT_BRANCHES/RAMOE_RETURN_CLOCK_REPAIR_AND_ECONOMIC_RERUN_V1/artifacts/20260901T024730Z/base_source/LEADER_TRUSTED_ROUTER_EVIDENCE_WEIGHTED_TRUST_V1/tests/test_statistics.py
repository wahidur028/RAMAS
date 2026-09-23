from __future__ import annotations

import unittest

import numpy as np

from src.statistics import circular_block_mean_test


class BlockBootstrapTests(unittest.TestCase):
    def test_deterministic_replay(self) -> None:
        values = np.sin(np.arange(300) / 7.0) * 0.001 + 0.0004
        first = circular_block_mean_test(values, block_length=14, resamples=999, seed=12029)
        second = circular_block_mean_test(values, block_length=14, resamples=999, seed=12029)
        self.assertEqual(first, second)

    def test_strong_positive_advantage_is_significant(self) -> None:
        values = np.full(300, 0.001)
        result = circular_block_mean_test(values, block_length=14, resamples=999, seed=12029)
        self.assertLess(result.one_sided_p_value, 0.05)
        self.assertGreater(result.confidence_interval_low, 0.0)

    def test_negative_advantage_does_not_pass(self) -> None:
        values = np.full(300, -0.001)
        result = circular_block_mean_test(values, block_length=14, resamples=999, seed=12029)
        self.assertGreaterEqual(result.one_sided_p_value, 0.05)


if __name__ == "__main__":
    unittest.main()
