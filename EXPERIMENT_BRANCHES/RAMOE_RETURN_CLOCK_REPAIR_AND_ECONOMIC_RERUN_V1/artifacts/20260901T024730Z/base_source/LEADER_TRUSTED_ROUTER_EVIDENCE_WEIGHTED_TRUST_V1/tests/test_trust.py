from __future__ import annotations

import unittest

import numpy as np

from src.trust import (
    _exact_tail_signal,
    effective_sample_size,
    exact_weighted_upper_tail_mean,
    risk_aware_trust_update,
)


class TrustUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trust = np.full((3, 6), 1.0 / 6.0)
        self.q = np.tile(np.array([0.2, 0.5, 0.3]), (100, 1))
        self.returns = np.zeros((100, 6))
        self.turnover = np.zeros((100, 6))
        self.mask = np.array([True, True, True, True, True, False])

    @staticmethod
    def update_kwargs() -> dict[str, float]:
        return {
            "eta": 0.5,
            "cvar_alpha": 0.95,
            "tail_penalty": 1.0,
            "turnover_penalty": 0.1,
            "minimum_posterior_mass": 20.0,
            "minimum_effective_sample_size": 20.0,
        }

    def test_exact_tail_signal_mean_matches_fractional_cvar(self) -> None:
        losses = np.array([0.01, 0.02, 0.03, 0.04, 0.20, 0.30])
        signal = _exact_tail_signal(losses, 0.25)
        expected = (0.30 + 0.5 * 0.20) / 1.5
        self.assertAlmostEqual(float(signal.mean()), expected)

    def test_weighted_tail_mean_uses_regime_distribution_directly(self) -> None:
        losses = np.array([0.01, 0.02, 0.20, 0.30])
        weights = np.array([9.0, 9.0, 1.0, 1.0])
        result = exact_weighted_upper_tail_mean(losses, weights, 0.10)
        self.assertAlmostEqual(result, 0.25)
        self.assertLessEqual(result, float(losses.max()))

    def test_effective_sample_size_detects_concentrated_mass(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.ones(20)), 20.0)
        concentrated = np.r_[1.0, np.zeros(19)]
        self.assertAlmostEqual(effective_sample_size(concentrated), 1.0)

    def test_rows_remain_stochastic_and_rejected_stays_zero(self) -> None:
        result = risk_aware_trust_update(
            self.trust,
            self.q,
            self.returns,
            self.turnover,
            self.mask,
            always_admitted_indices=(0, 1),
            **self.update_kwargs(),
        )
        self.assertTrue(np.allclose(result.trust_matrix.sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(result.trust_matrix[:, 5], np.zeros(3)))

    def test_tail_and_turnover_penalties_reduce_harmful_expert_trust(self) -> None:
        self.returns[:, 2] = 0.002
        self.returns[:5, 2] = -0.20
        self.turnover[:, 2] = 0.8
        result = risk_aware_trust_update(
            self.trust,
            self.q,
            self.returns,
            self.turnover,
            self.mask,
            always_admitted_indices=(0, 1),
            **{
                **self.update_kwargs(),
                "tail_penalty": 2.0,
                "turnover_penalty": 0.2,
            },
        )
        masked_initial = 0.2
        self.assertTrue((result.trust_matrix[:, 2] < masked_initial).all())

    def test_identical_evidence_preserves_relative_trust(self) -> None:
        result = risk_aware_trust_update(
            self.trust,
            self.q,
            self.returns,
            self.turnover,
            self.mask,
            always_admitted_indices=(0, 1),
            **self.update_kwargs(),
        )
        self.assertTrue(np.allclose(result.trust_matrix[:, :5], 0.2))

    def test_unsupported_regime_row_is_preserved_exactly(self) -> None:
        q = np.column_stack(
            [
                np.full(100, 0.499999995),
                np.r_[1e-6, np.full(99, 1e-12)],
                np.full(100, 0.500000005 - 1e-8),
            ]
        )
        q /= q.sum(axis=1, keepdims=True)
        self.returns[:, 1] = 0.01
        result = risk_aware_trust_update(
            self.trust,
            q,
            self.returns,
            self.turnover,
            self.mask,
            always_admitted_indices=(0, 1),
            **self.update_kwargs(),
        )
        self.assertFalse(bool(result.support_mask[1]))
        self.assertTrue(np.array_equal(result.trust_matrix[1, :5], np.full(5, 0.2)))

    def test_supported_rows_use_weighted_tail_and_update(self) -> None:
        self.returns[:, 1] = 0.01
        result = risk_aware_trust_update(
            self.trust,
            self.q,
            self.returns,
            self.turnover,
            self.mask,
            always_admitted_indices=(0, 1),
            **self.update_kwargs(),
        )
        self.assertTrue(result.support_mask.all())
        self.assertTrue((result.trust_matrix[:, 1] > 0.2).all())


if __name__ == "__main__":
    unittest.main()
