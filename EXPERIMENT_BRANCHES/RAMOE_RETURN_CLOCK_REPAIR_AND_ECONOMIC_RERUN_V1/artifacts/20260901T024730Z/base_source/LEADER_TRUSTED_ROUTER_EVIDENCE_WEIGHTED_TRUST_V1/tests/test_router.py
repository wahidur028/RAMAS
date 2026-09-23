from __future__ import annotations

import unittest

import numpy as np

from src.core import ContractError
from src.router import (
    aggregate_exposure,
    l1_allocation_change,
    masked_trust_matrix,
    route_experts,
    route_or_fallback,
    validate_admission_mask,
)


class RouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.q = np.array([0.2, 0.5, 0.3])
        self.trust = np.array(
            [
                [0.40, 0.20, 0.10, 0.10, 0.10, 0.10],
                [0.10, 0.35, 0.15, 0.15, 0.15, 0.10],
                [0.20, 0.25, 0.15, 0.15, 0.15, 0.10],
            ]
        )
        self.mask = np.array([True, True, True, True, True, False])
        self.always = (0, 1)

    def test_rejected_expert_gets_exact_zero(self) -> None:
        weights = route_experts(self.q, self.trust, self.mask, always_admitted_indices=self.always)
        self.assertEqual(weights[5], 0.0)
        self.assertAlmostEqual(weights.sum(), 1.0)

    def test_masked_trust_rows_are_stochastic(self) -> None:
        gated = masked_trust_matrix(self.trust, self.mask, always_admitted_indices=self.always)
        self.assertTrue(np.allclose(gated.sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(gated[:, 5], np.zeros(3)))

    def test_fail_safe_expert_cannot_be_removed(self) -> None:
        invalid = self.mask.copy()
        invalid[0] = False
        with self.assertRaises(ContractError):
            validate_admission_mask(invalid, expert_count=6, always_admitted_indices=self.always)

    def test_exposure_is_weighted_sum(self) -> None:
        weights = route_experts(self.q, self.trust, self.mask, always_admitted_indices=self.always)
        expert_exposure = np.array([0.0, 1.0, 1.0, 0.4, 0.2, 1.0])
        observed = aggregate_exposure(weights, expert_exposure)
        self.assertAlmostEqual(observed, float(weights @ expert_exposure))

    def test_invalid_router_uses_cash_fallback(self) -> None:
        result = route_or_fallback(
            None,
            self.trust,
            self.mask,
            np.array([0.0, 1.0, 1.0, 0.5, 0.2, 1.0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            always_admitted_indices=self.always,
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.desired_exposure, 0.0)

    def test_fallback_cannot_use_rejected_expert(self) -> None:
        with self.assertRaises(ContractError):
            route_or_fallback(
                self.q,
                self.trust,
                self.mask,
                np.zeros(6),
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                always_admitted_indices=self.always,
            )

    def test_fixed_router_mapping_is_l1_nonexpansive(self) -> None:
        before, after = l1_allocation_change(
            np.array([0.6, 0.3, 0.1]),
            np.array([0.5, 0.4, 0.1]),
            self.trust,
            self.mask,
            always_admitted_indices=self.always,
        )
        self.assertLessEqual(after, before + 1e-12)

    def test_short_or_leveraged_exposure_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            aggregate_exposure(np.full(6, 1.0 / 6.0), np.array([0, 1, -0.1, 0.5, 0.5, 1.2]))


if __name__ == "__main__":
    unittest.main()

