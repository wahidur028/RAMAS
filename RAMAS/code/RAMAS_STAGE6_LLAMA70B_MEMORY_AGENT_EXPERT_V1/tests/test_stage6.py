from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import run_stage6
from stage6lib.agent import ControlledProvider
from stage6lib.contracts import ContractError, validate_decision
from stage6lib.memory import EpisodicMemory, state_vector
from stage6lib.trust import RegimeTrust, blend_exposure
from stage6lib import source_adapter as source_adapter


def config() -> dict:
    return run_stage6.load_config(PACKAGE / "config.json")


def state(regime: str = "bull") -> dict:
    q = {
        "bull": (0.10, 0.80, 0.10),
        "bear": (0.80, 0.10, 0.10),
        "mix": (0.30, 0.30, 0.40),
    }[regime]
    return {
        "prob_bear": q[0],
        "prob_bull": q[1],
        "prob_mix": q[2],
        "hard_regime": regime,
        "router_entropy": 0.4,
        "router_transition_l1": 0.1,
        "return_7": 0.03,
        "return_30": 0.10,
        "realized_vol_30": 0.5,
        "drawdown_90": -0.05,
    }


class Stage6Tests(unittest.TestCase):
    def test_model_is_locked_to_llama70b_and_qwen_is_disabled(self) -> None:
        cfg = config()
        self.assertEqual(cfg["provider"]["model"], "llama3.3:70b")
        self.assertFalse(cfg["frozen_claims"]["qwen_used"])

    def test_valid_contract_accepts_only_visible_memory(self) -> None:
        raw = json.dumps(
            {
                "action": "BTC",
                "confidence": 0.75,
                "reason_codes": ["BULLISH_ROUTER", "MEMORY_SUPPORT"],
                "cited_memory_ids": ["episode-1"],
            }
        )
        value = validate_decision(raw, config()["agent"], {"episode-1"})
        self.assertEqual(value["action"], "BTC")

    def test_contract_rejects_unseen_memory(self) -> None:
        raw = json.dumps(
            {
                "action": "CASH",
                "confidence": 0.8,
                "reason_codes": ["MEMORY_WARNING"],
                "cited_memory_ids": ["future-episode"],
            }
        )
        with self.assertRaises(ContractError):
            validate_decision(raw, config()["agent"], set())

    def test_abstain_exactly_preserves_base_desired_exposure(self) -> None:
        self.assertAlmostEqual(blend_exposure(0.63, "ABSTAIN", 0.20), 0.63)

    def test_btc_and_cash_are_bounded_low_trust_perturbations(self) -> None:
        self.assertAlmostEqual(blend_exposure(0.50, "BTC", 0.05), 0.525)
        self.assertAlmostEqual(blend_exposure(0.50, "CASH", 0.05), 0.475)

    def test_memory_retrieves_only_completed_same_regime(self) -> None:
        memory = EpisodicMemory()
        for index, regime in enumerate(("bull", "bear"), start=1):
            s = state(regime)
            memory.add_completed(
                {
                    "episode_id": f"episode-{index}",
                    "decision_date": "2021-01-01",
                    "return_date": "2021-01-02",
                    "hard_regime": regime,
                    "state_vector": state_vector(s).tolist(),
                    "action": "BTC",
                    "confidence": 0.8,
                    "shadow_log_advantage_vs_ramoe": 0.01,
                    "asset_return": 0.02,
                }
            )
        found = memory.retrieve(state("bull"), 5)
        self.assertEqual([item["episode_id"] for item in found], ["episode-1"])

    def test_trust_updates_only_at_new_month_from_completed_nonabstain(self) -> None:
        cfg = config()["trust"].copy()
        cfg["minimum_completed_episodes_per_regime"] = 2
        trust = RegimeTrust(cfg)
        completed = [
            {"hard_regime": "bull", "action": "BTC", "shadow_log_advantage_vs_ramoe": 0.01},
            {"hard_regime": "bull", "action": "BTC", "shadow_log_advantage_vs_ramoe": 0.02},
        ]
        trust.maybe_update(date(2021, 1, 1), completed)
        trust.maybe_update(date(2021, 1, 31), completed)
        self.assertAlmostEqual(trust.value("bull"), 0.05)
        trust.maybe_update(date(2021, 2, 1), completed)
        self.assertAlmostEqual(trust.value("bull"), 0.075)

    def test_negative_completed_advantage_reduces_trust(self) -> None:
        cfg = config()["trust"].copy()
        cfg["minimum_completed_episodes_per_regime"] = 2
        trust = RegimeTrust(cfg)
        completed = [
            {"hard_regime": "bear", "action": "CASH", "shadow_log_advantage_vs_ramoe": -0.01},
            {"hard_regime": "bear", "action": "CASH", "shadow_log_advantage_vs_ramoe": -0.02},
        ]
        trust.maybe_update(date(2021, 1, 1), completed)
        trust.maybe_update(date(2021, 2, 1), completed)
        self.assertAlmostEqual(trust.value("bear"), 0.025)

    def test_controlled_semantic_preflight_covers_all_actions(self) -> None:
        summary, records = run_stage6.run_semantic_preflight(ControlledProvider(), config())
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(set(summary["observed_actions"]), {"BTC", "CASH", "ABSTAIN"})
        self.assertTrue(all(item["expected_action_matched"] for item in records))

    def test_deterministic_controller_is_transparent(self) -> None:
        self.assertEqual(run_stage6.deterministic_action(state("bull")), "BTC")
        self.assertEqual(run_stage6.deterministic_action(state("bear")), "CASH")
        mixed = state("mix")
        mixed["router_entropy"] = 0.95
        self.assertEqual(run_stage6.deterministic_action(mixed), "ABSTAIN")

    def test_bootstrap_is_deterministic(self) -> None:
        cfg = config()["bootstrap"].copy()
        cfg["resamples"] = 100
        values = np.linspace(-0.01, 0.02, 64)
        first = run_stage6.circular_block_test(values, cfg)
        second = run_stage6.circular_block_test(values, cfg)
        self.assertEqual(first, second)

    def test_expected_shortfall_uses_fractional_tail_mass(self) -> None:
        returns = np.asarray([-0.20, -0.10, 0.00, 0.10])
        # A 37.5% tail is 1.5 observations: 0.20 plus half of 0.10.
        expected = (0.20 + 0.5 * 0.10) / 1.5
        self.assertAlmostEqual(
            run_stage6.fractional_expected_shortfall_loss(returns, 0.375),
            expected,
        )

    def test_timeline_explicitly_blocks_post2021_confirmation(self) -> None:
        cfg = config()
        self.assertFalse(cfg["frozen_claims"]["post2021_confirmation_run"])
        self.assertFalse(cfg["frozen_claims"]["post2021_market_values_used"])
        self.assertFalse(cfg["frozen_claims"]["post2021_portfolio_rows_used"])

    def test_development_loop_reveals_completed_memory_only_on_next_decision(self) -> None:
        class Provider:
            kind = "controlled-test"

            def __init__(self) -> None:
                self.payloads = []

            def complete(self, payload):
                self.payloads.append(payload)
                raw = json.dumps(
                    {
                        "action": "BTC",
                        "confidence": 0.8,
                        "reason_codes": ["BULLISH_ROUTER"],
                        "cited_memory_ids": [],
                    }
                )
                return raw, {"message": {"content": raw}}, 0.0

        class Risk:
            @staticmethod
            def project_exposure(desired_exposure, drifted_pretrade_exposure, scenario_log_returns, **kwargs):
                exposure = float(np.clip(desired_exposure, 0.0, 1.0))
                return SimpleNamespace(
                    exposure=exposure,
                    turnover=abs(exposure - drifted_pretrade_exposure),
                    ambiguity_cvar=0.0,
                    fallback_used=False,
                )

        class Accounting:
            @staticmethod
            def net_return(exposure, pretrade, asset_return, cost_rate):
                return exposure * asset_return - cost_rate * abs(exposure - pretrade)

            @staticmethod
            def drifted_exposure(exposure, asset_return):
                wealth = 1.0 + exposure * asset_return
                return exposure * (1.0 + asset_return) / wealth

        cfg = config()
        dates = pd.to_datetime(["2021-01-01", "2021-01-02"])
        return_dates = pd.to_datetime(["2021-01-02", "2021-01-03"])
        q = np.asarray([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
        features = pd.DataFrame(
            {
                "return_1": [0.01, 0.02],
                "return_7": [0.03, 0.04],
                "return_30": [0.10, 0.11],
                "return_90": [0.20, 0.21],
                "realized_vol_30": [0.5, 0.5],
                "drawdown_90": [-0.05, -0.04],
                "router_entropy": [0.4, 0.4],
                "router_confidence": [0.8, 0.8],
                "router_transition_l1": [0.0, 0.0],
            }
        )
        asset = np.asarray([0.02, -0.01])
        base_returns = 0.5 * asset
        base_trace = pd.DataFrame(
            {
                "desired_exposure": [0.5, 0.5],
                "final_exposure": [0.5, 0.5],
                "portfolio_net_return": base_returns,
                "turnover": [0.5, 0.0],
                "asset_simple_return": asset,
            }
        )
        frame = pd.DataFrame(
            {
                "decision_date": dates,
                "return_date": return_dates,
                "exposure_volatility_target": [0.5, 0.5],
            }
        )
        period = source_adapter.PeriodInputs(
            "test", "test", frame, base_trace, pd.DatetimeIndex(dates),
            pd.DatetimeIndex(return_dates), q, asset,
            np.zeros((2, 3, 4)), features, {},
        )
        provider = Provider()
        base_config = {
            "transaction_cost_bps": 0.0,
            "cvar_alpha": 0.95,
            "cvar_limit": 1.0,
            "ambiguity_quantile": 0.95,
            "maximum_daily_turnover": 1.0,
            "exposure_grid_step": 0.05,
        }
        trace, calls, memory, _trust, _det_trust = run_stage6.run_development(
            period, provider, cfg, base_config, Accounting(), Risk()
        )
        self.assertEqual(len(trace), 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(memory.episodes), 2)
        self.assertEqual(provider.payloads[0]["memory"]["similar_completed_episodes"], [])
        self.assertEqual(
            provider.payloads[1]["memory"]["similar_completed_episodes"][0]["episode_id"],
            "episode-000001",
        )
        for payload in provider.payloads:
            encoded = json.dumps(payload, sort_keys=True)
            self.assertNotIn("asset_simple_return", encoded)
            self.assertNotIn("canonical_asset_simple_return", encoded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
