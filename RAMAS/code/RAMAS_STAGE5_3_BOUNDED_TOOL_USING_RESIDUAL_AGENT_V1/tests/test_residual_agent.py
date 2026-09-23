from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import run_residual_agent as stage53


class FakeRisk:
    @staticmethod
    def project_exposure(**kwargs):
        desired = float(kwargs["desired_exposure"])
        pretrade = float(kwargs["drifted_pretrade_exposure"])
        maximum = float(kwargs["maximum_turnover"])
        selected = float(
            np.clip(
                desired,
                max(0.0, pretrade - maximum),
                min(1.0, pretrade + maximum),
            )
        )
        return SimpleNamespace(
            exposure=selected,
            ambiguity_cvar=0.01 * selected,
            turnover=abs(selected - pretrade),
            fallback_used=False,
        )


class FakeAccounting:
    @staticmethod
    def net_return(exposure, pretrade, asset_return, transaction_cost_rate):
        return (
            (1.0 - transaction_cost_rate * abs(exposure - pretrade))
            * (1.0 + exposure * asset_return)
            - 1.0
        )

    @staticmethod
    def drifted_exposure(exposure, asset_return):
        return exposure * (1.0 + asset_return) / (1.0 + exposure * asset_return)


class FailingProvider:
    kind = "test_failure"

    def plan(self, payload, request_id):
        raise RuntimeError("provider unavailable")


class MustNotBeCalledProvider:
    kind = "must_not_be_called"

    def plan(self, payload, request_id):
        raise AssertionError("provider was called on a non-trigger day")


def base_config() -> dict:
    return {
        "transaction_cost_bps": 10.0,
        "cvar_alpha": 0.95,
        "cvar_limit": 0.045,
        "ambiguity_quantile": 0.9,
        "maximum_daily_turnover": 0.35,
        "exposure_grid_step": 0.05,
    }


def state(regime: str = "bull", confidence: float = 0.7) -> dict:
    probabilities = {
        "bear": [0.7, 0.2, 0.1],
        "bull": [0.1, 0.7, 0.2],
        "mix": [0.1, 0.2, 0.7],
    }[regime]
    value = {
        "decision_date": "2021-01-01",
        "target_return_date": "2021-01-02",
        "prob_bear": probabilities[0],
        "prob_bull": probabilities[1],
        "prob_mix": probabilities[2],
        "hard_regime": regime,
        "pretrade_exposure": 0.25,
        "transition_day": False,
        "return_1": 0.01,
        "return_7": 0.02,
        "return_30": 0.03,
        "return_90": 0.04,
        "realized_vol_30": 0.5,
        "drawdown_90": -0.1,
        "router_entropy": 0.7,
        "router_confidence": confidence,
        "router_transition_l1": 0.1,
    }
    value["uncertainty_bucket"] = stage53.uncertainty_bucket(value)
    return value


def episode(index: int, asset_return: float, current_state: dict) -> dict:
    return {
        "episode_id": f"episode-{index:06d}",
        "decision_date": f"2021-01-{(index - 1) % 28 + 1:02d}",
        "return_date": f"2021-01-{index % 28 + 1:02d}",
        "state": dict(current_state),
        "state_vector": stage53.normalized_state_vector(current_state).tolist(),
        "decision_pretrade_exposure": 0.25,
        "scenario_log_returns": np.zeros((3, 5)).tolist(),
        "asset_simple_return": asset_return,
        "base_desired_exposure": 0.5,
        "model_proposed_residual": 0.0,
        "executed_residual": 0.0,
        "desired_exposure": 0.5,
        "final_exposure": 0.5,
        "portfolio_net_return": 0.0,
        "baseline_net_return": 0.0,
        "log_advantage_vs_transparent": 0.0,
        "completed": True,
    }


def synthetic_period(confidence: float, rows: int = 2) -> stage53.PeriodInputs:
    decisions = pd.date_range("2021-01-01", periods=rows, freq="D")
    features = pd.DataFrame(
        [
            {
                "return_1": 0.0,
                "return_7": 0.0,
                "return_30": 0.0,
                "return_90": 0.0,
                "realized_vol_30": 0.5,
                "drawdown_90": -0.1,
                "router_entropy": 0.8,
                "router_confidence": confidence,
                "router_transition_l1": 0.0,
                "transition_day": False,
            }
        ]
        * rows
    )
    return stage53.PeriodInputs(
        "synthetic",
        "MECHANICAL_TEST_ONLY",
        pd.DataFrame(index=range(rows)),
        pd.DataFrame(
            {
                "desired_exposure": [0.5] * rows,
                "final_exposure": [0.5] * rows,
                "turnover": [0.0] * rows,
                "portfolio_net_return": [0.0] * rows,
            }
        ),
        pd.DatetimeIndex(decisions),
        pd.DatetimeIndex(decisions + pd.Timedelta(days=1)),
        np.tile(np.asarray([[0.1, 0.8, 0.1]]), (rows, 1)),
        np.asarray([0.01, -0.01] * ((rows + 1) // 2))[:rows],
        np.zeros((rows, 3, 5)),
        features,
        {},
    )


class ResidualAgentTests(unittest.TestCase):
    def setUp(self):
        self.config = stage53.load_json(PACKAGE / "config.json")

    def test_frozen_contract_accepts_package_config(self):
        stage53.validate_config(self.config)

    def test_trigger_is_strict_and_transition_aware(self):
        trigger = self.config["trigger"]
        low = state(confidence=0.799999)
        boundary = state(confidence=0.8)
        transition = state(confidence=0.9)
        transition["transition_day"] = True
        self.assertTrue(stage53.is_triggered(low, trigger))
        self.assertFalse(stage53.is_triggered(boundary, trigger))
        self.assertTrue(stage53.is_triggered(transition, trigger))

    def test_plan_rejects_unknown_or_duplicate_tools(self):
        valid = stage53.validate_plan(
            json.dumps(
                {
                    "tool_names": list(stage53.TOOL_NAMES),
                    "planning_reason": "inspect evidence",
                }
            )
        )
        self.assertEqual(valid["tool_names"], list(stage53.TOOL_NAMES))
        with self.assertRaises(stage53.Stage53Error):
            stage53.validate_plan(
                json.dumps({"tool_names": ["invent_tool"], "planning_reason": "bad"})
            )
        with self.assertRaises(stage53.Stage53Error):
            stage53.validate_plan(
                json.dumps(
                    {
                        "tool_names": [
                            "inspect_current_risk",
                            "inspect_current_risk",
                        ],
                        "planning_reason": "bad",
                    }
                )
            )

    def test_decision_rejects_unseen_tool_citation_and_off_grid_residual(self):
        value = {
            "residual": 0.25,
            "confidence": 0.6,
            "cited_tool_result_ids": ["visible"],
            "lesson": {"action": "none", "text": ""},
        }
        parsed = stage53.validate_decision(json.dumps(value), {"visible"})
        self.assertEqual(parsed["residual"], 0.25)
        changed = dict(value)
        changed["cited_tool_result_ids"] = ["unseen"]
        with self.assertRaises(stage53.Stage53Error):
            stage53.validate_decision(json.dumps(changed), {"visible"})
        changed = dict(value)
        changed["residual"] = 0.1
        with self.assertRaises(stage53.Stage53Error):
            stage53.validate_decision(json.dumps(changed), {"visible"})

    def test_nonzero_residual_requires_all_citations_and_eligible_comparison(self):
        results = []
        for name in stage53.TOOL_NAMES:
            payload = {}
            if name == "compare_allowed_actions":
                payload = {
                    "residual_comparison": [
                        {"residual": 0.25, "eligible": True},
                        {"residual": 0.0, "eligible": True},
                    ]
                }
            results.append(
                {
                    "tool_name": name,
                    "tool_result_id": f"id:{name}",
                    "result": payload,
                }
            )
        decision = {
            "residual": 0.25,
            "cited_tool_result_ids": [item["tool_result_id"] for item in results],
        }
        residual, status = stage53.enforce_evidence_gate(decision, results, self.config)
        self.assertEqual(residual, 0.25)
        self.assertEqual(status, "NONZERO_RESIDUAL_ACCEPTED_BY_TOOL_EVIDENCE")
        decision["cited_tool_result_ids"] = decision["cited_tool_result_ids"][:-1]
        residual, status = stage53.enforce_evidence_gate(decision, results, self.config)
        self.assertEqual(residual, 0.0)
        self.assertEqual(status, "ZEROED_MISSING_REQUIRED_TOOL_CITATION")
        decision["cited_tool_result_ids"] = [item["tool_result_id"] for item in results]
        results[1]["result"]["residual_comparison"][0]["eligible"] = False
        residual, status = stage53.enforce_evidence_gate(decision, results, self.config)
        self.assertEqual(residual, 0.0)
        self.assertEqual(status, "ZEROED_TOOL_EVIDENCE_NOT_ELIGIBLE")

    def test_counterfactual_tool_evidence_is_deterministic(self):
        current_state = state()
        episodes = [episode(index + 1, 0.02, current_state) for index in range(20)]
        first = stage53.counterfactual_evidence(
            episodes,
            0.25,
            self.config["transparent_controller"],
            base_config(),
            FakeAccounting,
            FakeRisk,
        )
        second = stage53.counterfactual_evidence(
            episodes,
            0.25,
            self.config["transparent_controller"],
            base_config(),
            FakeAccounting,
            FakeRisk,
        )
        self.assertGreater(first["mean_log_advantage"], 0.0)
        self.assertTrue(first["cvar_not_worse"])
        self.assertTrue(first["two_positive_chronological_blocks"])
        np.testing.assert_array_equal(first["advantages"], second["advantages"])

    def test_memory_promotes_with_bootstrap_evidence_then_rolls_back(self):
        current_state = state()
        memory = stage53.EvidenceMemory(self.config["memory"])
        memory.propose(
            state=current_state,
            residual=0.25,
            text="Increase only when repeated mature incidents support it.",
            episode_id="episode-000001",
            decision_date="2021-01-01",
        )
        for index in range(30):
            memory.add_episode(episode(index + 1, 0.02, current_state))
        memory.review(
            state=current_state,
            decision_date="2021-02-01",
            controller=self.config["transparent_controller"],
            base_config=base_config(),
            accounting=FakeAccounting,
            risk=FakeRisk,
        )
        self.assertEqual(list(memory.lessons.values())[0]["status"], "ACTIVE")
        for index in range(30, 50):
            memory.add_episode(episode(index + 1, -0.20, current_state))
        memory.review(
            state=current_state,
            decision_date="2021-03-01",
            controller=self.config["transparent_controller"],
            base_config=base_config(),
            accounting=FakeAccounting,
            risk=FakeRisk,
        )
        self.assertEqual(list(memory.lessons.values())[0]["status"], "ROLLED_BACK")
        self.assertTrue(
            any(item["event"] == "LESSON_ROLLED_BACK" for item in memory.events)
        )

    def test_fractional_tail_mean_uses_exact_tail_mass(self):
        losses = np.asarray([1.0, 2.0, 10.0])
        result = stage53.exact_tail_mean(losses, 0.5, upper=True)
        self.assertAlmostEqual(result, (10.0 + 0.5 * 2.0) / 1.5)

    def test_future_prices_cannot_change_past_features(self):
        dates = pd.date_range("2020-01-01", periods=220, freq="D")
        close = 100.0 * np.exp(np.linspace(0.0, 0.5, len(dates)))
        raw = pd.DataFrame({"date": dates, "close": close})
        decisions = pd.DatetimeIndex(dates[190:200])
        q = np.tile(np.asarray([[0.2, 0.6, 0.2]]), (len(decisions), 1))
        first, _ = stage53.frozen_pipeline.build_causal_market_features(raw, decisions, q)
        changed = raw.copy()
        changed.loc[changed["date"] > decisions[-1], "close"] *= 1000.0
        second, _ = stage53.frozen_pipeline.build_causal_market_features(changed, decisions, q)
        pd.testing.assert_frame_equal(first, second)

    def test_provider_failure_and_nontrigger_both_use_transparent_controller(self):
        triggered = synthetic_period(confidence=0.7)
        result = stage53.run_agent_period(
            triggered,
            FailingProvider(),
            stage53.EvidenceMemory(self.config["memory"]),
            self.config,
            base_config(),
            FakeAccounting,
            FakeRisk,
        )
        self.assertTrue(np.allclose(result.trace["executed_residual"], 0.0))
        self.assertTrue(
            (
                result.trace["evidence_gate_status"]
                == "FAIL_CLOSED_TO_TRANSPARENT_CONTROLLER"
            ).all()
        )
        nontriggered = synthetic_period(confidence=0.9)
        result = stage53.run_agent_period(
            nontriggered,
            MustNotBeCalledProvider(),
            stage53.EvidenceMemory(self.config["memory"]),
            self.config,
            base_config(),
            FakeAccounting,
            FakeRisk,
        )
        self.assertFalse(result.trace["agent_triggered"].any())
        self.assertTrue(np.allclose(result.trace["executed_residual"], 0.0))

    def test_controlled_agent_plans_uses_tools_and_can_pass_residual_gate(self):
        period = synthetic_period(confidence=0.7, rows=40)
        period.asset_returns[:] = 0.02
        result = stage53.run_agent_period(
            period,
            stage53.ControlledProvider(),
            stage53.EvidenceMemory(self.config["memory"]),
            self.config,
            base_config(),
            FakeAccounting,
            FakeRisk,
        )
        self.assertTrue(result.trace["agent_triggered"].all())
        self.assertTrue(result.trace["plan_valid"].all())
        self.assertTrue(result.trace["decision_valid"].all())
        self.assertGreater(int((result.trace["executed_residual"] != 0.0).sum()), 0)
        supported = result.trace["executed_residual"] != 0.0
        self.assertTrue(
            (
                result.trace.loc[supported, "evidence_gate_status"]
                == "NONZERO_RESIDUAL_ACCEPTED_BY_TOOL_EVIDENCE"
            ).all()
        )

    def test_preflight_selection_is_triggered_and_covers_all_regimes(self):
        rows = 90
        decisions = pd.date_range("2021-01-01", periods=rows, freq="D")
        probabilities = np.vstack(
            [
                np.tile([0.7, 0.2, 0.1], (30, 1)),
                np.tile([0.1, 0.7, 0.2], (30, 1)),
                np.tile([0.1, 0.2, 0.7], (30, 1)),
            ]
        )
        features = pd.DataFrame(
            {
                "router_confidence": [0.7] * rows,
                "transition_day": [False] * rows,
            }
        )
        period = stage53.PeriodInputs(
            "corrected_pre2024",
            "DEVELOPMENT_EVIDENCE_ONLY",
            pd.DataFrame(index=range(rows)),
            pd.DataFrame(index=range(rows)),
            pd.DatetimeIndex(decisions),
            pd.DatetimeIndex(decisions + pd.Timedelta(days=1)),
            probabilities,
            np.zeros(rows),
            np.zeros((rows, 3, 5)),
            features,
            {},
        )
        selected = stage53.representative_triggered_indices(
            period, self.config["trigger"], 30
        )
        hard = np.argmax(probabilities[selected], axis=1)
        self.assertEqual(len(selected), 30)
        self.assertTrue(all(int((hard == item).sum()) > 0 for item in range(3)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
