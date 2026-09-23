from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

import run_event_agent
from src.agent import ControlledProvider, run_agent_cycle
from src.contracts import AgentDecision, ContractError, MarketState
from src.events import REQUIRED_COLUMNS, EventRecord, content_sha256, events_available_asof, load_event_ledger, validate_manifest
from src.memory import MemoryEpisode, MemoryStore, counterfactual_score, make_context_key
from src.shield import evaluate_shield
from src.tools import TOOL_NAMES
from src.triggers import evaluate_trigger
from src.trust import apply_operator, btc_exposure, row_l1


def config() -> dict:
    return json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))


def state(*, confidence: float = 0.55, transition: bool = False) -> MarketState:
    return MarketState.from_dict(
        {
            "decision_time_utc": "2023-07-01T23:59:00+00:00",
            "probabilities": {"bear": 0.55, "bull": 0.20, "mix": 0.25},
            "hard_regime": "bear",
            "transition_day": transition,
            "router_confidence": confidence,
            "router_entropy": 0.80,
            "expert_exposures": {"cash": 0.0, "buy_hold": 1.0, "atp": 0.0},
            "realized_volatility_z": 1.0,
            "drawdown_90": -0.10,
            "atp_signal": 0.0,
        }
    )


def event(*, available_offset_hours: int = -1, severity: int = 4) -> EventRecord:
    decision = datetime(2023, 7, 1, 23, 59, tzinfo=timezone.utc)
    headline = "Verified exchange event"
    body = "Contemporaneous raw event text."
    available = decision + timedelta(hours=available_offset_hours)
    return EventRecord(
        "event-001",
        available - timedelta(minutes=1),
        available,
        "test-source",
        "https://example.test/event",
        "exchange",
        severity,
        headline,
        body,
        content_sha256(headline, body),
    )


class InterventionProvider:
    kind = "test_intervention"

    def __init__(self, tools=TOOL_NAMES, decision=None):
        self.tools = list(tools)
        self._decision = decision

    def plan(self, payload):
        return {"tool_names": self.tools, "planning_reason": "collect bounded evidence"}

    def decide(self, payload):
        return self._decision or {
            "operator": "DEFENSIVE_SHRINK",
            "target_regime": "bear",
            "strength": 0.05,
            "horizon_days": 7,
            "confidence": 0.7,
            "hypothesis": "Verified stress may make the current trust row too aggressive.",
            "invalidation_condition": "Stress resolves and expert downside statistics normalize.",
        }


class MustNotBeCalledProvider:
    kind = "must_not_be_called"

    def plan(self, payload):
        raise AssertionError("provider called on an untriggered state")

    def decide(self, payload):
        raise AssertionError("provider called on an untriggered state")


class FailingProvider:
    kind = "failing"

    def plan(self, payload):
        raise RuntimeError("planning unavailable")

    def decide(self, payload):
        raise RuntimeError("decision unavailable")


class EventTrustAgentTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        run_event_agent.validate_config(self.config)
        self.trust = run_event_agent.uniform_trust()

    def test_direct_exposure_output_is_rejected(self):
        value = {
            "operator": "KEEP",
            "target_regime": "bear",
            "strength": 0.0,
            "horizon_days": 7,
            "confidence": 0.5,
            "hypothesis": "",
            "invalidation_condition": "",
            "target_exposure": 0.25,
        }
        with self.assertRaises(ContractError):
            AgentDecision.from_dict(value)

    def test_event_ledger_hash_and_asof_filter_are_causal(self):
        first = event(available_offset_hours=-2)
        future = event(available_offset_hours=2)
        future = EventRecord(
            "event-002",
            future.published_at,
            future.available_at,
            future.source,
            future.source_url,
            future.category,
            future.severity,
            future.headline,
            future.body,
            future.content_sha256,
        )
        visible = events_available_asof([first, future], "2023-07-01T23:59:00+00:00")
        self.assertEqual([item.event_id for item in visible], ["event-001"])

    def test_runtime_hides_future_event_even_if_caller_supplies_it(self):
        future = event(available_offset_hours=2)
        quiet = MarketState.from_dict(
            {
                "decision_time_utc": "2023-07-01T23:59:00+00:00",
                "probabilities": {"bear": 0.05, "bull": 0.90, "mix": 0.05},
                "hard_regime": "bull",
                "transition_day": False,
                "router_confidence": 0.90,
                "router_entropy": 0.20,
                "expert_exposures": {"cash": 0.3, "buy_hold": 0.5, "atp": 0.4},
                "realized_volatility_z": 0.5,
                "drawdown_90": -0.05,
                "atp_signal": 1.0,
            }
        )
        trace = run_agent_cycle(
            state=quiet,
            visible_events=[future],
            current_trust=self.trust,
            frozen_trust=self.trust,
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
            active_intervention=None,
            memory=MemoryStore(self.config["memory"]),
            provider=MustNotBeCalledProvider(),
            config=self.config,
        )
        self.assertEqual(trace["future_events_hidden"], 1)
        self.assertEqual(trace["status"], "NOT_TRIGGERED")

    def test_manifest_and_ledger_validation(self):
        record = event()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "events.csv"
            with ledger.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "event_id": record.event_id,
                        "published_at_utc": record.published_at.isoformat(),
                        "available_at_utc": record.available_at.isoformat(),
                        "source": record.source,
                        "source_url": record.source_url,
                        "category": record.category,
                        "severity": record.severity,
                        "headline": record.headline,
                        "body": record.body,
                        "content_sha256": record.content_sha256,
                    }
                )
            digest = hashlib.sha256(ledger.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset_id": "test-point-in-time-events-v1",
                        "event_csv_sha256": digest,
                        "collection_method": "archived raw documents",
                        "revision_policy": "first observed version only",
                        "raw_text_preserved": True,
                        "point_in_time_availability_verified": True,
                        "coverage_start_utc": record.available_at.isoformat(),
                        "coverage_end_utc": record.available_at.isoformat(),
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(load_event_ledger(ledger)), 1)
            self.assertEqual(validate_manifest(manifest, ledger)["dataset_id"], "test-point-in-time-events-v1")

    def test_trigger_combines_router_expert_risk_and_event_reasons(self):
        result = evaluate_trigger(state(), [event()], self.config["trigger"])
        self.assertTrue(result.triggered)
        self.assertIn("LOW_ROUTER_CONFIDENCE", result.reasons)
        self.assertIn("EXPERT_DISAGREEMENT", result.reasons)
        self.assertIn("HIGH_SEVERITY_EVENT", result.reasons)

    def test_trust_operator_changes_only_one_row(self):
        proposed = apply_operator(
            self.trust,
            self.trust,
            operator="DEFENSIVE_SHRINK",
            target_regime="bear",
            strength=0.05,
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
        )
        self.assertGreater(row_l1(self.trust["bear"], proposed["bear"]), 0.0)
        self.assertEqual(self.trust["bull"], proposed["bull"])
        self.assertEqual(self.trust["mix"], proposed["mix"])

    def test_shield_rejects_large_change_and_active_overlap(self):
        aggressive = {
            **self.trust,
            "bear": {"cash": 1.0, "buy_hold": 0.0, "atp": 0.0},
        }
        result = evaluate_shield(
            state(),
            self.trust,
            aggressive,
            target_regime="bear",
            config=self.config["shield"],
            active_intervention=True,
            evidence_complete=True,
        )
        self.assertFalse(result.accepted)
        self.assertIn("ACTIVE_INTERVENTION_EXISTS", result.reasons)
        self.assertIn("TRUST_ROW_CHANGE_TOO_LARGE", result.reasons)

    def test_runtime_uses_automatic_provenance_and_accepts_bounded_action(self):
        trace = run_agent_cycle(
            state=state(),
            visible_events=[event()],
            current_trust=self.trust,
            frozen_trust=self.trust,
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
            active_intervention=None,
            memory=MemoryStore(self.config["memory"]),
            provider=InterventionProvider(),
            config=self.config,
        )
        self.assertTrue(trace["plan_valid"])
        self.assertTrue(trace["decision_valid"])
        self.assertTrue(trace["operator_evidence_complete"])
        self.assertEqual(len(trace["automatic_evidence_ids"]), len(TOOL_NAMES))
        self.assertTrue(trace["accepted"])

    def test_missing_operator_tools_fail_closed_without_manual_citation_rule(self):
        trace = run_agent_cycle(
            state=state(),
            visible_events=[event()],
            current_trust=self.trust,
            frozen_trust=self.trust,
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
            active_intervention=None,
            memory=MemoryStore(self.config["memory"]),
            provider=InterventionProvider(tools=("inspect_current_state", "get_verified_events")),
            config=self.config,
        )
        self.assertFalse(trace["operator_evidence_complete"])
        self.assertFalse(trace["accepted"])
        self.assertEqual(trace["proposed_trust"], self.trust)

    def test_provider_failure_keeps_trust_unchanged(self):
        trace = run_agent_cycle(
            state=state(),
            visible_events=[event()],
            current_trust=self.trust,
            frozen_trust=self.trust,
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
            active_intervention=None,
            memory=MemoryStore(self.config["memory"]),
            provider=FailingProvider(),
            config=self.config,
        )
        self.assertFalse(trace["plan_valid"])
        self.assertFalse(trace["decision_valid"])
        self.assertFalse(trace["accepted"])
        self.assertEqual(trace["proposed_trust"], self.trust)

    def test_nontrigger_does_not_call_provider(self):
        quiet = MarketState.from_dict(
            {
                "decision_time_utc": "2023-07-01T23:59:00+00:00",
                "probabilities": {"bear": 0.05, "bull": 0.90, "mix": 0.05},
                "hard_regime": "bull",
                "transition_day": False,
                "router_confidence": 0.90,
                "router_entropy": 0.20,
                "expert_exposures": {"cash": 0.3, "buy_hold": 0.5, "atp": 0.4},
                "realized_volatility_z": 0.5,
                "drawdown_90": -0.05,
                "atp_signal": 1.0,
            }
        )
        trace = run_agent_cycle(
            state=quiet,
            visible_events=[],
            current_trust=self.trust,
            frozen_trust=self.trust,
            expert_risk_scores={"cash": 0.1, "buy_hold": 0.2, "atp": 0.15},
            active_intervention=None,
            memory=MemoryStore(self.config["memory"]),
            provider=MustNotBeCalledProvider(),
            config=self.config,
        )
        self.assertEqual(trace["status"], "NOT_TRIGGERED")

    def test_memory_promotes_only_repeated_counterfactual_success(self):
        memory = MemoryStore(self.config["memory"])
        key = make_context_key("bear", ["HIGH_SEVERITY_EVENT"], ["exchange"])
        for index in range(5):
            score = counterfactual_score(
                agent_net_log_return=0.03,
                baseline_net_log_return=0.01,
                agent_max_drawdown=-0.02,
                baseline_max_drawdown=-0.02,
                extra_turnover=0.01,
                downside_penalty=1.0,
                turnover_penalty=0.001,
            )
            memory.add_completed_episode(
                MemoryEpisode(
                    f"episode-{index}",
                    key,
                    f"2023-01-{index + 1:02d}T00:00:00+00:00",
                    "DEFENSIVE_SHRINK",
                    7,
                    0.03,
                    0.01,
                    -0.02,
                    -0.02,
                    0.01,
                    score,
                )
            )
        self.assertEqual(memory.review(key)["status"], "ACTIVE")

    def test_controlled_preflight_is_mechanical_not_economic(self):
        report = run_event_agent.run_preflight(self.config, "controlled", None)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["economic_claim_permitted"])
        self.assertEqual(report["accepted_interventions"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
