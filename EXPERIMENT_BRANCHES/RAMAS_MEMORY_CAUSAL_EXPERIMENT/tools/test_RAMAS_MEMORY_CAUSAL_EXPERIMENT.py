#!/usr/bin/env python3
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import RAMAS_MEMORY_CAUSAL_EXPERIMENT as exp


def example_state():
    return {
        "state_id": "2022-01-03",
        "decision_date": "2022-01-03",
        "return_date": "2022-01-04",
        "hard_regime": "bull",
        "core_desired_exposure": 0.5,
        "beta_adaptive": 0.2,
        "beta_fixed": 0.05,
        "asset_simple_return": 0.012,
        "initial_pretrade_exposure": 0.35,
        "action_targets": {"BTC": 1.0, "CASH": 0.0, "ABSTAIN": 0.5},
        "state_view": {
            "router_probabilities": {"Bear": 0.1, "Bull": 0.75, "Mix": 0.15},
            "router_uncertainty": 0.2,
            "portfolio": {"pretrade_exposure": 0.35, "drawdown": 0.03},
            "features": {"return_1d": 0.004, "volatility_20d": 0.18},
        },
        "retrieved_memory": [{
            "memory_id": "episode_20211220_bull_001",
            "episode_date": "2021-12-20",
            "regime": "bull",
            "outcome": {"net_return": 0.008},
            "text": "Prior bull episode; turnover was costly.",
        }],
    }


class ExperimentTests(unittest.TestCase):
    def test_four_cells_and_prompt_isolation(self):
        state = exp.validate_state_snapshots([example_state()], expected_states=1)[0]
        rows = [
            exp.make_call_row(state, authority, condition, 1, "a" * 64)
            for authority in exp.AUTHORITY_MODES
            for condition in exp.MEMORY_CONDITIONS
        ]
        diagnostics = exp.validate_manifest(rows, expected_states=1)
        self.assertEqual(diagnostics["call_count"], 4)
        self.assertEqual(diagnostics["memory_informative_states"], 1)
        off = next(row for row in rows if row["memory_condition"] == "off")
        on = next(row for row in rows if row["memory_condition"] == "on")
        self.assertNotIn("episode_20211220_bull_001", off["prompt"])
        self.assertIn("episode_20211220_bull_001", on["prompt"])
        self.assertEqual(
            len({row["visible_state_sha256"] for row in rows}), 1
        )
        self.assertEqual(
            rows[0]["prompt_sha256"],
            next(row for row in rows if row["authority_mode"] == "fixed" and row["memory_condition"] == rows[0]["memory_condition"])["prompt_sha256"],
        )

    def test_future_memory_is_rejected(self):
        state = example_state()
        state["retrieved_memory"][0]["episode_date"] = "2022-01-03"
        with self.assertRaises(exp.ExperimentError):
            exp.validate_state_snapshots([state], expected_states=1)

    def test_advisor_output_contract(self):
        output = exp.validate_advisor_output({
            "action": "btc",
            "confidence": 0.8,
            "reason_codes": ["REGIME_SIGNAL", "MEMORY_SUPPORTS"],
            "cited_memory_ids": ["episode_20211220_bull_001"],
        }, "on", ["episode_20211220_bull_001"])
        self.assertEqual(output["action"], "BTC")
        with self.assertRaises(exp.ExperimentError):
            exp.validate_advisor_output({
                "action": "BTC",
                "confidence": 0.8,
                "reason_codes": ["REGIME_SIGNAL"],
                "cited_memory_ids": ["not_retrieved"],
            }, "on", ["episode_20211220_bull_001"])

    def test_controller_off_replay(self):
        state = exp.validate_state_snapshots([example_state()], expected_states=1)[0]
        row = exp.make_call_row(state, "adaptive", "off", 1, None)
        record = {
            "call_id": row["call_id"],
            "advisor_output": {
                "action": "BTC",
                "confidence": 1.0,
                "reason_codes": ["REGIME_SIGNAL"],
                "cited_memory_ids": [],
            },
        }
        replay = exp.replay_controller_off([row], {row["call_id"]: record}, 0.001)
        self.assertIsNotNone(replay)
        self.assertGreater(replay["final_wealth"], 1.0)
        self.assertEqual(replay["n_days"], 1)

    def test_compact_analysis_runs_on_complete_four_cell_input(self):
        state = exp.validate_state_snapshots([example_state()], expected_states=1)[0]
        rows = [
            exp.make_call_row(state, authority, condition, 1, "a" * 64)
            for authority in exp.AUTHORITY_MODES
            for condition in exp.MEMORY_CONDITIONS
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            calls = root / "calls.jsonl"
            summary = root / "summary.json"
            exp.write_jsonl(manifest, rows)
            call_records = []
            for row in rows:
                on = row["memory_condition"] == "on"
                call_records.append({
                    "record_type": "ramas_memory_experiment_result",
                    "record_version": 1,
                    "status": "success",
                    "valid": True,
                    "call_id": row["call_id"],
                    "pair_id": row["pair_id"],
                    "state_id": row["state_id"],
                    "state_key": row["state_key"],
                    "state_ordinal": row["state_ordinal"],
                    "decision_date": row["decision_date"],
                    "return_date": row["return_date"],
                    "hard_regime": row["hard_regime"],
                    "authority_mode": row["authority_mode"],
                    "memory_condition": row["memory_condition"],
                    "beta": row["beta"],
                    "prompt_sha256": row["prompt_sha256"],
                    "advisor_output": {
                        "action": "BTC" if on else "CASH",
                        "confidence": 0.8 if on else 0.5,
                        "reason_codes": ["MEMORY_SUPPORTS"] if on else ["REGIME_SIGNAL"],
                        "cited_memory_ids": ["episode_20211220_bull_001"] if on else [],
                    },
                })
            exp.write_jsonl(calls, call_records)
            exp.analyze_experiment(Namespace(
                manifest=str(manifest),
                calls=str(calls),
                output=str(summary),
                pair_output=None,
                controller_ledger=None,
                bootstrap=10,
                block_days=1,
                seed=7,
                cost_rate=0.001,
                allow_incomplete=False,
                expected_states=1,
                allow_nonstandard=False,
                allow_authority_prompt_difference=False,
                allow_missing_memory_hash=True,
            ))
            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["successful_calls"], 4)
            self.assertEqual(
                result["memory_reasoning_effect_by_authority"]["adaptive"][
                    "primary_informative_memory_pairs"
                ]["n_pairs"], 1
            )


if __name__ == "__main__":
    unittest.main()
