import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage7_runner", ROOT / "run_stage7.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Stage7ProtocolTests(unittest.TestCase):
    def test_arm_specs_keep_memory_as_the_only_advisor_input_intervention(self):
        specs = MODULE.arm_specs("closed_loop")
        self.assertEqual(specs[0]["memory"], "expanding")
        self.assertEqual(specs[1]["memory"], "none")
        self.assertEqual(specs[0]["trust"], specs[1]["trust"])
        self.assertEqual(specs[0]["fixed_beta"], specs[1]["fixed_beta"], 0.05)
        self.assertEqual(specs[0]["core"], specs[1]["core"])
        self.assertEqual(specs[0]["risk"], specs[1]["risk"])


    def test_transmission_counts_are_date_matched(self):
        rows = []
        for i, day in enumerate(pd.date_range("2021-12-30", periods=3, freq="D")):
            for arm in ("left", "right"):
                rows.append({
                    "arm": arm,
                    "return_date": day,
                    "hard_regime": "bull",
                    "action": "BTC" if arm == "left" and i == 1 else "CASH",
                    "desired_exposure": 0.5 if arm == "left" and i == 1 else 0.0,
                    "exposure": 0.5 if arm == "left" and i == 1 else 0.0,
                    "net_return_after_trading_costs": 0.01 if arm == "left" and i == 1 else 0.0,
                    "retrieved_memory_count": 1 if arm == "left" else 0,
                    "cited_memory_count": 0,
                })
        table = MODULE.transmission(pd.DataFrame(rows), "left", "right", "TEST")
        all_row = table[table.scope == "ALL"].iloc[0]
        self.assertEqual(all_row.action_different_days, 1)
        self.assertEqual(all_row.final_exposure_different_days, 1)


    def test_memory_audit_rejects_future_episode(self):
        frame = pd.DataFrame([
            {
                "arm": "memory",
                "decision_date": "2022-01-01",
                "retrieved_memory_ids": '["episode-1"]',
                "retrieved_memory_count": 1,
                "cited_memory_count": 0,
            }
        ])
        episodes = {"memory": [{"episode_id": "episode-1", "decision_date": "2022-01-01", "return_date": "2022-01-02"}]}
        result = MODULE.audit_memory_rows(frame, episodes)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["violations"][0]["reason"], "FUTURE_OR_SAME_DAY_EPISODE")
