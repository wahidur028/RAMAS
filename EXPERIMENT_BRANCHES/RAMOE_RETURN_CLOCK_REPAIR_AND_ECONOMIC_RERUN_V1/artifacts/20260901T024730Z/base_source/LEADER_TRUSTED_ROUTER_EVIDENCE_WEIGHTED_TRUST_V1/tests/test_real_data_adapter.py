from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import ContractError
from src.real_data import (
    align_baseline_and_router,
    build_past_only_scenarios,
    build_real_router_inputs,
    derive_rule_exposures,
    load_baseline,
    load_daily_router,
    load_latest_stage0_evidence,
    production_admission_mask,
)


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RealDataAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

    def baseline_frame(self, rows: int = 500) -> pd.DataFrame:
        decisions = pd.date_range("2020-01-02", periods=rows, freq="D")
        index = np.arange(rows, dtype=float)
        return pd.DataFrame(
            {
                "information_date": decisions - pd.Timedelta(days=1),
                "decision_date": decisions,
                "target_date": decisions + pd.Timedelta(days=1),
                "target_log_return": 0.0002 + 0.01 * np.sin(index / 13.0),
                "momentum_30": np.sin(index / 17.0),
                "momentum_90": np.sin(index / 29.0),
                "realized_vol_30": 0.01 + 0.005 * (1.0 + np.sin(index / 23.0)),
                "drawdown_90": -0.25 * (0.5 + 0.5 * np.sin(index / 31.0)),
            }
        )

    def router_frame(self, baseline: pd.DataFrame, start: int = 200) -> pd.DataFrame:
        selected = baseline.iloc[start:].copy()
        rows = len(selected)
        q = np.tile(np.array([0.2, 0.5, 0.3]), (rows, 1))
        return pd.DataFrame(
            {
                "model": "logreg_balanced_fixed",
                "decision_date": selected["information_date"].to_numpy(),
                "target_date": selected["decision_date"].to_numpy(),
                "daily_prob_bear": q[:, 0],
                "daily_prob_bull": q[:, 1],
                "daily_prob_mix": q[:, 2],
                "prob_bear": 0.9,
                "prob_bull": 0.05,
                "prob_mix": 0.05,
            }
        )

    def write_stage0(self, root: Path) -> Path:
        artifact = root / self.config["stage0"]["artifact_roots"][0] / "20260831T020856Z"
        artifact.mkdir(parents=True)
        decision = {
            "scientific_decision": self.config["stage0"]["expected_decision"],
            "stage0_modalities_passed": ["derivatives_carry", "trade_flow_liquidity", "onchain", "cross_market"],
            "information_modalities_passed": [],
        }
        decision_path = artifact / "06_SCIENTIFIC_DECISION.json"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
        audit = pd.DataFrame(
            {
                "modality": list(self.config["modality_files"]),
                "status": ["PASS", "PASS", "FAIL", "PASS", "PASS"],
            }
        )
        audit_path = artifact / "01_MODALITY_AUDIT.csv"
        audit.to_csv(audit_path, index=False)
        complete = {
            "status": "PASS",
            "artifact_sha256": {
                decision_path.name: digest(decision_path),
                audit_path.name: digest(audit_path),
            },
        }
        (artifact / "RUN_COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
        return artifact

    def test_daily_posterior_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "router.csv"
            pd.DataFrame(
                {
                    "decision_date": ["2021-01-01"],
                    "target_date": ["2021-01-02"],
                    "prob_bear": [0.2], "prob_bull": [0.5], "prob_mix": [0.3],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ContractError, "daily posterior"):
                load_daily_router(path, self.config)

    def test_router_target_is_baseline_decision_day(self) -> None:
        baseline = self.baseline_frame(300)
        router_raw = self.router_frame(baseline, 200)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "router.csv"
            router_raw.to_csv(path, index=False)
            router = load_daily_router(path, {**self.config, "minimum_replay_rows": 50})
        aligned = align_baseline_and_router(
            baseline,
            router,
            {**self.config, "minimum_replay_rows": 50},
        )
        self.assertTrue((aligned["router_target_date"] == aligned["decision_date"]).all())
        self.assertTrue((aligned["router_information_date"] == aligned["information_date"]).all())

    def test_rule_exposures_are_bounded_and_atp_is_zero(self) -> None:
        exposures = derive_rule_exposures(self.baseline_frame(40), self.config)
        self.assertEqual(exposures.shape, (40, 6))
        self.assertTrue(np.isfinite(exposures).all())
        self.assertTrue(((exposures >= 0.0) & (exposures <= 1.0)).all())
        self.assertTrue(np.array_equal(exposures[:, 0], np.zeros(40)))
        self.assertTrue(np.array_equal(exposures[:, 1], np.ones(40)))
        self.assertTrue(np.array_equal(exposures[:, 5], np.zeros(40)))

    def test_future_return_changes_cannot_change_past_scenarios(self) -> None:
        frame = self.baseline_frame(500)
        config = deepcopy(self.config)
        config["risk_scenarios"] = {
            "past_only_windows": [10, 20, 50],
            "quantile_samples": 31,
            "minimum_history": 50,
        }
        dates = pd.DatetimeIndex(frame["decision_date"].iloc[100:160])
        original = build_past_only_scenarios(frame, dates, config)
        changed = frame.copy()
        changed.loc[changed["target_date"] > dates[30], "target_log_return"] *= -7.0
        modified = build_past_only_scenarios(changed, dates, config)
        self.assertTrue(np.array_equal(original[:31], modified[:31]))

    def test_post_development_baseline_is_rejected(self) -> None:
        frame = self.baseline_frame(10)
        frame.loc[9, "information_date"] = "2023-12-30"
        frame.loc[9, "decision_date"] = "2023-12-31"
        frame.loc[9, "target_date"] = "2024-01-01"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "baseline.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ContractError, "post-development"):
                load_baseline(path, self.config)

    def test_stage0_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = self.write_stage0(root)
            (artifact / "06_SCIENTIFIC_DECISION.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "hash failed"):
                load_latest_stage0_evidence(root, self.config)

    def test_stage0_rejection_keeps_all_specialists_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.write_stage0(root)
            evidence = load_latest_stage0_evidence(root, self.config)
            mask = production_admission_mask(self.config, evidence)
            self.assertTrue(np.array_equal(mask, np.array([True, True, False, False, False, False])))

    def test_end_to_end_real_input_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = deepcopy(self.config)
            baseline = self.baseline_frame(500)
            router = self.router_frame(baseline, 200)
            baseline_path = root / "baseline.csv"
            router_path = root / "router.csv"
            baseline.to_csv(baseline_path, index=False)
            router.to_csv(router_path, index=False)
            config["baseline"]["candidates"] = ["baseline.csv"]
            config["router"]["prediction_candidates"] = ["router.csv"]
            config["minimum_replay_rows"] = 100
            config["risk_scenarios"] = {
                "past_only_windows": [10, 20, 50],
                "quantile_samples": 31,
                "minimum_history": 50,
            }
            self.write_stage0(root)
            inputs, evidence = build_real_router_inputs(root, config)
            self.assertEqual(len(inputs.decision_dates), 300)
            self.assertEqual(inputs.router_posteriors.shape, (300, 3))
            self.assertEqual(inputs.expert_exposures.shape, (300, 6))
            self.assertEqual(inputs.scenario_log_returns.shape, (300, 3, 31))
            self.assertEqual(evidence.decision["information_modalities_passed"], [])


if __name__ == "__main__":
    unittest.main()
