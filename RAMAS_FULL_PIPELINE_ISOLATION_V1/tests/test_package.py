import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load("ramas_isolation_runner", "RAMAS_FULL_PIPELINE_ISOLATION.py")
verify = load("ramas_isolation_verify", "RAMAS_VERIFY_RESULTS.py")


class PackageTests(unittest.TestCase):
    def test_arm_matrix_is_complete(self):
        specs = runner.specs()
        self.assertEqual(len(specs), 8)
        cells = {
            (s["trust"], s["factor_memory"], s["factor_controller"])
            for s in specs
        }
        self.assertEqual(len(cells), 8)
        off = [s for s in specs if s["factor_controller"] == "off"]
        self.assertTrue(all(s["risk"] == "none" for s in off))
        self.assertEqual(len(runner.numerical_specs()), 2)

    def test_default_call_budget(self):
        config = json.loads((ROOT / "config" / "experiment.json").read_text())
        config = runner.validate_config(config)
        plan = runner.estimate_plan(config, ["m"], [42])
        self.assertEqual(plan["llm_calls_per_model_seed"], 1608 * 8)
        self.assertEqual(plan["maximum_llm_calls"], 12864)

    def test_seed_domains_are_separate(self):
        config = json.loads((ROOT / "config" / "experiment.json").read_text())
        config["numerical_seed"] = config["sampling"]["sampling_seeds"][0]
        with self.assertRaises(runner.IsolationError):
            runner.validate_config(config)

    def test_model_name_matching(self):
        self.assertTrue(runner.model_matches("qwen3:8b", "qwen3:8b"))
        self.assertTrue(runner.model_matches("glm-4.7-flash", "glm-4.7-flash:latest"))
        self.assertFalse(runner.model_matches("qwen3:8b", "llama3.3:70b"))

    def test_invalid_model_output_stops_before_scoring(self):
        provider = object.__new__(runner.StrictOllamaProvider)
        provider.model = "fixture"
        provider.root = "http://unused"
        provider.expected_digest = None
        provider.sampling = {
            "system_prompt_mode": "model_neutralized_frozen_v1",
            "temperature": 0.0, "top_p": 1.0, "seed": 1,
            "num_ctx": 1024, "num_predict": 64, "timeout_seconds": 1,
            "identity_check_every_calls": 100,
        }
        provider.seed = 1
        provider.agent = {"actions": ["BTC", "CASH", "ABSTAIN"], "reason_codes": ["OK"]}
        provider.modules = {
            "SYSTEM_PROMPT": "You are the bounded Llama expert in a fixture.",
            "validate_decision": lambda *_: (_ for _ in ()).throw(ValueError("bad output")),
        }
        provider.calls = 1
        provider.failure_dir = None
        provider.pinned = {}
        provider.system_prompt = "You are the bounded language-model expert in a fixture."
        provider.request = lambda *args, **kwargs: {
            "done": True, "model": "fixture",
            "message": {"content": "not-valid"},
        }
        with self.assertRaises(runner.IsolationError):
            provider.complete({"memory": {"similar_completed_episodes": []}})
        self.assertEqual(provider.calls, 1)

    def test_independent_accounting_and_controller_off(self):
        dates = pd.date_range("2024-01-02", periods=5, freq="D")
        decisions = dates - pd.Timedelta(days=1)
        assets = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
        exposures = np.array([0.2, 0.4, 0.1, 0.6, 0.3])
        pretrade = [0.0]
        for x, r in zip(exposures[:-1], assets[:-1]):
            pretrade.append(float(verify.drifted(x, r)))
        pretrade = np.array(pretrade)
        turnover = np.abs(exposures - pretrade)
        cost = 0.001 * turnover
        net = (1 - cost) * (1 + exposures * assets) - 1
        before = np.r_[1.0, np.cumprod(1 + net)[:-1]]
        after = np.cumprod(1 + net)
        frame = pd.DataFrame({
            "decision_date": decisions, "return_date": dates,
            "hard_regime": ["bull"] * 5, "asset_simple_return": assets,
            "pretrade_exposure": pretrade, "exposure": exposures,
            "net_return_after_trading_costs": net, "turnover": turnover,
            "cost_fraction": cost, "wealth_before": before, "wealth_after": after,
            "action": ["BTC"] * 5, "beta": [0.05] * 5,
            "core_desired_exposure": exposures, "desired_exposure": exposures,
            "request_sha256": ["a"] * 5, "risk_mode": ["none"] * 5,
        })
        errors = verify.verify_ledger(frame, 0.001, 5, "fixture")
        self.assertLessEqual(max(errors.values()), 1e-12)
        summary = verify.metrics(frame, "2024-01-02", "2024-01-06")
        self.assertEqual(summary["observations"], 5)
        self.assertTrue(math.isfinite(summary["net_return_pct"]))

    def test_bootstrap_is_reproducible(self):
        values = np.linspace(-1, 1, 60)
        a = verify.bootstrap(values, 10, 100, 17)
        b = verify.bootstrap(values, 10, 100, 17)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
