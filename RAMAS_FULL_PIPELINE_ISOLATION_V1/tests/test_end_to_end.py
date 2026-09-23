import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
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


runner = load("ramas_e2e_runner", "RAMAS_FULL_PIPELINE_ISOLATION.py")
verify = load("ramas_e2e_verify", "RAMAS_VERIFY_RESULTS.py")


def make_frame(spec, offset):
    dates = pd.date_range("2024-01-02", periods=5, freq="D")
    decisions = dates - pd.Timedelta(days=1)
    assets = np.array([0.01, -0.02, 0.03, 0.005, -0.01])
    desired = np.clip(np.array([0.2, 0.4, 0.1, 0.6, 0.3]) + offset, 0, 1)
    exposures = desired.copy()
    if spec["risk"] == "standard":
        exposures = np.round(desired / 0.05) * 0.05
    pretrade = [0.0]
    for x, r in zip(exposures[:-1], assets[:-1]):
        pretrade.append(float(verify.drifted(x, r)))
    pretrade = np.array(pretrade)
    turnover = np.abs(exposures - pretrade)
    cost = 0.001 * turnover
    net = (1 - cost) * (1 + exposures * assets) - 1
    before = np.r_[1.0, np.cumprod(1 + net)[:-1]]
    after = np.cumprod(1 + net)
    return pd.DataFrame({
        "decision_date": decisions.strftime("%Y-%m-%d"),
        "return_date": dates.strftime("%Y-%m-%d"),
        "hard_regime": ["bull", "mix", "bull", "bear", "mix"],
        "asset_simple_return": assets,
        "pretrade_exposure": pretrade,
        "exposure": exposures,
        "net_return_after_trading_costs": net,
        "turnover": turnover,
        "cost_fraction": cost,
        "wealth_before": before,
        "wealth_after": after,
        "action": ["BTC", "ABSTAIN", "BTC", "CASH", "ABSTAIN"],
        "beta": [0.05] * 5,
        "core_desired_exposure": desired,
        "desired_exposure": desired,
        "request_sha256": [f"req-{i}" for i in range(5)],
        "risk_mode": [spec["risk"]] * 5,
        "retrieved_memory_count": [1 if spec.get("factor_memory") == "memory" else 0] * 5,
        "input_token_count": [10] * 5,
        "output_token_count": [4] * 5,
    })


def write_arm(directory, spec, identity, offset, with_calls):
    directory.mkdir(parents=True)
    frame = make_frame(spec, offset)
    frame.to_csv(directory / "DAILY_LEDGER.csv", index=False)
    if with_calls:
        calls = directory / "calls"
        calls.mkdir()
        for i in range(5):
            request = {
                "state": {"day": i, "offset": offset},
                "safe_exposure_previews": {"BTC": {"x": 1}},
                "memory": {"similar_completed_episodes": ([{"episode_id": f"e{i}"}]
                                                             if spec["factor_memory"] == "memory" else [])},
            }
            raw = json.dumps({
                "action": frame.action.iloc[i], "confidence": 0.5,
                "reason_codes": ["BULLISH_ROUTER"],
                "cited_memory_ids": [],
            }, sort_keys=True)
            item = {
                "identity": identity,
                "request": request,
                "request_sha256": verify.digest(request),
                "raw_response": raw,
                "response_sha256": verify.digest(raw),
                "provider_response": {"prompt_eval_count": 10, "eval_count": 4},
                "latency_seconds": 0.1,
            }
            item["record_sha256"] = verify.digest(item)
            (calls / f"{i+1:06d}-{spec['name']}.json").write_text(json.dumps(item))
    hashes = {
        path.relative_to(directory).as_posix(): verify.sha256(path)
        for path in directory.rglob("*") if path.is_file()
    }
    (directory / "RUN_COMPLETE.json").write_text(json.dumps({
        "status": "COMPLETE", "identity": identity, "spec": spec,
        "artifact_sha256": hashes,
    }))


class EndToEndVerifierTest(unittest.TestCase):
    def test_complete_factorial_is_recomputed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            output = root / "verify"
            config = json.loads((ROOT / "config" / "experiment.json").read_text())
            config["expected_full_rows"] = 5
            config["expected_evaluation_rows"] = 5
            config["evaluation_start"] = "2024-01-02"
            config["evaluation_end"] = "2024-01-06"
            config["bootstrap"]["resamples"] = 20
            config["bootstrap"]["block_days"] = 2
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))

            for index, spec in enumerate(runner.numerical_specs()):
                identity = {"model": "NUMERICAL_ONLY", "sampling_seed": None}
                write_arm(run / "numerical_controls" / spec["name"], spec, identity, index * 0.001, False)
            for index, spec in enumerate(runner.specs()):
                identity = {"model": "fixture-model", "sampling_seed": 42}
                write_arm(run / "cells" / "fixture" / "arms" / spec["name"], spec,
                          identity, (index + 1) * 0.001, True)
            run.mkdir(exist_ok=True)
            (run / "RUN_COMPLETE.json").write_text(json.dumps({"status": "COMPLETE"}))

            verify.verify(Namespace(run=run, config=config_path, output=output))
            report = json.loads((output / "VERIFICATION_REPORT.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["arms_verified"], 10)
            self.assertEqual(report["model_call_records_verified"], 40)
            self.assertTrue((output / "04_PAIRED_BLOCK_BOOTSTRAP.csv").is_file())
            self.assertTrue((output / "09_PROJECTION_ONLY_DAILY.csv").is_file())


if __name__ == "__main__":
    unittest.main()
