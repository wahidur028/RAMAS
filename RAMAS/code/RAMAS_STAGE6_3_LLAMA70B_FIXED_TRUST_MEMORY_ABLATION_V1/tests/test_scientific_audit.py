from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stage63lib.fixture import FixtureAccounting, build_fixture
from stage63lib.inputs import InputAuditError
from stage63lib.legacy import SYSTEM_PROMPT, source
from stage63lib.scientific_audit import (
    _clock_and_numeric_audit, _matched_contract, _reference_accounting_audit,
    _verify_adaptive_archive, audit_controlled_fixture,
)


class ScientificAuditTests(unittest.TestCase):
    def test_future_return_clock_is_rejected(self):
        period, *_ = build_fixture()
        period.return_dates = period.return_dates + pd.Timedelta(days=1)
        with self.assertRaisesRegex(InputAuditError, "exactly one day"):
            _clock_and_numeric_audit(period)

    def test_simplex_and_future_feature_are_rejected(self):
        period, *_ = build_fixture()
        period.router_probabilities[0, 0] = 0.2
        with self.assertRaisesRegex(InputAuditError, "simplex"):
            _clock_and_numeric_audit(period)
        period, *_ = build_fixture()
        period.features["tomorrow_close"] = 100.0
        with self.assertRaisesRegex(InputAuditError, "Unexpected feature schema"):
            _clock_and_numeric_audit(period)

    def test_controlled_success_cannot_claim_market_validation(self):
        period, base, accounting, _ = build_fixture()
        audit, rows = audit_controlled_fixture(period, base, accounting)
        self.assertEqual(rows, [])
        self.assertFalse(audit["economic_evidence"])
        self.assertFalse(audit["scientific_validation_established"])
        self.assertIsNone(audit["adaptive_reference_identity"])
        period.source_audit["synthetic"] = False
        with self.assertRaisesRegex(InputAuditError, "explicitly synthetic"):
            audit_controlled_fixture(period, base, accounting)

    @staticmethod
    def reference_period(seam_date="2024-01-01", reset=True):
        returns = pd.date_range(pd.Timestamp(seam_date) - pd.Timedelta(days=1), periods=3)
        xs, rs = np.array([0.45, 0.35, 0.45]), np.array([0.02, 0.04, -0.01])
        p, recorded_p, nets = 0., [], []
        for i, (x, r) in enumerate(zip(xs, rs)):
            if i == 1 and reset:
                p = 0.
            recorded_p.append(p)
            nets.append(FixtureAccounting.net_return(x, p, r, .001))
            p = FixtureAccounting.drifted_exposure(x, r)
        return SimpleNamespace(
            frame=pd.DataFrame({"row": range(3)}),
            decision_dates=returns - pd.Timedelta(days=1), return_dates=returns,
            asset_returns=rs,
            current_trace=pd.DataFrame({"final_exposure": xs, "pretrade_exposure": recorded_p,
                                        "portfolio_net_return": nets}),
        )

    def test_known_seam_detected_without_modifying_reference(self):
        period = self.reference_period()
        original = period.current_trace.copy(deep=True)
        audit = _reference_accounting_audit(period, .001, FixtureAccounting, require_known_seam=True)
        self.assertEqual(audit["known_reference_discontinuity_count"], 1)
        self.assertEqual(audit["known_reference_discontinuities"][0]["return_date"], "2024-01-01")
        self.assertFalse(audit["reference_repaired"])
        pd.testing.assert_frame_equal(original, period.current_trace)

    def test_unexpected_reference_seam_and_silent_repair_rejected(self):
        period = self.reference_period("2024-02-01")
        with self.assertRaisesRegex(InputAuditError, "Unexpected legacy"):
            _reference_accounting_audit(period, .001, FixtureAccounting, require_known_seam=True)
        period = self.reference_period(reset=False)
        with self.assertRaisesRegex(InputAuditError, "retain exactly"):
            _reference_accounting_audit(period, .001, FixtureAccounting, require_known_seam=True)

    def test_wrong_recorded_cost_return_rejected(self):
        period = self.reference_period()
        period.current_trace.loc[2, "portfolio_net_return"] += .001
        with self.assertRaisesRegex(InputAuditError, "recorded-pretrade accounting"):
            _reference_accounting_audit(period, .001, FixtureAccounting, require_known_seam=True)

    @staticmethod
    def contract_fixture():
        path = Path(__file__).resolve().parents[1] / "config.json"
        config = json.loads(path.read_text())
        old = copy.deepcopy(config)
        old["common_legacy_trust_rule"] = True
        identity = {
            "system_prompt_sha256": source.sha256_text(SYSTEM_PROMPT),
            "provider_identity": {"model": "llama3.3:70b", "model_digest": config["expected_model_digest"]},
            "python": "3.11.9", "numpy": "1.26.4", "pandas": "2.2.3",
        }
        return config, {"config": old, "identity": identity}, {"identity": copy.deepcopy(identity)}

    def test_fixed_beta_change_allowed_but_budget_change_rejected(self):
        config, contract, complete = self.contract_fixture()
        audit = _matched_contract(config, contract, complete)
        self.assertEqual(audit["adaptive_reference_identity"], complete["identity"])
        config["provider"]["num_predict"] += 1
        with self.assertRaisesRegex(InputAuditError, "provider setting.*num_predict"):
            _matched_contract(config, contract, complete)

    def test_diagnostic_cannot_be_relabelled_clean_confirmation(self):
        config, contract, complete = self.contract_fixture()
        config["scientific_scope"] = "CLEAN_CONFIRMATORY_TRADING"
        with self.assertRaisesRegex(InputAuditError, "diagnostic scope"):
            _matched_contract(config, contract, complete)

    def test_adaptive_archive_edit_rejected_before_returning_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["00_CONTRACT.json", "01_SOURCE_AUDIT.json", "03_DAILY_TRACE.csv", "10_FINAL_STATUS.json"]
            hashes = {}
            for name in names:
                (root / name).write_text("original\n")
                hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            complete = root / "RUN_COMPLETE.json"
            complete.write_text(json.dumps({"status": "PASS", "artifact_sha256": hashes}))
            config = {"expected_adaptive_complete_sha256": hashlib.sha256(complete.read_bytes()).hexdigest(),
                      "expected_adaptive_trace_sha256": hashes["03_DAILY_TRACE.csv"]}
            _verify_adaptive_archive(root, config)
            (root / "03_DAILY_TRACE.csv").write_text("changed\n")
            with self.assertRaisesRegex(InputAuditError, "checksum differs"):
                _verify_adaptive_archive(root, config)


if __name__ == "__main__":
    unittest.main()
