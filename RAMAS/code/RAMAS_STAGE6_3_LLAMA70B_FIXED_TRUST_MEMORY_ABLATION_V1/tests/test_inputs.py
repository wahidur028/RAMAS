from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from stage63lib import inputs


class InputBoundaryTests(unittest.TestCase):
    def test_source_relative_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(inputs.InputAuditError, "escapes"):
                inputs._inside(Path(directory), "../another_run")
            with self.assertRaisesRegex(inputs.InputAuditError, "relative path"):
                inputs._inside(Path(directory), "/another_run")

    def test_nonfinite_comparison_is_not_silently_equal(self):
        with self.assertRaisesRegex(inputs.InputAuditError, "non-finite"):
            inputs._close([float("nan")], [float("nan")], "test")

    def test_artifact_edit_rejected_even_when_completion_record_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ("00_CONTRACT.json", "01_SOURCE_AND_SEED_AUDIT.json", "03_DAILY_TRACE.csv", "08_LLM_CALLS.jsonl"):
                (root / name).write_text("original\n", encoding="utf-8")
                files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            (root / "RUN_COMPLETE.json").write_text(json.dumps({"status": "PASS", "artifact_sha256": files}), encoding="utf-8")
            config = {
                "expected_source_complete_sha256": hashlib.sha256((root / "RUN_COMPLETE.json").read_bytes()).hexdigest(),
                "expected_source_trace_sha256": files["03_DAILY_TRACE.csv"],
            }
            inputs._verify_archive(root, config)
            (root / "03_DAILY_TRACE.csv").write_text("edited\n", encoding="utf-8")
            with self.assertRaisesRegex(inputs.InputAuditError, "checksum differs"):
                inputs._verify_archive(root, config)

    def test_new_feature_field_requires_review_before_any_preview(self):
        columns = {key: [0.0] for key in inputs.FEATURE_KEYS}
        columns["tomorrow_return"] = [0.1]
        period = SimpleNamespace(frame=pd.DataFrame({"row": [0]}), features=pd.DataFrame(columns))
        with patch.object(inputs.legacy, "load_jsonl", return_value=[{}]):
            with self.assertRaisesRegex(inputs.InputAuditError, "Feature schema changed"):
                inputs._audit_archived_requests(Path("unused"), period, pd.DataFrame(), {}, None, None)


if __name__ == "__main__":
    unittest.main()
