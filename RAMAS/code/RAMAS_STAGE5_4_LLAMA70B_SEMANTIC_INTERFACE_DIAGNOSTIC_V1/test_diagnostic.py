#!/usr/bin/env python3
import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("diagnose_llama70b.py")
SPEC = importlib.util.spec_from_file_location("diagnose_llama70b", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DiagnosticTests(unittest.TestCase):
    def test_parser_accepts_valid_contract(self):
        raw = json.dumps({
            "residual": 0.25,
            "confidence": 0.6,
            "cited_tool_result_ids": ["a", "b"],
            "lesson": {"action": "none", "text": ""},
        })
        parsed = MODULE.parse_answer(raw, {"a", "b"})
        self.assertEqual(parsed["residual"], 0.25)

    def test_parser_rejects_unseen_citation(self):
        raw = json.dumps({
            "residual": 0.25,
            "confidence": 0.6,
            "cited_tool_result_ids": ["unseen"],
            "lesson": {"action": "none", "text": ""},
        })
        with self.assertRaises(ValueError):
            MODULE.parse_answer(raw, {"visible"})

    def test_expected_residual_uses_only_eligible_nonzero(self):
        call = {
            "request": {"executed_tool_results": [{
                "tool_name": "compare_allowed_actions",
                "result": {"residual_comparison": [
                    {"residual": -0.25, "eligible": False, "mean_log_advantage": 1.0},
                    {"residual": 0.0, "eligible": True, "mean_log_advantage": 0.0},
                    {"residual": 0.25, "eligible": True, "mean_log_advantage": 0.1},
                ]},
            }]},
        }
        self.assertEqual(MODULE.expected_residual(call), 0.25)

    def test_schema_does_not_force_citations(self):
        schema = MODULE.decision_schema(["a", "b", "c", "d"])
        citations = schema["properties"]["cited_tool_result_ids"]
        self.assertNotIn("minItems", citations)
        self.assertNotIn("maxItems", citations)


if __name__ == "__main__":
    unittest.main(verbosity=2)
