"""Local tests with an artificial provider; no model calls, temporary directories only."""
import json, sys, tempfile, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import runner as r  # noqa: E402


class FakeClient:
    """Deterministic fake Ollama: valid JSON for every call unless told otherwise."""
    def __init__(self, models, responder=None, fail_after=None):
        self.endpoint = "fake://"
        self.models = models
        self.responder = responder
        self.fail_after = fail_after
        self.calls = []
        self.unloaded = []

    def version(self):
        return "0.23.0-fake"

    def inspect(self, model):
        return {"requested": model, "observed": model, "digest": r.EXPECTED_DIGEST.get(model, "d" * 64),
                "details": {}, "capabilities": ["completion"], "template_sha256": "t" * 64,
                "modelfile_sha256": "m" * 64, "serving_metadata_sha256": "s" * 64}

    def unload(self, model):
        self.unloaded.append(model)

    def complete(self, body):
        self.calls.append(body)
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise r.TransportFailure("simulated outage")
        payload = json.loads(body["messages"][1]["content"])
        content = self.responder(body, payload) if self.responder else json.dumps(
            {"action": "ABSTAIN", "confidence": 0.5, "reason_codes": ["INSUFFICIENT_EVIDENCE"], "cited_memory_ids": []})
        return {"model": body["model"], "done": True, "done_reason": "stop", "message": {"role": "assistant", "content": content},
                "prompt_eval_count": 100, "eval_count": 20}, 0.01


def small_indices(bank, n=6):
    return r.pilot_indices(bank)[:n]


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, cls.agent, cls.prompt = r.load_inputs_cached()

    def test_schedule_is_complete_balanced_and_identical_across_models(self):
        idx = list(range(len(self.bank)))
        sched = r.build_schedule(idx, r.MODELS)
        self.assertEqual(len(sched), 2 * 2 * 3 * 1244)
        self.assertEqual(len({c["key"] for c in sched}), len(sched))
        for rep in (1, 2, 3):
            cells = [c for c in sched if c["repeat"] == rep]
            firsts = {c["state_index"]: c["first_condition"] for c in cells}
            self.assertEqual(sum(v == "exposed" for v in firsts.values()), 622)
            per_model = {}
            for c in cells:
                per_model.setdefault(c["model"], []).append(c["state_index"])
            orders = {m: [s for i, s in enumerate(v) if i % 2 == 0] for m, v in per_model.items()}
            self.assertEqual(orders[r.MODELS[0]], orders[r.MODELS[1]])  # same state order for both models
            # both conditions consecutive per state
            for i in range(0, len(cells), 2):
                self.assertEqual(cells[i]["state_index"], cells[i + 1]["state_index"])
                self.assertEqual({cells[i]["condition"], cells[i + 1]["condition"]}, {"exposed", "hidden"})
            models_in_order = [c["model"] for c in cells]
            self.assertEqual(models_in_order[0], r.model_order(r.MODELS, rep)[0])
        self.assertEqual(r.model_order(r.MODELS, 2), tuple(reversed(r.MODELS)))
        self.assertEqual(r.build_schedule(idx, r.MODELS), sched)  # deterministic

    def test_hidden_request_changes_only_memory_and_exposed_is_bank_identical(self):
        state = self.bank[10]
        b_exp, p_exp, vis_exp = r.build_body("qwen3:8b", state["request"], "exposed", self.prompt, self.agent, False)
        b_hid, p_hid, vis_hid = r.build_body("qwen3:8b", state["request"], "hidden", self.prompt, self.agent, False)
        self.assertEqual(p_exp, state["request"])
        self.assertEqual(r.sha256_text(r.canonical(p_exp)), state["request_sha256"])
        self.assertEqual(p_hid["memory"], r.HIDDEN_MEMORY)
        diff = {k for k in p_exp if p_exp[k] != p_hid[k]}
        self.assertEqual(diff, {"memory"})
        self.assertTrue(vis_exp and not vis_hid)
        self.assertIs(b_hid["think"], False)
        b_llama, _, _ = r.build_body("llama3.3:70b", state["request"], "exposed", self.prompt, self.agent, None)
        self.assertNotIn("think", b_llama)
        self.assertEqual(b_llama["options"], r.OPTIONS)
        self.assertEqual(b_llama["messages"][0]["content"], self.prompt)

    def test_validation_rejects_bad_outputs_and_accepts_good(self):
        vis = {"episode-000001"}
        good = json.dumps({"action": "BTC", "confidence": 0.7, "reason_codes": ["POSITIVE_MOMENTUM"], "cited_memory_ids": ["episode-000001"]})
        self.assertEqual(r.validate_decision(good, self.agent, vis)["action"], "BTC")
        for bad in ["not json", json.dumps({"action": "HOLD", "confidence": 0.5, "reason_codes": ["X"], "cited_memory_ids": []}),
                    json.dumps({"action": "BTC", "confidence": 0.5, "reason_codes": ["MEMORY_SUPPORT"], "cited_memory_ids": ["episode-999999"]}),
                    json.dumps({"action": "BTC", "confidence": 1.5, "reason_codes": ["MEMORY_SUPPORT"], "cited_memory_ids": []})]:
            with self.assertRaises(r.ContractError):
                r.validate_decision(bad, self.agent, vis)

    def test_pilot_run_completes_and_records_have_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            client = FakeClient(r.MODELS)
            code = r.run_experiment("pilot", out, client, r.MODELS, r.pilot_indices(self.bank))
            self.assertEqual(code, 0)
            complete = json.loads((out / "complete.json").read_text())
            self.assertEqual(complete["status"], "COMPLETE_ALL_VALID")
            self.assertEqual(complete["valid"], 48)
            contract, schedule, records = r.load_run(out, self.bank, self.agent, self.prompt)
            self.assertEqual(len(records), 48)
            for rec in records:
                self.assertEqual(rec["record_sha256"], r.record_hash(rec))
            # tampering is detected
            p = next(iter((out / "calls").glob("*.json")))
            data = json.loads(p.read_text()); data["valid"] = not data["valid"]; p.write_text(json.dumps(data))
            with self.assertRaises(r.RunnerError):
                r.load_records(out)

    def test_interruption_resume_never_resamples_saved_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            idx = r.pilot_indices(self.bank)
            client = FakeClient(r.MODELS, fail_after=7)
            code = r.run_experiment("pilot", out, client, r.MODELS, idx)
            self.assertEqual(code, 3)
            blocked = json.loads((out / "blocked.json").read_text())
            self.assertEqual(blocked["latest"]["status"], "PAUSED_TRANSPORT_FAILURE")
            saved = {p.name: p.read_bytes() for p in (out / "calls").glob("*.json")}
            self.assertEqual(len(saved), 8)  # 7 valid + 1 transport-failure record
            # simulate a crash mid-request: leave a pending marker without a record
            pend = out / "pending" / "r1-llama3.3_70b-s9999-exposed.json"
            pend.write_text(json.dumps({"cell": {"key": "r1-llama3.3_70b-s9999-exposed", "repeat": 1, "model": "llama3.3:70b",
                                                 "state_index": idx[0], "condition": "exposed", "first_condition": "exposed", "order": 0},
                                        "request": {}, "request_sha256": "x", "payload_sha256": "y", "started_at": "t"}))
            client2 = FakeClient(r.MODELS)
            code = r.run_experiment("pilot", out, client2, r.MODELS, idx)
            self.assertEqual(code, 0)
            for name, blob in saved.items():
                self.assertEqual((out / "calls" / name).read_bytes(), blob)  # untouched
            self.assertEqual(len(client2.calls), 48 - 8)  # only missing keys were queried
            unknown = json.loads((out / "calls" / "r1-llama3.3_70b-s9999-exposed.json").read_text())
            self.assertTrue(unknown["transport_outcome_unknown"]); self.assertFalse(unknown["valid"])
            complete = json.loads((out / "complete.json").read_text())
            self.assertEqual(complete["status"], "COMPLETE_WITH_INVALID_OR_MISSING")

    def test_consecutive_invalid_pauses_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            client = FakeClient(r.MODELS, responder=lambda body, payload: "not json at all")
            code = r.run_experiment("pilot", out, client, r.MODELS, r.pilot_indices(self.bank))
            self.assertEqual(code, 3)
            self.assertEqual(json.loads((out / "blocked.json").read_text())["latest"]["status"], "PAUSED_CONSECUTIVE_INVALID")
            self.assertEqual(len(client.calls), 3)

    def test_analysis_undefined_transmission_and_between_context_effect(self):
        import analyze
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            def responder(body, payload):
                exposed = bool(payload["memory"]["similar_completed_episodes"])
                action = "ABSTAIN" if exposed else "BTC"       # memory always flips the vote
                return json.dumps({"action": action, "confidence": 0.9, "reason_codes": ["MEMORY_WARNING" if exposed else "POSITIVE_MOMENTUM"], "cited_memory_ids": []})
            code = r.run_experiment("pilot", out, FakeClient(r.MODELS, responder=responder), r.MODELS, r.pilot_indices(self.bank))
            self.assertEqual(code, 0)
            # a (fake) main run over the pilot states: three repeats, so within-condition pairs exist
            main = Path(tmp) / "main"
            code = r.run_experiment("run", main, FakeClient(r.MODELS, responder=responder), r.MODELS, r.pilot_indices(self.bank), pilot_dir=out)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((main / "contract.json").read_text())["repeats"], 3)
            summary = analyze.analyze(out)
            self.assertTrue(summary["all_valid"]); self.assertIsNotNone(summary.get("estimated_main_hours_from_pilot"))
            out = main
            summary = analyze.analyze(out)
            self.assertTrue(summary["all_valid"])
            import csv
            rows = list(csv.DictReader((out / "analysis" / "paired_summary.csv").open()))
            within = [x for x in rows if x["comparison"] == "within_exposed" and x["regime"] == "all" and x["scope"] == "available_valid_pairs"]
            self.assertTrue(within and all(x["conditional_transmission_rate"] == "" for x in within))  # undefined, not zero
            between = [x for x in rows if x["comparison"] == "between_contexts" and x["regime"] == "all" and x["scope"] == "available_valid_pairs"]
            self.assertTrue(between and all(float(x["advice_disagreement_rate"]) == 1.0 for x in between))
            self.assertTrue(all(float(x["abstention_difference_pp"]) == 100.0 for x in between))


if __name__ == "__main__":
    unittest.main()
