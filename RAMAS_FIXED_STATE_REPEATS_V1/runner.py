#!/usr/bin/env python3
"""Fixed-state repeated-serving runner — PROTOCOL.md, version 1.

Written on the experiment server (2026-09-24) because the archive's own runner never
arrived (see UPLOAD_TRUNCATION_NOTE.md).  Standard library only.  Serves the interface
that the shipped analyze.py and launch.sh expect: MODELS, CONDITIONS, TOL, canonical,
atomic, load_inputs_cached, load_run.

Modes:  check | preflight | pilot | run | extension | status | analyze | export
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from frozen_contracts import ContractError, validate_decision  # noqa: E402

PROTOCOL_VERSION = "fixed_state_repeats_v1"
MODELS = ("llama3.3:70b", "qwen3:8b")
CONDITIONS = ("exposed", "hidden")
REPEATS = 3
EXPECTED_DIGEST = {"llama3.3:70b": "a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa"}
OPTIONS = {"temperature": 0.0, "top_p": 1.0, "seed": 16061, "num_ctx": 8192, "num_predict": 512}
ENDPOINT = os.environ.get("RAMAS_OLLAMA", "http://127.0.0.1:11434").rstrip("/")
TIMEOUT_SECONDS = 900
TOL = 1e-10  # exported-exposure comparison threshold (audit threshold, PROTOCOL.md)
HIDDEN_MEMORY = {"similar_completed_episodes": [], "summary": "NO_COMPLETED_SIMILAR_EPISODES"}
EXPECTED_PROMPT_SHA256 = "569897b2f09b09acda6d3b52645904f7eb7a179bb3751c46b79d51f091c8aedd"
EXPECTED_ROWS = 1244
PILOT_PER_REGIME = 4
IDENTITY_EVERY = 100
MAX_CONSECUTIVE_INVALID = 3
MIN_VALID_RATE = 0.98
MIN_CALLS_FOR_RATE = 50
SCHEDULE_SEED = 16061
PROGRESS_EVERY = 10

BANK_FILE = HERE / "data" / "fixed_memory_states.jsonl"
AGENT_FILE = HERE / "data" / "legacy_agent_config.json"
PROMPT_FILE = HERE / "data" / "system_prompt.txt"


# ----------------------------------------------------------------------------- helpers
def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log(message: str) -> None:
    print(f"[{utcnow()}] {message}", flush=True)


class RunnerError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- inputs
_INPUTS = None


def load_inputs():
    """(bank, agent, prompt).  bank = list of 1,244 state records in date order."""
    rows = [json.loads(line) for line in BANK_FILE.read_text().splitlines() if line.strip()]
    if len(rows) != EXPECTED_ROWS:
        raise RunnerError(f"state bank has {len(rows)} rows, expected {EXPECTED_ROWS}")
    dates = []
    for i, row in enumerate(rows):
        if row.get("cohort") != "fixed_memory":
            raise RunnerError(f"bank row {i} cohort={row.get('cohort')!r}; expected fixed_memory")
        if sha256_text(canonical(row["request"])) != row["request_sha256"]:
            raise RunnerError(f"bank row {i}: request_sha256 does not match its request")
        dates.append(row["request"]["state"]["decision_date"])
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise RunnerError("state bank is not strictly date-ordered")
    agent = json.loads(AGENT_FILE.read_text())["agent"]
    if tuple(agent["actions"]) != ("BTC", "CASH", "ABSTAIN"):
        raise RunnerError("agent action contract changed")
    prompt = PROMPT_FILE.read_text()
    if sha256_text(prompt) != EXPECTED_PROMPT_SHA256:
        raise RunnerError("system_prompt.txt hash differs from the neutral prompt in PROTOCOL.md")
    return rows, agent, prompt


def load_inputs_cached():
    global _INPUTS
    if _INPUTS is None:
        _INPUTS = load_inputs()
    return _INPUTS


def input_identity(bank, agent, prompt):
    return {
        "bank_file": BANK_FILE.name,
        "bank_sha256": sha256_file(BANK_FILE),
        "bank_rows": len(bank),
        "bank_first_decision_date": bank[0]["request"]["state"]["decision_date"],
        "bank_last_decision_date": bank[-1]["request"]["state"]["decision_date"],
        "bank_first_return_date": bank[0]["request"]["state"]["target_return_date"],
        "bank_last_return_date": bank[-1]["request"]["state"]["target_return_date"],
        "agent_config_sha256": sha256_file(AGENT_FILE),
        "system_prompt_sha256": sha256_text(prompt),
        "system_prompt_mode": "model_neutralized_frozen_v1",
        "hidden_memory": HIDDEN_MEMORY,
    }


# ----------------------------------------------------------------------------- schedule
def pilot_indices(bank):
    """12 states: four per regime, spaced through that regime's date sequence."""
    by_regime: dict[str, list[int]] = {}
    for i, row in enumerate(bank):
        by_regime.setdefault(row["request"]["state"]["hard_regime"], []).append(i)
    chosen = []
    for regime in sorted(by_regime):
        seq = by_regime[regime]
        if len(seq) < PILOT_PER_REGIME:
            raise RunnerError(f"regime {regime} has only {len(seq)} states")
        chosen += [seq[(2 * k + 1) * len(seq) // (2 * PILOT_PER_REGIME)] for k in range(PILOT_PER_REGIME)]
    return sorted(chosen)


def model_order(models, repeat):
    models = list(models)
    return tuple(models if repeat % 2 == 1 else list(reversed(models)))


def build_schedule(indices, models, repeats=REPEATS, seed=SCHEDULE_SEED):
    """Deterministic: per repeat one shuffled state order (shared by both models),
    exposed-first / hidden-first balanced, both conditions consecutive per state,
    models sequential with alternating order across repeats."""
    cells = []
    order_number = 0
    for repeat in range(1, repeats + 1):
        rng = random.Random(f"{seed}-{PROTOCOL_VERSION}-repeat{repeat}")
        state_order = list(indices)
        rng.shuffle(state_order)
        firsts = ["exposed"] * (len(indices) // 2) + ["hidden"] * (len(indices) - len(indices) // 2)
        rng.shuffle(firsts)
        first_condition = dict(zip(state_order, firsts))
        for model in model_order(models, repeat):
            slug = model.replace(":", "_").replace("/", "_")
            for state_index in state_order:
                first = first_condition[state_index]
                for condition in (first, "hidden" if first == "exposed" else "exposed"):
                    order_number += 1
                    cells.append({
                        "key": f"r{repeat}-{slug}-s{state_index:04d}-{condition}",
                        "repeat": repeat, "model": model, "state_index": state_index,
                        "condition": condition, "first_condition": first, "order": order_number,
                    })
    return cells


# ----------------------------------------------------------------------------- provider
def decision_schema(agent):
    return {
        "type": "object", "additionalProperties": False,
        "required": ["action", "confidence", "reason_codes", "cited_memory_ids"],
        "properties": {
            "action": {"type": "string", "enum": list(agent["actions"])},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_codes": {"type": "array", "minItems": 1,
                             "items": {"type": "string", "enum": list(agent["reason_codes"])}},
            "cited_memory_ids": {"type": "array", "items": {"type": "string"}},
        },
    }


def build_body(model, request, condition, prompt, agent, think):
    payload = json.loads(json.dumps(request))  # deep copy, JSON-value level
    if condition == "hidden":
        payload["memory"] = json.loads(json.dumps(HIDDEN_MEMORY))
    elif condition != "exposed":
        raise RunnerError(f"unknown condition {condition!r}")
    body = {
        "model": model, "stream": False, "format": decision_schema(agent),
        "messages": [{"role": "system", "content": prompt},
                     {"role": "user", "content": json.dumps(payload, sort_keys=True, allow_nan=False)}],
        "options": dict(OPTIONS),
    }
    if think is not None:
        body["think"] = bool(think)
    visible = {e["episode_id"] for e in payload["memory"]["similar_completed_episodes"]}
    return body, payload, visible


class TransportFailure(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, endpoint=ENDPOINT, timeout=TIMEOUT_SECONDS):
        self.endpoint = endpoint
        self.timeout = timeout

    def _request(self, suffix, value=None, timeout=None):
        req = urllib.request.Request(
            self.endpoint + suffix,
            data=None if value is None else json.dumps(value, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if value is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as response:
                result = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise TransportFailure(f"{type(exc).__name__}: {exc}") from exc
        if not isinstance(result, dict):
            raise TransportFailure(f"Ollama {suffix} returned a non-object")
        return result

    def version(self):
        return self._request("/api/version", timeout=30).get("version")

    def inspect(self, model):
        tags = self._request("/api/tags", timeout=30)
        matches = [m for m in tags.get("models", []) if m.get("name") == model
                   or m.get("name") == model + ":latest" or m.get("model") == model]
        if len(matches) != 1 or not matches[0].get("digest"):
            available = sorted(str(m.get("name")) for m in tags.get("models", []))
            raise RunnerError(f"model {model!r} missing or ambiguous; available={available}; no substitution")
        show = self._request("/api/show", {"model": model}, timeout=60)
        # Ollama 0.23.0 emits the `parameters` block (and the generated `modelfile`) with its
        # lines in a different order on every call; sort lines so the identity hash is stable.
        parameters = "\n".join(sorted((show.get("parameters") or "").splitlines()))
        modelfile = "\n".join(sorted((show.get("modelfile") or "").splitlines()))
        stable = {"parameters_sorted_lines": parameters, "template": show.get("template"), "system": show.get("system"),
                  "model_info": show.get("model_info"), "details": show.get("details"), "capabilities": show.get("capabilities")}
        return {
            "requested": model, "observed": matches[0].get("name"), "digest": matches[0]["digest"],
            "details": matches[0].get("details"), "capabilities": show.get("capabilities"),
            "parameters_sorted_lines": parameters,
            "template_sha256": sha256_text(show.get("template") or ""),
            "modelfile_sorted_lines_sha256": sha256_text(modelfile),
            "serving_metadata_sha256": sha256_text(canonical(stable)),
        }

    def unload(self, model):
        try:
            self._request("/api/chat", {"model": model, "messages": [], "keep_alive": 0}, timeout=120)
        except (TransportFailure, RunnerError):
            pass

    def complete(self, body):
        started = time.monotonic()
        outer = self._request("/api/chat", body)
        return outer, time.monotonic() - started


def hardware():
    info = {"host": platform.node(), "platform": platform.platform(), "python": platform.python_version()}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = None
    return info


def serving_identity(client, models):
    identity = {"endpoint": client.endpoint, "ollama_version": client.version(), "models": {}}
    for model in models:
        info = client.inspect(model)
        expected = EXPECTED_DIGEST.get(model)
        if expected and info["digest"] != expected:
            raise RunnerError(f"{model} digest {info['digest']} != pinned {expected}")
        identity["models"][model] = info
    return identity


def think_flag(model, identity):
    if model in MODELS:
        return {"llama3.3:70b": None, "qwen3:8b": False}[model]
    caps = identity["models"][model].get("capabilities") or []
    return False if "thinking" in caps else None


# ----------------------------------------------------------------------------- run state
def paths(output: Path):
    return {"contract": output / "contract.json", "schedule": output / "schedule.json",
            "calls": output / "calls", "pending": output / "pending",
            "progress": output / "progress.json", "complete": output / "complete.json",
            "blocked": output / "blocked.json", "lock": output / "run.lock",
            "identity_log": output / "identity_checks.jsonl"}


def record_hash(record):
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    return sha256_text(canonical(body))


def load_records(output: Path):
    records = []
    calls = paths(output)["calls"]
    if calls.is_dir():
        for p in sorted(calls.glob("*.json")):
            rec = json.loads(p.read_text())
            if rec.get("record_sha256") != record_hash(rec):
                raise RunnerError(f"record checksum mismatch: {p.name}")
            records.append(rec)
    return records


def load_run(output, bank, agent, prompt):
    """(contract, schedule, records) — interface used by analyze.py."""
    global MODELS
    output = Path(output)
    contract = json.loads(paths(output)["contract"].read_text())
    MODELS = tuple(contract["models"])
    schedule = build_schedule(contract["indices"], contract["models"], contract["repeats"], contract["schedule_seed"])
    if sha256_text(canonical(schedule)) != contract["schedule_sha256"]:
        raise RunnerError("rebuilt schedule differs from the contract's schedule hash")
    ident = input_identity(bank, agent, prompt)
    for key in ("bank_sha256", "system_prompt_sha256", "agent_config_sha256"):
        if ident[key] != contract["inputs"][key]:
            raise RunnerError(f"input {key} differs from the run contract")
    return contract, schedule, load_records(output)


PILOT_REPEATS = 1  # PROTOCOL.md: 12 states x 2 models x 2 conditions = 48 separate calls


def repeats_for(mode):
    return PILOT_REPEATS if mode == "pilot" else REPEATS


def make_contract(mode, indices, models, identity, inputs, pilot_reference=None):
    repeats = repeats_for(mode)
    schedule = build_schedule(indices, models, repeats)
    return {
        "protocol_version": PROTOCOL_VERSION, "mode": mode, "created_at": utcnow(),
        "models": list(models), "conditions": list(CONDITIONS), "repeats": repeats,
        "model_order_by_repeat": {str(r): list(model_order(models, r)) for r in range(1, repeats + 1)},
        "indices": list(indices), "planned_calls": len(schedule), "schedule_seed": SCHEDULE_SEED,
        "schedule_sha256": sha256_text(canonical(schedule)),
        "options": dict(OPTIONS), "think": {m: think_flag(m, identity) for m in models},
        "timeout_seconds": TIMEOUT_SECONDS, "identity_check_every_calls": IDENTITY_EVERY,
        "pause_rules": {"max_consecutive_invalid_per_model": MAX_CONSECUTIVE_INVALID,
                        "min_valid_rate_after_calls": [MIN_VALID_RATE, MIN_CALLS_FOR_RATE]},
        "inputs": inputs, "identity": {"serving": identity, "hardware": hardware()},
        "pilot_reference": pilot_reference,
        "runner": {"file": "runner.py", "sha256": sha256_file(Path(__file__).resolve()),
                   "origin": "written on the experiment server; the archive's runner was not received"},
        "claim_boundaries": {"portfolio_performance": False, "fresh_out_of_sample": False,
                             "semantic_content_isolated_from_prompt_length": False,
                             "model_family_effect_isolated": False,
                             "repeats_are_independent_seeds": False},
    }


def identity_matches(a, b):
    def strip(x):
        return {m: {k: v for k, v in info.items()} for m, info in x["models"].items()}
    return a["ollama_version"] == b["ollama_version"] and strip(a) == strip(b)


def write_blocked(output, status, detail, extra=None):
    blocked = {"status": status, "detail": detail, "at": utcnow()}
    if extra:
        blocked.update(extra)
    history = paths(output)["blocked"]
    previous = json.loads(history.read_text()) if history.exists() else {"events": []}
    previous.setdefault("events", []).append(blocked)
    previous["latest"] = blocked
    atomic(history, previous)
    log(f"BLOCKED {status}: {detail}")


def progress_snapshot(schedule, records, per_model, started_at):
    by_key = {r["key"]: r for r in records}
    remaining = [c for c in schedule if c["key"] not in by_key]
    eta_hours = 0.0
    for model, stats in per_model.items():
        n_left = sum(1 for c in remaining if c["model"] == model)
        mean = stats["latency_sum"] / stats["latency_n"] if stats["latency_n"] else None
        stats["mean_latency_seconds"] = mean
        stats["remaining_calls"] = n_left
        if mean is not None:
            eta_hours += n_left * mean / 3600
    return {"at": utcnow(), "started_at": started_at, "planned": len(schedule), "recorded": len(records),
            "valid": sum(1 for r in records if r["valid"]), "remaining": len(remaining),
            "per_model": per_model, "estimated_remaining_hours_from_observed_latency": round(eta_hours, 2)}


def run_experiment(mode, output: Path, client, models, indices, pilot_dir=None, label=None):
    output.mkdir(parents=True, exist_ok=True)
    P = paths(output)
    lock_handle = P["lock"].open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_handle.close()
        raise RunnerError(f"another writer holds {P['lock']}")
    try:
        return _run_locked(mode, output, client, models, indices, pilot_dir, label)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


def _run_locked(mode, output: Path, client, models, indices, pilot_dir, label):
    P = paths(output)
    bank, agent, prompt = load_inputs_cached()
    inputs = input_identity(bank, agent, prompt)
    identity_now = serving_identity(client, models)
    pilot_reference = None
    if mode in ("run", "extension"):
        if pilot_dir is None:
            raise RunnerError("main runs require --pilot <completed pilot directory>")
        pilot_dir = Path(pilot_dir)
        pc_path = pilot_dir / "complete.json"
        if not pc_path.is_file():
            raise RunnerError("pilot has no complete.json")
        pilot_complete = json.loads(pc_path.read_text())
        pilot_contract = json.loads((pilot_dir / "contract.json").read_text())
        if pilot_complete.get("status") != "COMPLETE_ALL_VALID":
            raise RunnerError(f"pilot status {pilot_complete.get('status')}; main run refused")
        if list(pilot_contract["models"]) != list(models):
            raise RunnerError("pilot models differ from this run's models")
        if not identity_matches(pilot_contract["identity"]["serving"], identity_now):
            raise RunnerError("live serving identity differs from the pilot's; main run refused")
        pilot_reference = {"directory": str(pilot_dir), "contract_sha256": sha256_file(pilot_dir / "contract.json"),
                           "complete_status": pilot_complete["status"], "valid": pilot_complete["valid"]}
    if P["contract"].exists():
        contract = json.loads(P["contract"].read_text())
        if contract["mode"] != mode or list(contract["models"]) != list(models) or contract["indices"] != list(indices):
            raise RunnerError("existing contract in this directory describes a different experiment")
        for key in ("bank_sha256", "system_prompt_sha256", "agent_config_sha256"):
            if contract["inputs"][key] != inputs[key]:
                raise RunnerError(f"input {key} changed since the contract was written")
        if not identity_matches(contract["identity"]["serving"], identity_now):
            write_blocked(output, "PAUSED_IDENTITY_CHANGED", "serving identity differs from the contract",
                          {"contract": contract["identity"]["serving"], "now": identity_now})
            return 3
        log(f"resuming {mode} in {output}")
    else:
        contract = make_contract(mode, indices, models, identity_now, inputs, pilot_reference)
        if label:
            contract["label"] = label
        atomic(P["contract"], contract)
        atomic(P["schedule"], build_schedule(indices, models, contract["repeats"]))
        log(f"new {mode} in {output}: {contract['planned_calls']} planned calls")
    schedule = build_schedule(contract["indices"], contract["models"], contract["repeats"], contract["schedule_seed"])
    P["calls"].mkdir(exist_ok=True)
    P["pending"].mkdir(exist_ok=True)
    started_at = contract["created_at"]

    # interrupted requests: unknown outcome, never resubmitted
    for marker in sorted(P["pending"].glob("*.json")):
        key = marker.stem
        if not (P["calls"] / f"{key}.json").exists():
            pending = json.loads(marker.read_text())
            rec = {**pending["cell"], "request": pending["request"], "request_sha256": pending["request_sha256"],
                   "payload_sha256": pending["payload_sha256"], "started_at": pending["started_at"],
                   "finished_at": None, "latency_seconds": None, "provider_response": None, "raw_content": None,
                   "valid": False, "decision": None, "error": "INTERRUPTED_UNKNOWN_OUTCOME",
                   "transport_outcome_unknown": True, "identity_check_sha256": pending.get("identity_check_sha256")}
            rec["record_sha256"] = record_hash(rec)
            atomic(P["calls"] / f"{key}.json", rec)
            log(f"pending request {key} recorded as unknown outcome (not resubmitted)")
        marker.unlink()

    records = load_records(output)
    by_key = {r["key"]: r for r in records}
    per_model = {m: {"recorded": 0, "valid": 0, "invalid": 0, "consecutive_invalid": 0, "latency_sum": 0.0, "latency_n": 0}
                 for m in contract["models"]}
    for r in records:
        s = per_model[r["model"]]
        s["recorded"] += 1
        if r["valid"]:
            s["valid"] += 1
            if r["latency_seconds"] is not None:
                s["latency_sum"] += r["latency_seconds"]; s["latency_n"] += 1
        else:
            s["invalid"] += 1

    def identity_check(reason):
        now = serving_identity(client, contract["models"])
        ok = identity_matches(contract["identity"]["serving"], now)
        with P["identity_log"].open("a") as fh:
            fh.write(json.dumps({"at": utcnow(), "reason": reason, "match": ok, "sha256": sha256_text(canonical(now))}) + "\n")
        if not ok:
            write_blocked(output, "PAUSED_IDENTITY_CHANGED", f"identity changed ({reason})", {"now": now})
            raise RunnerError("identity changed")
        return sha256_text(canonical(now))

    ident_sha = identity_check("start")
    scheduled_calls = 0
    current_model = None
    for cell in schedule:
        if cell["key"] in by_key:
            continue
        if cell["model"] != current_model:
            if current_model is not None:
                client.unload(current_model)
                ident_sha = identity_check(f"model change {current_model} -> {cell['model']}")
            current_model = cell["model"]
            log(f"model block: {current_model} (repeat {cell['repeat']})")
        elif scheduled_calls and scheduled_calls % IDENTITY_EVERY == 0:
            ident_sha = identity_check(f"every {IDENTITY_EVERY} calls")
        scheduled_calls += 1
        state = bank[cell["state_index"]]
        body, payload, visible = build_body(cell["model"], state["request"], cell["condition"], prompt, agent,
                                            contract["think"][cell["model"]])
        request_sha = sha256_text(canonical(body))
        payload_sha = sha256_text(canonical(payload))
        if cell["condition"] == "exposed" and payload_sha != state["request_sha256"]:
            raise RunnerError("exposed payload is not byte-equivalent to the state bank")
        pending = {"cell": cell, "request": body, "request_sha256": request_sha, "payload_sha256": payload_sha,
                   "started_at": utcnow(), "identity_check_sha256": ident_sha}
        marker = P["pending"] / f"{cell['key']}.json"
        atomic(marker, pending)
        rec = {**cell, "decision_date": state["request"]["state"]["decision_date"],
               "hard_regime": state["request"]["state"]["hard_regime"], "request": body,
               "request_sha256": request_sha, "payload_sha256": payload_sha, "started_at": pending["started_at"],
               "identity_check_sha256": ident_sha, "transport_outcome_unknown": False}
        try:
            outer, latency = client.complete(body)
        except TransportFailure as exc:
            rec.update(finished_at=utcnow(), latency_seconds=None, provider_response=None, raw_content=None,
                       valid=False, decision=None, error=f"TRANSPORT_FAILURE: {exc}")
            rec["record_sha256"] = record_hash(rec)
            atomic(P["calls"] / f"{cell['key']}.json", rec)
            marker.unlink()
            records.append(rec); by_key[cell["key"]] = rec
            per_model[cell["model"]]["recorded"] += 1; per_model[cell["model"]]["invalid"] += 1
            atomic(P["progress"], progress_snapshot(schedule, records, per_model, started_at))
            write_blocked(output, "PAUSED_TRANSPORT_FAILURE", str(exc), {"key": cell["key"]})
            return 3
        raw = (outer.get("message") or {}).get("content")
        error = None
        decision = None
        try:
            if outer.get("error"):
                raise ContractError(f"provider error: {outer['error']}")
            if outer.get("model") not in (cell["model"], cell["model"] + ":latest"):
                raise ContractError(f"response model {outer.get('model')!r} != {cell['model']!r}")
            if not outer.get("done"):
                raise ContractError("response not done")
            if outer.get("done_reason") == "length":
                raise ContractError("TRUNCATED: output hit num_predict")
            if not isinstance(raw, str):
                raise ContractError("message.content missing")
            decision = validate_decision(raw, agent, visible)
        except ContractError as exc:
            error = f"{type(exc).__name__}: {exc}"
        valid = error is None
        rec.update(finished_at=utcnow(), latency_seconds=latency, provider_response=outer, raw_content=raw,
                   valid=valid, decision=decision, error=error)
        rec["record_sha256"] = record_hash(rec)
        atomic(P["calls"] / f"{cell['key']}.json", rec)
        marker.unlink()
        records.append(rec); by_key[cell["key"]] = rec
        s = per_model[cell["model"]]
        s["recorded"] += 1
        if valid:
            s["valid"] += 1; s["consecutive_invalid"] = 0; s["latency_sum"] += latency; s["latency_n"] += 1
        else:
            s["invalid"] += 1; s["consecutive_invalid"] += 1
        log(f"{len(records)}/{len(schedule)} {cell['key']} valid={valid} "
            f"action={decision['action'] if decision else '-'} latency={latency:.1f}s"
            f"{'' if valid else ' error=' + str(error)[:120]}")
        if len(records) % PROGRESS_EVERY == 0:
            atomic(P["progress"], progress_snapshot(schedule, records, per_model, started_at))
        if s["consecutive_invalid"] >= MAX_CONSECUTIVE_INVALID:
            atomic(P["progress"], progress_snapshot(schedule, records, per_model, started_at))
            write_blocked(output, "PAUSED_CONSECUTIVE_INVALID", f"{cell['model']}: {MAX_CONSECUTIVE_INVALID} consecutive invalid")
            return 3
        if s["recorded"] >= MIN_CALLS_FOR_RATE and s["valid"] / s["recorded"] < MIN_VALID_RATE:
            atomic(P["progress"], progress_snapshot(schedule, records, per_model, started_at))
            write_blocked(output, "PAUSED_VALID_RATE", f"{cell['model']}: valid rate {s['valid']/s['recorded']:.4f} < {MIN_VALID_RATE}")
            return 3
    if current_model is not None:
        client.unload(current_model)
    identity_check("completion")
    atomic(P["progress"], progress_snapshot(schedule, records, per_model, started_at))
    all_valid = len(records) == len(schedule) and all(r["valid"] for r in records)
    complete = {"status": "COMPLETE_ALL_VALID" if all_valid else "COMPLETE_WITH_INVALID_OR_MISSING",
                "mode": mode, "planned": len(schedule), "recorded": len(records),
                "valid": sum(1 for r in records if r["valid"]),
                "invalid_or_missing": len(schedule) - sum(1 for r in records if r["valid"]),
                "completed_at": utcnow(), "per_model": {m: {k: v for k, v in s.items() if k != "consecutive_invalid"}
                                                        for m, s in per_model.items()}}
    atomic(P["complete"], complete)
    log(f"COMPLETE {complete['status']} valid={complete['valid']}/{complete['planned']}")
    return 0


# ----------------------------------------------------------------------------- commands
SERVER_WRITTEN = {"runner.py", "tests/test_runner.py"}  # replacements for files the archive did not deliver


def cmd_check(args):
    manifest = HERE / "MANIFEST.sha256"
    ok = bad = 0
    absent, replaced, mismatched = [], [], []
    for line in manifest.read_text().splitlines():
        h, name = line.split(maxsplit=1)
        p = HERE / name
        if name in SERVER_WRITTEN:
            replaced.append(name); continue
        if not p.is_file():
            absent.append(name); continue
        if sha256_file(p) == h:
            ok += 1
        else:
            bad += 1; mismatched.append(name)
    bank, agent, prompt = load_inputs()
    result = {"manifest_entries": ok + bad + len(absent) + len(replaced), "manifest_ok": ok,
              "manifest_mismatch": bad, "mismatched": mismatched,
              "manifest_absent_not_received": len(absent),
              "server_written_replacements": {n: sha256_file(HERE / n) for n in replaced if (HERE / n).is_file()},
              "inputs": input_identity(bank, agent, prompt), "pilot_indices": pilot_indices(bank),
              "planned_main_calls": len(build_schedule(range(len(bank)), MODELS)),
              "planned_pilot_calls": len(build_schedule(pilot_indices(bank), MODELS, PILOT_REPEATS)),
              "status": "CHECK_PASS" if bad == 0 else "CHECK_FAIL_MANIFEST_MISMATCH"}
    print(json.dumps(result, indent=2))
    return 0 if bad == 0 else 1


def cmd_preflight(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    client = OllamaClient()
    bank, agent, prompt = load_inputs()
    models = tuple(args.models.split(",")) if args.models else MODELS
    try:
        identity = serving_identity(client, models)
    except (RunnerError, TransportFailure) as exc:
        atomic(output / "blocked.json", {"status": "PREFLIGHT_BLOCKED", "detail": str(exc), "at": utcnow()})
        print(json.dumps({"status": "PREFLIGHT_BLOCKED", "detail": str(exc)}, indent=2)); return 2
    report = {"status": "PREFLIGHT_PASS", "at": utcnow(), "models": list(models), "identity": identity,
              "think": {m: think_flag(m, identity) for m in models}, "options": OPTIONS, "hardware": hardware(),
              "inputs": input_identity(bank, agent, prompt), "pilot_indices": pilot_indices(bank),
              "pilot_states": [{"index": i, "decision_date": bank[i]["request"]["state"]["decision_date"],
                                "regime": bank[i]["request"]["state"]["hard_regime"]} for i in pilot_indices(bank)],
              "planned_calls": {"pilot": len(build_schedule(pilot_indices(bank), models, PILOT_REPEATS)),
                                "main": len(build_schedule(range(len(bank)), models))},
              "live_model_calls_made": 0}
    atomic(output / "preflight.json", report)
    print(json.dumps({k: report[k] for k in ("status", "models", "think", "planned_calls")}, indent=2))
    print("ollama", identity["ollama_version"], {m: v["digest"][:16] for m, v in identity["models"].items()})
    return 0


def _models_arg(args):
    return tuple(args.models.split(",")) if getattr(args, "models", None) else MODELS


def cmd_pilot(args):
    bank, _, _ = load_inputs_cached()
    return run_experiment("pilot", Path(args.output), OllamaClient(), _models_arg(args), pilot_indices(bank),
                          label=getattr(args, "label", None))


def cmd_run(args):
    bank, _, _ = load_inputs_cached()
    mode = "extension" if args.command == "extension" else "run"
    return run_experiment(mode, Path(args.output), OllamaClient(), _models_arg(args), list(range(len(bank))),
                          pilot_dir=args.pilot, label=getattr(args, "label", None))


def cmd_status(args):
    output = Path(args.output); P = paths(output)
    out = {}
    for name in ("contract", "progress", "complete", "blocked"):
        if P[name].exists():
            data = json.loads(P[name].read_text())
            out[name] = {k: data[k] for k in data if k in ("mode", "models", "planned_calls", "created_at", "status",
                                                           "recorded", "valid", "remaining", "planned",
                                                           "estimated_remaining_hours_from_observed_latency", "at",
                                                           "latest", "per_model")}
    out["recorded_calls_on_disk"] = len(list(P["calls"].glob("*.json"))) if P["calls"].is_dir() else 0
    out["pending_markers"] = len(list(P["pending"].glob("*.json"))) if P["pending"].is_dir() else 0
    print(json.dumps(out, indent=2, default=str)); return 0


def cmd_analyze(args):
    import analyze  # shipped in the archive
    analyze.analyze(Path(args.output)); return 0


def cmd_export(args):
    output = Path(args.output).resolve()
    target = output.parent / f"{output.name}_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p.name != "run.lock" and not p.name.endswith(".tmp"):
                z.write(p, str(Path(output.name) / p.relative_to(output)))
    print(json.dumps({"export": str(target), "sha256": sha256_file(target)})); return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("check")
    p = sub.add_parser("preflight"); p.add_argument("--output", required=True); p.add_argument("--models")
    p = sub.add_parser("pilot"); p.add_argument("--output", required=True); p.add_argument("--models"); p.add_argument("--label")
    for name in ("run", "extension"):
        p = sub.add_parser(name); p.add_argument("--output", required=True); p.add_argument("--pilot", required=True)
        p.add_argument("--models"); p.add_argument("--label")
    for name in ("status", "analyze", "export"):
        p = sub.add_parser(name); p.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    handler = {"check": cmd_check, "preflight": cmd_preflight, "pilot": cmd_pilot, "run": cmd_run,
               "extension": cmd_run, "status": cmd_status, "analyze": cmd_analyze, "export": cmd_export}[args.command]
    try:
        return handler(args)
    except RunnerError as exc:
        log(f"ERROR {exc}")
        if getattr(args, "output", None) and args.command in ("pilot", "run", "extension"):
            write_blocked(Path(args.output), "STOPPED_RUNNER_ERROR", str(exc))
        return 1
    except Exception as exc:  # unexpected: record and stop, never silently continue
        log(f"UNEXPECTED {type(exc).__name__}: {exc}")
        if getattr(args, "output", None) and args.command in ("pilot", "run", "extension"):
            write_blocked(Path(args.output), "STOPPED_UNEXPECTED", f"{type(exc).__name__}: {exc}",
                          {"traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    sys.exit(main())
