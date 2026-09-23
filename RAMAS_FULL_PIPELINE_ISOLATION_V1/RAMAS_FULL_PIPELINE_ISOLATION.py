#!/usr/bin/env python3
"""Run the RAMAS memory/authority/controller factorial on the verified pipeline.

This program deliberately reuses the frozen Stage-6.4 implementation for the
RAMAS state transition, retrieval, trust update, risk controller, and
accounting.  It changes only declared experimental factors and the model/seed
serving identity.  It never substitutes a simplified reimplementation of the
allocator or controller.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKAGE = Path(__file__).resolve().parent
ARM_FACTORS = tuple(
    (trust, memory, controller)
    for trust in ("adaptive", "fixed")
    for memory in ("memory", "no_memory")
    for controller in ("on", "off")
)


class IsolationError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IsolationError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IsolationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IsolationError(f"Expected a JSON object: {path}")
    return value


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    if not text:
        raise IsolationError(f"Cannot create a safe identifier from {value!r}")
    return text


def parse_int_list(value: str | None, default: Iterable[int]) -> list[int]:
    if value is None:
        return [int(x) for x in default]
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result or len(result) != len(set(result)):
        raise IsolationError("Seed selection must be a nonempty list of unique integers")
    return result


def parse_str_list(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None:
        return [str(x) for x in default]
    result = [x.strip() for x in value.split(",") if x.strip()]
    if not result or len(result) != len(set(result)):
        raise IsolationError("Model selection must be a nonempty list of unique names")
    return result


def verify_package_manifest() -> str | None:
    manifest = PACKAGE / "PACKAGE_MANIFEST.sha256"
    if not manifest.is_file():
        return None
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip().lstrip("*")
        path = (PACKAGE / relative).resolve()
        if path != PACKAGE and PACKAGE not in path.parents:
            raise IsolationError(f"Package manifest path escapes package: {relative}")
        if not path.is_file() or sha256(path) != expected:
            raise IsolationError(f"Package file differs from manifest: {relative}")
    return sha256(manifest)


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "experiment_id", "project_root", "stage64_code_root",
        "full_dataset", "expected_stage64_manifest_sha256", "expected_full_rows",
        "evaluation_start", "evaluation_end", "expected_evaluation_rows",
        "cost_rate", "models", "sampling", "numerical_seed", "bootstrap",
        "execution", "claim_boundaries",
    }
    missing = sorted(required - set(config))
    if missing:
        raise IsolationError(f"Configuration is missing: {missing}")
    if config["scientific_scope"] != "RETROSPECTIVE_REUSED_HISTORY_MECHANISM_DIAGNOSTIC":
        raise IsolationError("Scientific scope label must not be promoted")
    if not math.isclose(float(config["cost_rate"]), 0.001, abs_tol=1e-15, rel_tol=0):
        raise IsolationError("The matched experiment requires the frozen 10-bps cost rate")
    if int(config["expected_full_rows"]) != 1608 or int(config["expected_evaluation_rows"]) != 1244:
        raise IsolationError("The historical contract requires 1,608 full rows and 1,244 evaluation rows")
    sampling = config["sampling"]
    seeds = sampling.get("sampling_seeds")
    if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) or not isinstance(x, int) for x in seeds):
        raise IsolationError("sampling_seeds must be a nonempty integer list")
    if len(seeds) != len(set(seeds)):
        raise IsolationError("sampling_seeds contains duplicates")
    if int(config["numerical_seed"]) in set(seeds):
        raise IsolationError("numerical_seed must be separate from LLM sampling seeds")
    if int(config["bootstrap"]["seed"]) in set(seeds) | {int(config["numerical_seed"])}:
        raise IsolationError("bootstrap seed must be separate from sampling/numerical seeds")
    if not 0 <= float(sampling["temperature"]):
        raise IsolationError("temperature must be nonnegative")
    if sampling.get("system_prompt_mode") != "model_neutralized_frozen_v1":
        raise IsolationError("Cross-model runs require the declared model-neutral prompt mode")
    if not 0 < float(sampling["top_p"]) <= 1:
        raise IsolationError("top_p must lie in (0,1]")
    models = config["models"]
    if not isinstance(models, list) or not models:
        raise IsolationError("At least one model must be configured")
    names = []
    for item in models:
        if not isinstance(item, dict) or not item.get("name") or not item.get("api_url"):
            raise IsolationError("Every model needs name and api_url")
        names.append(str(item["name"]))
        expected = item.get("expected_digest")
        if expected is not None and not re.fullmatch(r"[0-9a-f]{64}", str(expected).lower()):
            raise IsolationError(f"Invalid expected_digest for {item['name']}")
    if len(names) != len(set(names)):
        raise IsolationError("Model names must be unique")
    boundaries = config["claim_boundaries"]
    forbidden_true = [
        "fresh_out_of_sample", "router_temporal_validity_resolved",
        "feasible_information_to_fill_clock_resolved",
        "causal_memory_effect_from_closed_loop_factorial",
    ]
    if any(boundaries.get(key) is not False for key in forbidden_true):
        raise IsolationError("Unresolved scientific boundaries must remain false")
    return config


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    config = validate_config(load_json(path.resolve()))
    return config, sha256(path.resolve())


def verify_stage64_root(root: Path, expected_manifest: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = root / "PACKAGE_MANIFEST.sha256"
    if not manifest.is_file():
        raise IsolationError(f"Stage-6.4 package manifest is missing: {manifest}")
    actual_manifest = sha256(manifest)
    if actual_manifest != expected_manifest:
        raise IsolationError(
            f"Stage-6.4 manifest identity differs: expected={expected_manifest} actual={actual_manifest}"
        )
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = (root / relative.strip().lstrip("*")).resolve()
        if root not in target.parents:
            raise IsolationError(f"Stage-6.4 manifest path escapes package: {relative}")
        if not target.is_file() or sha256(target) != expected:
            raise IsolationError(f"Stage-6.4 package file differs: {target}")
        checked += 1
    return {"root": str(root), "manifest_sha256": actual_manifest, "files_verified": checked}


def import_stage64(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from stage63lib.inputs import load_inputs
    from stage63lib.journal import Journal
    from stage63lib.legacy import SYSTEM_PROMPT, legacy, validate_decision
    from suite64.core_variants import build_core_variants
    from suite64.engine import run_arm
    return {
        "load_inputs": load_inputs,
        "Journal": Journal,
        "legacy": legacy,
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "validate_decision": validate_decision,
        "build_core_variants": build_core_variants,
        "run_arm": run_arm,
    }


def source_context(config: dict[str, Any]) -> dict[str, Any]:
    stage_root = Path(config["stage64_code_root"]).expanduser().resolve()
    stage_audit = verify_stage64_root(stage_root, config["expected_stage64_manifest_sha256"])
    modules = import_stage64(stage_root)
    source_config = load_json(stage_root / "source_config.json")
    source_config["checkpoint_every"] = int(config["execution"]["checkpoint_every"])
    source_config["minimum_valid_action_rate"] = float(config["execution"]["minimum_valid_action_rate"])
    period, base_config, accounting, risk, input_audit = modules["load_inputs"](
        Path(config["project_root"]), source_config, Path(config["full_dataset"])
    )
    cores, core_audit, core_trace = modules["build_core_variants"](
        period, base_config, Path(input_audit["corrected_source_run"])
    )
    full_rows = len(period.frame)
    return_dates = period.return_dates
    evaluation = (return_dates >= config["evaluation_start"]) & (return_dates <= config["evaluation_end"])
    eval_rows = int(evaluation.sum())
    if full_rows != int(config["expected_full_rows"]) or eval_rows != int(config["expected_evaluation_rows"]):
        raise IsolationError(f"Date contract mismatch: full={full_rows}, evaluation={eval_rows}")
    if not math.isclose(float(source_config["cost_rate"]), float(config["cost_rate"]), abs_tol=1e-15):
        raise IsolationError("Source and experiment transaction costs differ")
    return {
        "modules": modules,
        "source_config": source_config,
        "period": period,
        "base_config": base_config,
        "accounting": accounting,
        "risk": risk,
        "input_audit": input_audit,
        "stage64_audit": stage_audit,
        "cores": cores,
        "core_audit": core_audit,
        "core_trace": core_trace,
        "full_rows": full_rows,
        "evaluation_rows": eval_rows,
    }


def model_matches(requested: str, observed: str) -> bool:
    return observed == requested or observed == requested + ":latest" or requested == observed + ":latest"


class StrictOllamaProvider:
    """Model-generic Ollama provider with fail-closed output validation."""

    kind = "ollama"

    def __init__(self, model: dict[str, Any], sampling: dict[str, Any], seed: int,
                 agent: dict[str, Any], modules: dict[str, Any]):
        self.model = str(model["name"])
        self.root = str(model["api_url"]).rstrip("/").removesuffix("/api/chat")
        self.expected_digest = model.get("expected_digest")
        self.sampling = dict(sampling)
        self.seed = int(seed)
        self.agent = agent
        self.modules = modules
        original_prompt = str(self.modules["SYSTEM_PROMPT"])
        source_phrase = "bounded Llama expert"
        if source_phrase not in original_prompt:
            raise IsolationError("Frozen system prompt no longer contains the reviewed model-role phrase")
        self.system_prompt = original_prompt.replace(
            source_phrase, "bounded language-model expert", 1
        )
        self.calls = 0
        self.failure_dir: Path | None = None
        self.pinned = self.inspect()

    def set_failure_dir(self, path: Path) -> None:
        self.failure_dir = path

    def request(self, suffix: str, value: Any = None, timeout: float | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            self.root + suffix,
            data=None if value is None else json.dumps(value, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if value is None else "POST",
        )
        with urllib.request.urlopen(
            request, timeout=timeout or float(self.sampling["timeout_seconds"])
        ) as response:
            result = json.loads(response.read())
        if not isinstance(result, dict):
            raise IsolationError(f"Ollama {suffix} returned a non-object")
        return result

    def options(self) -> dict[str, Any]:
        return {
            "temperature": float(self.sampling["temperature"]),
            "top_p": float(self.sampling["top_p"]),
            "seed": self.seed,
            "num_ctx": int(self.sampling["num_ctx"]),
            "num_predict": int(self.sampling["num_predict"]),
        }

    def inspect(self) -> dict[str, Any]:
        tags = self.request("/api/tags", timeout=30)
        matches = [
            item for item in tags.get("models", [])
            if isinstance(item, dict) and model_matches(self.model, str(item.get("name", "")))
        ]
        if len(matches) != 1 or not matches[0].get("digest"):
            available = sorted(str(x.get("name")) for x in tags.get("models", []) if isinstance(x, dict))
            raise IsolationError(f"Model {self.model!r} is missing/ambiguous. Available={available}")
        actual_digest = str(matches[0]["digest"])
        if self.expected_digest is not None and actual_digest != self.expected_digest:
            raise IsolationError(
                f"Model digest differs for {self.model}: expected={self.expected_digest} actual={actual_digest}"
            )
        version = self.request("/api/version", timeout=30)
        show = self.request("/api/show", {"model": self.model}, timeout=60)
        stable = {key: show.get(key) for key in (
            "parameters", "template", "system", "model_info", "details", "capabilities"
        )}
        return {
            "requested_model": self.model,
            "observed_model": str(matches[0].get("name")),
            "model_digest": actual_digest,
            "ollama_version": version.get("version"),
            "serving_metadata_sha256": digest(stable),
            "system_prompt_mode": self.sampling["system_prompt_mode"],
            "system_prompt_sha256": hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest(),
            "options": self.options(),
        }

    def assert_identity(self) -> None:
        if self.inspect() != self.pinned:
            raise IsolationError(f"Model or serving metadata changed during run: {self.model}")

    def complete(self, payload: dict[str, Any]):
        every = int(self.sampling["identity_check_every_calls"])
        if every <= 0:
            raise IsolationError("identity_check_every_calls must be positive")
        if self.calls % every == 0:
            self.assert_identity()
        schema = {
            "type": "object", "additionalProperties": False,
            "required": ["action", "confidence", "reason_codes", "cited_memory_ids"],
            "properties": {
                "action": {"type": "string", "enum": self.agent["actions"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason_codes": {
                    "type": "array", "minItems": 1,
                    "items": {"type": "string", "enum": self.agent["reason_codes"]},
                },
                "cited_memory_ids": {"type": "array", "items": {"type": "string"}},
            },
        }
        body = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, sort_keys=True, allow_nan=False)},
            ],
            "options": self.options(),
        }
        started = time.monotonic()
        outer = self.request("/api/chat", body)
        latency = time.monotonic() - started
        if outer.get("error") or not outer.get("done"):
            raise IsolationError(f"Incomplete Ollama response: {outer.get('error') or outer.get('done_reason')}")
        if not model_matches(self.model, str(outer.get("model", ""))):
            raise IsolationError(f"Response model does not match request: {outer.get('model')}")
        raw = outer.get("message", {}).get("content")
        if not isinstance(raw, str):
            raise IsolationError("Ollama response has no string message.content")
        visible = {
            str(item["episode_id"])
            for item in payload.get("memory", {}).get("similar_completed_episodes", [])
        }
        try:
            self.modules["validate_decision"](raw, self.agent, visible)
        except Exception as exc:
            evidence = {
                "status": "INVALID_MODEL_OUTPUT_STOP_BEFORE_SCORING",
                "model": self.model,
                "seed": self.seed,
                "request_sha256": digest(payload),
                "raw_response": raw,
                "provider_response": outer,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_at": now(),
            }
            if self.failure_dir is not None:
                atomic_json(self.failure_dir / "INVALID_MODEL_OUTPUT.json", evidence)
            raise IsolationError(
                f"Invalid model output; stopped before fallback/scoring. model={self.model} error={exc}"
            ) from exc
        self.calls += 1
        return raw, outer, latency


def enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in config["models"] if item.get("enabled", True)]


def specs() -> list[dict[str, Any]]:
    result = []
    for trust, memory, controller in ARM_FACTORS:
        name = f"llm_{trust}_{memory}_controller_{controller}"
        result.append({
            "name": name,
            "advisor": "llama",
            "trust": trust,
            "memory": "expanding" if memory == "memory" else "none",
            "core": "original",
            "risk": "standard" if controller == "on" else "none",
            "fixed_beta": 0.05,
            "factor_memory": memory,
            "factor_controller": controller,
        })
    return result


def numerical_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": f"numerical_only_controller_{controller}",
            "advisor": "abstain", "trust": "fixed", "memory": "none",
            "core": "original", "risk": "standard" if controller == "on" else "none",
            "fixed_beta": 0.05, "factor_memory": "none", "factor_controller": controller,
        }
        for controller in ("on", "off")
    ]


def estimate_plan(config: dict[str, Any], model_names: list[str], seeds: list[int]) -> dict[str, Any]:
    rows = int(config["expected_full_rows"])
    llm_arms = len(specs())
    calls_per_cell = rows * llm_arms
    return {
        "models": model_names,
        "sampling_seeds": seeds,
        "full_sequential_rows_per_arm": rows,
        "evaluation_rows_per_arm": int(config["expected_evaluation_rows"]),
        "llm_arms_per_model_seed": llm_arms,
        "numerical_arms_run_once": len(numerical_specs()),
        "llm_calls_per_model_seed": calls_per_cell,
        "maximum_llm_calls": calls_per_cell * len(model_names) * len(seeds),
        "warning": "The full matrix is expensive. Run the determinism probe before authorizing repeated temperature-zero seeds.",
    }


def select_models(config: dict[str, Any], selected: str | None) -> list[dict[str, Any]]:
    available = {str(item["name"]): item for item in enabled_models(config)}
    names = parse_str_list(selected, available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise IsolationError(f"Requested models are not enabled in config: {unknown}")
    return [available[name] for name in names]


def command_plan(args: argparse.Namespace) -> None:
    config, _ = load_config(args.config)
    models = select_models(config, args.models)
    seeds = parse_int_list(args.seeds, config["sampling"]["sampling_seeds"])
    print(json.dumps(estimate_plan(config, [x["name"] for x in models], seeds), indent=2))


def command_preflight(args: argparse.Namespace) -> None:
    package_manifest = verify_package_manifest()
    config, config_hash = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = source_context(config)
    model_records = []
    resolved = copy.deepcopy(config)
    resolved_models = {str(x["name"]): x for x in resolved["models"]}
    for model in enabled_models(config):
        provider = StrictOllamaProvider(
            model, config["sampling"], config["sampling"]["sampling_seeds"][0],
            context["source_config"]["agent"], context["modules"],
        )
        model_records.append(provider.pinned)
        resolved_models[model["name"]]["expected_digest"] = provider.pinned["model_digest"]
    resolved["models"] = list(resolved_models.values())
    resolved_path = output / "RESOLVED_CONFIG.json"
    atomic_json(resolved_path, resolved)
    resolved_hash = sha256(resolved_path)
    report = {
        "status": "PASS_NO_MODEL_INFERENCE",
        "created_at": now(),
        "package_manifest_sha256": package_manifest,
        "config": str(args.config.resolve()),
        "input_config_sha256": config_hash,
        "config_sha256": resolved_hash,
        "resolved_config": str(resolved_path),
        "resolved_config_sha256": resolved_hash,
        "stage64": context["stage64_audit"],
        "source_input_audit": context["input_audit"],
        "core_audit": context["core_audit"],
        "full_rows": context["full_rows"],
        "evaluation_rows": context["evaluation_rows"],
        "models": model_records,
        "cross_model_prompt_control": {
            "mode": config["sampling"]["system_prompt_mode"],
            "change": "bounded Llama expert -> bounded language-model expert for every model",
            "task_or_schema_changed": False,
        },
        "declared_state_controlled_contract_identity": {
            "pipeline_version": config.get("expected_pipeline_version"),
            "pipeline_sha256": config.get("expected_pipeline_sha256"),
            "status": "RECORDED_ONLY_NOT_USED_AS_STAGE64_CODE_IDENTITY",
        },
        "plan": estimate_plan(
            config,
            [x["name"] for x in enabled_models(config)],
            list(config["sampling"]["sampling_seeds"]),
        ),
        "claim_boundaries": config["claim_boundaries"],
    }
    atomic_json(output / "PREFLIGHT.json", report)
    context["core_trace"].to_csv(output / "CORE_DAILY_STATE.csv", index=False, float_format="%.17g")
    print(json.dumps({
        "status": report["status"], "preflight": str(output / "PREFLIGHT.json"),
        "resolved_config": str(resolved_path), "full_rows": context["full_rows"],
        "evaluation_rows": context["evaluation_rows"],
    }, indent=2))


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or count > length:
        raise IsolationError("probe prompt_count must lie between 1 and archived call count")
    if count == 1:
        return [0]
    return sorted({round(i * (length - 1) / (count - 1)) for i in range(count)})


def archived_payloads(config: dict[str, Any], source_config: dict[str, Any], count: int) -> list[dict[str, Any]]:
    path = Path(config["project_root"]) / source_config["source_result_relative"] / "08_LLM_CALLS.jsonl"
    if not path.is_file():
        raise IsolationError(f"Archived exact-request source is missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            payload = row.get("request")
            if not isinstance(payload, dict):
                raise IsolationError(f"Archived call {line_number} has no request object")
            rows.append(payload)
    return [rows[index] for index in evenly_spaced_indices(len(rows), count)]


def command_probe(args: argparse.Namespace) -> None:
    config, config_hash = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    context = source_context(config)
    models = select_models(config, args.models)
    seeds = parse_int_list(args.seeds, config["sampling"]["sampling_seeds"])
    prompt_count = int(args.prompt_count or config["probe"]["prompt_count"])
    repeats = int(args.repeats or config["probe"]["repeats_per_seed"])
    if repeats < 2:
        raise IsolationError("Determinism probe requires at least two repeats per seed")
    payloads = archived_payloads(config, context["source_config"], prompt_count)
    records = []
    for model in models:
        for seed in seeds:
            provider = StrictOllamaProvider(
                model, config["sampling"], seed, context["source_config"]["agent"], context["modules"]
            )
            failure = output / safe_slug(model["name"]) / f"seed_{seed}"
            provider.set_failure_dir(failure)
            for prompt_index, payload in enumerate(payloads):
                for repeat in range(repeats):
                    raw, outer, latency = provider.complete(payload)
                    parsed = json.loads(raw)
                    records.append({
                        "model": model["name"], "model_digest": provider.pinned["model_digest"],
                        "sampling_seed": seed, "prompt_index": prompt_index, "repeat": repeat,
                        "request_sha256": digest(payload), "raw_response": raw,
                        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        "semantic_sha256": digest(parsed), "action": parsed["action"],
                        "latency_seconds": latency,
                        "input_tokens": int(outer.get("prompt_eval_count", 0) or 0),
                        "output_tokens": int(outer.get("eval_count", 0) or 0),
                    })
            provider.assert_identity()
    summaries = []
    for model in [x["name"] for x in models]:
        subset = [x for x in records if x["model"] == model]
        within = True
        for seed in seeds:
            for prompt_index in range(prompt_count):
                hashes = {
                    x["response_sha256"] for x in subset
                    if x["sampling_seed"] == seed and x["prompt_index"] == prompt_index
                }
                within &= len(hashes) == 1
        across = True
        for prompt_index in range(prompt_count):
            hashes = {
                x["response_sha256"] for x in subset
                if x["prompt_index"] == prompt_index and x["repeat"] == 0
            }
            across &= len(hashes) == 1
        summaries.append({
            "model": model, "temperature": config["sampling"]["temperature"],
            "within_seed_exact_repeatability": within,
            "across_seed_exact_equality": across,
            "seed_realizations_distinct_on_probe": not across,
            "interpretation": (
                "MULTIPLE_SEEDS_NOT_INDEPENDENT_ON_PROBE" if across
                else "SERVING_OUTPUT_VARIES_ACROSS_SEEDS_ON_PROBE"
            ),
        })
    report = {
        "status": "PASS", "created_at": now(), "config_sha256": config_hash,
        "prompts": prompt_count, "repeats_per_seed": repeats, "seeds": seeds,
        "summaries": summaries, "records": records,
    }
    atomic_json(output / "DETERMINISM_PROBE.json", report)
    print(json.dumps({"status": "PASS", "summaries": summaries,
                      "report": str(output / "DETERMINISM_PROBE.json")}, indent=2))


def artifact_map(directory: Path, exclusions: set[str] | None = None) -> dict[str, str]:
    exclusions = exclusions or set()
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in exclusions or path.name.endswith((".pending", ".tmp")):
            continue
        result[path.relative_to(directory).as_posix()] = sha256(path)
    return result


def verify_complete(directory: Path, filename: str = "RUN_COMPLETE.json") -> dict[str, Any]:
    complete = load_json(directory / filename)
    if complete.get("status") != "COMPLETE":
        raise IsolationError(f"Completion record is not COMPLETE: {directory / filename}")
    for relative, expected in complete.get("artifact_sha256", {}).items():
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents or not path.is_file() or sha256(path) != expected:
            raise IsolationError(f"Completed artifact differs: {path}")
    return complete


def write_frame(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def run_one_arm(context: dict[str, Any], output: Path, spec: dict[str, Any],
                identity: dict[str, Any], provider: StrictOllamaProvider | None,
                resume: bool):
    import pandas as pd

    complete_path = output / "RUN_COMPLETE.json"
    if complete_path.is_file():
        complete = verify_complete(output)
        if complete.get("identity") != identity or complete.get("spec") != spec:
            raise IsolationError(f"Completed arm identity differs: {output}")
        return pd.read_csv(output / "DAILY_LEDGER.csv", low_memory=False)
    if output.exists() and any(output.iterdir()) and not resume:
        raise IsolationError(f"Arm output exists but is incomplete; rerun with --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if provider is not None:
        provider.set_failure_dir(output)
    journal = None
    if spec["advisor"] == "llama":
        journal = context["modules"]["Journal"](output, provider, identity)
    rows, arm = context["modules"]["run_arm"](
        context["period"], context["source_config"], context["base_config"],
        context["accounting"], context["risk"], journal, output, spec,
        core_desired=context["cores"]["original"],
    )
    frame = pd.DataFrame(rows)
    if len(frame) != context["full_rows"]:
        raise IsolationError(f"Arm row count differs: {output} rows={len(frame)}")
    if spec["advisor"] == "llama" and not bool(frame["valid"].all()):
        raise IsolationError(f"Invalid model output entered ledger: {output}")
    write_frame(frame, output / "DAILY_LEDGER.csv")
    complete = {
        "status": "COMPLETE", "created_at": now(), "identity": identity,
        "spec": spec, "rows": len(frame),
        "new_calls": int(getattr(journal, "new_calls", 0)),
        "reused_calls": int(getattr(journal, "reused_calls", 0)),
        "final_wealth": float(arm.wealth),
        "artifact_sha256": artifact_map(output, {"RUN_COMPLETE.json", "RUN.lock"}),
    }
    atomic_json(complete_path, complete)
    return frame


def probe_allows_seeds(probe: dict[str, Any], model: str) -> bool:
    record = next((x for x in probe.get("summaries", []) if x.get("model") == model), None)
    if record is None:
        raise IsolationError(f"Determinism probe does not cover model: {model}")
    return bool(record.get("seed_realizations_distinct_on_probe"))


def command_run(args: argparse.Namespace) -> None:
    import pandas as pd

    package_manifest = verify_package_manifest()
    config, config_hash = load_config(args.config)
    preflight = load_json(args.preflight.resolve())
    if preflight.get("status") != "PASS_NO_MODEL_INFERENCE" or preflight.get("config_sha256") != config_hash:
        raise IsolationError("Preflight is missing, failed, or belongs to another configuration")
    models = select_models(config, args.models)
    seeds = parse_int_list(args.seeds, config["sampling"]["sampling_seeds"])
    if len(seeds) > 1 and float(config["sampling"]["temperature"]) == 0:
        if args.probe is None:
            raise IsolationError("Multiple temperature-zero seeds require --probe")
        probe = load_json(args.probe.resolve())
        if probe.get("status") != "PASS" or probe.get("config_sha256") != config_hash:
            raise IsolationError("Determinism probe failed or belongs to another resolved configuration")
        for model in models:
            if not probe_allows_seeds(probe, model["name"]) and not args.allow_redundant_seeds:
                raise IsolationError(
                    f"{model['name']} produced identical outputs across seeds in the probe. "
                    "Use one seed, or --allow-redundant-seeds only to document exact replication."
                )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "RUN.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise IsolationError(f"Another process is using {output}") from exc
    identity = {
        "experiment_id": config["experiment_id"], "config_sha256": config_hash,
        "preflight_sha256": sha256(args.preflight.resolve()),
        "package_manifest_sha256": package_manifest,
        "models": [x["name"] for x in models], "sampling_seeds": seeds,
        "numerical_seed": int(config["numerical_seed"]),
        "runtime": {"python": platform.python_version(), "numpy": __import__("numpy").__version__,
                    "pandas": pd.__version__},
    }
    contract_path = output / "00_RUN_CONTRACT.json"
    if contract_path.is_file():
        if not args.resume:
            raise IsolationError(f"Run exists; use --resume: {output}")
        if load_json(contract_path).get("identity") != identity:
            raise IsolationError("Resume identity changed")
        if (output / "RUN_COMPLETE.json").is_file():
            verify_complete(output)
            print(json.dumps({"status": "ALREADY_COMPLETE_NO_NEW_CALLS", "output": str(output)}, indent=2))
            return
    else:
        if args.resume:
            raise IsolationError("--resume requested but no run contract exists")
        atomic_json(contract_path, {
            "identity": identity, "started_at": now(), "factorial": {
                "memory": ["memory", "no_memory"], "authority": ["adaptive", "fixed"],
                "controller": ["on", "off"],
            }, "controller_off": "bypass CVaR, grid and turnover limit; retain [0,1] and costs",
            "claim_boundaries": config["claim_boundaries"],
        })
    try:
        context = source_context(config)
        atomic_json(output / "01_SOURCE_AUDIT.json", context["input_audit"])
        atomic_json(output / "02_CORE_AUDIT.json", context["core_audit"])
        write_frame(context["core_trace"], output / "CORE_DAILY_STATE.csv")
        all_frames = []

        numerical_root = output / "numerical_controls"
        for spec in numerical_specs():
            arm_identity = {
                "run": identity, "model": "NUMERICAL_ONLY",
                "sampling_seed": None, "numerical_seed": int(config["numerical_seed"]),
                "provider_identity": {"kind": "none", "model_calls": 0},
            }
            frame = run_one_arm(context, numerical_root / spec["name"], spec, arm_identity, None, args.resume)
            frame = frame.copy()
            frame["model"] = "NUMERICAL_ONLY"
            frame["model_digest"] = ""
            frame["sampling_seed"] = ""
            frame["numerical_seed"] = int(config["numerical_seed"])
            frame["logical_arm"] = spec["name"]
            frame["arm"] = spec["name"]
            all_frames.append(frame)

        for model in models:
            for seed in seeds:
                cell_id = f"{safe_slug(model['name'])}__seed_{seed}"
                cell_root = output / "cells" / cell_id
                provider = StrictOllamaProvider(
                    model, config["sampling"], seed, context["source_config"]["agent"], context["modules"]
                )
                for spec in specs():
                    arm_identity = {
                        "run": identity, "cell_id": cell_id, "model": model["name"],
                        "sampling_seed": seed, "provider_identity": provider.pinned,
                    }
                    frame = run_one_arm(
                        context, cell_root / "arms" / spec["name"], spec,
                        arm_identity, provider, args.resume,
                    )
                    frame = frame.copy()
                    frame["model"] = model["name"]
                    frame["model_digest"] = provider.pinned["model_digest"]
                    frame["sampling_seed"] = seed
                    frame["numerical_seed"] = ""
                    frame["logical_arm"] = spec["name"]
                    frame["arm"] = f"{cell_id}__{spec['name']}"
                    all_frames.append(frame)
                    print(f"CELL={cell_id} ARM={spec['name']} STATUS=COMPLETE", flush=True)
                provider.assert_identity()
                atomic_json(cell_root / "CELL_COMPLETE.json", {
                    "status": "COMPLETE", "cell_id": cell_id, "model": model["name"],
                    "sampling_seed": seed, "provider_identity": provider.pinned,
                    "arms": [x["name"] for x in specs()], "completed_at": now(),
                })

        ledger = pd.concat(all_frames, ignore_index=True)
        write_frame(ledger, output / "ALL_DAILY_LEDGER.csv")
        expected_arms = len(numerical_specs()) + len(models) * len(seeds) * len(specs())
        if ledger["arm"].nunique() != expected_arms:
            raise IsolationError("Consolidated arm count differs from the declared matrix")
        complete = {
            "status": "COMPLETE", "completed_at": now(), "identity": identity,
            "full_rows_per_arm": context["full_rows"],
            "evaluation_rows_per_arm": context["evaluation_rows"],
            "arms": int(ledger["arm"].nunique()), "rows": len(ledger),
            "llm_calls_expected": context["full_rows"] * len(models) * len(seeds) * len(specs()),
            "scientific_scope": config["scientific_scope"],
            "artifact_sha256": artifact_map(output, {"RUN_COMPLETE.json", "RUN.lock"}),
        }
        atomic_json(output / "RUN_COMPLETE.json", complete)
        print(json.dumps({"status": "COMPLETE", "output": str(output),
                          "arms": complete["arms"], "rows": complete["rows"]}, indent=2))
    except Exception as exc:
        atomic_json(output / "FAILURE.json", {
            "status": "STOP_TECHNICAL_OR_SOURCE", "created_at": now(),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "economic_stop": False,
        })
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="print arm/call budget without loading project data")
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--models")
    plan.add_argument("--seeds")
    plan.set_defaults(function=command_plan)

    preflight = sub.add_parser("preflight", help="read-only source/model identity audit")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(function=command_preflight)

    probe = sub.add_parser("probe", help="test repeatability and seed distinctness")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    probe.add_argument("--models")
    probe.add_argument("--seeds")
    probe.add_argument("--prompt-count", type=int)
    probe.add_argument("--repeats", type=int)
    probe.set_defaults(function=command_probe)

    run = sub.add_parser("run", help="run or resume the full sequential factorial")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--preflight", type=Path, required=True)
    run.add_argument("--probe", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--models")
    run.add_argument("--seeds")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--allow-redundant-seeds", action="store_true")
    run.set_defaults(function=command_run)
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; durable calls remain resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
