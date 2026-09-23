#!/usr/bin/env python3
"""Read-only audit of archived RAMAS memory/no-memory LLaMA call pairs.

This does not call Ollama and does not alter any experiment artifacts. It
checks whether the archived memory and no-memory calls have identical
non-memory requests, whether memory was actually retrieved, whether retrieved
episodes precede the decision, and whether observable advisor outputs differ.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


ROOT = Path("/home/infonet/wahid/leader_router_fresh")
START_DATE = "2022-01-01"
END_DATE = "2025-05-28"
OUTPUT = ROOT / "ramas_existing_call_pair_audit.json"

BRANCHES = {
    "adaptive": ROOT
    / "EXPERIMENT_BRANCHES/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/artifacts",
    "fixed": ROOT
    / "EXPERIMENT_BRANCHES/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/artifacts",
}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def latest_call_dir(artifacts: Path) -> Path:
    candidates = sorted(path for path in artifacts.glob("*/calls") if path.is_dir())
    if not candidates:
        raise RuntimeError(f"No calls directory found under {artifacts}")
    return candidates[-1]


def collect_calls(call_dir: Path) -> Dict[str, Dict[str, Path]]:
    result: Dict[str, Dict[str, Path]] = {}
    pattern = re.compile(r"^(\d+)-(memory|no_memory)\.json$")
    for path in sorted(call_dir.glob("*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        index, condition = match.groups()
        result.setdefault(index, {})[condition] = path
    return result


def strip_memory(request: Mapping[str, Any]) -> Dict[str, Any]:
    value = copy.deepcopy(dict(request))
    value.pop("memory", None)
    return value


def response_object(record: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = [
        record.get("raw_response"),
        ((record.get("provider_response") or {}).get("message") or {}).get("content"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        text = candidate.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"_parse_error": True}


def output_signature(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "action": value.get("action"),
        "confidence": value.get("confidence"),
        "reason_codes": value.get("reason_codes"),
        "cited_memory_ids": value.get("cited_memory_ids"),
    }


def memory_episodes(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    request = record.get("request") or {}
    memory = request.get("memory") or {}
    episodes = memory.get("similar_completed_episodes") or []
    if not isinstance(episodes, list):
        return []
    return [episode for episode in episodes if isinstance(episode, dict)]


def episode_date(episode: Mapping[str, Any]) -> Optional[str]:
    for key in ("episode_date", "decision_date", "return_date"):
        value = episode.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def diff_fields(left: Mapping[str, Any], right: Mapping[str, Any]) -> List[str]:
    fields = ["action", "confidence", "reason_codes", "cited_memory_ids"]
    return [field for field in fields if left.get(field) != right.get(field)]


def analyze_arm(label: str, artifacts: Path) -> Dict[str, Any]:
    call_dir = latest_call_dir(artifacts)
    calls = collect_calls(call_dir)
    total_pairs = 0
    window_pairs = 0
    complete_pairs = 0
    state_matches = 0
    model_contracts: Counter[str] = Counter()
    memory_nonempty = 0
    memory_empty = 0
    future_episode_count = 0
    invalid_episode_date_count = 0
    observable_change_count = 0
    action_change_count = 0
    confidence_change_count = 0
    reason_change_count = 0
    citation_change_count = 0
    exact_output_same_count = 0
    examples: List[Dict[str, Any]] = []

    for index in sorted(calls):
        pair = calls[index]
        total_pairs += 1
        if "memory" not in pair or "no_memory" not in pair:
            continue

        memory_record = read_json(pair["memory"])
        no_memory_record = read_json(pair["no_memory"])
        memory_request = memory_record.get("request") or {}
        no_memory_request = no_memory_record.get("request") or {}
        memory_state = memory_request.get("state") or {}
        no_memory_state = no_memory_request.get("state") or {}
        decision_date = str(memory_state.get("decision_date", ""))
        if not (START_DATE <= decision_date <= END_DATE):
            continue

        window_pairs += 1
        if memory_record.get("status", "success") != no_memory_record.get("status", "success"):
            continue
        complete_pairs += 1

        state_equal = strip_memory(memory_request) == strip_memory(no_memory_request)
        if state_equal:
            state_matches += 1

        identity = memory_record.get("identity") or {}
        provider = identity.get("provider_identity") or {}
        options = provider.get("options") or {}
        contract = {
            "model": provider.get("model"),
            "model_digest": provider.get("model_digest"),
            "temperature": options.get("temperature"),
            "seed": options.get("seed"),
            "num_ctx": options.get("num_ctx"),
            "num_predict": options.get("num_predict"),
            "system_prompt_sha256": identity.get("system_prompt_sha256"),
        }
        model_contracts[canonical(contract)] += 1

        episodes = memory_episodes(memory_record)
        if episodes:
            memory_nonempty += 1
        else:
            memory_empty += 1

        for episode in episodes:
            date = episode_date(episode)
            if date is None:
                invalid_episode_date_count += 1
            elif date >= decision_date:
                future_episode_count += 1

        memory_output = output_signature(response_object(memory_record))
        no_memory_output = output_signature(response_object(no_memory_record))
        changed = diff_fields(memory_output, no_memory_output)
        if changed:
            observable_change_count += 1
            action_change_count += int("action" in changed)
            confidence_change_count += int("confidence" in changed)
            reason_change_count += int("reason_codes" in changed)
            citation_change_count += int("cited_memory_ids" in changed)
        else:
            exact_output_same_count += 1

        if len(examples) < 25 and episodes and changed:
            examples.append(
                {
                    "index": index,
                    "decision_date": decision_date,
                    "memory_episode_count": len(episodes),
                    "changed_fields": changed,
                    "memory_output": memory_output,
                    "no_memory_output": no_memory_output,
                    "first_memory_episode": episodes[0],
                }
            )

    first_contract = json.loads(next(iter(model_contracts))) if model_contracts else None
    return {
        "authority": label,
        "call_directory": str(call_dir),
        "retrospective_only": True,
        "evaluation_window": {
            "start": START_DATE,
            "end": END_DATE,
            "expected_dates": 1244,
        },
        "pair_counts": {
            "all_index_pairs": total_pairs,
            "window_pairs": window_pairs,
            "complete_pairs": complete_pairs,
            "non_memory_request_matches": state_matches,
            "non_memory_request_mismatch": window_pairs - state_matches,
        },
        "memory_retrieval": {
            "nonempty_memory_pairs": memory_nonempty,
            "empty_memory_pairs": memory_empty,
            "future_or_same_day_episode_count": future_episode_count,
            "missing_episode_date_count": invalid_episode_date_count,
        },
        "observable_output_change": {
            "pairs_with_any_change": observable_change_count,
            "pairs_with_exactly_same_observable_output": exact_output_same_count,
            "action_changes": action_change_count,
            "confidence_changes": confidence_change_count,
            "reason_code_changes": reason_change_count,
            "citation_changes": citation_change_count,
        },
        "provider_contract_example": first_contract,
        "provider_contract_variants": len(model_contracts),
        "examples": examples,
    }


def main() -> None:
    report = {
        "experiment": "RAMAS_existing_memory_no_memory_call_pair_audit",
        "script_version": "1.0",
        "read_only": True,
        "note": (
            "This audits archived RETROSPECTIVE_LLAMA70B calls. It does not "
            "replace the planned prospective state-controlled rerun."
        ),
        "arms": {
            label: analyze_arm(label, artifacts)
            for label, artifacts in BRANCHES.items()
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    for label, result in report["arms"].items():
        print(
            label,
            json.dumps(
                {
                    "window_pairs": result["pair_counts"]["window_pairs"],
                    "state_matches": result["pair_counts"]["non_memory_request_matches"],
                    "memory_nonempty": result["memory_retrieval"]["nonempty_memory_pairs"],
                    "future_episode_count": result["memory_retrieval"]["future_or_same_day_episode_count"],
                    "observable_changes": result["observable_output_change"]["pairs_with_any_change"],
                },
                sort_keys=True,
            ),
        )


if __name__ == "__main__":
    main()
