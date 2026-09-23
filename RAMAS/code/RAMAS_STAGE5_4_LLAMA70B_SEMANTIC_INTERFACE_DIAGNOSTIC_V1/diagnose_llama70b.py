#!/usr/bin/env python3
"""Isolate Llama-70B decision compliance on frozen RAMAS Stage 5.3 contexts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "llama3.3:70b"
SEED = 12031
ALLOWED_RESIDUALS = (-0.25, 0.0, 0.25)
SYSTEM_PROMPT = """You are the decision component of a bounded, tool-using Bitcoin allocation agent.
The transparent controller already selected the default exposure. Choose only a residual from -0.25, 0, or 0.25.
Residual zero means abstain and keep the transparent controller. A nonzero residual must be marked eligible by compare_allowed_actions and must cite the result identifier of every required tool.
Tool results are past-only evidence. Confidence and explanation cannot override the code-side evidence gate.
Do not optimize prediction accuracy. Seek after-cost growth while avoiding worse worst-day loss CVaR.
Return exactly one JSON object with residual, confidence, cited_tool_result_ids, and lesson.
lesson must be {\"action\":\"none\",\"text\":\"\"} unless proposing one concise causal hypothesis for the chosen nonzero residual.
No prose outside JSON."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decision_schema(tool_result_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["residual", "confidence", "cited_tool_result_ids", "lesson"],
        "properties": {
            "residual": {"type": "number", "enum": list(ALLOWED_RESIDUALS)},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "cited_tool_result_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": sorted(tool_result_ids)},
            },
            "lesson": {
                "type": "object",
                "additionalProperties": False,
                "required": ["action", "text"],
                "properties": {
                    "action": {"type": "string", "enum": ["propose", "none"]},
                    "text": {"type": "string", "maxLength": 300},
                },
            },
        },
    }


def parse_answer(raw: str, allowed_ids: set[str]) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("response is not an object")
    if set(value) != {"residual", "confidence", "cited_tool_result_ids", "lesson"}:
        raise ValueError("response keys do not match the frozen contract")
    residual = value["residual"]
    if isinstance(residual, bool) or not isinstance(residual, (int, float)):
        raise ValueError("residual is not numeric")
    matches = [candidate for candidate in ALLOWED_RESIDUALS if abs(float(residual) - candidate) <= 1e-12]
    if len(matches) != 1:
        raise ValueError("residual is outside the frozen grid")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        raise ValueError("confidence is invalid")
    citations = value["cited_tool_result_ids"]
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise ValueError("citations are invalid")
    if len(citations) != len(set(citations)) or not set(citations).issubset(allowed_ids):
        raise ValueError("citations are duplicated or not visible")
    lesson = value["lesson"]
    if not isinstance(lesson, dict) or set(lesson) != {"action", "text"}:
        raise ValueError("lesson is invalid")
    if lesson["action"] not in {"propose", "none"} or not isinstance(lesson["text"], str):
        raise ValueError("lesson fields are invalid")
    return {
        "residual": float(matches[0]),
        "confidence": float(confidence),
        "cited_tool_result_ids": citations,
        "lesson": lesson,
    }


def comparison_rows(call: dict[str, Any]) -> list[dict[str, Any]]:
    tools = call["request"]["executed_tool_results"]
    comparison = next(item for item in tools if item["tool_name"] == "compare_allowed_actions")
    return comparison["result"]["residual_comparison"]


def expected_residual(call: dict[str, Any]) -> float:
    eligible = [
        row for row in comparison_rows(call)
        if float(row["residual"]) != 0.0 and bool(row["eligible"])
    ]
    if not eligible:
        return 0.0
    return float(max(eligible, key=lambda row: float(row["mean_log_advantage"]))["residual"])


def qwen_residual(call: dict[str, Any]) -> float:
    return float(json.loads(call["raw_response"])["residual"])


def evenly_select(items: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    positions = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)] if count > 1 else [0]
    return [items[position] for position in positions]


def select_cases(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(calls, key=lambda call: call["request"]["state"]["decision_date"])
    eligible = [call for call in ordered if expected_residual(call) != 0.0]
    if not eligible:
        raise ValueError("no eligible nonzero context exists; the diagnostic cannot test acceptance")
    ineligible_nonzero = [call for call in ordered if expected_residual(call) == 0.0 and qwen_residual(call) != 0.0]
    abstentions = [call for call in ordered if qwen_residual(call) == 0.0]
    chosen = eligible + evenly_select(ineligible_nonzero, 19) + evenly_select(abstentions, 10)
    unique: dict[str, dict[str, Any]] = {call["request_id"]: call for call in chosen}
    if len(unique) != 30:
        raise ValueError(f"expected 30 unique diagnostic cases, found {len(unique)}")
    return sorted(unique.values(), key=lambda call: call["request"]["state"]["decision_date"])


def ollama_decide(endpoint: str, model: str, call: dict[str, Any], timeout: float) -> tuple[str, dict[str, Any], float]:
    request_payload = call["request"]
    ids = [item["tool_result_id"] for item in request_payload["executed_tool_results"]]
    body = {
        "model": model,
        "stream": False,
        "format": decision_schema(ids),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(request_payload, sort_keys=True, allow_nan=False)},
        ],
        "options": {"temperature": 0.0, "seed": SEED},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, allow_nan=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        outer = json.loads(response.read().decode("utf-8"))
    latency = time.monotonic() - start
    if "error" in outer:
        raise RuntimeError(str(outer["error"]))
    raw = outer.get("message", {}).get("content")
    if not isinstance(raw, str):
        raise ValueError("Ollama response has no message content")
    return raw, outer, latency


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/chat")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    all_calls = [json.loads(line) for line in args.calls.open(encoding="utf-8") if line.strip()]
    decision_calls = [call for call in all_calls if call.get("stage") == "DECISION" and call.get("status") == "OK"]
    cases = select_cases(decision_calls)
    contract = {
        "experiment_id": "RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1",
        "scientific_label": "MODEL_INTERFACE_DIAGNOSTIC_ONLY_NO_ECONOMIC_CLAIM",
        "source_calls": str(args.calls.resolve()),
        "source_calls_sha256": sha256_file(args.calls),
        "model": args.model,
        "seed": SEED,
        "case_count": len(cases),
        "selection": {
            "eligible_nonzero_cases": sum(expected_residual(case) != 0.0 for case in cases),
            "qwen_ineligible_nonzero_cases": sum(expected_residual(case) == 0.0 and qwen_residual(case) != 0.0 for case in cases),
            "qwen_abstention_cases": sum(qwen_residual(case) == 0.0 for case in cases),
        },
        "full_replay_performed": False,
        "oos_opened": False,
    }
    write_json(args.output / "00_CONTRACT.json", contract)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, call in enumerate(cases, 1):
        state = call["request"]["state"]
        ids = {item["tool_result_id"] for item in call["request"]["executed_tool_results"]}
        expected = expected_residual(call)
        raw = ""
        outer: dict[str, Any] = {}
        latency = 0.0
        error = ""
        answer: dict[str, Any] | None = None
        try:
            raw, outer, latency = ollama_decide(args.endpoint, args.model, call, args.timeout)
            answer = parse_answer(raw, ids)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        residual = answer["residual"] if answer else None
        citations = set(answer["cited_tool_result_ids"]) if answer else set()
        full_citations = bool(answer is not None and ids.issubset(citations))
        selected_row = next(
            (row for row in comparison_rows(call) if residual is not None and abs(float(row["residual"]) - residual) <= 1e-12),
            None,
        )
        tool_eligible = bool(residual == 0.0 or (selected_row is not None and selected_row["eligible"]))
        semantic_accept = bool(answer is not None and tool_eligible and (residual == 0.0 or full_citations))
        agreement = bool(answer is not None and abs(float(residual) - expected) <= 1e-12)
        row = {
            "case": index,
            "request_id": call["request_id"],
            "decision_date": state["decision_date"],
            "hard_regime": state["hard_regime"],
            "uncertainty_bucket": state["uncertainty_bucket"],
            "qwen_residual": qwen_residual(call),
            "expected_residual": expected,
            "llama_residual": "" if residual is None else residual,
            "valid_json_contract": answer is not None,
            "expected_action_agreement": agreement,
            "all_required_citations": full_citations,
            "selected_action_tool_eligible": tool_eligible,
            "semantic_accept": semantic_accept,
            "citation_count": len(citations),
            "latency_seconds": latency,
            "error": error,
        }
        rows.append(row)
        records.append({
            "case": index,
            "source_request_id": call["request_id"],
            "request": call["request"],
            "raw_response": raw,
            "parsed_response": answer,
            "provider_response": outer,
            "diagnostic": row,
        })
        print(
            f"LLAMA70B_PROGRESS={index}/30 DATE={state['decision_date']} "
            f"EXPECTED={expected:+.2f} LLAMA={residual} ACCEPTED={semantic_accept}",
            flush=True,
        )

    with (args.output / "01_CASE_RESULTS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "02_LLAMA70B_CALLS.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    valid_rate = sum(bool(row["valid_json_contract"]) for row in rows) / len(rows)
    agreement_rate = sum(bool(row["expected_action_agreement"]) for row in rows) / len(rows)
    semantic_rate = sum(bool(row["semantic_accept"]) for row in rows) / len(rows)
    nonzero_rows = [row for row in rows if row["llama_residual"] not in ("", 0.0)]
    citation_rate = (
        sum(bool(row["all_required_citations"]) for row in nonzero_rows) / len(nonzero_rows)
        if nonzero_rows else None
    )
    eligible_case = next(row for row in rows if row["expected_residual"] != 0.0)
    eligible_accepted = bool(
        eligible_case["expected_action_agreement"]
        and eligible_case["all_required_citations"]
        and eligible_case["semantic_accept"]
    )
    if not eligible_accepted and eligible_case["llama_residual"] == 0.0:
        decision = "LLAMA70B_ALWAYS_ABSTAINED_ON_THE_ELIGIBLE_CASE"
    elif valid_rate < 0.95 or agreement_rate < 0.90 or semantic_rate < 0.95 or not eligible_accepted:
        decision = "LLAMA70B_SEMANTIC_INTERFACE_FAILED"
    else:
        decision = "LLAMA70B_INTERFACE_BETTER_NO_ECONOMIC_CLAIM"
    summary = {
        "status": "PASS",
        "decision": decision,
        "model": args.model,
        "cases": len(rows),
        "valid_json_contract_rate": valid_rate,
        "expected_action_agreement_rate": agreement_rate,
        "semantic_accept_rate": semantic_rate,
        "nonzero_response_count": len(nonzero_rows),
        "nonzero_full_citation_rate": citation_rate,
        "eligible_case_accepted": eligible_accepted,
        "total_latency_seconds": sum(float(row["latency_seconds"]) for row in rows),
        "economic_claim_permitted": False,
        "full_replay_performed": False,
        "oos_opened": False,
    }
    write_json(args.output / "03_SUMMARY.json", summary)
    report = [
        "# RAMAS Stage 5.4 Llama-70B semantic-interface diagnostic",
        "",
        f"Decision: **{decision}**",
        "",
        f"- Valid JSON/contract rate: {valid_rate:.1%}",
        f"- Agreement with deterministic tool eligibility: {agreement_rate:.1%}",
        f"- Semantically accepted decision rate: {semantic_rate:.1%}",
        f"- Nonzero responses: {len(nonzero_rows)}",
        f"- Full-citation rate among nonzero responses: {'N/A' if citation_rate is None else f'{citation_rate:.1%}'}",
        f"- Single eligible case accepted correctly: {eligible_accepted}",
        "",
        "This is an interface and instruction-following comparison only. It neither reruns the portfolio nor changes the failed Stage 5.3 economic conclusion.",
    ]
    (args.output / "04_PLAIN_ENGLISH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    artifacts = {}
    for path in sorted(args.output.iterdir()):
        if path.is_file():
            artifacts[path.name] = sha256_file(path)
    write_json(args.output / "RUN_COMPLETE.json", {"status": "PASS", "decision": decision, "artifact_sha256": artifacts})
    print("RAMAS_STAGE5_4_LLAMA70B_DIAGNOSTIC_STATUS=PASS", flush=True)
    print(f"DECISION={decision}", flush=True)
    print(f"OUTPUT={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
