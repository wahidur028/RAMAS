from __future__ import annotations

import json
import math
from typing import Any


class ContractError(ValueError):
    pass


def validate_decision(raw: str, config: dict[str, Any], visible_ids: set[str]) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"response is not JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("response must be one JSON object")
    required = {"action", "confidence", "reason_codes", "cited_memory_ids"}
    if set(value) != required:
        raise ContractError(f"response keys must be exactly {sorted(required)}")
    action = value["action"]
    if action not in config["actions"]:
        raise ContractError(f"invalid action={action!r}")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ContractError("confidence must be numeric")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ContractError("confidence must be finite and within [0,1]")
    reasons = value["reason_codes"]
    if not isinstance(reasons, list) or not reasons or any(not isinstance(item, str) for item in reasons):
        raise ContractError("reason_codes must be a non-empty string list")
    allowed_reasons = set(config["reason_codes"])
    if any(item not in allowed_reasons for item in reasons):
        raise ContractError("reason_codes contain an unknown value")
    citations = value["cited_memory_ids"]
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise ContractError("cited_memory_ids must be a string list")
    if len(citations) != len(set(citations)):
        raise ContractError("cited_memory_ids contain duplicates")
    if any(item not in visible_ids for item in citations):
        raise ContractError("response cites memory that was not shown")
    return {
        "action": action,
        "confidence": confidence,
        "reason_codes": reasons,
        "cited_memory_ids": citations,
    }
