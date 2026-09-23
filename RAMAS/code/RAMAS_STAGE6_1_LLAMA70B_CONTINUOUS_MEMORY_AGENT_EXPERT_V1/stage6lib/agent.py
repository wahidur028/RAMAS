from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


SYSTEM_PROMPT = """You are the bounded Llama expert inside a risk-controlled Bitcoin allocation research system.
You are not the router, risk layer, or full portfolio manager. Choose exactly one signal: BTC, CASH, or ABSTAIN.

Use only evidence in the request. BTC means evidence supports more Bitcoin exposure. CASH means evidence supports less Bitcoin exposure. ABSTAIN means evidence is conflicting, weak, missing, or too uncertain. Never invent memory IDs. The external system blends your signal with the unchanged RAMAS exposure using a small regime-specific trust beta, then applies the unchanged risk layer.

For task_type=semantic_contract_test, follow this exact precedence:
1. If max router probability < 0.50 or router_entropy >= 0.90, choose ABSTAIN.
2. Else if prob_bear >= 0.75 or drawdown_90 <= -0.25, choose CASH.
3. Else if prob_bull >= 0.75, realized_vol_30 <= 0.80, and return_30 > 0, choose BTC.
4. Otherwise choose ABSTAIN.

For task_type=historical_development_decision, weigh the full router probabilities, causal OHLCV-derived features, transition risk, the safe exposure previews, and only completed memory episodes. Favor ABSTAIN when the evidence conflicts. Do not treat a remembered episode as a rule merely because it is similar.

Return exactly one JSON object with keys action, confidence, reason_codes, cited_memory_ids. Do not add prose or markdown."""


class OllamaProvider:
    kind = "ollama"

    def __init__(self, config: dict[str, Any], agent_config: dict[str, Any]):
        self.config = config
        self.agent_config = agent_config

    def complete(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action", "confidence", "reason_codes", "cited_memory_ids"],
            "properties": {
                "action": {"type": "string", "enum": ["BTC", "CASH", "ABSTAIN"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reason_codes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "enum": self.agent_config["reason_codes"]},
                },
                "cited_memory_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        }
        body = {
            "model": self.config["model"],
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, sort_keys=True, allow_nan=False)},
            ],
            "options": {
                "temperature": float(self.config["temperature"]),
                "seed": int(self.config["seed"]),
            },
        }
        request = urllib.request.Request(
            self.config["endpoint"],
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=float(self.config["timeout_seconds"])) as response:
                outer = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        latency = time.monotonic() - started
        if "error" in outer:
            raise RuntimeError(f"Ollama returned error: {outer['error']}")
        content = outer.get("message", {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response has no message.content")
        return content, outer, latency


class ControlledProvider:
    kind = "controlled"

    def complete(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
        state = payload["state"]
        if max(state["prob_bear"], state["prob_bull"], state["prob_mix"]) < 0.50 or state["router_entropy"] >= 0.90:
            action, reason = "ABSTAIN", "MIXED_OR_UNCERTAIN"
        elif state["prob_bear"] >= 0.75 or state["drawdown_90"] <= -0.25:
            action, reason = "CASH", "BEARISH_ROUTER"
        elif state["prob_bull"] >= 0.75 and state["realized_vol_30"] <= 0.80 and state["return_30"] > 0:
            action, reason = "BTC", "BULLISH_ROUTER"
        else:
            action, reason = "ABSTAIN", "INSUFFICIENT_EVIDENCE"
        value = {
            "action": action,
            "confidence": 0.8 if action != "ABSTAIN" else 0.5,
            "reason_codes": [reason],
            "cited_memory_ids": [],
        }
        raw = json.dumps(value, sort_keys=True)
        return raw, {"message": {"content": raw}, "controlled": True}, 0.0
