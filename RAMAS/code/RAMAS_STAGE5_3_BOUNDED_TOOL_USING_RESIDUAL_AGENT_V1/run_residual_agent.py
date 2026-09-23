#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import frozen_pipeline
from frozen_pipeline import (
    TOL,
    PeriodInputs,
    Stage53Error,
    atomic_csv,
    atomic_json,
    atomic_jsonl,
    import_base,
    load_json,
    load_period_inputs,
    normalized_state_vector,
    resolve_base_dir,
    resolve_corrected_run,
    resolve_raw_dataset,
    sha256_file,
    sha256_text,
)


PACKAGE = Path(__file__).resolve().parent
TOOL_NAMES = (
    "retrieve_mature_incidents",
    "compare_allowed_actions",
    "inspect_current_risk",
    "inspect_evidence_ledger",
)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("phase") != "EXPLORATORY_DEVELOPMENT_ONLY":
        raise Stage53Error("Stage 5.3 must remain exploratory and development-only")
    if config.get("expected_raw_sha256") != (
        "b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e"
    ):
        raise Stage53Error("Raw dataset identity changed")
    controller = config.get("transparent_controller", {})
    if controller.get("mapping") != {"bear": 0.25, "bull": 0.5, "mix": 0.25}:
        raise Stage53Error("The transparent controller changed")
    if controller.get("allowed_residuals") != [-0.25, 0.0, 0.25]:
        raise Stage53Error("The residual action grid changed")
    if float(controller.get("minimum_desired_exposure", -1.0)) != 0.0:
        raise Stage53Error("The minimum desired exposure changed")
    if float(controller.get("maximum_desired_exposure", -1.0)) != 0.75:
        raise Stage53Error("The maximum desired exposure changed")
    if config.get("agent", {}).get("required_tools_for_nonzero_residual") != list(
        TOOL_NAMES
    ):
        raise Stage53Error("The required tool contract changed")
    trigger = config.get("trigger", {})
    if trigger.get("transition_day") is not True:
        raise Stage53Error("The transition trigger changed")
    if abs(float(trigger.get("router_confidence_strictly_below", 0.0)) - 0.8) > TOL:
        raise Stage53Error("The uncertainty trigger changed")
    if config.get("frozen_claims") != {
        "router_changed": False,
        "return_clock_changed": False,
        "risk_layer_changed": False,
        "transaction_cost_changed": False,
        "existing_experts_weighted": False,
        "monthly_trust_updates_enabled": False,
        "dqn_enabled": False,
        "hourly_data_used": False,
        "multimodal_data_used": False,
        "multi_agent_enabled": False,
        "reused_oos_opened": False,
        "pre2024_is_independent_confirmation": False,
    }:
        raise Stage53Error("The frozen scientific boundary changed")
    preflight = config.get("interface_preflight", {})
    if int(preflight.get("sample_triggered_days", 0)) != 30:
        raise Stage53Error("Preflight size changed")
    if int(preflight.get("minimum_valid_plans", 0)) != 29:
        raise Stage53Error("Planning preflight threshold changed")
    if int(preflight.get("minimum_valid_decisions", 0)) != 29:
        raise Stage53Error("Decision preflight threshold changed")
    if abs(float(preflight.get("minimum_full_run_interface_rate", 0.0)) - 0.99) > TOL:
        raise Stage53Error("Full-run interface threshold changed")


def risk_kwargs(base_config: dict[str, Any]) -> dict[str, float]:
    return {
        "transaction_cost_rate": float(base_config["transaction_cost_bps"]) / 10000.0,
        "cvar_alpha": float(base_config["cvar_alpha"]),
        "cvar_limit": float(base_config["cvar_limit"]),
        "ambiguity_quantile": float(base_config["ambiguity_quantile"]),
        "maximum_turnover": float(base_config["maximum_daily_turnover"]),
        "exposure_grid_step": float(base_config["exposure_grid_step"]),
    }


def action_preview(
    risk: Any,
    scenarios: np.ndarray,
    pretrade: float,
    residuals: list[float],
    base_desired: float,
    controller: dict[str, Any],
    base_config: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for residual in residuals:
        desired = float(
            np.clip(
                base_desired + residual,
                float(controller["minimum_desired_exposure"]),
                float(controller["maximum_desired_exposure"]),
            )
        )
        projected = risk.project_exposure(
            desired_exposure=desired,
            drifted_pretrade_exposure=float(pretrade),
            scenario_log_returns=np.asarray(scenarios, dtype=float),
            **risk_kwargs(base_config),
        )
        output.append(
            {
                "residual": float(residual),
                "desired_exposure": desired,
                "safe_final_exposure": float(projected.exposure),
                "estimated_cvar": float(projected.ambiguity_cvar),
                "turnover": float(projected.turnover),
                "risk_fallback": bool(projected.fallback_used),
            }
        )
    return output


def exact_tail_mean(values: np.ndarray, probability: float, *, upper: bool) -> float:
    data = np.sort(np.asarray(values, dtype=float))
    if len(data) == 0:
        return 0.0
    if upper:
        data = data[::-1]
    mass = float(np.clip(probability, TOL, 1.0)) * len(data)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = float(np.sum(data[:whole])) if whole else 0.0
    if fraction > TOL and whole < len(data):
        total += fraction * float(data[whole])
    return total / mass


def daily_loss_cvar(returns: np.ndarray, tail_probability: float) -> float:
    return exact_tail_mean(-np.asarray(returns, dtype=float), tail_probability, upper=True)


def two_positive_blocks(values: np.ndarray) -> tuple[bool, list[float]]:
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return False, []
    first, second = np.array_split(data, 2)
    means = [float(np.mean(first)), float(np.mean(second))]
    return bool(all(value > 0.0 for value in means)), means


def circular_block_lower_bound(
    values: np.ndarray,
    *,
    block_length: int,
    resamples: int,
    seed: int,
    alpha: float,
) -> float:
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return float("-inf")
    length = max(1, min(int(block_length), len(data)))
    blocks_needed = int(math.ceil(len(data) / length))
    offsets = np.arange(length)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(resamples), dtype=float)
    for index in range(int(resamples)):
        starts = rng.integers(0, len(data), size=blocks_needed)
        selected = ((starts[:, None] + offsets[None, :]) % len(data)).ravel()[: len(data)]
        means[index] = float(np.mean(data[selected]))
    return float(np.quantile(means, float(alpha), method="linear"))


def uncertainty_bucket(state: dict[str, Any]) -> str:
    if bool(state["transition_day"]):
        return "transition"
    confidence = float(state["router_confidence"])
    if confidence < 0.65:
        return "low_confidence"
    if confidence < 0.80:
        return "moderate_confidence"
    return "high_confidence"


def context_key(state: dict[str, Any], residual: float) -> str:
    sign = "increase" if residual > 0.0 else "decrease"
    return f"{state['hard_regime']}:{uncertainty_bucket(state)}:{sign}"


def counterfactual_evidence(
    episodes: list[dict[str, Any]],
    residual: float,
    controller: dict[str, Any],
    base_config: dict[str, Any],
    accounting: Any,
    risk: Any,
) -> dict[str, Any]:
    candidate_returns: list[float] = []
    baseline_returns: list[float] = []
    advantages: list[float] = []
    cost_rate = float(base_config["transaction_cost_bps"]) / 10000.0
    for episode in episodes:
        state = episode["state"]
        baseline_desired = float(controller["mapping"][state["hard_regime"]])
        candidate_desired = float(
            np.clip(
                baseline_desired + residual,
                float(controller["minimum_desired_exposure"]),
                float(controller["maximum_desired_exposure"]),
            )
        )
        pretrade = float(episode["decision_pretrade_exposure"])
        scenarios = np.asarray(episode["scenario_log_returns"], dtype=float)
        baseline_projection = risk.project_exposure(
            desired_exposure=baseline_desired,
            drifted_pretrade_exposure=pretrade,
            scenario_log_returns=scenarios,
            **risk_kwargs(base_config),
        )
        candidate_projection = risk.project_exposure(
            desired_exposure=candidate_desired,
            drifted_pretrade_exposure=pretrade,
            scenario_log_returns=scenarios,
            **risk_kwargs(base_config),
        )
        asset_return = float(episode["asset_simple_return"])
        baseline_return = float(
            accounting.net_return(
                baseline_projection.exposure,
                pretrade,
                asset_return,
                cost_rate,
            )
        )
        candidate_return = float(
            accounting.net_return(
                candidate_projection.exposure,
                pretrade,
                asset_return,
                cost_rate,
            )
        )
        baseline_returns.append(baseline_return)
        candidate_returns.append(candidate_return)
        advantages.append(float(np.log1p(candidate_return) - np.log1p(baseline_return)))
    advantage_array = np.asarray(advantages, dtype=float)
    candidate_array = np.asarray(candidate_returns, dtype=float)
    baseline_array = np.asarray(baseline_returns, dtype=float)
    blocks_pass, block_means = two_positive_blocks(advantage_array)
    return {
        "residual": float(residual),
        "sample_count": int(len(episodes)),
        "mean_log_advantage": float(np.mean(advantage_array)) if len(episodes) else 0.0,
        "median_log_advantage": float(np.median(advantage_array)) if len(episodes) else 0.0,
        "positive_incident_rate": float(np.mean(advantage_array > 0.0)) if len(episodes) else 0.0,
        "candidate_daily_loss_cvar_95": daily_loss_cvar(candidate_array, 0.05),
        "baseline_daily_loss_cvar_95": daily_loss_cvar(baseline_array, 0.05),
        "cvar_not_worse": bool(
            daily_loss_cvar(candidate_array, 0.05)
            <= daily_loss_cvar(baseline_array, 0.05) + TOL
        ),
        "two_positive_chronological_blocks": blocks_pass,
        "chronological_block_means": block_means,
        "advantages": advantage_array,
    }


class EvidenceMemory:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.episodes: list[dict[str, Any]] = []
        self.proposals: dict[str, dict[str, Any]] = {}
        self.lessons: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []

    def add_episode(self, episode: dict[str, Any]) -> None:
        self.episodes.append(episode)

    def retrieve(self, state: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
        target = normalized_state_vector(state)
        regime = str(state["hard_regime"])
        bucket = uncertainty_bucket(state)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for episode in self.episodes:
            if not episode.get("completed", False):
                continue
            if episode["state"]["hard_regime"] != regime:
                continue
            if uncertainty_bucket(episode["state"]) != bucket:
                continue
            distance = float(
                np.linalg.norm(target - np.asarray(episode["state_vector"], dtype=float))
            )
            candidates.append((distance, episode))
        candidates.sort(key=lambda item: (item[0], item[1]["episode_id"]))
        output: list[dict[str, Any]] = []
        for distance, episode in candidates[: int(maximum)]:
            output.append(
                {
                    "episode_id": episode["episode_id"],
                    "decision_date": episode["decision_date"],
                    "hard_regime": episode["state"]["hard_regime"],
                    "uncertainty_bucket": uncertainty_bucket(episode["state"]),
                    "exact_context_bucket": True,
                    "state_distance": distance,
                    "executed_residual": episode["executed_residual"],
                    "portfolio_net_return": episode["portfolio_net_return"],
                    "baseline_net_return": episode["baseline_net_return"],
                }
            )
        return output

    def episodes_by_ids(self, identifiers: list[str]) -> list[dict[str, Any]]:
        wanted = set(identifiers)
        return [item for item in self.episodes if item["episode_id"] in wanted]

    def matching_context(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        regime = str(state["hard_regime"])
        bucket = uncertainty_bucket(state)
        return [
            item
            for item in self.episodes
            if item.get("completed", False)
            and item["state"]["hard_regime"] == regime
            and uncertainty_bucket(item["state"]) == bucket
        ]

    def propose(
        self,
        *,
        state: dict[str, Any],
        residual: float,
        text: str,
        episode_id: str,
        decision_date: str,
    ) -> None:
        if residual == 0.0 or not text.strip():
            return
        key = context_key(state, residual)
        if key in self.proposals or key in self.lessons:
            return
        proposal = {
            "context_key": key,
            "residual": float(residual),
            "text": text.strip()[:300],
            "episode_id": episode_id,
            "decision_date": decision_date,
            "status": "QUARANTINED",
            "last_review_support": 0,
        }
        self.proposals[key] = proposal
        self.events.append({"event": "LESSON_QUARANTINED", **proposal})

    def ledger(self, state: dict[str, Any]) -> dict[str, Any]:
        prefix = f"{state['hard_regime']}:{uncertainty_bucket(state)}:"
        relevant_lessons = [
            value
            for key, value in sorted(self.lessons.items())
            if key.startswith(prefix)
        ]
        return {
            "current_context_prefix": prefix,
            "active_relevant_lessons": [
                {
                    "context_key": item["context_key"],
                    "residual": item["residual"],
                    "text": item["text"],
                    "support": item["support"],
                    "mean_log_advantage": item["mean_log_advantage"],
                }
                for item in relevant_lessons
                if item["status"] == "ACTIVE"
            ],
            "quarantined_relevant": int(
                sum(
                    key.startswith(prefix) and value["status"] == "QUARANTINED"
                    for key, value in self.proposals.items()
                )
            ),
            "rolled_back_relevant": int(
                sum(
                    key.startswith(prefix) and value["status"] == "ROLLED_BACK"
                    for key, value in self.lessons.items()
                )
            ),
            "completed_incidents": len(self.episodes),
        }

    def review(
        self,
        *,
        state: dict[str, Any],
        decision_date: str,
        controller: dict[str, Any],
        base_config: dict[str, Any],
        accounting: Any,
        risk: Any,
    ) -> None:
        matching = self.matching_context(state)
        minimum = int(self.config["minimum_matching_incidents_for_promotion"])
        review_step = int(self.config["minimum_new_incidents_between_reviews"])
        rollback = int(self.config["rollback_additional_incidents"])
        prefix = f"{state['hard_regime']}:{uncertainty_bucket(state)}:"
        for key, proposal in sorted(self.proposals.items()):
            if not key.startswith(prefix) or proposal["status"] != "QUARANTINED":
                continue
            support = len(matching)
            if support < minimum or support - int(proposal["last_review_support"]) < review_step:
                continue
            proposal["last_review_support"] = support
            evidence = counterfactual_evidence(
                matching,
                float(proposal["residual"]),
                controller,
                base_config,
                accounting,
                risk,
            )
            lower_bound = circular_block_lower_bound(
                evidence["advantages"],
                block_length=int(self.config["bootstrap_block_length_days"]),
                resamples=int(self.config["bootstrap_resamples"]),
                seed=int(self.config["bootstrap_seed"]),
                alpha=float(self.config["bootstrap_alpha"]),
            )
            passes = bool(
                evidence["mean_log_advantage"] > 0.0
                and lower_bound > 0.0
                and evidence["cvar_not_worse"]
                and evidence["two_positive_chronological_blocks"]
            )
            if passes:
                proposal["status"] = "PROMOTED"
                lesson = {
                    "context_key": key,
                    "residual": float(proposal["residual"]),
                    "text": proposal["text"],
                    "status": "ACTIVE",
                    "support": support,
                    "activated_support": support,
                    "mean_log_advantage": float(evidence["mean_log_advantage"]),
                    "bootstrap_lower_bound": lower_bound,
                }
                self.lessons[key] = lesson
                self.events.append(
                    {"event": "LESSON_PROMOTED", "decision_date": decision_date, **lesson}
                )
        for key, lesson in sorted(self.lessons.items()):
            if not key.startswith(prefix) or lesson["status"] != "ACTIVE":
                continue
            support = len(matching)
            if support - int(lesson["activated_support"]) < rollback:
                continue
            evidence = counterfactual_evidence(
                matching,
                float(lesson["residual"]),
                controller,
                base_config,
                accounting,
                risk,
            )
            if evidence["mean_log_advantage"] <= 0.0 or not evidence["cvar_not_worse"]:
                lesson["status"] = "ROLLED_BACK"
                lesson["support"] = support
                lesson["mean_log_advantage"] = float(evidence["mean_log_advantage"])
                self.events.append(
                    {"event": "LESSON_ROLLED_BACK", "decision_date": decision_date, **lesson}
                )


def planner_system_prompt() -> str:
    return """You are the planning component of a bounded Bitcoin allocation agent.
The transparent regime controller owns the default exposure. You do not choose exposure in this step.
Choose deterministic tools needed to evaluate whether a residual adjustment has causal, past-only support.
Available tools are retrieve_mature_incidents, compare_allowed_actions, inspect_current_risk, inspect_evidence_ledger.
A later nonzero residual is possible only if all four tools are selected and their result identifiers are cited.
When evidence is likely insufficient, you may still inspect tools and later choose residual zero.
Return exactly one JSON object with tool_names and planning_reason. No prose outside JSON."""


def decision_system_prompt() -> str:
    return """You are the decision component of a bounded, tool-using Bitcoin allocation agent.
The transparent controller already selected the default exposure. Choose only a residual from -0.25, 0, or 0.25.
Residual zero means abstain and keep the transparent controller. A nonzero residual must be marked eligible by compare_allowed_actions and must cite the result identifier of every required tool.
Tool results are past-only evidence. Confidence and explanation cannot override the code-side evidence gate.
Do not optimize prediction accuracy. Seek after-cost growth while avoiding worse worst-day loss CVaR.
Return exactly one JSON object with residual, confidence, cited_tool_result_ids, and lesson.
lesson must be {\"action\":\"none\",\"text\":\"\"} unless proposing one concise causal hypothesis for the chosen nonzero residual.
No prose outside JSON."""


def plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_names", "planning_reason"],
        "properties": {
            "tool_names": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(TOOL_NAMES)},
            },
            "planning_reason": {"type": "string", "maxLength": 160},
        },
    }


def decision_schema(tool_result_ids: set[str]) -> dict[str, Any]:
    citations: dict[str, Any] = {
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string"},
    }
    if tool_result_ids:
        citations["items"]["enum"] = sorted(tool_result_ids)
    else:
        citations["maxItems"] = 0
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["residual", "confidence", "cited_tool_result_ids", "lesson"],
        "properties": {
            "residual": {"type": "number", "enum": [-0.25, 0.0, 0.25]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "cited_tool_result_ids": citations,
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


class ControlledProvider:
    kind = "controlled"

    def plan(self, payload: dict[str, Any], request_id: str) -> tuple[str, dict[str, Any], float]:
        value = {"tool_names": list(TOOL_NAMES), "planning_reason": "mechanical_test"}
        return json.dumps(value, sort_keys=True), {"controlled": True, "request_id": request_id}, 0.0

    def decide(
        self,
        payload: dict[str, Any],
        tool_results: list[dict[str, Any]],
        request_id: str,
    ) -> tuple[str, dict[str, Any], float]:
        comparison = next(
            item for item in tool_results if item["tool_name"] == "compare_allowed_actions"
        )
        eligible = [
            row
            for row in comparison["result"]["residual_comparison"]
            if row["residual"] != 0.0 and row["eligible"]
        ]
        chosen = max(eligible, key=lambda row: row["mean_log_advantage"], default=None)
        residual = float(chosen["residual"]) if chosen is not None else 0.0
        value = {
            "residual": residual,
            "confidence": 0.6 if chosen is not None else 0.0,
            "cited_tool_result_ids": [
                item["tool_result_id"] for item in tool_results
            ] if chosen is not None else [],
            "lesson": (
                {"action": "propose", "text": "Use this residual only when the same past-only evidence gate remains positive."}
                if chosen is not None
                else {"action": "none", "text": ""}
            ),
        }
        return json.dumps(value, sort_keys=True), {"controlled": True, "request_id": request_id}, 0.0


class OllamaProvider:
    kind = "ollama"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _complete(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
        request_id: str,
    ) -> tuple[str, dict[str, Any], float]:
        request_payload = {
            "model": self.config["model"],
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, sort_keys=True, allow_nan=False)},
            ],
            "options": {
                "temperature": float(self.config["temperature"]),
                "seed": int(self.config["seed"]),
            },
        }
        request = urllib.request.Request(
            str(self.config["endpoint"]),
            data=json.dumps(request_payload, allow_nan=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(
            request, timeout=float(self.config["timeout_seconds"])
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        latency = time.monotonic() - start
        raw = response_payload.get("message", {}).get("content")
        if not isinstance(raw, str):
            raise Stage53Error("Ollama response has no message content")
        response_payload["request_id"] = request_id
        return raw, response_payload, latency

    def plan(self, payload: dict[str, Any], request_id: str) -> tuple[str, dict[str, Any], float]:
        return self._complete(
            system_prompt=planner_system_prompt(),
            payload=payload,
            schema=plan_schema(),
            request_id=request_id,
        )

    def decide(
        self,
        payload: dict[str, Any],
        tool_results: list[dict[str, Any]],
        request_id: str,
    ) -> tuple[str, dict[str, Any], float]:
        decision_payload = {**payload, "executed_tool_results": tool_results}
        return self._complete(
            system_prompt=decision_system_prompt(),
            payload=decision_payload,
            schema=decision_schema({item["tool_result_id"] for item in tool_results}),
            request_id=request_id,
        )


def validate_plan(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Stage53Error(f"Planner returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage53Error("Planner response is not an object")
    names = value.get("tool_names")
    if not isinstance(names, list) or not names or len(names) > len(TOOL_NAMES):
        raise Stage53Error("Planner tool_names is invalid")
    if any(not isinstance(item, str) or item not in TOOL_NAMES for item in names):
        raise Stage53Error("Planner selected an unknown tool")
    if len(set(names)) != len(names):
        raise Stage53Error("Planner selected a tool more than once")
    reason = value.get("planning_reason")
    if not isinstance(reason, str):
        raise Stage53Error("Planner planning_reason is invalid")
    return {"tool_names": names, "planning_reason": reason[:160]}


def validate_decision(
    raw: str,
    allowed_tool_result_ids: set[str],
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Stage53Error(f"Decision returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise Stage53Error("Decision response is not an object")
    residual = value.get("residual")
    if isinstance(residual, bool) or not isinstance(residual, (int, float)):
        raise Stage53Error("Residual is not numeric")
    matching = [item for item in [-0.25, 0.0, 0.25] if abs(float(residual) - item) <= TOL]
    if not matching:
        raise Stage53Error("Residual is outside the frozen grid")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        raise Stage53Error("Confidence is invalid")
    citations = value.get("cited_tool_result_ids")
    if not isinstance(citations, list) or any(not isinstance(item, str) for item in citations):
        raise Stage53Error("Tool citations are invalid")
    if len(set(citations)) != len(citations):
        raise Stage53Error("Tool citations are duplicated")
    if not set(citations).issubset(allowed_tool_result_ids):
        raise Stage53Error("Decision cited a tool result that was not visible")
    lesson_raw = value.get("lesson")
    if not isinstance(lesson_raw, dict):
        raise Stage53Error("Lesson is invalid")
    action = lesson_raw.get("action")
    text = lesson_raw.get("text")
    if action not in {"propose", "none"} or not isinstance(text, str):
        raise Stage53Error("Lesson fields are invalid")
    if action == "none" and text != "":
        raise Stage53Error("A no-lesson response must have empty text")
    if action == "propose" and not text.strip():
        raise Stage53Error("A proposed lesson must have text")
    return {
        "residual": float(matching[0]),
        "confidence": float(np.clip(float(confidence), 0.0, 1.0)),
        "cited_tool_result_ids": citations,
        "lesson": {"action": action, "text": text.strip()[:300]},
    }


def execute_tools(
    *,
    selected_tools: list[str],
    request_id: str,
    state: dict[str, Any],
    memory: EvidenceMemory,
    scenarios: np.ndarray,
    controller: dict[str, Any],
    agent_config: dict[str, Any],
    base_config: dict[str, Any],
    accounting: Any,
    risk: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    retrieved_ids: list[str] = []
    for tool_name in TOOL_NAMES:
        if tool_name not in selected_tools:
            continue
        result_id = f"{request_id}:tool:{tool_name}"
        if tool_name == "retrieve_mature_incidents":
            incidents = memory.retrieve(
                state,
                int(agent_config["maximum_mature_incidents"]),
            )
            retrieved_ids = [item["episode_id"] for item in incidents]
            result: dict[str, Any] = {
                "mature_incident_count": len(incidents),
                "incidents": incidents,
                "past_only": True,
            }
        elif tool_name == "compare_allowed_actions":
            episodes = memory.episodes_by_ids(retrieved_ids)
            comparisons: list[dict[str, Any]] = []
            for residual in controller["allowed_residuals"]:
                evidence = counterfactual_evidence(
                    episodes,
                    float(residual),
                    controller,
                    base_config,
                    accounting,
                    risk,
                )
                eligible = bool(
                    float(residual) == 0.0
                    or (
                        evidence["sample_count"]
                        >= int(agent_config["minimum_incidents_for_nonzero_residual"])
                        and evidence["mean_log_advantage"] > 0.0
                        and evidence["cvar_not_worse"]
                        and evidence["two_positive_chronological_blocks"]
                    )
                )
                comparisons.append(
                    {
                        key: value
                        for key, value in evidence.items()
                        if key != "advantages"
                    }
                    | {"eligible": eligible}
                )
            result = {
                "retrieved_incident_ids": retrieved_ids,
                "residual_comparison": comparisons,
                "comparison_is_one_step_conditional": True,
                "future_current_return_used": False,
            }
        elif tool_name == "inspect_current_risk":
            base_desired = float(controller["mapping"][state["hard_regime"]])
            result = {
                "base_desired_exposure": base_desired,
                "risk_parameters": risk_kwargs(base_config),
                "action_preview": action_preview(
                    risk,
                    scenarios,
                    float(state["pretrade_exposure"]),
                    [float(item) for item in controller["allowed_residuals"]],
                    base_desired,
                    controller,
                    base_config,
                ),
            }
        elif tool_name == "inspect_evidence_ledger":
            result = memory.ledger(state)
        else:
            raise Stage53Error(f"Unknown tool={tool_name}")
        results.append(
            {
                "tool_name": tool_name,
                "tool_result_id": result_id,
                "result": result,
            }
        )
    return results


def enforce_evidence_gate(
    decision: dict[str, Any],
    tool_results: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[float, str]:
    residual = float(decision["residual"])
    if residual == 0.0:
        return 0.0, "MODEL_ABSTAINED_TO_TRANSPARENT_CONTROLLER"
    by_name = {item["tool_name"]: item for item in tool_results}
    required = set(config["agent"]["required_tools_for_nonzero_residual"])
    if not required.issubset(by_name):
        return 0.0, "ZEROED_MISSING_REQUIRED_TOOL"
    required_ids = {by_name[name]["tool_result_id"] for name in required}
    if not required_ids.issubset(set(decision["cited_tool_result_ids"])):
        return 0.0, "ZEROED_MISSING_REQUIRED_TOOL_CITATION"
    comparison_rows = by_name["compare_allowed_actions"]["result"][
        "residual_comparison"
    ]
    selected = [
        item for item in comparison_rows if abs(float(item["residual"]) - residual) <= TOL
    ]
    if len(selected) != 1 or not bool(selected[0]["eligible"]):
        return 0.0, "ZEROED_TOOL_EVIDENCE_NOT_ELIGIBLE"
    return residual, "NONZERO_RESIDUAL_ACCEPTED_BY_TOOL_EVIDENCE"


def build_state(period: PeriodInputs, index: int, pretrade: float) -> dict[str, Any]:
    feature = period.features.iloc[index]
    q = period.router_probabilities[index]
    hard_regime = ["bear", "bull", "mix"][int(np.argmax(q))]
    state = {
        "decision_date": str(period.decision_dates[index].date()),
        "target_return_date": str(period.return_dates[index].date()),
        "prob_bear": float(q[0]),
        "prob_bull": float(q[1]),
        "prob_mix": float(q[2]),
        "hard_regime": hard_regime,
        "pretrade_exposure": float(pretrade),
        "transition_day": bool(feature["transition_day"]),
    }
    for key, value in feature.to_dict().items():
        if key != "transition_day":
            state[key] = float(value)
    state["uncertainty_bucket"] = uncertainty_bucket(state)
    return state


def is_triggered(state: dict[str, Any], trigger_config: dict[str, Any]) -> bool:
    return bool(
        state["transition_day"]
        or float(state["router_confidence"])
        < float(trigger_config["router_confidence_strictly_below"])
    )


@dataclass
class ReplayResult:
    trace: pd.DataFrame
    calls: list[dict[str, Any]]
    checks: dict[str, bool]


def run_agent_period(
    period: PeriodInputs,
    provider: Any,
    memory: EvidenceMemory,
    package_config: dict[str, Any],
    base_config: dict[str, Any],
    accounting: Any,
    risk: Any,
) -> ReplayResult:
    controller = package_config["transparent_controller"]
    pretrade = 0.0
    baseline_pretrade = 0.0
    trace_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    triggered_count = 0
    valid_plans = 0
    valid_decisions = 0
    provider_failures = 0
    cost_rate = float(base_config["transaction_cost_bps"]) / 10000.0
    for index in range(len(period.frame)):
        state = build_state(period, index, pretrade)
        memory.review(
            state=state,
            decision_date=state["decision_date"],
            controller=controller,
            base_config=base_config,
            accounting=accounting,
            risk=risk,
        )
        triggered = is_triggered(state, package_config["trigger"])
        base_desired = float(controller["mapping"][state["hard_regime"]])
        plan_valid = False
        decision_valid = False
        plan_raw = ""
        decision_raw = ""
        plan_status = "NOT_TRIGGERED"
        decision_status = "NOT_TRIGGERED"
        provider_error = ""
        selected_tools: list[str] = []
        tool_results: list[dict[str, Any]] = []
        decision = {
            "residual": 0.0,
            "confidence": 0.0,
            "cited_tool_result_ids": [],
            "lesson": {"action": "none", "text": ""},
        }
        evidence_gate_status = "NOT_TRIGGERED_TRANSPARENT_CONTROLLER"
        request_payload = {
            "experiment_id": package_config["experiment_id"],
            "prompt_version": package_config["agent"]["prompt_version"],
            "state": state,
            "transparent_controller": {
                "base_desired_exposure": base_desired,
                "allowed_residuals": controller["allowed_residuals"],
                "desired_exposure_bounds": [
                    controller["minimum_desired_exposure"],
                    controller["maximum_desired_exposure"],
                ],
            },
            "tool_catalog": list(TOOL_NAMES),
            "constraints": {
                "past_only": True,
                "shorting": False,
                "leverage": False,
                "nonzero_requires_all_tool_citations": True,
            },
        }
        if "canonical_asset_simple_return" in json.dumps(request_payload, sort_keys=True):
            raise Stage53Error("Future target return leaked into the agent request")
        if triggered:
            triggered_count += 1
            request_id = f"{period.name}-{index + 1:06d}"
            try:
                plan_raw, plan_response, plan_latency = provider.plan(
                    request_payload, f"{request_id}:plan"
                )
                plan = validate_plan(plan_raw)
                plan_valid = True
                valid_plans += 1
                plan_status = "OK"
                selected_tools = list(plan["tool_names"])
                tool_results = execute_tools(
                    selected_tools=selected_tools,
                    request_id=request_id,
                    state=state,
                    memory=memory,
                    scenarios=period.scenarios[index],
                    controller=controller,
                    agent_config=package_config["agent"],
                    base_config=base_config,
                    accounting=accounting,
                    risk=risk,
                )
                call_rows.append(
                    {
                        "request_id": f"{request_id}:plan",
                        "stage": "PLAN",
                        "request": request_payload,
                        "raw_response": plan_raw,
                        "provider_response": plan_response,
                        "latency_seconds": plan_latency,
                        "status": "OK",
                        "request_sha256": sha256_text(
                            planner_system_prompt()
                            + "\n"
                            + json.dumps(request_payload, sort_keys=True)
                        ),
                        "response_sha256": sha256_text(plan_raw),
                    }
                )
                decision_raw, decision_response, decision_latency = provider.decide(
                    request_payload,
                    tool_results,
                    f"{request_id}:decision",
                )
                decision = validate_decision(
                    decision_raw,
                    {item["tool_result_id"] for item in tool_results},
                )
                decision_valid = True
                valid_decisions += 1
                decision_status = "OK"
                call_rows.append(
                    {
                        "request_id": f"{request_id}:decision",
                        "stage": "DECISION",
                        "request": {**request_payload, "executed_tool_results": tool_results},
                        "raw_response": decision_raw,
                        "provider_response": decision_response,
                        "latency_seconds": decision_latency,
                        "status": "OK",
                        "request_sha256": sha256_text(
                            decision_system_prompt()
                            + "\n"
                            + json.dumps(
                                {**request_payload, "executed_tool_results": tool_results},
                                sort_keys=True,
                            )
                        ),
                        "response_sha256": sha256_text(decision_raw),
                    }
                )
                executed_residual, evidence_gate_status = enforce_evidence_gate(
                    decision, tool_results, package_config
                )
            except Exception as exc:
                provider_failures += 1
                provider_error = f"{type(exc).__name__}: {exc}"
                if not plan_valid:
                    plan_status = "FAIL_CLOSED"
                else:
                    decision_status = "FAIL_CLOSED"
                executed_residual = 0.0
                evidence_gate_status = "FAIL_CLOSED_TO_TRANSPARENT_CONTROLLER"
                call_rows.append(
                    {
                        "request_id": f"{request_id}:failure",
                        "stage": "FAILURE",
                        "request": request_payload,
                        "raw_response": decision_raw or plan_raw,
                        "provider_response": {},
                        "latency_seconds": 0.0,
                        "status": "FAIL_CLOSED",
                        "error": provider_error,
                    }
                )
        else:
            executed_residual = 0.0
        desired = float(
            np.clip(
                base_desired + executed_residual,
                float(controller["minimum_desired_exposure"]),
                float(controller["maximum_desired_exposure"]),
            )
        )
        projected = risk.project_exposure(
            desired_exposure=desired,
            drifted_pretrade_exposure=pretrade,
            scenario_log_returns=period.scenarios[index],
            **risk_kwargs(base_config),
        )
        baseline_projected = risk.project_exposure(
            desired_exposure=base_desired,
            drifted_pretrade_exposure=baseline_pretrade,
            scenario_log_returns=period.scenarios[index],
            **risk_kwargs(base_config),
        )
        asset_return = float(period.asset_returns[index])
        portfolio_return = float(
            accounting.net_return(
                projected.exposure, pretrade, asset_return, cost_rate
            )
        )
        baseline_return = float(
            accounting.net_return(
                baseline_projected.exposure,
                baseline_pretrade,
                asset_return,
                cost_rate,
            )
        )
        episode_id = f"episode-{len(memory.episodes) + 1:06d}"
        episode = {
            "episode_id": episode_id,
            "decision_date": state["decision_date"],
            "return_date": state["target_return_date"],
            "state": state,
            "state_vector": normalized_state_vector(state).tolist(),
            "decision_pretrade_exposure": float(pretrade),
            "scenario_log_returns": np.asarray(period.scenarios[index], dtype=float).tolist(),
            "asset_simple_return": asset_return,
            "base_desired_exposure": base_desired,
            "model_proposed_residual": float(decision["residual"]),
            "executed_residual": float(executed_residual),
            "desired_exposure": desired,
            "final_exposure": float(projected.exposure),
            "portfolio_net_return": portfolio_return,
            "baseline_net_return": baseline_return,
            "log_advantage_vs_transparent": float(
                np.log1p(portfolio_return) - np.log1p(baseline_return)
            ),
            "completed": True,
        }
        memory.add_episode(episode)
        if (
            decision_valid
            and executed_residual != 0.0
            and decision["lesson"]["action"] == "propose"
        ):
            memory.propose(
                state=state,
                residual=executed_residual,
                text=decision["lesson"]["text"],
                episode_id=episode_id,
                decision_date=state["decision_date"],
            )
        trace_rows.append(
            {
                "decision_date": state["decision_date"],
                "return_date": state["target_return_date"],
                "prob_bear": state["prob_bear"],
                "prob_bull": state["prob_bull"],
                "prob_mix": state["prob_mix"],
                "hard_regime": state["hard_regime"],
                "transition_day": state["transition_day"],
                "router_confidence": state["router_confidence"],
                "router_entropy": state["router_entropy"],
                "uncertainty_bucket": state["uncertainty_bucket"],
                "agent_triggered": triggered,
                "plan_valid": plan_valid,
                "decision_valid": decision_valid,
                "plan_status": plan_status,
                "decision_status": decision_status,
                "selected_tools": "|".join(selected_tools),
                "tool_result_ids": "|".join(
                    item["tool_result_id"] for item in tool_results
                ),
                "model_proposed_residual": float(decision["residual"]),
                "executed_residual": float(executed_residual),
                "evidence_gate_status": evidence_gate_status,
                "confidence": float(decision["confidence"]),
                "cited_tool_result_ids": "|".join(
                    decision["cited_tool_result_ids"]
                ),
                "base_desired_exposure": base_desired,
                "desired_exposure": desired,
                "pretrade_exposure": float(pretrade),
                "final_exposure": float(projected.exposure),
                "turnover": float(projected.turnover),
                "ambiguity_cvar": float(projected.ambiguity_cvar),
                "risk_fallback_used": bool(projected.fallback_used),
                "asset_simple_return": asset_return,
                "portfolio_net_return": portfolio_return,
                "transparent_final_exposure": float(baseline_projected.exposure),
                "transparent_turnover": float(baseline_projected.turnover),
                "transparent_net_return": baseline_return,
                "log_advantage_vs_transparent": episode[
                    "log_advantage_vs_transparent"
                ],
                "provider_error": provider_error,
            }
        )
        pretrade = float(accounting.drifted_exposure(projected.exposure, asset_return))
        baseline_pretrade = float(
            accounting.drifted_exposure(baseline_projected.exposure, asset_return)
        )
        completed = index + 1
        if provider.kind == "ollama" and (
            completed == 1 or completed % 50 == 0 or completed == len(period.frame)
        ):
            print(
                "AGENT_PROGRESS="
                f"{period.name}:{completed}/{len(period.frame)} "
                f"TRIGGERED={triggered_count} VALID_PLANS={valid_plans} "
                f"VALID_DECISIONS={valid_decisions}",
                flush=True,
            )
    trace = pd.DataFrame(trace_rows)
    triggered_rows = trace["agent_triggered"].to_numpy(bool)
    checks = {
        "row_count_matches": len(trace) == len(period.frame),
        "return_clock_next_day": bool(
            np.all(
                (
                    pd.to_datetime(trace["return_date"])
                    - pd.to_datetime(trace["decision_date"])
                ).dt.days.to_numpy()
                == 1
            )
        ),
        "nontrigger_rows_have_zero_residual": bool(
            np.allclose(
                trace.loc[~triggered_rows, "executed_residual"].to_numpy(float),
                0.0,
            )
        ),
        "residuals_on_frozen_grid": bool(
            trace["executed_residual"].isin([-0.25, 0.0, 0.25]).all()
        ),
        "desired_exposure_bounded": bool(trace["desired_exposure"].between(0.0, 0.75).all()),
        "final_exposure_bounded": bool(trace["final_exposure"].between(0.0, 1.0).all()),
        "turnover_bounded": bool(
            (
                trace["turnover"].to_numpy(float)
                <= float(base_config["maximum_daily_turnover"]) + TOL
            ).all()
        ),
        "returns_finite": bool(np.isfinite(trace["portfolio_net_return"]).all()),
        "provider_failures_zero_residual": bool(
            np.allclose(
                trace.loc[
                    trace["evidence_gate_status"]
                    == "FAIL_CLOSED_TO_TRANSPARENT_CONTROLLER",
                    "executed_residual",
                ].to_numpy(float),
                0.0,
            )
        ),
        "nonzero_residuals_have_tool_support": bool(
            (
                trace.loc[trace["executed_residual"] != 0.0, "evidence_gate_status"]
                == "NONZERO_RESIDUAL_ACCEPTED_BY_TOOL_EVIDENCE"
            ).all()
        ),
        "frozen_risk_parameters_inherited": bool(
            float(base_config["transaction_cost_bps"]) == 10.0
            and float(base_config["cvar_limit"]) == 0.045
            and float(base_config["maximum_daily_turnover"]) == 0.35
        ),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise Stage53Error(f"Residual-agent integration checks failed: {failed}")
    return ReplayResult(trace=trace, calls=call_rows, checks=checks)


def variant_metrics(
    adapter: Any,
    label: str,
    returns: np.ndarray,
    exposure: np.ndarray,
    turnover: np.ndarray,
) -> dict[str, Any]:
    return adapter.metrics(
        label,
        np.asarray(returns, dtype=float),
        np.asarray(exposure, dtype=float),
        np.asarray(turnover, dtype=float),
    )


def evaluate_development(
    period: PeriodInputs,
    replay: ReplayResult,
    config: dict[str, Any],
    adapter: Any,
    statistics: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trace = replay.trace
    agent_returns = trace["portfolio_net_return"].to_numpy(float)
    agent_exposure = trace["final_exposure"].to_numpy(float)
    agent_turnover = trace["turnover"].to_numpy(float)
    baseline_returns = trace["transparent_net_return"].to_numpy(float)
    baseline_exposure = trace["transparent_final_exposure"].to_numpy(float)
    baseline_turnover = trace["transparent_turnover"].to_numpy(float)
    current_returns = period.current_trace["portfolio_net_return"].to_numpy(float)
    current_exposure = period.current_trace["final_exposure"].to_numpy(float)
    current_turnover = period.current_trace["turnover"].to_numpy(float)
    metrics = pd.DataFrame(
        [
            variant_metrics(
                adapter,
                "bounded_tool_residual_agent",
                agent_returns,
                agent_exposure,
                agent_turnover,
            ),
            variant_metrics(
                adapter,
                "transparent_controller",
                baseline_returns,
                baseline_exposure,
                baseline_turnover,
            ),
            variant_metrics(
                adapter,
                "current_ramoe_diagnostic",
                current_returns,
                current_exposure,
                current_turnover,
            ),
        ]
    )
    agent_metric = metrics.loc[
        metrics["variant"] == "bounded_tool_residual_agent"
    ].iloc[0]
    baseline_metric = metrics.loc[
        metrics["variant"] == "transparent_controller"
    ].iloc[0]
    advantage = np.log1p(agent_returns) - np.log1p(baseline_returns)
    bootstrap_spec = config["bootstrap"]
    bootstrap = statistics.circular_block_mean_test(
        advantage,
        block_length=int(bootstrap_spec["block_length_days"]),
        resamples=int(bootstrap_spec["resamples"]),
        seed=int(bootstrap_spec["seed"]),
    )
    years = period.return_dates.year.to_numpy()
    yearly_rows: list[dict[str, Any]] = []
    positive_years = 0
    for year in sorted(np.unique(years)):
        selected = years == year
        agent_growth = float(np.prod(1.0 + agent_returns[selected]) - 1.0)
        baseline_growth = float(np.prod(1.0 + baseline_returns[selected]) - 1.0)
        positive_years += int(agent_growth > baseline_growth)
        yearly_rows.append(
            {
                "year": int(year),
                "agent_growth": agent_growth,
                "transparent_growth": baseline_growth,
                "agent_minus_transparent_growth": agent_growth - baseline_growth,
                "mean_executed_residual": float(
                    trace.loc[selected, "executed_residual"].mean()
                ),
                "triggered_days": int(trace.loc[selected, "agent_triggered"].sum()),
                "nonzero_residual_days": int(
                    (trace.loc[selected, "executed_residual"] != 0.0).sum()
                ),
            }
        )
    hard = trace["hard_regime"].to_numpy(str)
    regime_rows: list[dict[str, Any]] = []
    for regime in ["bear", "bull", "mix"]:
        selected = hard == regime
        regime_rows.append(
            {
                "regime": regime,
                "rows": int(selected.sum()),
                "agent_growth": float(np.prod(1.0 + agent_returns[selected]) - 1.0),
                "transparent_growth": float(
                    np.prod(1.0 + baseline_returns[selected]) - 1.0
                ),
                "mean_log_advantage": float(np.mean(advantage[selected])),
                "triggered_days": int(trace.loc[selected, "agent_triggered"].sum()),
                "nonzero_residual_days": int(
                    (trace.loc[selected, "executed_residual"] != 0.0).sum()
                ),
            }
        )
    triggered = trace["agent_triggered"].to_numpy(bool)
    triggered_count = int(triggered.sum())
    plan_rate = float(trace.loc[triggered, "plan_valid"].mean()) if triggered_count else 1.0
    decision_rate = (
        float(trace.loc[triggered, "decision_valid"].mean()) if triggered_count else 1.0
    )
    interface_rate = min(plan_rate, decision_rate)
    gate_config = config["economic_gate"]
    checks = {
        "full_run_interface_valid": bool(
            interface_rate + TOL
            >= float(config["interface_preflight"]["minimum_full_run_interface_rate"])
        ),
        "positive_growth_advantage": bool(
            agent_metric["terminal_growth"] > baseline_metric["terminal_growth"]
        ),
        "sharpe_improvement_at_least_0_05": bool(
            agent_metric["sharpe_zero_cash_rate"]
            >= baseline_metric["sharpe_zero_cash_rate"]
            + float(gate_config["minimum_sharpe_improvement"])
        ),
        "one_sided_block_bootstrap_p_below_0_05": bool(
            bootstrap.one_sided_p_value < float(bootstrap_spec["one_sided_alpha"])
        ),
        "daily_loss_cvar_not_worse": bool(
            agent_metric["daily_loss_cvar_95"]
            <= baseline_metric["daily_loss_cvar_95"] + TOL
        ),
        "positive_advantage_in_at_least_two_years": bool(
            positive_years >= int(gate_config["minimum_positive_advantage_years"])
        ),
    }
    gate = {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "primary_comparator": "transparent_controller",
        "interface": {
            "triggered_days": triggered_count,
            "valid_plans": int(trace.loc[triggered, "plan_valid"].sum()),
            "valid_decisions": int(trace.loc[triggered, "decision_valid"].sum()),
            "plan_rate": plan_rate,
            "decision_rate": decision_rate,
            "combined_rate": interface_rate,
        },
        "behavior": {
            "llm_calls": int(len(replay.calls)),
            "nonzero_residual_days": int((trace["executed_residual"] != 0.0).sum()),
            "model_nonzero_proposals": int(
                (trace["model_proposed_residual"] != 0.0).sum()
            ),
            "tool_evidence_zeroed_days": int(
                trace["evidence_gate_status"].str.startswith("ZEROED_").sum()
            ),
        },
        "positive_advantage_years": positive_years,
        "inference": {
            "observed_mean_log_advantage": bootstrap.observed_mean_log_advantage,
            "confidence_interval_low": bootstrap.confidence_interval_low,
            "confidence_interval_high": bootstrap.confidence_interval_high,
            "one_sided_p_value": bootstrap.one_sided_p_value,
            "block_length_days": int(bootstrap_spec["block_length_days"]),
            "resamples": int(bootstrap_spec["resamples"]),
            "seed": int(bootstrap_spec["seed"]),
        },
    }
    return metrics, pd.DataFrame(yearly_rows), pd.DataFrame(regime_rows), gate


def representative_triggered_indices(
    period: PeriodInputs,
    trigger_config: dict[str, Any],
    count: int,
) -> np.ndarray:
    hard = np.argmax(period.router_probabilities, axis=1)
    transition = period.features["transition_day"].to_numpy(bool)
    confidence = period.features["router_confidence"].to_numpy(float)
    triggered = transition | (
        confidence < float(trigger_config["router_confidence_strictly_below"])
    )
    candidates = np.flatnonzero(triggered)
    if len(candidates) < int(count):
        raise Stage53Error("Insufficient triggered development days for preflight")
    offsets = np.rint(np.linspace(0, len(candidates) - 1, int(count))).astype(int)
    selected = candidates[offsets]
    if len(np.unique(selected)) != int(count):
        raise Stage53Error("Triggered preflight selection is not unique")
    if not np.all(triggered[selected]):
        raise Stage53Error("Preflight selected a non-triggered day")
    regime_counts = [int((hard[selected] == item).sum()) for item in range(3)]
    if any(value == 0 for value in regime_counts):
        raise Stage53Error("Triggered preflight does not cover every hard regime")
    return np.asarray(selected, dtype=int)


def subset_period(period: PeriodInputs, indices: np.ndarray, label: str) -> PeriodInputs:
    selected = np.asarray(indices, dtype=int)
    return PeriodInputs(
        name="interface_preflight",
        label=label,
        frame=period.frame.iloc[selected].reset_index(drop=True),
        current_trace=period.current_trace.iloc[selected].reset_index(drop=True),
        decision_dates=pd.DatetimeIndex(period.decision_dates[selected]),
        return_dates=pd.DatetimeIndex(period.return_dates[selected]),
        router_probabilities=period.router_probabilities[selected].copy(),
        asset_returns=period.asset_returns[selected].copy(),
        scenarios=period.scenarios[selected].copy(),
        features=period.features.iloc[selected].reset_index(drop=True),
        source_audit={
            "source_period": period.name,
            "selected_source_indices": [int(item) for item in selected],
            "scientific_label": label,
        },
    )


def preflight_summary(
    replay: ReplayResult,
    config: dict[str, Any],
    config_path: Path,
    provider: Any,
) -> dict[str, Any]:
    trace = replay.trace
    valid_plans = int(trace["plan_valid"].sum())
    valid_decisions = int(trace["decision_valid"].sum())
    spec = config["interface_preflight"]
    passed = bool(
        len(trace) == int(spec["sample_triggered_days"])
        and trace["agent_triggered"].all()
        and valid_plans >= int(spec["minimum_valid_plans"])
        and valid_decisions >= int(spec["minimum_valid_decisions"])
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "CONTINUE_TO_FULL_DEVELOPMENT_REPLAY"
            if passed
            else "STOP_RESIDUAL_AGENT_INTERFACE_PREFLIGHT_FAILED"
        ),
        "scientific_label": spec["scientific_label"],
        "economic_claim_permitted": False,
        "experiment_id": config["experiment_id"],
        "provider": provider.kind,
        "model": (
            config["provider"]["model"]
            if provider.kind == "ollama"
            else "CONTROLLED_MECHANICAL_PROVIDER"
        ),
        "package_config_sha256": sha256_file(config_path),
        "agent_skill_sha256": sha256_file(PACKAGE / "AGENT_SKILL.md"),
        "planner_prompt_sha256": sha256_text(planner_system_prompt()),
        "decision_prompt_sha256": sha256_text(decision_system_prompt()),
        "sampled_triggered_days": int(len(trace)),
        "valid_plans": valid_plans,
        "valid_decisions": valid_decisions,
        "minimum_valid_plans": int(spec["minimum_valid_plans"]),
        "minimum_valid_decisions": int(spec["minimum_valid_decisions"]),
        "regime_counts": {
            regime: int((trace["hard_regime"] == regime).sum())
            for regime in ["bear", "bull", "mix"]
        },
    }


def write_preflight(output: Path, summary: dict[str, Any], replay: ReplayResult) -> None:
    output.mkdir(parents=True)
    atomic_json(output / "00_PREFLIGHT.json", summary)
    atomic_csv(output / "01_PREFLIGHT_TRACE.csv", replay.trace)
    atomic_jsonl(output / "02_PREFLIGHT_CALLS.jsonl", replay.calls)
    atomic_json(
        output / "03_INTEGRATION_CHECKS.json",
        {"status": "PASS", "checks": replay.checks},
    )
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    atomic_json(
        output / "RUN_COMPLETE.json",
        {
            "status": summary["status"],
            "decision": summary["decision"],
            "provider": summary["provider"],
            "artifact_sha256": {
                str(path.relative_to(output)): sha256_file(path) for path in artifacts
            },
        },
    )


def verify_preflight(
    evidence_path: Path,
    config: dict[str, Any],
    config_path: Path,
    provider: Any,
) -> dict[str, Any]:
    path = evidence_path.expanduser().resolve()
    if not path.is_file():
        raise Stage53Error(f"Preflight evidence is missing: {path}")
    complete = load_json(path.parent / "RUN_COMPLETE.json")
    if complete.get("status") != "PASS":
        raise Stage53Error("Preflight status is not PASS")
    if complete.get("artifact_sha256", {}).get(path.name) != sha256_file(path):
        raise Stage53Error("Preflight evidence hash mismatch")
    evidence = load_json(path)
    expected = {
        "status": "PASS",
        "experiment_id": config["experiment_id"],
        "provider": provider.kind,
        "model": config["provider"]["model"],
        "package_config_sha256": sha256_file(config_path),
        "agent_skill_sha256": sha256_file(PACKAGE / "AGENT_SKILL.md"),
        "planner_prompt_sha256": sha256_text(planner_system_prompt()),
        "decision_prompt_sha256": sha256_text(decision_system_prompt()),
    }
    for key, expected_value in expected.items():
        if evidence.get(key) != expected_value:
            raise Stage53Error(f"Preflight identity mismatch: {key}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "valid_plans": int(evidence["valid_plans"]),
        "valid_decisions": int(evidence["valid_decisions"]),
    }


def plain_report(
    provider_kind: str,
    decision: str,
    metrics: pd.DataFrame,
    gate: dict[str, Any],
    memory: EvidenceMemory,
) -> str:
    agent = metrics.loc[metrics["variant"] == "bounded_tool_residual_agent"].iloc[0]
    baseline = metrics.loc[metrics["variant"] == "transparent_controller"].iloc[0]
    lines = [
        "# RAMAS Stage 5.3 bounded tool-using residual-agent result",
        "",
        f"Decision: **{decision}**",
        "",
        "The frozen uncertainty router, corrected next-day clock, costs, and risk layer were unchanged. The transparent regime controller owned every default action. The LLM-agent was called only on predeclared uncertainty or transition days and could propose only a bounded residual.",
        "",
    ]
    if provider_kind != "ollama":
        lines.extend(
            [
                "This run used the controlled mechanical provider. Its returns cannot support an economic claim.",
                "",
            ]
        )
    lines.extend(
        [
            "## Development comparison",
            "",
            f"- Residual-agent growth: {agent['terminal_growth'] * 100:.2f}%",
            f"- Transparent-controller growth: {baseline['terminal_growth'] * 100:.2f}%",
            f"- Growth difference: {(agent['terminal_growth'] - baseline['terminal_growth']) * 100:.2f}%",
            f"- Residual-agent Sharpe: {agent['sharpe_zero_cash_rate']:.3f}",
            f"- Transparent-controller Sharpe: {baseline['sharpe_zero_cash_rate']:.3f}",
            f"- One-sided block-bootstrap p-value: {gate['inference']['one_sided_p_value']:.4f}",
            f"- Triggered days: {gate['interface']['triggered_days']}",
            f"- Nonzero residual days: {gate['behavior']['nonzero_residual_days']}",
            f"- Active lessons at end: {sum(item['status'] == 'ACTIVE' for item in memory.lessons.values())}",
            "",
            "## Scientific boundary",
            "",
            "Only the 2021–2023 development period was evaluated. The reused 2024–2025 window was not opened and cannot rescue a failed development result. Multi-agent work remains prohibited unless this repaired single agent passes its fixed gate.",
            "",
        ]
    )
    return "\n".join(lines)


def final_decision(provider_kind: str, gate_passed: bool, interface_valid: bool) -> str:
    if provider_kind != "ollama":
        return "MECHANICAL_TEST_ONLY_NO_ECONOMIC_CLAIM"
    if not interface_valid:
        return "STOP_INVALID_RESIDUAL_AGENT_INTERFACE"
    if not gate_passed:
        return "STOP_AGENT_ECONOMIC_BRANCH_DEVELOPMENT_GATE_FAILED"
    return "FREEZE_SINGLE_AGENT_FOR_PROSPECTIVE_CONFIRMATION"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--corrected-run", type=Path)
    parser.add_argument("--full-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=["ollama", "controlled"], default="ollama")
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-evidence", type=Path)
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise Stage53Error(f"Refusing to overwrite output={output}")
    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    validate_config(config)
    corrected_run = resolve_corrected_run(project_root, args.corrected_run, config)
    base_dir = resolve_base_dir(corrected_run)
    adapter, accounting, risk, statistics = import_base(base_dir)
    base_config = adapter.load_config(base_dir / "config.json")
    raw, raw_audit = resolve_raw_dataset(args.full_dataset, corrected_run, config)
    period_spec = config["expected_period"]
    period_name = str(period_spec["name"])
    period = load_period_inputs(
        corrected_run / "results" / period_name,
        period_name,
        period_spec,
        raw,
        base_config,
    )
    provider: Any = (
        ControlledProvider()
        if args.provider == "controlled"
        else OllamaProvider(config["provider"])
    )

    if args.preflight_only:
        indices = representative_triggered_indices(
            period,
            config["trigger"],
            int(config["interface_preflight"]["sample_triggered_days"]),
        )
        preflight_period = subset_period(
            period,
            indices,
            config["interface_preflight"]["scientific_label"],
        )
        replay = run_agent_period(
            preflight_period,
            provider,
            EvidenceMemory(config["memory"]),
            config,
            base_config,
            accounting,
            risk,
        )
        summary = preflight_summary(replay, config, config_path, provider)
        write_preflight(output, summary, replay)
        print("RAMAS_STAGE5_3_INTERFACE_PREFLIGHT_STATUS=PASS", flush=True)
        print(f"DECISION={summary['decision']}", flush=True)
        print(f"VALID_PLANS={summary['valid_plans']}", flush=True)
        print(f"VALID_DECISIONS={summary['valid_decisions']}", flush=True)
        print(f"OUTPUT={output}", flush=True)
        if summary["status"] != "PASS":
            raise SystemExit(4)
        return

    preflight_audit: dict[str, Any] | None = None
    if provider.kind == "ollama":
        if args.preflight_evidence is None:
            raise Stage53Error("Full Ollama replay requires passed preflight evidence")
        preflight_audit = verify_preflight(
            args.preflight_evidence,
            config,
            config_path,
            provider,
        )
    memory = EvidenceMemory(config["memory"])
    replay = run_agent_period(
        period,
        provider,
        memory,
        config,
        base_config,
        accounting,
        risk,
    )
    metrics, yearly, regimes, gate = evaluate_development(
        period,
        replay,
        config,
        adapter,
        statistics,
    )
    interface_valid = bool(gate["checks"]["full_run_interface_valid"])
    decision = final_decision(provider.kind, bool(gate["passed"]), interface_valid)
    output.mkdir(parents=True)
    atomic_json(
        output / "00_CONTRACT.json",
        {
            "experiment_id": config["experiment_id"],
            "phase": config["phase"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": provider.kind,
            "model": (
                config["provider"]["model"]
                if provider.kind == "ollama"
                else "CONTROLLED_MECHANICAL_PROVIDER"
            ),
            "corrected_source_run": str(corrected_run),
            "base_source": str(base_dir),
            "base_config_sha256": sha256_file(base_dir / "config.json"),
            "package_config_sha256": sha256_file(config_path),
            "agent_skill_sha256": sha256_file(PACKAGE / "AGENT_SKILL.md"),
            "preflight": preflight_audit,
            "transparent_controller": config["transparent_controller"],
            "trigger": config["trigger"],
            "frozen_claims": config["frozen_claims"],
        },
    )
    atomic_json(
        output / "01_SOURCE_AND_INTEGRITY_AUDIT.json",
        {
            "raw_dataset": raw_audit,
            "period": period.source_audit,
            "integration_checks": replay.checks,
            "reused_oos_opened": False,
        },
    )
    atomic_csv(output / "02_POLICY_METRICS.csv", metrics)
    atomic_csv(output / "03_YEARLY_RESULTS.csv", yearly)
    atomic_csv(output / "04_REGIME_RESULTS.csv", regimes)
    atomic_json(output / "05_DEVELOPMENT_GATE.json", gate)
    atomic_jsonl(output / "06_AGENT_CALLS.jsonl", replay.calls)
    atomic_csv(output / "07_DAILY_TRACE.csv", replay.trace)
    atomic_jsonl(output / "08_MEMORY_EPISODES.jsonl", memory.episodes)
    atomic_jsonl(output / "09_MEMORY_EVENTS.jsonl", memory.events)
    atomic_json(
        output / "10_MEMORY_STATE.json",
        {
            "proposals": memory.proposals,
            "lessons": memory.lessons,
            "active_lessons": int(
                sum(item["status"] == "ACTIVE" for item in memory.lessons.values())
            ),
            "rolled_back_lessons": int(
                sum(
                    item["status"] == "ROLLED_BACK"
                    for item in memory.lessons.values()
                )
            ),
            "completed_episodes": len(memory.episodes),
        },
    )
    atomic_json(
        output / "11_FINAL_DECISION.json",
        {
            "decision": decision,
            "provider": provider.kind,
            "development_gate_passed": bool(gate["passed"]),
            "agent_interface_valid": interface_valid,
            "reused_oos_opened": False,
            "multi_agent_permitted": False,
            "economic_claim_permitted": bool(
                provider.kind == "ollama" and interface_valid
            ),
        },
    )
    (output / "12_PLAIN_ENGLISH_REPORT.md").write_text(
        plain_report(provider.kind, decision, metrics, gate, memory),
        encoding="utf-8",
    )
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    atomic_json(
        output / "RUN_COMPLETE.json",
        {
            "status": "PASS",
            "decision": decision,
            "provider": provider.kind,
            "artifact_sha256": {
                str(path.relative_to(output)): sha256_file(path) for path in artifacts
            },
        },
    )
    print("RAMAS_STAGE5_3_BOUNDED_RESIDUAL_AGENT_STATUS=PASS", flush=True)
    print(f"PROVIDER={provider.kind.upper()}", flush=True)
    print(f"DECISION={decision}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={str(gate['passed']).upper()}", flush=True)
    print(f"REUSED_OOS_OPENED=FALSE", flush=True)
    print(f"ROUTER_CHANGED=FALSE", flush=True)
    print(f"RISK_LAYER_CHANGED=FALSE", flush=True)
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
