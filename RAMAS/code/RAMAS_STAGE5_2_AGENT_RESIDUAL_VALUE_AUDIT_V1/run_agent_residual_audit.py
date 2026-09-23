#!/usr/bin/env python3
"""RAMAS Stage 5.2: zero-LLM-call agent residual value audit.

This program does not tune a new policy and does not open OOS data.  It uses the
already-recorded Stage 5.1 development actions to answer one narrow question:
did the LLM agent add value beyond the transparent regime mapping that explains
most of its actions?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import statistics
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


EXPERIMENT_ID = "RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1"
REQUIRED_TRACE_COLUMNS = {
    "decision_date",
    "return_date",
    "hard_regime",
    "transition_day",
    "router_confidence",
    "requested_exposure",
    "final_exposure",
    "current_ramoe_desired_exposure",
    "pretrade_exposure",
    "turnover",
    "asset_simple_return",
    "portfolio_net_return",
    "confidence",
    "reason_codes",
    "cited_memory_ids",
    "active_lessons_seen",
}
ALLOWED_EXPOSURES = {0.0, 0.25, 0.5, 0.75, 1.0}
STATIC_MAPPING = {"bear": 0.25, "bull": 0.50, "mix": 0.25}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"unsafe archive path: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"links are not accepted in result archives: {member.name}")
        try:
            tf.extractall(destination, filter="data")
        except TypeError:  # Python 3.10 compatibility after the explicit checks above.
            tf.extractall(destination)


def locate_results(source: Path, temporary_parent: Path) -> tuple[Path, str]:
    source = source.resolve()
    if source.is_file():
        extract_dir = temporary_parent / "stage5_1_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        safe_extract_tar(source, extract_dir)
        candidates = list(extract_dir.rglob("corrected_pre2024/01_DAILY_TRACE.csv"))
        if len(candidates) != 1:
            raise ValueError(f"expected one Stage 5.1 trace in archive, found {len(candidates)}")
        return candidates[0].parents[1], sha256_file(source)

    candidates = []
    for candidate in (source, source / "results"):
        if (candidate / "corrected_pre2024/01_DAILY_TRACE.csv").is_file():
            candidates.append(candidate)
    if len(candidates) != 1:
        nested = list(source.rglob("corrected_pre2024/01_DAILY_TRACE.csv"))
        if len(nested) != 1:
            raise ValueError(f"expected one Stage 5.1 trace under directory, found {len(nested)}")
        candidates = [nested[0].parents[1]]
    return candidates[0], "DIRECTORY_INPUT"


def read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_TRACE_COLUMNS - fields
        if missing:
            raise ValueError(f"trace missing columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = dict(raw)
            for key in (
                "router_confidence",
                "requested_exposure",
                "final_exposure",
                "current_ramoe_desired_exposure",
                "pretrade_exposure",
                "turnover",
                "asset_simple_return",
                "portfolio_net_return",
                "confidence",
                "active_lessons_seen",
            ):
                row[key] = float(raw[key])
            row["transition_day"] = as_bool(raw["transition_day"])
            row["hard_regime"] = raw["hard_regime"].strip().lower()
            rows.append(row)
    if not rows:
        raise ValueError("empty trace")
    return rows


def next_day_clock_valid(rows: Sequence[dict[str, Any]]) -> bool:
    return all(
        date.fromisoformat(row["return_date"]) > date.fromisoformat(row["decision_date"])
        for row in rows
    )


def replay_policy(
    rows: Sequence[dict[str, Any]],
    targets: Sequence[float],
    cost_rate: float,
) -> list[dict[str, float]]:
    if len(rows) != len(targets):
        raise ValueError("target length does not match trace")
    pretrade = 0.0
    output: list[dict[str, float]] = []
    for row, target_raw in zip(rows, targets):
        target = float(target_raw)
        if not (0.0 <= target <= 1.0) or not math.isfinite(target):
            raise ValueError(f"invalid target exposure: {target}")
        asset_return = float(row["asset_simple_return"])
        if asset_return <= -1.0 or not math.isfinite(asset_return):
            raise ValueError(f"invalid asset return: {asset_return}")
        turnover = abs(target - pretrade)
        net = (1.0 - cost_rate * turnover) * (1.0 + target * asset_return) - 1.0
        denominator = 1.0 + target * asset_return
        next_pretrade = target * (1.0 + asset_return) / denominator
        output.append(
            {
                "target_exposure": target,
                "pretrade_exposure": pretrade,
                "turnover": turnover,
                "net_return": net,
            }
        )
        pretrade = next_pretrade
    return output


def fractional_expected_shortfall(returns: Sequence[float], alpha: float = 0.05) -> float:
    if not returns:
        return 0.0
    values = sorted(float(x) for x in returns)
    mass = alpha * len(values)
    whole = int(math.floor(mass))
    fraction = mass - whole
    total = sum(values[:whole])
    if fraction > 0.0:
        total += fraction * values[whole]
    return -total / mass


def metrics(returns: Sequence[float], exposures: Sequence[float], turnovers: Sequence[float]) -> dict[str, float]:
    values = [float(x) for x in returns]
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in values:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak)
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    annualizer = math.sqrt(365.25)
    return {
        "rows": n,
        "terminal_growth": wealth - 1.0,
        "annualized_return": wealth ** (365.25 / n) - 1.0,
        "annualized_volatility": std * annualizer,
        "sharpe": (mean / std * annualizer) if std > 0.0 else 0.0,
        "maximum_drawdown": max_drawdown,
        "daily_loss_cvar_95": fractional_expected_shortfall(values, 0.05),
        "mean_exposure": statistics.fmean(exposures),
        "total_turnover": sum(turnovers),
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def circular_block_bootstrap(
    differences: Sequence[float],
    block_length: int,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    data = [float(x) for x in differences]
    n = len(data)
    observed = statistics.fmean(data)
    centered = [x - observed for x in data]
    rng = random.Random(seed)
    null_means: list[float] = []
    raw_means: list[float] = []
    for _ in range(resamples):
        null_total = 0.0
        raw_total = 0.0
        count = 0
        while count < n:
            start = rng.randrange(n)
            take = min(block_length, n - count)
            for offset in range(take):
                index = (start + offset) % n
                null_total += centered[index]
                raw_total += data[index]
            count += take
        null_means.append(null_total / n)
        raw_means.append(raw_total / n)
    exceedances = sum(value >= observed for value in null_means)
    return {
        "observed_mean_log_advantage": observed,
        "one_sided_p_value": (exceedances + 1.0) / (resamples + 1.0),
        "confidence_interval_low": percentile(raw_means, 0.025),
        "confidence_interval_high": percentile(raw_means, 0.975),
        "block_length_days": block_length,
        "resamples": resamples,
        "seed": seed,
    }


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    return numerator / (dx * dy) if dx > 0.0 and dy > 0.0 else 0.0


def summarize_policy(name: str, replay: Sequence[dict[str, float]]) -> dict[str, Any]:
    result = metrics(
        [x["net_return"] for x in replay],
        [x["target_exposure"] for x in replay],
        [x["turnover"] for x in replay],
    )
    return {"policy": name, **result}


def yearly_rows(
    rows: Sequence[dict[str, Any]],
    policy_replays: dict[str, Sequence[dict[str, float]]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["return_date"][:4]].append(index)
    output: list[dict[str, Any]] = []
    for year, indices in sorted(groups.items()):
        for name, replay in policy_replays.items():
            subset = [replay[i] for i in indices]
            output.append({"year": year, **summarize_policy(name, subset)})
    return output


def regime_rows(
    rows: Sequence[dict[str, Any]],
    policy_replays: dict[str, Sequence[dict[str, float]]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["hard_regime"]].append(index)
        if row["transition_day"]:
            groups["transition_day"].append(index)
    output: list[dict[str, Any]] = []
    for regime, indices in sorted(groups.items()):
        for name, replay in policy_replays.items():
            subset = [replay[i] for i in indices]
            output.append({"regime": regime, **summarize_policy(name, subset)})
    return output


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))
    return result


def verify_source_manifest(results_root: Path) -> dict[str, Any]:
    manifest_path = results_root / "RUN_COMPLETE.json"
    if not manifest_path.is_file():
        return {"present": False, "passed": False, "checked_files": 0, "mismatches": ["RUN_COMPLETE.json missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("artifact_sha256", {})
    mismatches = []
    checked = 0
    for relative, expected_hash in expected.items():
        artifact = results_root / relative
        checked += 1
        if not artifact.is_file():
            mismatches.append(f"missing:{relative}")
        elif sha256_file(artifact) != expected_hash:
            mismatches.append(f"hash:{relative}")
    return {
        "present": True,
        "passed": bool(expected) and not mismatches and manifest.get("status") == "PASS",
        "checked_files": checked,
        "mismatches": mismatches,
        "source_status": manifest.get("status"),
        "source_decision": manifest.get("decision"),
    }


def build_behavior_audit(
    rows: Sequence[dict[str, Any]],
    agent_core: Sequence[dict[str, float]],
    static_core: Sequence[dict[str, float]],
    memory_events: Sequence[dict[str, Any]],
    memory_state: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    static_actions = [STATIC_MAPPING[row["hard_regime"]] for row in rows]
    agent_actions = [float(row["requested_exposure"]) for row in rows]
    deviations: list[dict[str, Any]] = []
    deviation_log_advantages: list[float] = []
    reason_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter(f"{value:.2f}" for value in agent_actions)
    memory_citations = 0
    active_rows = 0
    active_rows_that_deviated = 0
    confidence: list[float] = []
    daily_log_advantage: list[float] = []
    transition_reason_rows = 0
    transition_reason_true = 0

    for i, row in enumerate(rows):
        reasons = [x for x in str(row["reason_codes"]).split("|") if x]
        reason_counts.update(reasons)
        if str(row["cited_memory_ids"]).strip():
            memory_citations += 1
        if float(row["active_lessons_seen"]) > 0:
            active_rows += 1
            if abs(agent_actions[i] - static_actions[i]) > 1e-12:
                active_rows_that_deviated += 1
        if "TRANSITION_RISK" in reasons:
            transition_reason_rows += 1
            if row["transition_day"]:
                transition_reason_true += 1
        advantage = math.log1p(agent_core[i]["net_return"]) - math.log1p(static_core[i]["net_return"])
        confidence.append(float(row["confidence"]))
        daily_log_advantage.append(advantage)
        if abs(agent_actions[i] - static_actions[i]) > 1e-12:
            deviation_log_advantages.append(advantage)
            deviations.append(
                {
                    "decision_date": row["decision_date"],
                    "return_date": row["return_date"],
                    "hard_regime": row["hard_regime"],
                    "transition_day": row["transition_day"],
                    "router_confidence": row["router_confidence"],
                    "static_action": static_actions[i],
                    "agent_action": agent_actions[i],
                    "asset_simple_return": row["asset_simple_return"],
                    "agent_minus_static_log_return": advantage,
                    "agent_better": advantage > 1e-15,
                }
            )

    event_counts = Counter(str(event.get("event", "UNKNOWN")) for event in memory_events)
    active_lessons = memory_state.get("active_lessons", []) if memory_state else []
    proposals = [event for event in memory_events if event.get("event") == "LESSON_QUARANTINED"]
    proposal_text_counts = Counter(str(event.get("text", "")) for event in proposals)
    audit = {
        "rows": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "static_mapping": STATIC_MAPPING,
        "static_mapping_exact_rows": len(rows) - len(deviations),
        "static_mapping_fidelity": (len(rows) - len(deviations)) / len(rows),
        "agent_deviation_rows": len(deviations),
        "agent_deviation_better_rows": sum(bool(row["agent_better"]) for row in deviations),
        "agent_deviation_mean_log_advantage": statistics.fmean(deviation_log_advantages) if deviation_log_advantages else 0.0,
        "memory_citation_rows": memory_citations,
        "memory_citation_rate": memory_citations / len(rows),
        "active_lesson_rows": active_rows,
        "active_lesson_rows_that_changed_static_action": active_rows_that_deviated,
        "active_lessons_at_end": len(active_lessons),
        "memory_event_counts": dict(sorted(event_counts.items())),
        "quarantined_proposals": len(proposals),
        "unique_quarantined_lesson_texts": len(proposal_text_counts),
        "maximum_repeated_lesson_text_count": max(proposal_text_counts.values(), default=0),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "transition_reason_rows": transition_reason_rows,
        "actual_transition_rows": sum(bool(row["transition_day"]) for row in rows),
        "transition_reason_precision": transition_reason_true / transition_reason_rows if transition_reason_rows else 0.0,
        "confidence_log_advantage_correlation": pearson(confidence, daily_log_advantage),
    }
    return audit, deviations


def fixed_trigger_diagnostics(
    rows: Sequence[dict[str, Any]],
    agent_actions: Sequence[float],
    static_actions: Sequence[float],
    cost_rate: float,
) -> list[dict[str, Any]]:
    triggers: dict[str, Callable[[dict[str, Any]], bool]] = {
        "STATIC_NO_AGENT": lambda row: False,
        "FULL_RECORDED_AGENT": lambda row: True,
        "TRANSITION_ONLY": lambda row: bool(row["transition_day"]),
        "LOW_ROUTER_CONFIDENCE_LT_0_80_ONLY": lambda row: float(row["router_confidence"]) < 0.80,
        "BEAR_ONLY": lambda row: row["hard_regime"] == "bear",
        "BULL_ONLY": lambda row: row["hard_regime"] == "bull",
        "MIX_ONLY": lambda row: row["hard_regime"] == "mix",
        "UNCERTAINTY_OR_TRANSITION": lambda row: bool(row["transition_day"]) or float(row["router_confidence"]) < 0.80,
    }
    result = []
    for name, trigger in triggers.items():
        chosen = [agent_actions[i] if trigger(row) else static_actions[i] for i, row in enumerate(rows)]
        replay = replay_policy(rows, chosen, cost_rate)
        triggered = sum(trigger(row) for row in rows)
        changed = sum(trigger(row) and abs(agent_actions[i] - static_actions[i]) > 1e-12 for i, row in enumerate(rows))
        result.append({"trigger_policy": name, "triggered_rows": triggered, "changed_rows": changed, **summarize_policy(name, replay)})
    return result


def build_report(
    decision: dict[str, Any],
    metrics_by_name: dict[str, dict[str, Any]],
    behavior: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    agent = metrics_by_name["RECORDED_AGENT_CORE_NO_SAFETY_LAYER"]
    static = metrics_by_name["DISTILLED_STATIC_CORE_NO_SAFETY_LAYER"]
    deployed = metrics_by_name["RECORDED_AGENT_DEPLOYED"]
    lines = [
        "# RAMAS Stage 5.2 result",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "This audit made zero LLM calls and did not read the reused 2024–2025 window.",
        "It compares the recorded LLM core actions with the transparent regime rule under one common accounting engine.",
        "The core comparison intentionally removes the safety layer from both sides; it is a mechanism test, not a new deployment claim.",
        "",
        "## Headline",
        "",
        f"- Recorded deployed agent growth: {deployed['terminal_growth']:.2%}",
        f"- Agent core growth: {agent['terminal_growth']:.2%}",
        f"- Static core growth: {static['terminal_growth']:.2%}",
        f"- Agent-minus-static core growth: {agent['terminal_growth'] - static['terminal_growth']:+.2%}",
        f"- Agent core Sharpe: {agent['sharpe']:.3f}",
        f"- Static core Sharpe: {static['sharpe']:.3f}",
        f"- One-sided block-bootstrap p-value: {gate['inference']['one_sided_p_value']:.4f}",
        f"- Static-rule fidelity: {behavior['static_mapping_fidelity']:.2%}",
        f"- Agent deviations: {behavior['agent_deviation_rows']}",
        f"- Active lessons remaining at end: {behavior['active_lessons_at_end']}",
        "",
        "## Interpretation",
        "",
        decision["plain_english"],
        "",
        "## Constraint",
        "",
        "The 2024–2025 diagnostic remains sealed. No multi-agent experiment is permitted by this stage.",
        "The next allowed model experiment is one bounded residual-agent repair defined in AGENT_REPAIR_CONTRACT.md.",
    ]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("config experiment_id is not frozen")
    configured_mapping = {str(k): float(v) for k, v in config.get("distilled_static_mapping", {}).items()}
    if configured_mapping != STATIC_MAPPING:
        raise ValueError("config static mapping differs from the frozen mapping")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    temporary_parent = Path(tempfile.mkdtemp(prefix="ramas_stage5_2_"))
    try:
        results_root, source_archive_sha = locate_results(Path(args.stage5_source), temporary_parent)
        trace_path = results_root / "corrected_pre2024/01_DAILY_TRACE.csv"
        rows = read_trace(trace_path)
        source_contract = json.loads((results_root / "00_CONTRACT.json").read_text(encoding="utf-8"))
        source_decision = json.loads((results_root / "05_FINAL_DECISION.json").read_text(encoding="utf-8"))
        source_manifest = verify_source_manifest(results_root)
        if source_contract.get("experiment_id") != config["source_experiment_id"]:
            raise ValueError("unexpected source experiment")
        if source_decision.get("reused_oos_diagnostic_run") is not False:
            raise ValueError("source opened reused OOS; audit refused")

        cost_rate = float(config["accounting"]["transaction_cost_rate"])
        agent_actions = [float(row["requested_exposure"]) for row in rows]
        deployed_actions = [float(row["final_exposure"]) for row in rows]
        static_actions = [STATIC_MAPPING[row["hard_regime"]] for row in rows]
        current_actions = [float(row["current_ramoe_desired_exposure"]) for row in rows]

        deployed_replay = replay_policy(rows, deployed_actions, cost_rate)
        agent_core_replay = replay_policy(rows, agent_actions, cost_rate)
        static_core_replay = replay_policy(rows, static_actions, cost_rate)
        current_core_replay = replay_policy(rows, current_actions, cost_rate)
        policy_replays = {
            "RECORDED_AGENT_DEPLOYED": deployed_replay,
            "RECORDED_AGENT_CORE_NO_SAFETY_LAYER": agent_core_replay,
            "DISTILLED_STATIC_CORE_NO_SAFETY_LAYER": static_core_replay,
            "CURRENT_RAMOE_CORE_NO_SAFETY_LAYER": current_core_replay,
        }
        policy_metrics = [summarize_policy(name, replay) for name, replay in policy_replays.items()]
        metrics_by_name = {row["policy"]: row for row in policy_metrics}

        reconstructed = [row["net_return"] for row in deployed_replay]
        recorded = [float(row["portfolio_net_return"]) for row in rows]
        max_return_error = max(abs(a - b) for a, b in zip(reconstructed, recorded))
        max_pretrade_error = max(
            abs(replay["pretrade_exposure"] - float(row["pretrade_exposure"]))
            for replay, row in zip(deployed_replay, rows)
        )
        allowed_actions = all(any(abs(float(row["requested_exposure"]) - x) < 1e-12 for x in ALLOWED_EXPOSURES) for row in rows)
        integrity = {
            "status": "PASS",
            "checks": {
                "source_experiment_matches": True,
                "source_internal_hash_manifest": source_manifest["passed"],
                "reused_oos_not_run": True,
                "next_day_clock": next_day_clock_valid(rows),
                "decision_dates_unique": len({row["decision_date"] for row in rows}) == len(rows),
                "allowed_action_grid": allowed_actions,
                "deployed_accounting_reconstruction": max_return_error < 1e-12,
                "deployed_pretrade_reconstruction": max_pretrade_error < 1e-12,
            },
            "maximum_return_reconstruction_error": max_return_error,
            "maximum_pretrade_reconstruction_error": max_pretrade_error,
            "source_manifest": source_manifest,
        }
        if not all(integrity["checks"].values()):
            integrity["status"] = "FAIL"
            raise ValueError(f"integrity checks failed: {integrity}")

        memory_events = parse_jsonl(results_root / "09_MEMORY_EVENTS.jsonl")
        memory_state_path = results_root / "10_MEMORY_STATE.json"
        memory_state = json.loads(memory_state_path.read_text(encoding="utf-8")) if memory_state_path.is_file() else {}
        behavior, deviations = build_behavior_audit(rows, agent_core_replay, static_core_replay, memory_events, memory_state)

        log_differences = [
            math.log1p(a["net_return"]) - math.log1p(s["net_return"])
            for a, s in zip(agent_core_replay, static_core_replay)
        ]
        bootstrap_cfg = config["inference"]
        inference = circular_block_bootstrap(
            log_differences,
            int(bootstrap_cfg["block_length_days"]),
            int(bootstrap_cfg["resamples"]),
            int(bootstrap_cfg["seed"]),
        )
        agent_metrics = metrics_by_name["RECORDED_AGENT_CORE_NO_SAFETY_LAYER"]
        static_metrics = metrics_by_name["DISTILLED_STATIC_CORE_NO_SAFETY_LAYER"]
        yearly = yearly_rows(rows, policy_replays)
        agent_years = {row["year"]: row for row in yearly if row["policy"] == "RECORDED_AGENT_CORE_NO_SAFETY_LAYER"}
        static_years = {row["year"]: row for row in yearly if row["policy"] == "DISTILLED_STATIC_CORE_NO_SAFETY_LAYER"}
        positive_advantage_years = sum(
            agent_years[year]["terminal_growth"] > static_years[year]["terminal_growth"]
            for year in agent_years
        )
        requirements = config["agent_value_gate"]
        gate_checks = {
            "positive_growth_advantage": agent_metrics["terminal_growth"] > static_metrics["terminal_growth"],
            "sharpe_improvement_margin": agent_metrics["sharpe"] - static_metrics["sharpe"] >= float(requirements["minimum_sharpe_improvement"]),
            "daily_loss_cvar_not_worse": agent_metrics["daily_loss_cvar_95"] <= static_metrics["daily_loss_cvar_95"] + 1e-15,
            "one_sided_block_bootstrap_p_below_alpha": inference["one_sided_p_value"] < float(requirements["alpha"]),
            "minimum_positive_advantage_years": positive_advantage_years >= int(requirements["minimum_positive_advantage_years"]),
            "minimum_agent_deviation_rows": behavior["agent_deviation_rows"] >= int(requirements["minimum_agent_deviation_rows"]),
        }
        agent_value_passed = all(gate_checks.values())
        promoted = int(behavior["memory_event_counts"].get("LESSON_PROMOTED", 0))
        rolled_back = int(behavior["memory_event_counts"].get("LESSON_ROLLED_BACK", 0))
        memory_checks = {
            "active_lesson_survived": behavior["active_lessons_at_end"] > 0,
            "more_promotions_than_rollbacks": promoted > rolled_back,
            "active_lessons_changed_actions": behavior["active_lesson_rows_that_changed_static_action"] > 0,
        }
        memory_value_passed = all(memory_checks.values())
        gate = {
            "passed": agent_value_passed,
            "scope": "CORE_POLICY_MECHANISM_ONLY_NOT_DEPLOYMENT_CLAIM",
            "checks": gate_checks,
            "positive_advantage_years": positive_advantage_years,
            "agent_minus_static": {
                "terminal_growth": agent_metrics["terminal_growth"] - static_metrics["terminal_growth"],
                "sharpe": agent_metrics["sharpe"] - static_metrics["sharpe"],
                "maximum_drawdown": agent_metrics["maximum_drawdown"] - static_metrics["maximum_drawdown"],
                "daily_loss_cvar_95": agent_metrics["daily_loss_cvar_95"] - static_metrics["daily_loss_cvar_95"],
            },
            "inference": inference,
            "memory_gate": {"passed": memory_value_passed, "checks": memory_checks},
        }

        if agent_value_passed and memory_value_passed:
            decision_code = "ADVANCE_TO_BOUNDED_RESIDUAL_AGENT_PREFLIGHT"
            plain = "Recorded agent residual decisions and memory both passed the predeclared mechanism gates. A bounded residual-agent preflight is allowed; OOS and multi-agent work remain prohibited."
        elif agent_value_passed:
            decision_code = "REPAIR_MEMORY_BEFORE_ANY_NEW_AGENT_REPLAY"
            plain = "Recorded agent actions added measurable value, but the memory mechanism did not. Replace memory promotion with evidence-gated sequential validation before another model replay."
        else:
            decision_code = "REDESIGN_AS_TOOL_USING_RESIDUAL_AGENT_NO_OOS"
            plain = "The current LLM policy did not establish incremental value over the transparent regime rule. Do not repeat the same target-exposure prompt. The only defensible continuation is one bounded, tool-using residual-agent repair with a hard kill gate."
        decision = {
            "decision": decision_code,
            "agent_incremental_value_established": agent_value_passed,
            "memory_incremental_value_established": memory_value_passed,
            "new_llm_calls_performed": False,
            "reused_oos_opened": False,
            "multi_agent_experiment_permitted": False,
            "one_bounded_single_agent_repair_permitted": True,
            "plain_english": plain,
        }

        trigger_rows = fixed_trigger_diagnostics(rows, agent_actions, static_actions, cost_rate)
        contract = {
            "experiment_id": EXPERIMENT_ID,
            "completed_at_utc": utc_now(),
            "phase": "EXPLORATORY_MECHANISM_AUDIT",
            "source_experiment_id": source_contract["experiment_id"],
            "source_archive_sha256": source_archive_sha,
            "source_trace_sha256": sha256_file(trace_path),
            "rows": len(rows),
            "frozen_claims": {
                "new_llm_calls": False,
                "router_changed": False,
                "risk_layer_changed": False,
                "transaction_cost_changed": False,
                "return_clock_changed": False,
                "reused_oos_opened": False,
                "fixed_trigger_diagnostics_are_candidate_selection": False,
                "core_policy_audit_is_deployment_evidence": False,
            },
            "mechanism_definition": {
                "agent_core": "recorded requested_exposure",
                "static_core": STATIC_MAPPING,
                "common_accounting": "drifted pretrade exposure plus symmetric turnover cost",
                "safety_layer": "removed from both core policies only for mechanism isolation",
            },
        }

        metric_fields = ["policy", "rows", "terminal_growth", "annualized_return", "annualized_volatility", "sharpe", "maximum_drawdown", "daily_loss_cvar_95", "mean_exposure", "total_turnover"]
        write_json(output / "00_CONTRACT.json", contract)
        write_json(output / "01_INTEGRITY_CHECKS.json", integrity)
        write_csv(output / "02_CORE_POLICY_METRICS.csv", policy_metrics, metric_fields)
        write_csv(output / "03_YEARLY_RESULTS.csv", yearly, ["year", *metric_fields])
        write_csv(output / "04_REGIME_RESULTS.csv", regime_rows(rows, policy_replays), ["regime", *metric_fields])
        write_csv(
            output / "05_AGENT_DEVIATIONS.csv",
            deviations,
            ["decision_date", "return_date", "hard_regime", "transition_day", "router_confidence", "static_action", "agent_action", "asset_simple_return", "agent_minus_static_log_return", "agent_better"],
        )
        write_json(output / "06_BEHAVIOR_AND_MEMORY_AUDIT.json", behavior)
        write_csv(output / "07_FIXED_TRIGGER_DIAGNOSTICS.csv", trigger_rows, ["trigger_policy", "triggered_rows", "changed_rows", *metric_fields])
        write_json(output / "08_INCREMENTAL_AGENT_GATE.json", gate)
        write_json(output / "09_FINAL_DECISION.json", decision)
        (output / "10_PLAIN_ENGLISH_REPORT.md").write_text(build_report(decision, metrics_by_name, behavior, gate), encoding="utf-8")
        write_json(output / "RUN_COMPLETE.json", {"status": "PASS", "decision": decision_code, "completed_at_utc": utc_now()})

        artifact_paths = sorted(path for path in output.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
        (output / "SHA256SUMS.txt").write_text(
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in artifact_paths),
            encoding="utf-8",
        )
        return output
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage5-source", required=True, help="Stage 5.1 result archive or extracted run directory")
    parser.add_argument("--output", required=True, help="new, non-existing output directory")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    return parser.parse_args()


if __name__ == "__main__":
    result_path = run(parse_args())
    decision = json.loads((result_path / "09_FINAL_DECISION.json").read_text(encoding="utf-8"))
    print("RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_STATUS=PASS")
    print(f"DECISION={decision['decision']}")
    print(f"AGENT_INCREMENTAL_VALUE_ESTABLISHED={str(decision['agent_incremental_value_established']).upper()}")
    print(f"MEMORY_INCREMENTAL_VALUE_ESTABLISHED={str(decision['memory_incremental_value_established']).upper()}")
    print("NEW_LLM_CALLS=0")
    print("REUSED_OOS_OPENED=FALSE")
    print(f"OUTPUT={result_path}")
