"""One independently reconstructed arm, with pluggable credit label and retrieval.

The daily loop mirrors ``suite64.engine.run_arm`` and reuses that module's own
trust update, projection and checkpoint helpers, so legacy mode reproduces the
archived arms exactly.  Only two seams are replaced, both behind explicit spec
fields: which episodes are retrieved, and what value is stored next to them.
"""
from __future__ import annotations

from datetime import date
import math

import numpy as np

from stage63lib.engine import Arm, make_state
from stage63lib.journal import atomic_json, digest
from stage63lib.legacy import blend_exposure, legacy, state_vector, validate_decision
from suite64.engine import (
    REGIMES, _checkpoint, _new_arm, _update_trust, _validate_spec,
    project_action, retrieval_evidence,
)

from . import labels as label_module
from . import retrieval as retrieval_module

LABEL_MODES = ("legacy", "counterfactual")
RETRIEVAL_MODES = ("legacy", "balanced")


class CorrectedArmError(RuntimeError):
    pass


def validate_corrections(spec: dict) -> dict:
    """Validate the two correction fields and return a normalized copy."""
    _validate_spec(spec)
    corrections = {
        "label": spec.get("label", "legacy"),
        "retrieval": spec.get("retrieval", "legacy"),
        "per_action": int(spec.get("per_action", 2)),
        "regime_filter": spec.get("regime_filter", "same"),
        "label_authority": float(spec.get("label_authority", 1.0)),
    }
    if corrections["label"] not in LABEL_MODES:
        raise CorrectedArmError(f"Unknown label mode: {corrections['label']!r}")
    if corrections["retrieval"] not in RETRIEVAL_MODES:
        raise CorrectedArmError(f"Unknown retrieval mode: {corrections['retrieval']!r}")
    if corrections["regime_filter"] not in {"same", "none"}:
        raise CorrectedArmError("regime_filter must be 'same' or 'none'")
    if not 1 <= corrections["per_action"] <= 20:
        raise CorrectedArmError("per_action must lie between 1 and 20")
    if not 0.0 < corrections["label_authority"] <= 1.0:
        raise CorrectedArmError("label_authority must lie in (0,1]")
    if spec["memory"] == "none" and corrections["retrieval"] == "balanced":
        # A no-memory arm must stay blind regardless of retrieval style.
        corrections["retrieval"] = "balanced_suppressed"
    return corrections


def build_evidence(arm, state, spec, config, corrections):
    """Retrieve episodic evidence under the arm's declared retrieval mode."""
    if spec["memory"] == "none":
        return {"similar_completed_episodes": [], "summary": retrieval_module.NO_EVIDENCE}
    if corrections["retrieval"] == "legacy":
        return retrieval_evidence(
            arm.ledger.episodes, state, spec["memory"],
            config["agent"]["maximum_similar_episodes"],
        )
    current = date.fromisoformat(state["decision_date"])
    completed = [
        e for e in arm.ledger.episodes
        if date.fromisoformat(e["return_date"]) <= current
        and date.fromisoformat(e["decision_date"]) < current
    ]
    retrieved = retrieval_module.balanced_retrieve(
        completed, state,
        per_action=corrections["per_action"],
        regime_filter=corrections["regime_filter"],
    )
    return retrieval_module.summarize(retrieved)


def payload_for(arm, period, i, config, base_config, risk, spec, corrections, base_desired):
    """Byte-identical to the frozen payload except for the memory block."""
    decision_date = period.decision_dates[i].date()
    _update_trust(arm, decision_date, config)
    regime = REGIMES[int(np.argmax(period.router_probabilities[i]))]
    beta = 0.0 if spec["advisor"] == "abstain" else arm.trust.value(regime)
    state = make_state(period, i, arm.pretrade, base_desired, beta)
    evidence = build_evidence(arm, state, spec, config, corrections)
    previews = {}
    for action in config["agent"]["actions"]:
        projected = project_action(action, beta, base_desired, arm.pretrade,
                                   period.scenarios[i], base_config, risk, spec["risk"])
        previews[action] = dict(
            blended_desired_exposure=blend_exposure(base_desired, action, beta),
            risk_limited_exposure=float(projected.exposure),
            turnover=float(projected.turnover),
            ambiguity_cvar=None if projected.ambiguity_cvar is None else float(projected.ambiguity_cvar))
    return dict(
        task_type="historical_development_decision",
        objective="add incremental risk-adjusted value over the pre-agent RAMoE control after costs",
        state=state, memory=evidence, safe_exposure_previews=previews,
        constraints=dict(allowed_actions=["BTC", "CASH", "ABSTAIN"], shorting=False,
                         leverage=False, invalid_or_failed_response="ABSTAIN_TO_PRE_AGENT_RAMOE"))


def run_arm(period, config, base_config, accounting, risk, journal, output, spec, core_desired=None):
    """Replay one independent arm from empty memory, cash holdings and initial trust.

    Returns ``(rows, arm)``.  Every original Stage 6.4 ledger column keeps its
    frozen meaning; corrected-design fields are added alongside so a legacy-mode
    run stays directly comparable to the archived arms.
    """
    corrections = validate_corrections(spec)
    if spec["advisor"] == "llama" and journal is None:
        raise CorrectedArmError("Llama arm requires a durable journal")
    if set(config["agent"]["actions"]) != {"BTC", "CASH", "ABSTAIN"}:
        raise CorrectedArmError("All three legacy actions are required")
    cost = float(config["cost_rate"])
    if not math.isfinite(cost) or not 0 <= cost < 1:
        raise CorrectedArmError("Invalid proportional trading cost")
    total = len(period.frame)
    if not total:
        raise CorrectedArmError("Cannot run an empty period")
    decisions = [d.date() for d in period.decision_dates]
    if any((b - a).days != 1 for a, b in zip(decisions, decisions[1:])):
        raise CorrectedArmError("Continuous replay requires ordered consecutive decision days")
    if core_desired is None:
        core = np.asarray(period.current_trace["desired_exposure"], dtype=float)
    else:
        core = np.asarray(core_desired, dtype=float)
    if core.shape != (total,) or not np.all(np.isfinite(core)) or np.any((core < 0) | (core > 1)):
        raise CorrectedArmError("core_desired must contain one finite bounded exposure per day")

    arm = _new_arm(config, spec)
    rows = []
    every = int(config.get("checkpoint_every", 25))
    if every <= 0:
        raise CorrectedArmError("checkpoint_every must be positive")

    for i in range(total):
        base_desired = float(core[i])
        payload = payload_for(arm, period, i, config, base_config, risk, spec, corrections, base_desired)
        visible = {e["episode_id"] for e in payload["memory"]["similar_completed_episodes"]}
        decision = dict(action="ABSTAIN", confidence=0.0,
                        reason_codes=["INSUFFICIENT_EVIDENCE"], cited_memory_ids=[])
        outer, latency, error = {}, 0.0, ""
        valid = True
        if spec["advisor"] == "llama":
            raw, outer, latency = journal.complete(f"{i+1:06d}-{arm.name}", payload)
            try:
                if outer.get("done_reason") == "length":
                    raise ValueError("Output hit the token limit")
                decision = validate_decision(raw, config["agent"], visible)
            except (ValueError, TypeError) as exc:
                valid = False
                error = f"{type(exc).__name__}: {exc}"
        elif spec["advisor"] == "rule":
            decision["action"] = legacy.deterministic_action(payload["state"])
            decision["reason_codes"] = (["BULLISH_ROUTER", "POSITIVE_MOMENTUM"]
                                        if decision["action"] == "BTC" else ["INSUFFICIENT_EVIDENCE"])

        state, action = payload["state"], decision["action"]
        beta, base_desired = state["llama_trust_beta"], state["base_ramoe_desired_exposure"]
        projected = project_action(action, beta, base_desired, arm.pretrade,
                                   period.scenarios[i], base_config, risk, spec["risk"])
        shadow = project_action(action, 1.0 if action != "ABSTAIN" else 0.0, base_desired,
                                arm.shadow_pretrade, period.scenarios[i], base_config, risk, spec["risk"])
        asset = float(period.asset_returns[i])
        net = float(accounting.net_return(projected.exposure, arm.pretrade, asset, cost))
        shadow_net = float(accounting.net_return(shadow.exposure, arm.shadow_pretrade, asset, cost))
        reference_return = float(period.current_trace.iloc[i]["portfolio_net_return"])
        if not all(math.isfinite(v) and v > -1 for v in (asset, net, shadow_net, reference_return)):
            raise CorrectedArmError("Invalid asset/portfolio/reference return")
        if not 0 <= float(projected.exposure) <= 1:
            raise CorrectedArmError("Unbounded exposure")
        turnover = abs(float(projected.exposure) - arm.pretrade)
        if not math.isclose(turnover, float(projected.turnover), abs_tol=1e-12, rel_tol=0):
            raise CorrectedArmError("Projector turnover differs from own pretrade holdings")

        # Counterfactual credit: the taken action at full advisory authority
        # against the unchanged core, both projected from THIS arm's holdings.
        core_exposure = float(payload["safe_exposure_previews"]["ABSTAIN"]["risk_limited_exposure"])
        if action == "ABSTAIN":
            action_exposure = core_exposure
        else:
            action_exposure = float(project_action(
                action, corrections["label_authority"], base_desired, arm.pretrade,
                period.scenarios[i], base_config, risk, spec["risk"]).exposure)
        active_label, legacy_value, counterfactual_value = label_module.build_label(
            corrections["label"], action=action, shadow_net=shadow_net,
            reference_net=reference_return, action_exposure=action_exposure,
            core_exposure=core_exposure, pretrade=arm.pretrade, asset_return=asset,
            cost_rate=cost, accounting=accounting,
        )

        decision_date, return_date = state["decision_date"], state["target_return_date"]
        visible_episodes = {e["episode_id"]: e for e in arm.ledger.episodes if e["episode_id"] in visible}
        ages = [(date.fromisoformat(decision_date) - date.fromisoformat(e["return_date"])).days
                for e in visible_episodes.values()]
        previews = payload["safe_exposure_previews"]
        preview_values = [p["risk_limited_exposure"] for p in previews.values()]
        desired = blend_exposure(base_desired, action, beta)
        row = dict(
            index=i, arm=arm.name, decision_date=decision_date, return_date=return_date,
            hard_regime=state["hard_regime"], asset_simple_return=asset,
            pretrade_exposure=arm.pretrade, exposure=float(projected.exposure),
            net_return_after_trading_costs=net, net_log_return_after_trading_costs=math.log1p(net),
            gross_return_before_trading_costs=float(projected.exposure) * asset,
            turnover=turnover, cost_fraction=cost * turnover,
            trading_cost_in_wealth_units=arm.wealth * cost * turnover,
            wealth_before=arm.wealth, wealth_after=arm.wealth * (1 + net),
            action=action, beta=beta, core_desired_exposure=base_desired, desired_exposure=desired,
            shadow_pretrade_exposure=arm.shadow_pretrade, shadow_exposure=float(shadow.exposure),
            shadow_net_return=shadow_net, reference_net_return=reference_return,
            shadow_log_advantage_vs_ramoe=legacy_value,
            valid=valid, error=error, confidence=float(decision["confidence"]),
            cited_memory_ids=list(decision["cited_memory_ids"]),
            cited_memory_count=len(decision["cited_memory_ids"]),
            retrieved_memory_ids=sorted(visible), retrieved_memory_count=len(visible),
            memory_total_before=len(arm.ledger.episodes),
            retrieved_memory_cross_year_count=sum(e["return_date"][:4] < decision_date[:4] for e in visible_episodes.values()),
            retrieved_memory_cross_regime_count=sum(e["hard_regime"] != state["hard_regime"] for e in visible_episodes.values()),
            retrieved_memory_max_age_days=max(ages) if ages else None,
            request_sha256=digest(payload), latency_seconds=float(latency),
            input_token_count=int(outer.get("prompt_eval_count", 0) or 0),
            output_token_count=int(outer.get("eval_count", 0) or 0),
            action_headroom=max(preview_values) - min(preview_values) > 1e-12,
            projection_changes_selected_target=abs(float(projected.exposure) - desired) > 1e-12,
            advisor=spec["advisor"], trust_mode=spec["trust"], memory_mode=spec["memory"], risk_mode=spec["risk"],
            # --- corrected-design diagnostics, additive to the frozen schema ---
            counterfactual_log_advantage=counterfactual_value,
            active_credit_label=active_label,
            credit_label_mode=corrections["label"],
            retrieval_mode=corrections["retrieval"],
            retrieved_action_diversity=retrieval_module.action_diversity(payload["memory"]),
            core_only_exposure=core_exposure,
            label_action_exposure=action_exposure,
        )
        for preview_action, preview in previews.items():
            row["preview_" + preview_action.lower() + "_desired_exposure"] = preview["blended_desired_exposure"]
            row["preview_" + preview_action.lower() + "_exposure"] = preview["risk_limited_exposure"]

        episode = dict(
            episode_id=f"episode-{i+1:06d}", decision_date=decision_date, return_date=return_date,
            hard_regime=state["hard_regime"], state_vector=state_vector(state).tolist(),
            action=action, confidence=float(decision["confidence"]),
            shadow_log_advantage_vs_ramoe=active_label,
            legacy_shadow_log_advantage=legacy_value,
            counterfactual_log_advantage=counterfactual_value,
            asset_return=asset)
        arm.ledger.add_completed(episode)
        arm.pretrade = float(accounting.drifted_exposure(projected.exposure, asset))
        arm.shadow_pretrade = float(accounting.drifted_exposure(shadow.exposure, asset))
        arm.wealth = row["wealth_after"]
        arm.valid += int(valid)
        arm.invalid_streak = 0 if valid else arm.invalid_streak + 1
        rows.append(row)
        stop_invalid = arm.invalid_streak >= config.get("maximum_consecutive_invalid", 3)
        if i == 0 or (i + 1) % every == 0 or i + 1 == total or stop_invalid:
            _checkpoint(output, rows, arm, spec, total)
            print(f"STAGE8_ARM={arm.name} PROGRESS={i+1}/{total} VALID={arm.valid} "
                  f"NEW_CALLS={getattr(journal, 'new_calls', 0)} REUSED_CALLS={getattr(journal, 'reused_calls', 0)}",
                  flush=True)
        if stop_invalid:
            raise CorrectedArmError("Consecutive invalid responses; journal retained for investigation")

    from pathlib import Path
    output = Path(output)
    atomic_json(output / "episodes.json", arm.ledger.episodes)
    atomic_json(output / "trust_events.json", arm.trust.events)
    atomic_json(output / "ARM_STATE.json", dict(
        arm=arm.name, spec=spec, corrections=corrections, wealth=arm.wealth, pretrade=arm.pretrade,
        shadow_pretrade=arm.shadow_pretrade, beta=arm.trust.beta, episodes=len(arm.ledger.episodes),
        valid=arm.valid, stale_evidence_trust_changes=arm.stale_changes,
        outcome_reference="COUNTERFACTUAL_SAME_PATH_CORE_BASELINE" if corrections["label"] == "counterfactual"
        else "UNCHANGED_ARCHIVED_NUMERICAL_REFERENCE"))
    return rows, arm
