#!/usr/bin/env python3
"""RAMAS Stage 7: matched sequential-memory validation.

This runner is deliberately narrower than the Stage 6.4 component suite.  It
adds two fresh Llama-70B fixed-beta arms (memory and no-memory) and an optional
state-controlled pair.  The closed-loop pair is the primary test of whether
completed episodic memory changes later decisions and outcomes.  The
state-controlled pair is a diagnostic: it keeps the exogenous/current state
stream common so an action difference cannot be attributed to diverged
holdings.

The runner reuses the byte-pinned Stage 6.4 source and numerical engine.  It
does not repair the inherited legacy clock or claim fresh OOS evidence; those
require a separately audited source release.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parent
STAGE64 = PACKAGE / "RAMAS_STAGE6_4_COMPONENT_SUITE_V1"
sys.path.insert(0, str(STAGE64))

from stage63lib.engine import Arm, eligible_episodes, make_state  # noqa: E402
from stage63lib.fixed_trust import FixedTrust  # noqa: E402
from stage63lib.inputs import load_inputs  # noqa: E402
from stage63lib.journal import Journal, atomic_json, digest  # noqa: E402
from stage63lib.legacy import (  # noqa: E402
    EpisodicMemory,
    ControlledProvider,
    blend_exposure,
    legacy,
    state_vector,
    validate_decision,
)
from stage63lib.provider import OllamaProvider  # noqa: E402
from stage63lib.fixture import build_fixture  # noqa: E402
from suite64.core_variants import build_core_variants  # noqa: E402
from suite64.engine import payload_for, project_action, run_arm  # noqa: E402
from suite64.metrics import atomic_csv, paired_block_comparison, summarize_ledger  # noqa: E402


EXPERIMENT_ID = "RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1"
MODEL = "llama3.3:70b"
FIXED_BETA = 0.05
REGIMES = ("bear", "bull", "mix")


def sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source_config() -> dict:
    source = json.loads((STAGE64 / "source_config.json").read_text())
    if source["provider"]["model"] != MODEL or source.get("qwen_used") is not False:
        raise RuntimeError("The Stage 7 package is locked to llama3.3:70b and Qwen=false")
    # Make the inherited fixed-trust contract explicit for the fresh pair.
    source["experiment_id"] = EXPERIMENT_ID
    source["arms"] = ["memory", "no_memory"]
    source["continue_across_years"] = True
    source["common_legacy_trust_rule"] = False
    source["agent_trust_mode"] = "FIXED_POSITIVE"
    source["fixed_beta"] = FIXED_BETA
    source["qwen_used"] = False
    return source


def arm_specs(prefix: str) -> list[dict]:
    return [
        dict(
            name=f"{prefix}_memory",
            advisor="llama",
            trust="fixed",
            memory="expanding",
            core="original",
            risk="standard",
            fixed_beta=FIXED_BETA,
        ),
        dict(
            name=f"{prefix}_no_memory",
            advisor="llama",
            trust="fixed",
            memory="none",
            core="original",
            risk="standard",
            fixed_beta=FIXED_BETA,
        ),
    ]


def _write_json(path: Path, value: object) -> None:
    atomic_json(path, value)


def _ensure_same_dates(frame: pd.DataFrame, arms: tuple[str, str]) -> pd.DataFrame:
    selected = frame[frame["arm"].isin(arms)].copy()
    selected["return_date"] = pd.to_datetime(selected["return_date"])
    if selected.empty:
        raise ValueError("No rows for requested arms")
    sizes = selected.groupby("arm").size().to_dict()
    if sizes.get(arms[0]) != sizes.get(arms[1]):
        raise ValueError(f"Unmatched arm lengths: {sizes}")
    pivot = selected.pivot(index="return_date", columns="arm", values="net_return_after_trading_costs")
    if pivot.isna().any().any() or len(pivot) != sizes[arms[0]]:
        raise ValueError("Arms do not share one complete return date grid")
    return selected


def transmission(frame: pd.DataFrame, left: str, right: str, label: str) -> pd.DataFrame:
    _ensure_same_dates(frame, (left, right))
    l = frame[frame.arm == left].set_index("return_date")
    r = frame[frame.arm == right].set_index("return_date")
    l.index = pd.to_datetime(l.index)
    r.index = pd.to_datetime(r.index)
    common = l.index.intersection(r.index)
    rows = []
    for key, mask in [
        ("ALL", np.ones(len(common), dtype=bool)),
        ("POST2021", common >= pd.Timestamp("2022-01-01")),
        ("BEAR", l.loc[common, "hard_regime"].astype(str).str.lower().to_numpy() == "bear"),
        ("BULL", l.loc[common, "hard_regime"].astype(str).str.lower().to_numpy() == "bull"),
        ("MIX", l.loc[common, "hard_regime"].astype(str).str.lower().to_numpy() == "mix"),
    ]:
        ll, rr = l.loc[common].iloc[np.flatnonzero(mask)], r.loc[common].iloc[np.flatnonzero(mask)]
        rec = {
            "comparison": label,
            "scope": key,
            "days": int(mask.sum()),
            "action_different_days": int((ll["action"].to_numpy() != rr["action"].to_numpy()).sum()),
            "desired_exposure_different_days": int(
                (np.abs(ll["desired_exposure"].to_numpy(float) - rr["desired_exposure"].to_numpy(float)) > 1e-12).sum()
            ),
            "final_exposure_different_days": int(
                (np.abs(ll["exposure"].to_numpy(float) - rr["exposure"].to_numpy(float)) > 1e-12).sum()
            ),
            "mean_abs_final_exposure_difference": float(
                np.abs(ll["exposure"].to_numpy(float) - rr["exposure"].to_numpy(float)).mean()
            ) if len(ll) else None,
            "mean_abs_log_return_difference_bps": float(
                np.abs(
                    np.log1p(ll["net_return_after_trading_costs"].to_numpy(float))
                    - np.log1p(rr["net_return_after_trading_costs"].to_numpy(float))
                ).mean() * 10000.0
            ) if len(ll) else None,
            "memory_retrieved_days_left": int((ll.get("retrieved_memory_count", pd.Series(0, index=ll.index)) > 0).sum()),
            "cited_memory_days_left": int((ll.get("cited_memory_count", pd.Series(0, index=ll.index)) > 0).sum()),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def audit_memory_rows(frame: pd.DataFrame, episodes_by_arm: dict[str, list[dict]]) -> dict:
    violations = []
    no_memory_leaks = 0
    for arm_name, episodes in episodes_by_arm.items():
        by_id = {e["episode_id"]: e for e in episodes}
        part = frame[frame.arm == arm_name]
        for _, row in part.iterrows():
            retrieved = row.get("retrieved_memory_ids", [])
            if isinstance(retrieved, str):
                try:
                    retrieved = json.loads(retrieved)
                except json.JSONDecodeError:
                    retrieved = []
            retrieved = list(retrieved or [])
            if "no_memory" in arm_name:
                if retrieved or int(row.get("retrieved_memory_count", 0)) != 0 or int(row.get("cited_memory_count", 0)) != 0:
                    no_memory_leaks += 1
                continue
            decision = date.fromisoformat(str(row["decision_date"])[:10])
            for episode_id in retrieved:
                episode = by_id.get(episode_id)
                if episode is None:
                    violations.append({"arm": arm_name, "decision_date": str(row["decision_date"]), "episode_id": episode_id, "reason": "UNKNOWN_EPISODE"})
                    continue
                if not (
                    date.fromisoformat(str(episode["return_date"])[:10]) <= decision
                    and date.fromisoformat(str(episode["decision_date"])[:10]) < decision
                ):
                    violations.append({"arm": arm_name, "decision_date": str(row["decision_date"]), "episode_id": episode_id, "reason": "FUTURE_OR_SAME_DAY_EPISODE"})
    return {
        "status": "PASS" if not violations and not no_memory_leaks else "FAIL",
        "violations": violations,
        "no_memory_leak_rows": no_memory_leaks,
        "episode_counts": {k: len(v) for k, v in episodes_by_arm.items()},
    }


def run_state_controlled_pair(period, sc, base_config, accounting, risk, output: Path, provider_identity: dict | None, demo: bool = False):
    """Run a diagnostic pair with a common current-state stream.

    The reference pre-trade exposure is the deterministic source stream.  Each
    arm receives the same date, q/features, core desired exposure, beta, and
    risk scenarios; only visible completed memory differs.  The arm-specific
    action is recorded and its outcome is appended to that arm's private memory,
    but it does not feed back into the next day's common state.  This is not a
    closed-loop performance claim; it identifies direct memory-conditioned
    action sensitivity.
    """
    specs = arm_specs("state_controlled")
    arms = {spec["name"]: Arm(spec["name"], FixedTrust(dict(sc, fixed_beta=FIXED_BETA, agent_trust_mode="FIXED_POSITIVE", arms=["memory", "no_memory"], common_legacy_trust_rule=False, continue_across_years=True))) for spec in specs}
    journals = {}
    for spec in specs:
        name = spec["name"]
        if demo:
            from stage63lib.legacy import SYSTEM_PROMPT  # noqa: F401
            from stage63lib.provider import OllamaProvider as _Unused  # noqa: F401
            journals[name] = None
        else:
            provider = OllamaProvider(sc["provider"], sc["agent"])
            if provider_identity is not None and provider.pinned != provider_identity:
                raise RuntimeError("Ollama model/serving identity changed between state-controlled arms")
            journals[name] = Journal(output / "state_controlled" / name, provider, {"experiment_id": EXPERIMENT_ID, "arm": spec, "provider_identity": provider.pinned})

    rows = []
    # The deterministic stream has no portfolio_state field in all source
    # variants, so reconstruct a reproducible common pre-trade path from the
    # source core's final exposure and next-day drift.
    common_pretrade = 0.0
    for i in range(len(period.frame)):
        base_desired = float(period.current_trace.iloc[i]["desired_exposure"])
        for spec in specs:
            name = spec["name"]
            arm = arms[name]
            beta = FIXED_BETA
            state = make_state(period, i, common_pretrade, base_desired, beta)
            completed = eligible_episodes(arm.ledger.episodes, state["decision_date"])
            evidence = EpisodicMemory(completed).evidence_summary(state, sc["agent"]["maximum_similar_episodes"] if spec["memory"] != "none" else 0)
            previews = {}
            for action in sc["agent"]["actions"]:
                projected = legacy.project_action(action, beta, base_desired, common_pretrade, period.scenarios[i], base_config, risk)
                previews[action] = dict(
                    blended_desired_exposure=blend_exposure(base_desired, action, beta),
                    risk_limited_exposure=float(projected.exposure),
                    turnover=float(projected.turnover),
                    ambiguity_cvar=float(projected.ambiguity_cvar),
                )
            payload = dict(
                task_type="state_controlled_memory_sensitivity",
                objective="test whether visible completed memory changes a bounded action under an identical current state",
                state=state,
                memory=evidence,
                safe_exposure_previews=previews,
                constraints=dict(allowed_actions=["BTC", "CASH", "ABSTAIN"], shorting=False, leverage=False, invalid_or_failed_response="ABSTAIN_TO_PRE_AGENT_RAMOE"),
            )
            decision = dict(action="ABSTAIN", confidence=0.0, reason_codes=["INSUFFICIENT_EVIDENCE"], cited_memory_ids=[])
            valid = True
            error = ""
            outer = {}
            latency = 0.0
            if demo:
                # Controlled demo deliberately alternates a deterministic action;
                # it is a mechanical test and carries no economic meaning.
                decision["action"] = "BTC" if i % 2 == 0 else "CASH"
            else:
                raw, outer, latency = journals[name].complete(f"{i+1:06d}-{name}", payload)
                try:
                    decision = validate_decision(raw, sc["agent"], {e["episode_id"] for e in evidence["similar_completed_episodes"]})
                except (ValueError, TypeError) as exc:
                    valid = False
                    error = f"{type(exc).__name__}: {exc}"
            action = decision["action"]
            projected = legacy.project_action(action, beta, base_desired, common_pretrade, period.scenarios[i], base_config, risk)
            asset = float(period.asset_returns[i])
            net = float(accounting.net_return(projected.exposure, common_pretrade, asset, sc["cost_rate"]))
            visible_ids = [e["episode_id"] for e in evidence["similar_completed_episodes"]]
            row = dict(
                index=i,
                arm=name,
                decision_date=state["decision_date"],
                return_date=state["target_return_date"],
                hard_regime=state["hard_regime"],
                asset_simple_return=asset,
                pretrade_exposure=common_pretrade,
                exposure=float(projected.exposure),
                net_return_after_trading_costs=net,
                net_log_return_after_trading_costs=math.log1p(net),
                turnover=abs(float(projected.exposure) - common_pretrade),
                cost_fraction=sc["cost_rate"] * abs(float(projected.exposure) - common_pretrade),
                action=action,
                beta=beta,
                core_desired_exposure=base_desired,
                desired_exposure=blend_exposure(base_desired, action, beta),
                valid=valid,
                error=error,
                confidence=float(decision["confidence"]),
                cited_memory_count=len(decision["cited_memory_ids"]),
                retrieved_memory_ids=visible_ids,
                retrieved_memory_count=len(visible_ids),
                request_sha256=digest(payload),
                latency_seconds=float(latency),
                advisor="llama",
                trust_mode="fixed",
                memory_mode=spec["memory"],
                risk_mode="standard",
                state_controlled=True,
            )
            episode = dict(
                episode_id=f"episode-{i+1:06d}",
                decision_date=state["decision_date"],
                return_date=state["target_return_date"],
                hard_regime=state["hard_regime"],
                state_vector=state_vector(state).tolist(),
                action=action,
                confidence=float(decision["confidence"]),
                shadow_log_advantage_vs_ramoe=0.0,
                asset_return=asset,
            )
            arm.ledger.add_completed(episode)
            rows.append(row)
        # The state stream is common by construction.  Drift the deterministic
        # source core only; arm actions cannot contaminate the next state.
        reference_exposure = float(period.current_trace.iloc[i].get("final_exposure", period.current_trace.iloc[i]["desired_exposure"]))
        common_pretrade = float(accounting.drifted_exposure(reference_exposure, float(period.asset_returns[i])))
    return rows, arms


def execute(args) -> int:
    sc = load_source_config()
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    contract = out / "00_CONTRACT.json"
    identity = dict(
        experiment_id=EXPERIMENT_ID,
        mode=args.mode,
        demo=bool(args.demo),
        project_root=str(args.project_root.expanduser().resolve()),
        package_sha256=sha(PACKAGE / "PACKAGE_MANIFEST.sha256") if (PACKAGE / "PACKAGE_MANIFEST.sha256").exists() else None,
        stage64_manifest_sha256=sha(STAGE64 / "PACKAGE_MANIFEST.sha256"),
        model=MODEL,
        runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
    )
    if contract.exists():
        old = json.loads(contract.read_text())
        if old.get("identity") != identity:
            raise RuntimeError("Output contract identity changed; use a new output directory")
    else:
        _write_json(contract, {"identity": identity, "created_utc": datetime.now(timezone.utc).isoformat(), "no_yearly_reset": True, "pretrained_weights_updated": False})

    if args.demo:
        period, base_config, accounting, risk = build_fixture()
        audit = {"evidence_kind": "CONTROLLED_NON_ECONOMIC", "raw_market": {"sha256": "CONTROLLED_FIXTURE"}}
    else:
        period, base_config, accounting, risk, audit = load_inputs(args.project_root, sc, args.full_dataset)
    _write_json(out / "01_SOURCE_AUDIT.json", audit)
    cores = {"original": np.asarray(period.current_trace["desired_exposure"], dtype=float)}
    if not args.demo:
        cores, core_audit, core_trace = build_core_variants(period, base_config, Path(audit["corrected_source_run"]))
        _write_json(out / "02_CORE_AUDIT.json", core_audit)
        atomic_csv(core_trace, out / "CORE_DAILY_STATE.csv")

    provider_identity = None
    if not args.demo:
        probe = OllamaProvider(sc["provider"], sc["agent"])
        provider_identity = probe.pinned
        expected_digest = sc.get("expected_model_digest")
        if expected_digest and provider_identity.get("model_digest") != expected_digest:
            raise RuntimeError("Ollama model digest differs from the pinned Stage 6 source identity")
        _write_json(out / "03_PROVIDER_IDENTITY.json", provider_identity)

    all_rows: list[dict] = []
    episodes: dict[str, list[dict]] = {}
    arm_plan = []
    if args.mode in {"closed_loop", "full"}:
        arm_plan.extend(arm_specs("closed_loop"))
    for spec in arm_plan:
        armout = out / "closed_loop" / spec["name"]
        armout.mkdir(parents=True, exist_ok=True)
        if (armout / "ARM_STATE.json").exists() and (armout / "episodes.json").exists() and (armout / "DAILY_LEDGER.csv").exists():
            rows = pd.read_csv(armout / "DAILY_LEDGER.csv").to_dict("records")
            episodes[spec["name"]] = json.loads((armout / "episodes.json").read_text())
        else:
            provider = ControlledProvider() if args.demo else OllamaProvider(sc["provider"], sc["agent"])
            pid = {"kind": "controlled"} if args.demo else provider.pinned
            journal = Journal(armout, provider, {"experiment_id": EXPERIMENT_ID, "arm": spec, "source_audit_sha256": digest(audit), "provider_identity": pid})
            rows, arm = run_arm(period, sc, base_config, accounting, risk, journal, armout, spec, core_desired=cores["original"])
            atomic_csv(pd.DataFrame(rows), armout / "DAILY_LEDGER.csv")
            episodes[spec["name"]] = arm.ledger.episodes
        all_rows.extend(rows)

    if args.mode in {"state_controlled", "full"}:
        state_rows, state_arms = run_state_controlled_pair(period, sc, base_config, accounting, risk, out, provider_identity, demo=args.demo)
        for name, arm in state_arms.items():
            episodes[name] = arm.ledger.episodes
        all_rows.extend(state_rows)
        for name in sorted(state_arms):
            part = pd.DataFrame([r for r in state_rows if r["arm"] == name])
            armout = out / "state_controlled" / name
            armout.mkdir(parents=True, exist_ok=True)
            atomic_csv(part, armout / "DAILY_LEDGER.csv")
            _write_json(armout / "episodes.json", state_arms[name].ledger.episodes)

    if not all_rows:
        raise RuntimeError("No experiment arms were selected")
    ledger = pd.DataFrame(all_rows)
    atomic_csv(ledger, out / "daily_ledger.csv")
    metrics = summarize_ledger(ledger, transaction_cost_rate=sc["cost_rate"])
    atomic_csv(metrics, out / "all_metrics.csv")
    for scope, filename in [("FULL", "full_period_metrics.csv"), ("YEAR", "yearly_metrics.csv"), ("MONTH", "monthly_metrics.csv"), ("REGIME", "regime_metrics.csv"), ("YEAR_REGIME", "year_regime_metrics.csv")]:
        atomic_csv(metrics[metrics.scope == scope], out / filename)

    comparisons = []
    transmissions = []
    for prefix, label in [("closed_loop", "CLOSED_LOOP_MEMORY_EFFECT"), ("state_controlled", "STATE_CONTROLLED_MEMORY_EFFECT")]:
        left, right = f"{prefix}_memory", f"{prefix}_no_memory"
        if left in set(ledger.arm) and right in set(ledger.arm):
            result = paired_block_comparison(ledger, left, right, start_date="2022-01-01", block_length=30, n_resamples=5000, seed=16062)
            result.update({"comparison": label, "left": left, "right": right, "interpretation": "INCONCLUSIVE_UNLESS_INTERVAL_EXCLUDES_ZERO_AND_CLOCK_AUDIT_PASSES"})
            comparisons.append(result)
            transmissions.append(transmission(ledger, left, right, label))
    if comparisons:
        _write_json(out / "COMPARISONS.json", {"primary": "CLOSED_LOOP_MEMORY_MINUS_NO_MEMORY", "comparisons": comparisons, "bootstrap": "paired_circular_30_day_blocks_5000_resamples_seed_16062", "intervals": "pointwise_95_percent_not_multiplicity_adjusted"})
        atomic_csv(pd.concat(transmissions, ignore_index=True), out / "TRANSMISSION.csv")

    memory_audit = audit_memory_rows(ledger, episodes)
    _write_json(out / "MEMORY_ELIGIBILITY_AUDIT.json", memory_audit)
    final = {
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "mode": args.mode,
        "evidence_kind": "CONTROLLED_NON_ECONOMIC" if args.demo else "REUSED_LEGACY_CLOCK_WITH_FRESH_LLM_ABLATION",
        "model": MODEL,
        "qwen_used": False,
        "arms": sorted(ledger.arm.unique().tolist()),
        "rows": int(len(ledger)),
        "memory_eligibility_passed": memory_audit["status"] == "PASS",
        "continuous_learning_established": False,
        "economic_superiority_claim": False,
        "clock_repair_required_before_confirmatory_claim": True,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(out / "FINAL_STATUS.json", final)
    (out / "READ_RESULTS_FIRST.md").write_text(
        "# RAMAS Stage 7 results\n\n"
        "Headline accounting is net return after trading costs; daily net log return is its additive representation. "
        "The closed-loop pair is the primary memory-utility test. The state-controlled pair is a diagnostic of direct "
        "memory-conditioned action sensitivity, not a closed-loop performance estimate. A positive point estimate alone "
        "does not establish learning. Read COMPARISONS.json, TRANSMISSION.csv, MEMORY_ELIGIBILITY_AUDIT.json and the "
        "source/clock audit before interpreting results.\n"
    )
    _write_json(out / "RUN_COMPLETE.json", {"status": "COMPLETE", "identity": identity, "artifact_sha256": {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"RUN_COMPLETE.json", "SERVER.lock", "RUN.lock"}}})
    print("RAMAS_STAGE7_STATUS=COMPLETE")
    print("SCIENTIFIC_CLAIM_AUTOMATICALLY_PASSED=FALSE")
    print("OUTPUT=" + str(out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/home/infonet/wahid/leader_router_fresh"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-dataset", type=Path)
    parser.add_argument("--mode", choices=["closed_loop", "state_controlled", "full"], default="full")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    try:
        return execute(args)
    except Exception as exc:
        print(f"RAMAS_STAGE7_STATUS=STOP\nERROR={type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
