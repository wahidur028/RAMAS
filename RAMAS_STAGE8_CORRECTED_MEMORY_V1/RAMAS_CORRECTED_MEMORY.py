#!/usr/bin/env python3
"""RAMAS Stage 8 - corrected episodic-memory experiment.

Four subcommands:

  preflight  read-only source/model identity audit                (no model calls)
  validate   prove the engine reproduces a frozen Stage 6.4 arm    (no model calls)
  diagnose   measure retrieval collapse and credit-label validity  (no model calls)
  run        execute the corrected memory / no-memory pair         (model calls)

This package imports the verified Stage 6.4 code read-only and never writes
inside it or inside any existing artifact directory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent


class Stage8Error(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_config(path: Path) -> tuple[dict, str]:
    config = load_json(path)
    required = {"experiment_id", "project_root", "stage64_code_root",
                "expected_stage64_manifest_sha256", "full_dataset", "arms", "analysis"}
    missing = sorted(required - set(config))
    if missing:
        raise Stage8Error(f"Configuration is missing: {missing}")
    names = [a["name"] for a in config["arms"]]
    if len(names) != len(set(names)):
        raise Stage8Error("Arm names must be unique")
    return config, sha256(path)


PINNED_RUNTIME = {"python": "3.11.9", "numpy": "1.26.4", "pandas": "2.2.3"}


def assert_pinned_runtime(allow_mismatch: bool = False) -> dict:
    """Refuse to run on an interpreter that cannot reproduce the frozen arms.

    Bit-exact reproduction of the archived payload digests depends on the BLAS
    build behind numpy: a different interpreter changes the last bit of the
    retrieval distance, which changes the request hash and breaks every journal
    match. The frozen Stage 6.x runs pin 3.11.9 / 1.26.4 / 2.2.3.
    """
    import numpy
    import pandas
    actual = {"python": platform.python_version(),
              "numpy": numpy.__version__, "pandas": pandas.__version__}
    differing = {k: {"expected": v, "actual": actual[k]}
                 for k, v in PINNED_RUNTIME.items() if actual[k] != v}
    if differing and not allow_mismatch:
        raise Stage8Error(
            "Runtime does not match the pinned Stage 6.x environment: "
            f"{differing}. Use /home/infonet/anaconda3/envs/wahid_test/bin/python, "
            "or pass --allow-runtime-mismatch for a non-reproducing exploratory run."
        )
    return actual


def import_stage64(root: Path) -> dict:
    """Import the frozen Stage 6.4 package after verifying its manifest."""
    root = Path(root).expanduser().resolve()
    manifest = root / "PACKAGE_MANIFEST.sha256"
    if not manifest.is_file():
        raise Stage8Error(f"Stage 6.4 manifest missing: {manifest}")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = (root / relative.strip().lstrip("*")).resolve()
        if root not in target.parents:
            raise Stage8Error(f"Manifest path escapes package: {relative}")
        if not target.is_file() or sha256(target) != expected:
            raise Stage8Error(f"Stage 6.4 package file differs: {target}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from stage63lib.inputs import load_inputs
    from stage63lib.journal import Journal
    from suite64.core_variants import build_core_variants
    from corrected_memory import engine as corrected_engine
    from corrected_memory import replay as replay_module
    return {
        "load_inputs": load_inputs, "Journal": Journal,
        "build_core_variants": build_core_variants,
        "engine": corrected_engine, "replay": replay_module,
        "source_config": load_json(root / "source_config.json"),
        "manifest_sha256": sha256(manifest), "root": str(root),
    }


def build_context(config: dict, modules: dict) -> dict:
    source_config = dict(modules["source_config"])
    source_config["checkpoint_every"] = int(config.get("checkpoint_every", 25))
    period, base_config, accounting, risk, audit = modules["load_inputs"](
        Path(config["project_root"]), source_config, Path(config["full_dataset"]))
    cores, core_audit, core_trace = modules["build_core_variants"](
        period, base_config, Path(audit["corrected_source_run"]))
    return dict(period=period, base_config=base_config, accounting=accounting, risk=risk,
                input_audit=audit, cores=cores, core_audit=core_audit, core_trace=core_trace,
                source_config=source_config)


# ----------------------------------------------------------------- preflight
def command_preflight(args) -> None:
    config, config_hash = load_config(args.config)
    runtime = assert_pinned_runtime(getattr(args, 'allow_runtime_mismatch', False))
    modules = import_stage64(config["stage64_code_root"])
    if modules["manifest_sha256"] != config["expected_stage64_manifest_sha256"]:
        raise Stage8Error(
            f"Stage 6.4 manifest identity differs: expected="
            f"{config['expected_stage64_manifest_sha256']} actual={modules['manifest_sha256']}")
    context = build_context(config, modules)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PASS_NO_MODEL_INFERENCE", "created_at": now(), "runtime": runtime,
        "experiment_id": config["experiment_id"], "config_sha256": config_hash,
        "stage64_manifest_sha256": modules["manifest_sha256"],
        "stage64_root": modules["root"],
        "source_input_audit": context["input_audit"],
        "core_audit": context["core_audit"],
        "full_rows": len(context["period"].frame),
        "arms": config["arms"],
        "corrections_declared": {
            "credit_label": "counterfactual advantage of the taken action over the unchanged "
                            "core, projected from the arm's own pre-trade holdings",
            "retrieval": "k nearest completed episodes per action instead of k nearest overall",
            "everything_else": "UNCHANGED_FROM_STAGE64",
        },
        "claim_boundaries": {
            "fresh_out_of_sample": False,
            "router_temporal_validity_resolved": False,
            "feasible_information_to_fill_clock_resolved": False,
            "corrected_label_validated_as_improving_decisions": False,
        },
    }
    atomic_json(output / "PREFLIGHT.json", report)
    context["core_trace"].to_csv(output / "CORE_DAILY_STATE.csv", index=False, float_format="%.17g")
    print(json.dumps({"status": report["status"], "full_rows": report["full_rows"],
                      "preflight": str(output / "PREFLIGHT.json")}, indent=2))


# ------------------------------------------------------------------ validate
def command_validate(args) -> None:
    """Replay an archived Stage 6.4 arm in legacy mode and compare, no model calls."""
    import pandas as pd
    import numpy as np

    config, _ = load_config(args.config)
    runtime = assert_pinned_runtime(getattr(args, 'allow_runtime_mismatch', False))
    modules = import_stage64(config["stage64_code_root"])
    context = build_context(config, modules)
    archive = args.archived_arm.resolve()
    archived_state = load_json(archive / "ARM_STATE.json")
    spec = dict(archived_state["spec"])
    spec.update(label="legacy", retrieval="legacy")
    journal = modules["replay"].ReplayJournal(archive)
    output = args.output.resolve() / ("legacy_equivalence__" + spec["name"])
    output.mkdir(parents=True, exist_ok=True)
    print(f"Replaying archived arm {spec['name']} in legacy mode (no model calls)...", flush=True)
    rows, arm = modules["engine"].run_arm(
        context["period"], context["source_config"], context["base_config"],
        context["accounting"], context["risk"], journal, output, spec,
        core_desired=context["cores"][spec.get("core", "original")])
    rebuilt = pd.DataFrame(rows)
    original = pd.read_csv(archive / "DAILY_LEDGER.csv", low_memory=False)
    shared = [c for c in original.columns if c in rebuilt.columns]
    mismatches = {}
    for column in shared:
        left, right = original[column], rebuilt[column]
        try:
            l = pd.to_numeric(left).to_numpy(float)
            r = pd.to_numeric(right).to_numpy(float)
            if l.shape != r.shape:
                mismatches[column] = {"kind": "shape", "archived": l.shape, "rebuilt": r.shape}
                continue
            both_missing = np.isnan(l) & np.isnan(r)
            if not np.array_equal(np.isnan(l), np.isnan(r)):
                mismatches[column] = {"kind": "missingness"}
                continue
            comparable = ~both_missing
            worst = float(np.max(np.abs(l[comparable] - r[comparable]))) if comparable.any() else 0.0
            if worst > 1e-12:
                mismatches[column] = {"kind": "numeric", "max_abs_difference": worst}
        except (ValueError, TypeError):
            differing = int((left.astype(str) != right.astype(str)).sum())
            if differing:
                mismatches[column] = {"kind": "text", "differing_rows": differing}
    verdict = {
        "status": "PASS_ENGINE_REPRODUCES_ARCHIVED_ARM" if not mismatches else "FAIL_ENGINE_DIVERGES",
        "created_at": now(), "archived_arm": str(archive), "spec": spec,
        "rows": len(rebuilt), "columns_compared": len(shared),
        "archived_calls_matched_by_request_digest": journal.matched,
        "model_calls_made": 0,
        "terminal_wealth_archived": archived_state["wealth"],
        "terminal_wealth_rebuilt": float(arm.wealth),
        "terminal_wealth_absolute_difference": abs(float(arm.wealth) - float(archived_state["wealth"])),
        "mismatched_columns": mismatches,
    }
    atomic_json(output / "LEGACY_EQUIVALENCE.json", verdict)
    print(json.dumps({k: verdict[k] for k in (
        "status", "rows", "columns_compared", "archived_calls_matched_by_request_digest",
        "terminal_wealth_absolute_difference", "mismatched_columns")}, indent=2))
    if mismatches:
        raise Stage8Error("Engine does not reproduce the archived arm; do not spend GPU time")


# ------------------------------------------------------------------ diagnose
def command_diagnose(args) -> None:
    """Reproduce the two failure-mode measurements from archived artifacts."""
    import pandas as pd
    import numpy as np
    import re

    archive = args.archived_arm.resolve()
    ledger = pd.read_csv(archive / "DAILY_LEDGER.csv", low_memory=False)
    diversity, signal_rows = {0: 0, 1: 0, 2: 0, 3: 0}, []
    future = dict(zip(ledger["index"].astype(int), ledger["asset_simple_return"]))
    for path in sorted((archive / "calls").glob("*.json")):
        index = int(re.match(r"(\d+)-", path.name).group(1)) - 1
        memory = json.loads(path.read_text(encoding="utf-8"))["request"]["memory"]
        summary = memory["summary"]
        count = len(summary) if isinstance(summary, dict) else 0
        diversity[count] = diversity.get(count, 0) + 1
        if isinstance(summary, dict):
            vote = sum({"BTC": 1.0, "CASH": -1.0, "ABSTAIN": 0.0}[a]
                       * s["mean_log_advantage_vs_ramoe"] * s["count"] for a, s in summary.items())
            signal_rows.append((vote, future.get(index)))
    total = sum(diversity.values())
    frame = pd.DataFrame(signal_rows, columns=["signal", "outcome"]).dropna()
    frame = frame[frame["signal"] != 0]
    corr = float(np.corrcoef(frame["signal"], frame["outcome"])[0, 1]) if len(frame) > 2 else None
    hit = float(((frame["signal"] > 0) == (frame["outcome"] > 0)).mean()) if len(frame) else None

    r = ledger["asset_simple_return"].to_numpy(float)
    legacy_label = ledger["shadow_log_advantage_vs_ramoe"].to_numpy(float)
    report = {
        "status": "COMPLETE_NO_MODEL_CALLS", "created_at": now(), "archived_arm": str(archive),
        "retrieval_collapse": {
            "days_total": total,
            "days_by_distinct_retrieved_actions": {str(k): v for k, v in sorted(diversity.items())},
            "fraction_with_action_contrast": (
                sum(v for k, v in diversity.items() if k >= 2) / total if total else None),
            "interpretation": "A contrast between candidate actions is only expressible on days "
                              "where at least two distinct actions are retrieved.",
        },
        "credit_label_validity": {
            "legacy_label_corr_with_same_day_return": float(np.corrcoef(legacy_label, r)[0, 1]),
            "legacy_label_positive_fraction": float((legacy_label > 0).mean()),
            "note": "A label that measures decision quality should correlate positively with the "
                    "return when the taken action increased exposure. A negative correlation "
                    "indicates the label is dominated by the holdings gap against the reference.",
        },
        "memory_signal_predictiveness": {
            "days_with_nonzero_signal": len(frame),
            "corr_with_informed_return": corr,
            "directional_agreement_rate": hit,
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "MEMORY_DIAGNOSTICS.json", report)
    print(json.dumps(report, indent=2, default=str))


# ----------------------------------------------------------------------- run
def command_run(args) -> None:
    import pandas as pd

    config, config_hash = load_config(args.config)
    runtime = assert_pinned_runtime(getattr(args, 'allow_runtime_mismatch', False))
    modules = import_stage64(config["stage64_code_root"])
    if modules["manifest_sha256"] != config["expected_stage64_manifest_sha256"]:
        raise Stage8Error("Stage 6.4 manifest identity differs; refusing to run")
    preflight = load_json(args.preflight.resolve())
    if preflight.get("status") != "PASS_NO_MODEL_INFERENCE" or preflight.get("config_sha256") != config_hash:
        raise Stage8Error("Preflight is missing, failed, or belongs to another configuration")

    output = args.output.resolve()
    for protected in (PACKAGE, Path(config["stage64_code_root"]).resolve()):
        if output == protected or protected in output.parents:
            raise Stage8Error("Output cannot be inside a code package")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "RUN.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise Stage8Error(f"Another process is using {output}") from exc

    try:
        from stage63lib.provider import OllamaProvider
        context = build_context(config, modules)
        source_config = context["source_config"]
        provider_config = dict(source_config["provider"])
        if args.seed is not None:
            provider_config["seed"] = int(args.seed)
        provider = OllamaProvider(provider_config, source_config["agent"])
        identity = {
            "experiment_id": config["experiment_id"], "config_sha256": config_hash,
            "stage64_manifest_sha256": modules["manifest_sha256"],
            "provider_identity": provider.pinned,
            "runtime": {"python": platform.python_version(),
                        "numpy": __import__("numpy").__version__, "pandas": pd.__version__},
        }
        contract = output / "00_RUN_CONTRACT.json"
        if contract.is_file():
            if not args.resume:
                raise Stage8Error(f"Run exists; use --resume: {output}")
            if load_json(contract).get("identity") != identity:
                raise Stage8Error("Resume identity changed")
        else:
            if args.resume:
                raise Stage8Error("--resume requested but no run contract exists")
            atomic_json(contract, {"identity": identity, "started_at": now(),
                                   "arms": config["arms"],
                                   "scientific_scope": "RETROSPECTIVE_REUSED_HISTORY_MECHANISM_DIAGNOSTIC",
                                   "claim_boundaries": preflight["claim_boundaries"]})
        atomic_json(output / "01_SOURCE_AUDIT.json", context["input_audit"])
        atomic_json(output / "02_CORE_AUDIT.json", context["core_audit"])

        frames = []
        for spec in config["arms"]:
            arm_out = output / "arms" / spec["name"]
            done = arm_out / "RUN_COMPLETE.json"
            if done.is_file():
                record = load_json(done)
                if record.get("spec") != spec or record.get("identity") != identity:
                    raise Stage8Error(f"Completed arm identity or spec changed: {arm_out}")
                frames.append(pd.read_csv(arm_out / "DAILY_LEDGER.csv", low_memory=False))
                print(f"ARM={spec['name']} STATUS=REUSED_COMPLETE", flush=True)
                continue
            arm_out.mkdir(parents=True, exist_ok=True)
            journal = modules["Journal"](arm_out, provider, dict(identity, spec=spec))
            print(f"ARM={spec['name']} STATUS=START", flush=True)
            rows, arm = modules["engine"].run_arm(
                context["period"], source_config, context["base_config"],
                context["accounting"], context["risk"], journal, arm_out, spec,
                core_desired=context["cores"][spec.get("core", "original")])
            frame = pd.DataFrame(rows)
            if not bool(frame["valid"].all()):
                raise Stage8Error(f"Invalid model output entered the ledger: {arm_out}")
            frame.to_csv(arm_out / "DAILY_LEDGER.csv", index=False, float_format="%.17g")
            atomic_json(done, {"status": "COMPLETE", "created_at": now(), "identity": identity,
                               "spec": spec, "rows": len(frame), "final_wealth": float(arm.wealth),
                               "new_calls": journal.new_calls, "reused_calls": journal.reused_calls})
            provider.assert_identity()
            frames.append(frame)
            print(f"ARM={spec['name']} STATUS=COMPLETE WEALTH={arm.wealth:.6f}", flush=True)

        ledger = pd.concat(frames, ignore_index=True)
        ledger.to_csv(output / "ALL_DAILY_LEDGER.csv", index=False, float_format="%.17g")
        contrasts = []
        if len(config["arms"]) >= 2:
            from suite64.metrics import paired_block_comparison, prepare_daily_ledger
            prepared = prepare_daily_ledger(ledger, float(source_config["cost_rate"]))
            analysis = config["analysis"]
            for a, b in analysis.get("contrasts", []):
                contrasts.append(paired_block_comparison(
                    prepared, a, b, block_length=int(analysis["block_days"]),
                    n_resamples=int(analysis["bootstrap_resamples"]), seed=int(analysis["seed"])))
        atomic_json(output / "COMPARISONS.json", {
            "primary": config["analysis"].get("primary"),
            "intervals": "POINTWISE_95_PERCENT_NOT_MULTIPLICITY_ADJUSTED",
            "contrasts": contrasts})
        atomic_json(output / "RUN_COMPLETE.json", {
            "status": "COMPLETE", "completed_at": now(), "identity": identity,
            "arms": [a["name"] for a in config["arms"]], "rows": len(ledger),
            "corrected_label_validated_as_improving_decisions": False})
        print(json.dumps({"status": "COMPLETE", "output": str(output),
                          "rows": len(ledger)}, indent=2))
    except Exception as exc:
        atomic_json(output / "FAILURE.json", {"status": "STOP", "created_at": now(),
                                              "error_type": type(exc).__name__, "error": str(exc),
                                              "traceback": traceback.format_exc()})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    default_config = PACKAGE / "config" / "experiment.json"

    p = sub.add_parser("preflight", help="read-only source/model identity audit")
    p.add_argument("--config", type=Path, default=default_config)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-runtime-mismatch", action="store_true")
    p.set_defaults(function=command_preflight)

    p = sub.add_parser("validate", help="prove engine reproduces an archived arm (no model calls)")
    p.add_argument("--config", type=Path, default=default_config)
    p.add_argument("--archived-arm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-runtime-mismatch", action="store_true")
    p.set_defaults(function=command_validate)

    p = sub.add_parser("diagnose", help="measure retrieval collapse and label validity")
    p.add_argument("--archived-arm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(function=command_diagnose)

    p = sub.add_parser("run", help="execute the corrected memory / no-memory pair")
    p.add_argument("--config", type=Path, default=default_config)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-runtime-mismatch", action="store_true")
    p.set_defaults(function=command_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.function(args)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; durable calls remain resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
