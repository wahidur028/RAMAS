"""Reconstruct the admitted policy matrix, then intervene on allocation only.

The historical numerical reference was simulated in two pieces. Its trust W
was carried between pieces, while policy holdings were reset. We reproduce that
reference exactly before constructing a narrower aggregation-only ablation.
No edited q is supplied to trust learning, Llama context, or memory retrieval.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CoreSourceError(RuntimeError):
    """The frozen source does not support the requested reconstruction."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_modules(corrected_root: Path):
    from stage63lib.legacy import source

    base_dir = source.resolve_base_dir(corrected_root)
    # Use a private namespace: load_inputs already imports the original src.
    package_name = "_ramas64_core_" + hashlib.sha256(str(base_dir).encode()).hexdigest()[:16]
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, base_dir / "src" / "__init__.py",
            submodule_search_locations=[str(base_dir / "src")],
        )
        if spec is None or spec.loader is None:
            raise CoreSourceError("Cannot import the verified numerical source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)
    return (
        importlib.import_module(package_name + ".accounting"),
        importlib.import_module(package_name + ".trust"),
        base_dir,
        source,
    )


def _close(left, right, label: str, tolerance=1e-12) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise CoreSourceError(f"{label}: incompatible shapes or non-finite values")
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    if error > tolerance:
        raise CoreSourceError(f"{label}: error={error}, tolerance={tolerance}")
    return error


def _arrays(q, weights, proposals, initial_weights, admission):
    """Return desired exposures only; each trajectory projects its own holdings."""
    q = np.asarray(q, dtype=float)
    weights = np.asarray(weights, dtype=float)
    proposals = np.asarray(proposals, dtype=float)
    initial_weights = np.asarray(initial_weights, dtype=float)
    admission = np.asarray(admission, dtype=bool)
    if q.ndim != 2 or proposals.ndim != 2:
        raise CoreSourceError("q and policy proposals must be matrices")
    days, regimes = q.shape
    experts = proposals.shape[1]
    if (weights.shape != (days, regimes, experts)
            or proposals.shape[0] != days
            or initial_weights.shape != (regimes, experts)
            or admission.shape != (experts,)):
        raise CoreSourceError("Policy aggregation dimensions do not align")
    if not all(np.isfinite(x).all() for x in (q, weights, proposals, initial_weights)):
        raise CoreSourceError("Non-finite policy inputs")
    if ((q < 0).any() or (weights < 0).any() or (initial_weights < 0).any()
            or (proposals < 0).any() or (proposals > 1).any()):
        raise CoreSourceError("Invalid probability, weight, or exposure bound")
    _close(q.sum(axis=1), np.ones(days), "q sums")
    _close(weights.sum(axis=2), np.ones((days, regimes)), "W sums")
    _close(initial_weights.sum(axis=1), np.ones(regimes), "initial W sums")
    if np.any(weights[:, :, ~admission] != 0) or np.any(initial_weights[:, ~admission] != 0):
        raise CoreSourceError("Rejected policy received nonzero weight")
    mixed = np.einsum("dr,dre->de", q, weights)
    hard_weights = weights[np.arange(days), np.argmax(q, axis=1)]
    uniform_weights = weights.mean(axis=1)
    frozen_weights = q @ initial_weights
    variants = {
        "original": np.einsum("de,de->d", mixed, proposals),
        "hard_allocator": np.einsum("de,de->d", hard_weights, proposals),
        "uniform_allocator": np.einsum("de,de->d", uniform_weights, proposals),
        "frozen_W": np.einsum("de,de->d", frozen_weights, proposals),
    }
    if any(((v < -1e-12) | (v > 1 + 1e-12)).any() for v in variants.values()):
        raise CoreSourceError("Desired exposure outside [0,1]")
    return variants, mixed


def _period_weights(frame, initial_weights, config, admission, accounting, trust_module):
    """Mirror simulator.simulate's trust clock without replaying portfolio risk."""
    names = config["expert_names"]
    q = frame[["prob_" + x for x in config["regime_names"]]].to_numpy(float)
    proposals = frame[["exposure_" + x for x in names]].to_numpy(float)
    asset_returns = frame["canonical_asset_simple_return"].to_numpy(float)
    months = pd.to_datetime(frame["return_date"], errors="raise").dt.to_period("M")
    W = np.asarray(initial_weights, dtype=float).copy()
    policy_pretrade = np.zeros(len(names), dtype=float)
    past_q, past_returns, past_turnover = [], [], []
    daily_weights, monthly_updates = [], []
    always = tuple(names.index(x) for x in config["always_admitted_experts"])

    def update(completed_month):
        nonlocal W
        if len(past_q) < 2:
            return
        result = trust_module.risk_aware_trust_update(
            W, np.asarray(past_q), np.asarray(past_returns), np.asarray(past_turnover), admission,
            always_admitted_indices=always,
            eta=float(config["trust_eta"]),
            cvar_alpha=float(config["cvar_alpha"]),
            tail_penalty=float(config["trust_tail_penalty"]),
            turnover_penalty=float(config["trust_turnover_penalty"]),
            minimum_posterior_mass=float(config["trust_minimum_posterior_mass"]),
            minimum_effective_sample_size=float(config["trust_minimum_effective_sample_size"]),
        )
        W = result.trust_matrix.copy()
        monthly_updates.append({
            "completed_return_month": str(completed_month),
            "supported_regimes": result.support_mask.tolist(),
            "posterior_mass": result.posterior_mass.tolist(),
            "effective_sample_size": result.effective_sample_size.tolist(),
        })

    for i in range(len(frame)):
        if i and months.iloc[i] != months.iloc[i - 1]:
            update(months.iloc[i - 1])
            past_q, past_returns, past_turnover = [], [], []
        daily_weights.append(W.copy())
        turnover = np.abs(proposals[i] - policy_pretrade)
        returns = np.asarray([
            accounting.net_return(x, p, asset_returns[i], float(config["transaction_cost_bps"]) / 10000)
            for x, p in zip(proposals[i], policy_pretrade)
        ])
        past_q.append(q[i].copy())
        past_returns.append(returns)
        past_turnover.append(turnover)
        policy_pretrade = np.asarray([
            accounting.drifted_exposure(x, asset_returns[i]) for x in proposals[i]
        ])
    if len(frame):
        update(months.iloc[-1])
    return np.asarray(daily_weights), W, monthly_updates


def build_core_variants(period, base_config: dict[str, Any], corrected_root: Path):
    """Return (desired-exposure arrays, audit, per-day q/W/proposal table).

    `corrected_root` is audit['corrected_source_run'] returned by load_inputs.
    Validates original desired exposures and both terminal W files before any
    intervention is released. W itself carries across the 2024 source seam;
    the independent source-policy holdings reset there in the historical code.
    """
    corrected_root = Path(corrected_root).expanduser().resolve()
    try:
        accounting, trust_module, base_dir, source = _load_modules(corrected_root)
        disk_config = json.loads((base_dir / "config.json").read_text())
        for key in (
            "regime_names", "expert_names", "always_admitted_experts", "production_default_admission",
            "transaction_cost_bps", "trust_eta", "cvar_alpha", "trust_tail_penalty",
            "trust_turnover_penalty", "trust_minimum_posterior_mass", "trust_minimum_effective_sample_size",
        ):
            if base_config[key] != disk_config[key]:
                raise CoreSourceError(f"Core reconstruction config differs from frozen source: {key}")
        names = base_config["expert_names"]
        regimes = base_config["regime_names"]
        if names != ["cash", "buy_and_hold", "trend", "volatility_target", "drawdown_control", "specialist_atp"]:
            raise CoreSourceError("Unreviewed policy registry; do not assume the two-policy scope")
        admission = np.asarray([base_config["production_default_admission"][x] for x in names], dtype=bool)
        if not np.array_equal(admission, [True, True, False, False, False, False]):
            raise CoreSourceError("This component contrast requires the frozen two-policy admission mask")
        initial = np.tile(admission.astype(float) / admission.sum(), (len(regimes), 1))
        W = initial.copy()
        frames, weight_arrays, period_audits = [], [], []
        source_periods = period.source_audit.get("periods", [])
        if not source_periods:
            raise CoreSourceError("Period source audit has no historical split identities")
        for spec in source_periods:
            directory = (corrected_root / "results" / spec["name"]).resolve()
            if not directory.is_relative_to(corrected_root):
                raise CoreSourceError("Source-period path escapes corrected root")
            input_file, trace_file = source.verify_child_artifact(directory)
            if (_sha(input_file) != spec["corrected_inputs_sha256"]
                    or _sha(trace_file) != spec["base_trace_sha256"]):
                raise CoreSourceError("Source-period hash differs from matched input audit")
            frame = pd.read_csv(input_file, low_memory=False)
            if len(frame) != int(spec["rows"]):
                raise CoreSourceError("Source-period row count changed")
            daily_W, W, updates = _period_weights(frame, W, base_config, admission, accounting, trust_module)
            complete = source.load_json(directory / "RUN_COMPLETE.json")
            terminal_file = directory / "08_TERMINAL_TRUST.json"
            if not terminal_file.is_file() or complete.get("artifact_sha256", {}).get(terminal_file.name) != _sha(terminal_file):
                raise CoreSourceError("Terminal policy matrix is missing or its hash differs")
            terminal_error = _close(W, source.load_json(terminal_file)["matrix"], "terminal policy W")
            frames.append(frame)
            weight_arrays.append(daily_W)
            period_audits.append({
                "name": spec["name"], "rows": len(frame),
                "initial_policy_holdings": [0.0] * len(names),
                "trust_carried_from_previous_period": len(frames) > 1,
                "terminal_W_maximum_error": terminal_error,
                "trust_updates": updates,
                "terminal_W_sha256": _sha(terminal_file),
            })
        frame = pd.concat(frames, ignore_index=True)
        if len(frame) != len(period.frame):
            raise CoreSourceError("Source reconstruction does not span the matched period")
        for key, dates in (("decision_date", period.decision_dates), ("return_date", period.return_dates)):
            if not np.array_equal(pd.to_datetime(frame[key]).to_numpy(), pd.DatetimeIndex(dates).to_numpy()):
                raise CoreSourceError(f"Core reconstruction {key} differs from matched period")
        q = frame[["prob_" + x for x in regimes]].to_numpy(float)
        _close(q, period.router_probabilities, "router stream")
        _close(frame["canonical_asset_simple_return"], period.asset_returns, "return stream")
        proposals = frame[["exposure_" + x for x in names]].to_numpy(float)
        weights = np.concatenate(weight_arrays)
        variants, mixed = _arrays(q, weights, proposals, initial, admission)
        desired_error = _close(variants["original"], period.current_trace["desired_exposure"], "original desired exposure")
        component_trace = frame[["decision_date", "return_date"]].copy()
        for k, regime in enumerate(regimes):
            component_trace["prob_" + regime] = q[:, k]
            for j, name in enumerate(names):
                component_trace[f"W_{regime}_{name}"] = weights[:, k, j]
        for j, name in enumerate(names):
            component_trace["proposal_" + name] = proposals[:, j]
            component_trace["admitted_" + name] = admission[j]
            component_trace["mixed_weight_" + name] = mixed[:, j]
            component_trace["weighted_exposure_" + name] = mixed[:, j] * proposals[:, j]
        for name, values in variants.items():
            component_trace["desired_" + name] = values
        audit = {
            "status": "EXACT_FROZEN_CORE_RECONSTRUCTED",
            "maximum_original_desired_error": desired_error,
            "policy_registry": names, "regime_order": regimes,
            "initial_admitted_W": initial.tolist(),
            "production_admission_mask": admission.tolist(),
            "periods": period_audits,
            "numerical_source": str(base_dir),
            "numerical_manifest_sha256": _sha(base_dir / "MANIFEST.sha256"),
            "W_update_dependencies": ["completed_original_q", "independent_policy_returns", "independent_policy_turnover"],
            "W_update_depends_on_Llama_or_final_portfolio": False,
            "hard_allocator": "Replace q only in q@W@e with one-hot(argmax(q)); historical W stream unchanged",
            "uniform_allocator": "Replace q only in q@W@e with [1/3,1/3,1/3]; historical W stream unchanged",
            "frozen_W": "Use initial admitted W throughout; original q and actual proposals unchanged",
            "q_used_in_W_updates_prompts_retrieval": "ORIGINAL_UNCHANGED",
            "hard_tie_break": "numpy.argmax: first tied index in recorded regime order",
            "W_resets_at_2024_source_boundary": False,
            "historical_policy_holdings_reset_at_source_boundary": True,
            "new_portfolio_holdings_must_continue_across_source_boundary": True,
            "inactive_policy_proposals_preserved": True,
            "inactive_policy_weighted_exposures_exactly_zero": bool(np.all((mixed * proposals)[:, ~admission] == 0)),
            "masked_policy_zero_effect_is_structural_not_economic_rejection": True,
            "scope": "ALLOCATION_COMPONENT_INTERVENTION_WITH_COMMON_HISTORICAL_POLICY_WEIGHT_STREAM",
        }
        return variants, audit, component_trace
    except CoreSourceError:
        raise
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        raise CoreSourceError(f"Cannot reconstruct frozen policy matrix: {exc}") from exc
