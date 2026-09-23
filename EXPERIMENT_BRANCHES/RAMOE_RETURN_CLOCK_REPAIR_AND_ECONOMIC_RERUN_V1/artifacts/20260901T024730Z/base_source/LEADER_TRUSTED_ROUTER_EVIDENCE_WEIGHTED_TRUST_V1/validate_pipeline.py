from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import atomic_write_csv, atomic_write_json, sha256_file
from src.fixture import MechanicalFixture, build_fixture
from src.simulator import SimulationConfig, SimulationResult, simulate


PACKAGE = Path(__file__).resolve().parent


def load_config() -> dict[str, object]:
    return json.loads((PACKAGE / "config.json").read_text(encoding="utf-8"))


def vectors(config: dict[str, object]) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    names = list(config["expert_names"])
    fixture_admission = config["mechanical_fixture_admission"]
    fallback = config["fallback_expert_weights"]
    mask = np.asarray([bool(fixture_admission[name]) for name in names], dtype=bool)
    fallback_weights = np.asarray([float(fallback[name]) for name in names], dtype=float)
    always = tuple(names.index(name) for name in config["always_admitted_experts"])
    return mask, fallback_weights, always


def simulation_config(config: dict[str, object], always: tuple[int, ...]) -> SimulationConfig:
    return SimulationConfig(
        always_admitted_indices=always,
        transaction_cost_rate=float(config["transaction_cost_bps"]) / 10000.0,
        cvar_alpha=float(config["cvar_alpha"]),
        cvar_limit=float(config["cvar_limit"]),
        ambiguity_quantile=float(config["ambiguity_quantile"]),
        maximum_turnover=float(config["maximum_daily_turnover"]),
        exposure_grid_step=float(config["exposure_grid_step"]),
        trust_eta=float(config["trust_eta"]),
        trust_tail_penalty=float(config["trust_tail_penalty"]),
        trust_turnover_penalty=float(config["trust_turnover_penalty"]),
        trust_minimum_posterior_mass=float(config["trust_minimum_posterior_mass"]),
        trust_minimum_effective_sample_size=float(config["trust_minimum_effective_sample_size"]),
    )


def run_variant(
    fixture: MechanicalFixture,
    mask: np.ndarray,
    fallback: np.ndarray,
    sim_config: SimulationConfig,
    variant: str,
) -> SimulationResult:
    flags = {
        "full": (True, True, True),
        "no_admission": (False, True, True),
        "no_trust_updates": (True, False, True),
        "no_risk_projection": (True, True, False),
    }
    admission, trust, risk = flags[variant]
    return simulate(
        decision_dates=fixture.decision_dates,
        return_dates=fixture.return_dates,
        router_posteriors=fixture.router_posteriors,
        expert_exposures=fixture.expert_exposures,
        asset_simple_returns=fixture.asset_simple_returns,
        scenario_log_returns=fixture.scenario_log_returns,
        initial_trust=fixture.initial_trust,
        admission_mask=mask,
        fallback_weights=fallback,
        config=sim_config,
        enable_admission=admission,
        enable_trust_updates=trust,
        enable_risk_projection=risk,
    )


def summarize(name: str, result: SimulationResult) -> dict[str, object]:
    trace = result.trace
    wealth = np.cumprod(1.0 + trace["portfolio_net_return"].to_numpy(float))
    peak = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdown = np.r_[1.0, wealth] / peak - 1.0
    return {
        "variant": name,
        "rows": int(len(trace)),
        "terminal_growth": float(wealth[-1] - 1.0),
        "maximum_drawdown": float(-np.min(drawdown)),
        "mean_exposure": float(trace["final_exposure"].mean()),
        "mean_turnover": float(trace["turnover"].mean()),
        "trust_updates": int(result.trust_updates),
        "router_fallback_days": int(trace["router_fallback_used"].sum()),
        "risk_fallback_days": int(trace["risk_fallback_used"].sum()),
    }


def causal_prefix_check(
    fixture: MechanicalFixture,
    baseline: SimulationResult,
    mask: np.ndarray,
    fallback: np.ndarray,
    sim_config: SimulationConfig,
) -> bool:
    cutoff = len(fixture.decision_dates) // 2
    changed_q = fixture.router_posteriors.copy()
    changed_exposures = fixture.expert_exposures.copy()
    changed_returns = fixture.asset_simple_returns.copy()
    changed_scenarios = fixture.scenario_log_returns.copy()
    changed_q[cutoff:] = changed_q[cutoff:, ::-1]
    changed_exposures[cutoff:] = 1.0 - changed_exposures[cutoff:]
    changed_returns[cutoff:] = np.clip(-changed_returns[cutoff:], -0.8, 0.8)
    changed_scenarios[cutoff:] = -changed_scenarios[cutoff:]
    modified = simulate(
        decision_dates=fixture.decision_dates,
        return_dates=fixture.return_dates,
        router_posteriors=changed_q,
        expert_exposures=changed_exposures,
        asset_simple_returns=changed_returns,
        scenario_log_returns=changed_scenarios,
        initial_trust=fixture.initial_trust,
        admission_mask=mask,
        fallback_weights=fallback,
        config=sim_config,
    )
    columns = [column for column in baseline.trace.columns if column != "ambiguity_cvar"]
    return baseline.trace.loc[: cutoff - 1, columns].equals(
        modified.trace.loc[: cutoff - 1, columns]
    ) and np.array_equal(
        baseline.trace.loc[: cutoff - 1, "ambiguity_cvar"].to_numpy(),
        modified.trace.loc[: cutoff - 1, "ambiguity_cvar"].to_numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PACKAGE.parent)
    args = parser.parse_args()
    config = load_config()
    mask, fallback, always = vectors(config)
    sim_config = simulation_config(config, always)
    fixture_config = config["fixture"]
    fixture = build_fixture(
        seed=int(fixture_config["seed"]),
        rows=int(fixture_config["rows"]),
        scenario_models=int(fixture_config["scenario_models"]),
        scenario_samples=int(fixture_config["scenario_samples"]),
    )

    variants = {
        name: run_variant(fixture, mask, fallback, sim_config, name)
        for name in ["full", "no_admission", "no_trust_updates", "no_risk_projection"]
    }
    replay = run_variant(fixture, mask, fallback, sim_config, "full")
    full = variants["full"]
    names = list(config["expert_names"])
    atp_index = names.index("specialist_atp")
    checks = {
        "deterministic_trace": full.trace.equals(replay.trace),
        "deterministic_terminal_trust": np.array_equal(full.terminal_trust, replay.terminal_trust),
        "deterministic_trust_update_audit": full.trust_update_audit.equals(replay.trust_update_audit),
        "causal_future_invariance": causal_prefix_check(fixture, full, mask, fallback, sim_config),
        "weights_sum_to_one": bool(
            np.allclose(
                full.trace[[f"weight_{i}" for i in range(len(names))]].sum(axis=1),
                1.0,
                atol=1e-12,
            )
        ),
        "rejected_atp_weight_exactly_zero": bool(
            np.array_equal(full.trace[f"weight_{atp_index}"].to_numpy(), np.zeros(len(full.trace)))
        ),
        "no_admission_ablation_restores_atp_weight": bool(
            (variants["no_admission"].trace[f"weight_{atp_index}"] > 0.0).any()
        ),
        "risk_projection_changes_exposure": bool(
            not np.array_equal(
                full.trace["final_exposure"].to_numpy(),
                variants["no_risk_projection"].trace["final_exposure"].to_numpy(),
            )
        ),
        "unsupported_trust_rows_are_preserved": bool(
            np.array_equal(
                full.trust_update_audit.loc[
                    ~full.trust_update_audit["support_passed"], "trust_before"
                ].to_numpy(float),
                full.trust_update_audit.loc[
                    ~full.trust_update_audit["support_passed"], "trust_after"
                ].to_numpy(float),
            )
        ),
        "exposure_bounded": bool(full.trace["final_exposure"].between(0.0, 1.0).all()),
        "turnover_bounded": bool(
            (full.trace["turnover"] <= sim_config.maximum_turnover + 1e-12).all()
        ),
        "terminal_trust_row_stochastic": bool(
            np.allclose(full.terminal_trust.sum(axis=1), 1.0, atol=1e-12)
        ),
        "terminal_rejected_atp_trust_zero": bool(
            np.array_equal(full.terminal_trust[:, atp_index], np.zeros(full.terminal_trust.shape[0]))
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Mechanical validation failed: {failed}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.project_root
        / "EXPERIMENT_BRANCHES"
        / str(config["experiment_id"])
        / "artifacts"
        / stamp
    )
    output.mkdir(parents=True, exist_ok=False)
    contract = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "config_sha256": sha256_file(PACKAGE / "config.json"),
        "final_evaluation_period_selected": False,
        "specialist_training_performed": False,
        "dqn_training_performed": False,
        "fixture_is_scientific_evidence": False,
    }
    atomic_write_json(output / "00_CONTRACT.json", contract)
    atomic_write_json(output / "01_MECHANICAL_CHECKS.json", {"status": "PASS", "checks": checks})
    summary = pd.DataFrame([summarize(name, result) for name, result in variants.items()])
    atomic_write_csv(output / "02_ABLATION_SANITY.csv", summary)
    atomic_write_csv(output / "03_CAUSAL_TRACE.csv", full.trace)
    atomic_write_csv(output / "03A_TRUST_UPDATE_AUDIT.csv", full.trust_update_audit)
    atomic_write_json(
        output / "04_TERMINAL_TRUST.json",
        {
            "expert_names": names,
            "regime_names": config["regime_names"],
            "matrix": full.terminal_trust.tolist(),
        },
    )
    report = (
        "# Trusted Router Mechanical Validation\n\n"
        "The pipeline passed deterministic, causal, masking, trust, risk, turnover, "
        "and accounting checks on a synthetic fixture.\n\n"
        "This result validates software mechanics only. It does not establish predictive "
        "information, economic benefit, or a final evaluation protocol.\n"
    )
    (output / "05_PLAIN_ENGLISH_REPORT.md").write_text(report, encoding="utf-8")
    artifacts = [
        "00_CONTRACT.json",
        "01_MECHANICAL_CHECKS.json",
        "02_ABLATION_SANITY.csv",
        "03_CAUSAL_TRACE.csv",
        "03A_TRUST_UPDATE_AUDIT.csv",
        "04_TERMINAL_TRUST.json",
        "05_PLAIN_ENGLISH_REPORT.md",
    ]
    complete = {
        "status": "PASS",
        "scientific_status": "MECHANICALLY_VALID_NOT_ECONOMICALLY_VALID",
        "artifact_sha256": {name: sha256_file(output / name) for name in artifacts},
    }
    atomic_write_json(output / "RUN_COMPLETE.json", complete)
    print("LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_MECHANICAL_STATUS=PASS")
    print("SCIENTIFIC_STATUS=MECHANICALLY_VALID_NOT_ECONOMICALLY_VALID")
    print("FINAL_EVALUATION_PERIOD_SELECTED=FALSE")
    print("SPECIALIST_TRAINING_PERFORMED=FALSE")
    print("DQN_TRAINING_PERFORMED=FALSE")
    print(f"OUTPUT={output}")


if __name__ == "__main__":
    main()
