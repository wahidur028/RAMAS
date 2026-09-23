from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.accounting import drifted_exposure, net_return
from src.core import atomic_write_csv, atomic_write_json, exact_upper_tail_mean, sha256_file
from src.real_data import build_real_router_inputs, production_admission_mask
from src.simulator import SimulationConfig, SimulationResult, simulate
from src.statistics import circular_block_mean_test


PACKAGE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    claims = config["claims"]
    if claims.get("dqn_closed") is not True or claims.get("dqn_training_permitted") is not False:
        raise RuntimeError("DQN closure contract failed")
    if claims.get("stage0_decision_may_be_overridden") is not False:
        raise RuntimeError("Stage-0 decision override is prohibited")
    if claims.get("daily_router_posterior_required") is not True:
        raise RuntimeError("Daily Router posterior contract is not frozen")
    required = int(ceil(1.0 / (1.0 - float(config["cvar_alpha"]))))
    if (
        float(config["trust_minimum_posterior_mass"]) != required
        or float(config["trust_minimum_effective_sample_size"]) != required
        or config.get("trust_tail_estimator") != "exact_regime_posterior_weighted_upper_tail_mean"
    ):
        raise RuntimeError("Evidence-weighted trust support contract failed")
    if claims.get("unsupported_regime_trust_may_update") is not False:
        raise RuntimeError("Unsupported regime trust updates are prohibited")
    return config


def simulation_config(config: dict[str, Any]) -> SimulationConfig:
    names = list(config["expert_names"])
    always = tuple(names.index(name) for name in config["always_admitted_experts"])
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
    inputs,
    mask: np.ndarray,
    fallback: np.ndarray,
    sim_config: SimulationConfig,
    *,
    trust_updates: bool,
    risk_projection: bool,
) -> SimulationResult:
    return simulate(
        decision_dates=inputs.decision_dates,
        return_dates=inputs.return_dates,
        router_posteriors=inputs.router_posteriors,
        expert_exposures=inputs.expert_exposures,
        asset_simple_returns=inputs.asset_simple_returns,
        scenario_log_returns=inputs.scenario_log_returns,
        initial_trust=inputs.initial_trust,
        admission_mask=mask,
        fallback_weights=fallback,
        config=sim_config,
        enable_admission=True,
        enable_trust_updates=trust_updates,
        enable_risk_projection=risk_projection,
    )


def metrics(label: str, returns: np.ndarray, exposures: np.ndarray, turnover: np.ndarray) -> dict[str, Any]:
    values = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + values)
    path = np.r_[1.0, wealth]
    peak = np.maximum.accumulate(path)
    drawdown = path / peak - 1.0
    years = len(values) / 365.25
    annualized = float(wealth[-1] ** (1.0 / years) - 1.0) if wealth[-1] > 0.0 else -1.0
    volatility = float(np.std(values, ddof=1) * np.sqrt(365.25)) if len(values) > 1 else 0.0
    sharpe = float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(365.25)) if np.std(values, ddof=1) > 0.0 else 0.0
    return {
        "variant": label,
        "rows": int(len(values)),
        "terminal_growth": float(wealth[-1] - 1.0),
        "annualized_return": annualized,
        "annualized_volatility": volatility,
        "sharpe_zero_cash_rate": sharpe,
        "maximum_drawdown": float(-np.min(drawdown)),
        "daily_loss_cvar_95": float(exact_upper_tail_mean(-values, 0.05)),
        "mean_exposure": float(np.mean(exposures)),
        "mean_daily_turnover": float(np.mean(turnover)),
        "total_turnover": float(np.sum(turnover)),
    }


def constant_exposure_returns(asset_returns: np.ndarray, exposure: float, cost_rate: float) -> tuple[np.ndarray, np.ndarray]:
    output = np.empty(len(asset_returns), dtype=float)
    turnover = np.empty(len(asset_returns), dtype=float)
    pretrade = 0.0
    for index, asset_return in enumerate(asset_returns):
        turnover[index] = abs(exposure - pretrade)
        output[index] = net_return(exposure, pretrade, float(asset_return), cost_rate)
        pretrade = drifted_exposure(exposure, float(asset_return))
    return output, turnover


def report_text(
    decision: dict[str, Any],
    checks: dict[str, bool],
    metrics_frame: pd.DataFrame,
    inference: dict[str, Any],
) -> str:
    full = metrics_frame.loc[metrics_frame["variant"] == "trusted_router"].iloc[0]
    matched = metrics_frame.loc[metrics_frame["variant"] == "constant_matched_mean_exposure"].iloc[0]
    return (
        "# Evidence-Weighted Trust Repair Result\n\n"
        f"**Status: {decision['scientific_status']}**\n\n"
        "The real pre-2024 BTC clock was replayed after replacing the invalid global-tail-then-condition "
        "trust update with exact regime-posterior-weighted CVaR and an evidence-support gate.\n\n"
        f"- Replay rows: {int(full['rows'])}\n"
        f"- Repaired Router terminal growth: {float(full['terminal_growth']):.6f}\n"
        f"- Matched-exposure terminal growth: {float(matched['terminal_growth']):.6f}\n"
        f"- Repaired Router maximum drawdown: {float(full['maximum_drawdown']):.6f}\n"
        f"- Mean exposure: {float(full['mean_exposure']):.6f}\n"
        f"- One-sided block-bootstrap p-value: {float(inference['one_sided_p_value']):.6f}\n"
        f"- Supported regime-month updates: {int(decision['supported_regime_month_updates'])}\n"
        f"- Skipped regime-month updates: {int(decision['skipped_regime_month_updates'])}\n"
        f"- Integration checks passed: {sum(checks.values())}/{len(checks)}\n"
        "- Admitted experts: Cash and Buy-and-Hold only\n"
        "- Trend, volatility, drawdown, and Specialist ATP weights: exactly zero\n"
        "- Specialist training performed: false\n"
        "- DQN training performed: false\n\n"
        "The trust repair is accepted economically only when every predeclared gate passes. "
        "Specialist admission remains closed regardless of this diagnostic.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = load_config(config_path)
    inputs, stage0 = build_real_router_inputs(project_root, config)
    names = list(config["expert_names"])
    mask = production_admission_mask(config, stage0)
    fallback = np.asarray([config["fallback_expert_weights"][name] for name in names], dtype=float)
    sim_config = simulation_config(config)

    full = run_variant(inputs, mask, fallback, sim_config, trust_updates=True, risk_projection=True)
    replay = run_variant(inputs, mask, fallback, sim_config, trust_updates=True, risk_projection=True)
    no_trust = run_variant(inputs, mask, fallback, sim_config, trust_updates=False, risk_projection=True)
    no_risk = run_variant(inputs, mask, fallback, sim_config, trust_updates=True, risk_projection=False)
    weight_columns = [f"weight_{index}" for index in range(len(names))]
    rejected = np.flatnonzero(~mask)
    trust_audit = full.trust_update_audit
    unsupported = trust_audit.loc[~trust_audit["support_passed"]]
    derived_minimum = int(ceil(1.0 / (1.0 - sim_config.cvar_alpha)))
    checks = {
        "deterministic_trace": full.trace.equals(replay.trace),
        "deterministic_terminal_trust": bool(np.array_equal(full.terminal_trust, replay.terminal_trust)),
        "deterministic_trust_update_audit": full.trust_update_audit.equals(replay.trust_update_audit),
        "unsupported_trust_rows_preserved_exactly": bool(
            np.array_equal(
                unsupported["trust_before"].to_numpy(float),
                unsupported["trust_after"].to_numpy(float),
            )
        ),
        "support_threshold_matches_tail_requirement": bool(
            sim_config.trust_minimum_posterior_mass == derived_minimum
            and sim_config.trust_minimum_effective_sample_size == derived_minimum
        ),
        "supported_and_skipped_regime_months_present": bool(
            0 < full.trust_rows_updated < full.trust_updates * len(config["regime_names"])
        ),
        "conditional_tail_components_finite_nonnegative": bool(
            np.isfinite(trust_audit["tail_component"]).all()
            and (trust_audit["tail_component"] >= 0.0).all()
        ),
        "daily_router_only": inputs.input_contract["router_source"] == "daily posterior only",
        "execution_clock_aligned": bool(
            (
                inputs.aligned_frame["router_information_date"].isna()
                | (inputs.aligned_frame["router_information_date"] == inputs.aligned_frame["information_date"])
            ).all()
        ),
        "return_dates_after_decisions": bool(np.all(inputs.return_dates.values > inputs.decision_dates.values)),
        "weights_sum_to_one": bool(np.allclose(full.trace[weight_columns].sum(axis=1), 1.0, atol=1e-12)),
        "rejected_weights_exactly_zero": bool(
            all(np.array_equal(full.trace[f"weight_{index}"].to_numpy(), np.zeros(len(full.trace))) for index in rejected)
        ),
        "rejected_terminal_trust_zero": bool(
            all(np.array_equal(full.terminal_trust[:, index], np.zeros(full.terminal_trust.shape[0])) for index in rejected)
        ),
        "exposure_bounded": bool(full.trace["final_exposure"].between(0.0, 1.0).all()),
        "turnover_bounded": bool((full.trace["turnover"] <= sim_config.maximum_turnover + 1e-12).all()),
        "cost_accounting_finite": bool(np.isfinite(full.trace["portfolio_net_return"]).all()),
        "stage0_decision_bound": stage0.decision["scientific_decision"] == config["stage0"]["expected_decision"],
        "specialist_atp_rejected": not bool(mask[names.index("specialist_atp")]),
        "no_late_modality_values": bool((inputs.modality_alignment["availability_violations"] == 0).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Real-data adapter checks failed: {[key for key, value in checks.items() if not value]}")

    cost_rate = sim_config.transaction_cost_rate
    cash_returns, cash_turnover = constant_exposure_returns(inputs.asset_simple_returns, 0.0, cost_rate)
    half_returns, half_turnover = constant_exposure_returns(inputs.asset_simple_returns, 0.5, cost_rate)
    bh_returns, bh_turnover = constant_exposure_returns(inputs.asset_simple_returns, 1.0, cost_rate)
    matched_exposure = float(full.trace["final_exposure"].mean())
    matched_returns, matched_turnover = constant_exposure_returns(
        inputs.asset_simple_returns,
        matched_exposure,
        cost_rate,
    )
    metric_rows = [
        metrics("trusted_router", full.trace["portfolio_net_return"].to_numpy(float), full.trace["final_exposure"].to_numpy(float), full.trace["turnover"].to_numpy(float)),
        metrics("no_trust_updates", no_trust.trace["portfolio_net_return"].to_numpy(float), no_trust.trace["final_exposure"].to_numpy(float), no_trust.trace["turnover"].to_numpy(float)),
        metrics("no_risk_projection", no_risk.trace["portfolio_net_return"].to_numpy(float), no_risk.trace["final_exposure"].to_numpy(float), no_risk.trace["turnover"].to_numpy(float)),
        metrics("cash", cash_returns, np.zeros(len(cash_returns)), cash_turnover),
        metrics(
            "constant_matched_mean_exposure",
            matched_returns,
            np.full(len(matched_returns), matched_exposure),
            matched_turnover,
        ),
        metrics("constant_50_percent", half_returns, np.full(len(half_returns), 0.5), half_turnover),
        metrics("buy_and_hold", bh_returns, np.ones(len(bh_returns)), bh_turnover),
    ]
    metrics_frame = pd.DataFrame(metric_rows)

    router_returns = full.trace["portfolio_net_return"].to_numpy(float)
    paired_log_advantage = np.log1p(router_returns) - np.log1p(matched_returns)
    gate_config = config["economic_gate"]
    bootstrap = circular_block_mean_test(
        paired_log_advantage,
        block_length=int(gate_config["block_length_days"]),
        resamples=int(gate_config["bootstrap_resamples"]),
        seed=int(gate_config["bootstrap_seed"]),
    )
    inference = {
        "primary_comparator": gate_config["primary_comparator"],
        "block_length_days": int(gate_config["block_length_days"]),
        "bootstrap_resamples": int(gate_config["bootstrap_resamples"]),
        "bootstrap_seed": int(gate_config["bootstrap_seed"]),
        "observed_mean_log_advantage": bootstrap.observed_mean_log_advantage,
        "confidence_interval_low": bootstrap.confidence_interval_low,
        "confidence_interval_high": bootstrap.confidence_interval_high,
        "one_sided_p_value": bootstrap.one_sided_p_value,
    }
    yearly_rows: list[dict[str, Any]] = []
    return_years = inputs.return_dates.year.to_numpy()
    for year in sorted(np.unique(return_years)):
        year_mask = return_years == year
        router_growth = float(np.prod(1.0 + router_returns[year_mask]) - 1.0)
        matched_growth = float(np.prod(1.0 + matched_returns[year_mask]) - 1.0)
        yearly_rows.append(
            {
                "year": int(year),
                "trusted_router_growth": router_growth,
                "matched_exposure_growth": matched_growth,
                "growth_advantage": router_growth - matched_growth,
                "router_wins": bool(router_growth > matched_growth),
            }
        )
    yearly_frame = pd.DataFrame(yearly_rows)
    router_metric = metrics_frame.loc[metrics_frame["variant"] == "trusted_router"].iloc[0]
    matched_metric = metrics_frame.loc[
        metrics_frame["variant"] == "constant_matched_mean_exposure"
    ].iloc[0]
    economic_checks = {
        "positive_full_period_growth_advantage": bool(
            router_metric["terminal_growth"] > matched_metric["terminal_growth"]
        ),
        "one_sided_block_bootstrap_p_below_alpha": bool(
            bootstrap.one_sided_p_value < float(gate_config["one_sided_alpha"])
        ),
        "minimum_positive_years": bool(
            int(yearly_frame["router_wins"].sum()) >= int(gate_config["minimum_positive_years"])
        ),
        "maximum_drawdown_not_worse": bool(
            router_metric["maximum_drawdown"] <= matched_metric["maximum_drawdown"]
        ),
        "daily_cvar_not_worse": bool(
            router_metric["daily_loss_cvar_95"] <= matched_metric["daily_loss_cvar_95"]
        ),
    }
    economic_gate_passed = bool(all(economic_checks.values()))
    scientific_status = (
        "EVIDENCE_WEIGHTED_TRUST_ECONOMIC_GATE_PASSED_SPECIALISTS_REMAIN_CLOSED"
        if economic_gate_passed
        else "EVIDENCE_WEIGHTED_TRUST_REPAIR_VALID_ECONOMIC_GATE_FAILED"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output.resolve() if args.output else (
        project_root / "EXPERIMENT_BRANCHES" / config["experiment_id"] / "artifacts" / stamp
    )
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite output={output}")
    output.mkdir(parents=True)
    contract = {
        "experiment_id": config["experiment_id"],
        "phase": config["phase"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        **inputs.input_contract,
    }
    atomic_write_json(output / "00_INPUT_CONTRACT.json", contract)
    atomic_write_json(
        output / "01_STAGE0_ADMISSION.json",
        {
            "source_artifact": str(stage0.artifact_directory),
            "scientific_decision": stage0.decision["scientific_decision"],
            "expert_names": names,
            "admission_mask": mask.tolist(),
            "admitted_experts": [name for name, admitted in zip(names, mask) if admitted],
            "rejected_experts": [name for name, admitted in zip(names, mask) if not admitted],
            "override_permitted": False,
        },
    )
    atomic_write_csv(output / "02_MODALITY_ALIGNMENT.csv", inputs.modality_alignment)
    expert_input = inputs.aligned_frame[["information_date", "decision_date", "target_date", *config["router"]["posterior_columns"]]].copy()
    expert_input["asset_simple_return"] = inputs.asset_simple_returns
    for index, name in enumerate(names):
        expert_input[f"exposure_{name}"] = inputs.expert_exposures[:, index]
    atomic_write_csv(output / "03_REAL_ROUTER_INPUTS.csv", expert_input)
    atomic_write_csv(output / "04_REAL_ROUTER_TRACE.csv", full.trace)
    atomic_write_csv(output / "04A_TRUST_UPDATE_AUDIT.csv", trust_audit)
    atomic_write_json(output / "05_INTEGRATION_CHECKS.json", {"status": "PASS", "checks": checks})
    atomic_write_csv(output / "06_DEVELOPMENT_METRICS.csv", metrics_frame)
    atomic_write_csv(output / "06A_YEARLY_MATCHED_EXPOSURE.csv", yearly_frame)
    atomic_write_json(
        output / "06B_ECONOMIC_GATE.json",
        {
            "status": "PASS" if economic_gate_passed else "FAIL",
            "economic_gate_passed": economic_gate_passed,
            "checks": economic_checks,
            "inference": inference,
        },
    )
    atomic_write_json(
        output / "07_TERMINAL_TRUST.json",
        {"regime_names": config["regime_names"], "expert_names": names, "matrix": full.terminal_trust.tolist()},
    )
    decision = {
        "scientific_status": scientific_status,
        "real_data_adapter_valid": True,
        "trust_repair_mechanically_valid": True,
        "economic_gate_passed": economic_gate_passed,
        "economic_specialist_validated": False,
        "supported_regime_month_updates": int(full.trust_rows_updated),
        "skipped_regime_month_updates": int(
            full.trust_updates * len(config["regime_names"]) - full.trust_rows_updated
        ),
        "admitted_experts": [name for name, admitted in zip(names, mask) if admitted],
        "specialist_training_performed": False,
        "dqn_training_performed": False,
        "interpretation": (
            "evidence-weighted trust repair passed its economic development gate; Stage-0 specialist rejection remains binding"
            if economic_gate_passed
            else "evidence-weighted trust repair is mechanically valid but failed its economic development gate; Stage-0 specialist rejection remains binding"
        ),
    }
    atomic_write_json(output / "08_DECISION.json", decision)
    (output / "09_PLAIN_ENGLISH_REPORT.md").write_text(
        report_text(decision, checks, metrics_frame, inference), encoding="utf-8"
    )
    artifacts = [
        "00_INPUT_CONTRACT.json", "01_STAGE0_ADMISSION.json", "02_MODALITY_ALIGNMENT.csv",
        "03_REAL_ROUTER_INPUTS.csv", "04_REAL_ROUTER_TRACE.csv", "04A_TRUST_UPDATE_AUDIT.csv",
        "05_INTEGRATION_CHECKS.json", "06_DEVELOPMENT_METRICS.csv",
        "06A_YEARLY_MATCHED_EXPOSURE.csv", "06B_ECONOMIC_GATE.json",
        "07_TERMINAL_TRUST.json", "08_DECISION.json", "09_PLAIN_ENGLISH_REPORT.md",
    ]
    atomic_write_json(
        output / "RUN_COMPLETE.json",
        {
            "status": "PASS",
            "scientific_status": decision["scientific_status"],
            "artifact_sha256": {name: sha256_file(output / name) for name in artifacts},
        },
    )
    print("LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_STATUS=PASS", flush=True)
    print(f"SCIENTIFIC_STATUS={decision['scientific_status']}", flush=True)
    print(f"REAL_REPLAY_ROWS={len(full.trace)}", flush=True)
    print(f"ECONOMIC_GATE_PASSED={str(economic_gate_passed).upper()}", flush=True)
    print(f"SUPPORTED_REGIME_MONTH_UPDATES={full.trust_rows_updated}", flush=True)
    print("ADMITTED_EXPERTS=CASH,BUY_AND_HOLD", flush=True)
    print("SPECIALIST_TRAINING_PERFORMED=FALSE", flush=True)
    print("DQN_TRAINING_PERFORMED=FALSE", flush=True)
    print(f"OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
