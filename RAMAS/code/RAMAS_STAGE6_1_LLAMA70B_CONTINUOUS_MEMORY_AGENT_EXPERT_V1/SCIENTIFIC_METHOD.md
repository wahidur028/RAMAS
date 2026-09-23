# Scientific method

## Question

Does a bounded Llama-70B expert with completed-episode memory and adaptive,
regime-specific trust add technically valid and economically useful information to
the frozen causal RAMoE allocation pipeline?

## Prequential clock

For decision date `t`, the prompt may contain only router probabilities, OHLCV-derived
features, risk previews, trust, and memory available at `t`. The action controls the
return at `t+1`. Only after that return completes is the episode added to memory.
Trust updates occur at a later month boundary from completed earlier episodes.

The exact Stage 6 Llama prompt, action contract, trust rule, transaction cost, router,
and risk layer remain frozen. The proposed final exposure is a bounded blend of the
pre-agent RAMoE desired exposure and the Llama action, followed by the unchanged risk
projection.

## Continuous learning timeline

- 2015–2020: market and router history only. The frozen source does not provide an
  agent-compatible RAMoE decision trace for these years, so this package makes no
  false claim that the agent learned from completed episodes then.
- 2021: burn-in; the verified Stage 6 state is reused.
- 2022–2023: continuous walk-forward development evidence.
- 2024–2025: continuous reused-OOS diagnostic evidence.

Memory, agent trust, and portfolio pre-trade exposure are never reset at January 1.
Annual boundaries are reporting checkpoints only.

## Evidence hierarchy

1. Mathematical and causal validity.
2. Technical mechanism validity and failure behavior.
3. Method attribution using internal controls and ablations.
4. Economic findings by year and full horizon, including uncertainty.
5. External baseline comparison after the main pipeline is validated.
6. Novelty positioning by a separate literature audit.

No single weak metric rejects a research architecture. Mixed results are preserved
and reported. Conversely, a technically correct run does not prove economic value,
and performance does not prove novelty.

## Comparators in this package

`pre_agent_ramoe_internal_control` is an internal mechanism control. Buy-and-hold and
cash-only are descriptive external references. ML/RL and full-LLM baselines are not
implemented here and must not be claimed as completed comparisons.

## Statistical output

The package reports terminal growth, zero-cash-rate Sharpe, maximum drawdown loss,
daily 95% loss CVaR, exposure, turnover, paired daily log-return advantage, a
deterministic circular-block bootstrap, effective interventions, memory growth, and
trust events. Results are emitted for every return year and for the full horizon.
