# RAMAS Stage 5.3: bounded tool-using residual LLM-agent

This package implements the single-agent repair authorized by the Stage 5.2
mechanism audit. It does not repeat the failed unrestricted target-exposure
prompt.

## What changed

The transparent regime rule now owns the default exposure: Bear 0.25, Bull
0.50, Mix 0.25. The Qwen3-Coder 30B agent is called only when the frozen router
is transitioning or its confidence is below 0.80. It first plans deterministic
tool use, observes the tool results, and then proposes only a residual from
`{-0.25, 0, +0.25}`.

A nonzero residual is accepted only when all four tools were executed and
cited, at least 12 same-context mature incidents were compared, mean past-only
log advantage is positive, daily-loss CVaR is not worse, and both chronological
halves have positive advantage. All other outputs become residual zero.

## Scientific boundary

- Router: frozen.
- Corrected next-day return clock: frozen.
- Transaction cost and risk layer: frozen.
- Development data: 2021–2023 only.
- Reused 2024–2025 diagnostic: never opened by this package.
- Multi-agent collaboration: prohibited in this stage.
- Model weights and router weights: never updated online.

The runtime memory is event-driven. A lesson needs at least 30 matching mature
incidents, positive mean advantage, a positive 95% circular-block-bootstrap
lower bound, non-worse CVaR, and positive advantage in two chronological blocks.
It rolls back after 20 additional incidents if cumulative advantage is no
longer positive or CVaR worsens.

## Run on the research server

```bash
cd /home/infonet/wahid/leader_router_fresh

sha256sum -c RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1_CODE.tar.gz

nohup bash RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1/run_server.sh \
  > RAMAS_STAGE5_3_RUN.log 2>&1 &
echo $!
```

Monitor it with:

```bash
tail -f /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE5_3_RUN.log
```

The preflight uses 30 triggered development observations. If at least 29 plans
and 29 decisions are not schema-valid, the run stops before economic replay.
Preflight memory is discarded.

The full replay should make far fewer LLM calls than Stage 5.1 because only
triggered days use the two-call plan/decision loop. Progress is printed every
50 market days.

## Result decision

The repaired agent must beat the transparent controller on all fixed checks:
positive growth advantage, Sharpe improvement of at least 0.05, one-sided
30-day circular-block-bootstrap p-value below 0.05, non-worse daily-loss CVaR,
and positive advantage in at least two calendar years.

Failure produces `STOP_AGENT_ECONOMIC_BRANCH_DEVELOPMENT_GATE_FAILED`. It does
not authorize another prompt tweak, OOS rescue, or multi-agent experiment.
