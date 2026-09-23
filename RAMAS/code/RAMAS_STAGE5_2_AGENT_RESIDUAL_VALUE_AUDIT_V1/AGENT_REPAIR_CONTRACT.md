# RAMAS bounded residual-agent repair contract

This is the only new single-agent design permitted after Stage 5.2. It is not
permission to run it before the Stage 5.2 audit has completed.

## What changes

The transparent controller owns the default action:

- Bear: 0.25 exposure
- Bull: 0.50 exposure
- Mix: 0.25 exposure

The LLM agent no longer predicts an unrestricted target exposure. It can only
return a residual adjustment from `{-0.25, 0.00, +0.25}`. The final desired
exposure remains clipped to `[0.00, 0.75]`. Residual zero is abstention.

## Why this remains an agent

The model must choose and use deterministic tools before it can propose a
nonzero residual:

1. `retrieve_mature_incidents`: retrieve past-only, outcome-complete incidents
   matching the current regime and router-uncertainty state.
2. `compare_allowed_actions`: compare allowed candidate residuals on retrieved
   incidents using the frozen accounting rule.
3. `inspect_current_risk`: read the frozen safety-layer limits and current
   pretrade exposure.
4. `inspect_evidence_ledger`: read active, quarantined, and rolled-back lessons.

A nonzero residual is accepted only when the response cites valid tool-result
identifiers. Unsupported prose, model confidence, and generic reason codes
cannot alter exposure.

## Trigger

The agent is called only when the frozen router reports either:

- `transition_day = true`; or
- `router_confidence < 0.80`.

All other days use the transparent controller without an LLM call. This trigger
is frozen before the repair run and cannot be selected from Stage 5.2 results.

## Evidence-gated runtime memory

Every completed incident is appended immediately. A proposed lesson remains
quarantined until all conditions hold using only matured past outcomes:

- at least 30 matching incidents;
- positive mean residual log advantage over the transparent controller;
- 95% circular-block-bootstrap lower bound above zero;
- daily-loss CVaR is not worse;
- positive advantage in two non-overlapping chronological blocks.

An active lesson is automatically rolled back after 20 additional matching
incidents if its cumulative advantage becomes non-positive or its CVaR becomes
worse. Memory promotion is event-driven, not monthly, and never changes router
weights or model parameters.

## Hard kill gate

The repaired agent must beat the transparent controller on pre-2024 development
under all of the following predeclared conditions:

- positive growth advantage;
- Sharpe improvement at least 0.05;
- one-sided 30-day circular-block-bootstrap p-value below 0.05;
- daily-loss CVaR not worse;
- positive advantage in at least two calendar years.

If this gate fails, the LLM-agent economic branch stops. The 2024–2025 reused
window remains diagnostic-only and cannot rescue the method. Multi-agent work is
allowed only after the repaired single agent passes this gate and a later clean
prospective confirmation.
