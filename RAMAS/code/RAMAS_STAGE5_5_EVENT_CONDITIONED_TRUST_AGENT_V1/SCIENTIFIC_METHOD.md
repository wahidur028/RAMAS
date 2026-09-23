# Scientific method: RAMAS Stage 5.5

## Decision being tested

This branch does **not** ask an LLM to forecast the next BTC return or choose a
daily BTC exposure. Stages 5.1–5.4 showed no incremental economic value from
that formulation.

The new question is narrower:

> Can an evidence-grounded LLM-agent improve risk-adjusted allocation by
> deciding when a temporary, bounded change to one regime–expert trust row is
> justified by point-in-time events, quantitative risk, and verified past
> counterfactual episodes?

## Why an agent is potentially necessary

The frozen numerical router describes market state but cannot reliably
interpret asynchronous events such as exchange failures, regulatory actions,
stablecoin stress, liquidation cascades, or institutional-flow shocks. The
agent's possible informational advantage must come from point-in-time textual
and event evidence. Without that evidence, there is no scientific reason to
prefer an LLM-agent over a deterministic controller.

## Agent loop

On a triggered decision time, the model:

1. observes the router, risk, expert-disagreement, and event summary;
2. chooses which bounded tools to call;
3. receives tool results produced only from information available at the
   decision time;
4. selects one trust operator, target regime row, strength, and expiry;
5. states a falsifiable hypothesis and invalidation condition;
6. passes the proposal to deterministic evidence and risk gates.

The runtime, not the LLM, records tool-result identifiers. This removes the
manual citation-copying failure seen in Stages 5.3 and 5.4 while retaining a
complete evidence trail.

## Frozen action space

- `KEEP`: no trust change.
- `DEFENSIVE_SHRINK`: move a small fraction of one row toward Cash.
- `RISK_AWARE_REWEIGHT`: move one row toward a deterministic past-only
  risk-adjusted expert target.
- `ROLLBACK`: move one row toward the frozen trusted configuration.

Strength is restricted to `0`, `0.05`, or `0.10`. Intervention horizons are
restricted to 3, 7, or 14 days. The agent cannot emit exposure, leverage,
short positions, or arbitrary weights.

## Risk shield

Code rejects a proposal when any of the following is true:

- the operator-specific tools were not executed;
- another intervention is already active;
- more than one trust row changes;
- target-row L1 movement exceeds 0.10;
- implied BTC exposure moves by more than 0.10;
- implied BTC exposure leaves [0, 1];
- the response violates the frozen schema.

The unchanged RAMoE daily execution and existing risk projection remain the
only components allowed to produce deployable exposure.

## Counterfactual memory

After an intervention expires, its realized net log return, drawdown, and
turnover must be compared with the exact unchanged-controller counterfactual.
The score penalizes additional drawdown and turnover. A lesson remains
quarantined until at least five completed, non-overlapping matching episodes
have positive average score, a positive block-bootstrap lower bound, and two
positive chronological blocks. A previously active lesson is rolled back when
those requirements cease to hold.

Five episodes is a mechanism-development minimum, not sufficient evidence for
an economic paper claim. The full development gate separately requires at
least 30 accepted interventions.

## Required primary comparator

The primary comparator is not original RAMoE alone. It is a deterministic
event-triggered trust controller with exactly the same triggers, candidate
operators, action budget, accounting, and risk shield. This comparison is
required to identify incremental value from agent reasoning rather than value
from the new event data or the trust operators themselves.

## Evidence gates

### Gate A — mechanism

Schema, causal as-of filtering, trust conservation, fail-closed behavior,
automatic provenance, safety limits, and memory tests must pass.

### Gate B — LLM interface

At least 11 of 12 synthetic triggered cases must produce valid plans and valid
decisions. These cases test the interface only and permit no economic claim.

### Gate C — event data

A raw point-in-time event ledger must provide exact publication and
availability timestamps, source identity, raw text, content hashes, collection
method, revision policy, and verified causal availability. Hindsight summaries
or revised pages fail this gate.

### Gate D — development economics

Only after Gate C passes may a pre-2024 causal development adapter be built.
The LLM-agent must then beat the equal-budget deterministic controller on all
predeclared conditions: positive net growth advantage, Sharpe improvement of
at least 0.05, non-worse maximum drawdown and expected shortfall, positive
results in at least two non-overlapping blocks, positive block-bootstrap lower
bound, and at least 30 accepted interventions.

### Gate E — confirmation

The reused 2024–2025 OOS period is sealed. It cannot rescue a failed
development result. A successful development branch requires a new untouched
or prospective confirmation period.

## Current claim boundary

This package can establish only mechanism correctness, interface validity, and
event-data readiness. It cannot establish improved profit, lower risk, or
deployable performance.

