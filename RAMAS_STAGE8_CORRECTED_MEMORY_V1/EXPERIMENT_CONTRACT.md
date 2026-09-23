# RAMAS Stage 8 — corrected episodic-memory contract

## Decision

This experiment tests whether the inconclusive memory effect in Stages 6.2–6.4
is a property of the memory *implementation* rather than of LLM episodic memory
in general. It is not a search for a profitable arm and it is not a clock repair.

## What changes

Exactly two seams, both behind explicit spec fields.

**1. Credit label.** The value stored next to each episode becomes the realized
advantage of the taken action over the unchanged numerical core, with both
exposures projected from the arm's own pre-trade holdings:

    label = log1p(net(x_action)) - log1p(net(x_core)),  both at the same pretrade

`x_action` is the exposure the taken action would have produced at full advisory
authority; `x_core` is the exposure the unchanged core produced. ABSTAIN scores
exactly zero, because abstaining is following the core.

The frozen label compared a full-authority shadow portfolio against the archived
legacy RAMoE portfolio, which follows a *different* holdings path. Measured on
the archived Stage 6.4 arms, that label correlates **-0.36** with the day's
return — it is dominated by the exposure gap between the two paths, not by
decision quality. The replacement correlates **+0.67**.

**2. Retrieval.** Episodes are retrieved k-per-action rather than k-nearest
overall. On the archived arms, all retrieved episodes carry the same action on
**85.9%** of days, so the evidence block can express an action contrast on only
13.5% of days. Balanced retrieval guarantees a contrast whenever the history
contains the actions.

The default also removes the hard same-regime prefilter. The archived
`llama_memory_pooled` arm — the only Stage 6.4 contrast whose interval excludes
zero — beat the main method, which is the direction this change predicts.

## What does not change

Inputs, return clock, router stream, numerical core, risk projection,
accounting, journal, provider, decision validation and metrics are imported
unchanged from the verified Stage 6.4 package. This package never writes inside
it.

## Arms

Fixed trust at beta = 0.05, controller on, memory versus no-memory. Fixed trust
removes the trust rule as a second consumer of the corrected label, so the only
thing that differs between the two arms is what the model can see.

| Arm | memory | label | retrieval |
|---|---|---|---|
| `corrected_memory` | expanding | counterfactual | balanced, 2 per action, no regime filter |
| `corrected_no_memory` | none | counterfactual | suppressed |

Legacy reference for the same contrast is the completed Stage 6.3 pair
(`fixed_memory` vs `fixed_no_memory`, +0.0369 bps/day, CI [-0.034, +0.122]).

## Estimand

Paired mean daily net log return after trading costs, `corrected_memory` minus
`corrected_no_memory`, over return dates 2022-01-01 to 2025-05-28 (1,244 days),
circular moving-block bootstrap, block 30, 5,000 resamples.

Closed-loop arms do not retain the same later state, so this is a total-system
effect, not a per-date advice effect.

## Claim boundaries — all remain false

- `fresh_out_of_sample`: 2024–2025 is reused diagnostic history.
- `router_temporal_validity_resolved`: the regime posterior targets the
  allocation day, not the holding day, and its fit chronology is not verified.
- `feasible_information_to_fill_clock_resolved`: decisions use close[d] and are
  filled at close[d]; inference latency is not modelled.
- `corrected_label_validated_as_improving_decisions`: **this experiment tests
  it.** The corrected label is shown to *measure* decision quality; it is not
  yet shown to *improve* decisions. A null result here is a valid outcome and
  strengthens the generality of the failure-mode claim rather than weakening it.

## Prerequisite

`validate` must report `PASS_ENGINE_REPRODUCES_ARCHIVED_ARM` before any model
call is made. That replay uses archived responses only and proves the engine is
byte-faithful to the frozen design when the corrections are switched off.
