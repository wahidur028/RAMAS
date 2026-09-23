# Scientific method

## Question

Did the recorded Stage 5.1 LLM core policy add incremental value beyond the
simple Bear/Mix=0.25 and Bull=0.50 rule that explains most of its actions?

## Design

- Zero new LLM calls.
- Pre-2024 development rows only.
- Reused 2024–2025 data remain unopened.
- Same next-day return clock and 0.001 symmetric turnover cost.
- The recorded agent and static mapping are replayed through one shared,
  path-consistent accounting engine.
- The frozen deployment safety layer is removed from both core policies. This
  isolates policy decisions but cannot support a deployment-performance claim.
- Trigger variants are fixed diagnostics. The best variant is not selected.

## Primary gate

The recorded agent core must have positive total-growth advantage, improve
Sharpe by at least 0.05, avoid worse daily-loss CVaR, obtain a one-sided
30-day circular-block-bootstrap p-value below 0.05, win in at least two calendar
years, and contain at least 30 genuine deviations from the static rule.

## Memory gate

At least one lesson must survive, promotions must exceed rollbacks, and active
lessons must demonstrably change an action relative to the static mapping.

## Interpretation boundary

Passing is only permission for one bounded residual-agent preflight. Failing
prohibits repetition of the current prompt design. Neither outcome permits an
OOS claim or a multi-agent claim.
