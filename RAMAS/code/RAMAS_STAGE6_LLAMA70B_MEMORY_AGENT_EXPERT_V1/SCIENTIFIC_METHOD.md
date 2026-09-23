# Scientific method

## Question

Can a Llama 3.3 70B agent add incremental, risk-adjusted value to the existing RAMAS cryptocurrency allocation system when used as a bounded expert with causal memory and separately learned trust?

## Intervention

At the close of decision day `t`, the agent receives:

- the full causal next-day regime probability vector;
- lagged OHLCV-derived returns, volatility, and drawdown features available by `t`;
- the unchanged RAMAS desired exposure;
- safe exposure previews calculated by the unchanged risk layer;
- up to five similar episodes whose outcomes completed before the present decision.

It returns `BTC`, `CASH`, or `ABSTAIN`, confidence, reason codes, and optional citations to only the memory IDs shown in its request.

For regime `r`, the desired candidate is

`(1 - beta_r) * RAMAS_desired + beta_r * agent_target`,

where `agent_target` is 1 for BTC and 0 for CASH. `ABSTAIN` exactly preserves the RAMAS desired exposure. `beta_r` begins at 0.05, updates only at month boundaries from completed earlier outcomes, and is capped at 0.20. The original RAMAS trust matrix is never modified.

## Causal clock

The action for return date `t+1` can use prices through `t`. The realized `t -> t+1` return is added to memory only after that action has been recorded and is therefore visible no earlier than the following decision. Monthly trust updates likewise use only completed prior episodes.

## Controls

The 2021 pilot reports:

- original RAMAS;
- RAMAS + Llama memory expert with adaptive trust;
- the same Llama decisions with frozen trust;
- a transparent deterministic controller using the same market information and adaptive trust rule;
- the Llama signal as a shadow stand-alone expert;
- buy-and-hold and cash references.

All tradable variants use the frozen next-day return clock, symmetric transaction costs, drifted pre-trade exposure, and unchanged risk projection.

## Predeclared development gate

Every check must pass:

1. at least 98% valid Llama responses;
2. at least 30 non-abstaining decisions and at least two distinct actions;
3. terminal growth above original RAMAS;
4. Sharpe at least 0.05 above original RAMAS;
5. maximum drawdown no worse than original RAMAS;
6. 95% daily loss CVaR no worse than original RAMAS;
7. positive one-sided 95% circular-block-bootstrap lower bound for mean log-growth advantage over RAMAS;
8. terminal growth above the deterministic same-information controller.

Failure stops the branch before any 2022–2025 diagnostic. Passing permits a code/data freeze and a later year-by-year diagnostic, but does not itself establish economic value.

## Known limitation

The available market file begins in 2015 and historical router methodology supports expanding evaluation from 2018. However, the frozen corrected RAMAS base trace currently begins in 2021. A claim that the agent was trained from 2015 would therefore be false. Extending agent training to 2018–2020 requires regenerating and freezing corresponding causal RAMAS base traces without selecting parameters on later outcomes.
