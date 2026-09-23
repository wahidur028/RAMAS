# Stage 6.3 scientific protocol

Prepared after Stage 6.2 interpretation and before Stage 6.3 Llama inference. This is a frozen protocol for a new comparison on reused historical data, not untouched confirmation.

## Question and intervention

Does completed episodic information improve net account growth when agent influence is fixed at its existing initial value, beta 0.05?

Fresh arms: M1/T0 (memory, fixed positive trust) and M0/T0 (no memory, same fixed positive trust). Archived Stage 6.2 supplies M1/T1 and M0/T1 under adaptive trust for a secondary interaction. No old response seeds a new arm.

Both new arms retain the frozen prompt, contract, features, forecast posteriors, numerical desired-allocation stream, risk projection, cost model, episode representation and five-nearest-same-regime retrieval. Only agent beta is changed from Stage 6.2's adaptive rule. Both maintain a private outcome ledger; only M1 exposes eligible episodes and summary. Shadow/reference outcomes remain in the same memory content, but never update fixed beta.

Start all cash, wealth1, empty experiences and beta0.05 for Bear/Bull/Mix. Carry state across years. Eligible outcomes must satisfy decision_date < current decision and return_date <= current decision. The inherited close-based convention permits the prior decision's just-completed outcome; feasible execution delay remains unresolved.

## Accounting and example

For numerical desired BTC exposure b, agent action a and beta=.05:

- ABSTAIN: desired=b.
- BTC: desired=(1-beta)b+beta.
- CASH: desired=(1-beta)b.

Example: b=.45 and BTC advice gives .4775 before projection. ABSTAIN preserves .45. Risk/grid projection can still map different targets to the same final weight. Beta0.05 is not a 5% BTC holding or guaranteed influence on every day.

For projected weight x, drifted pre-trade weight p, BTC simple return r and fee c=.001:

`net = (1-c*abs(x-p))*(1+x*r)-1`

`p_next = x*(1+r)/(1+x*r)`

Wealth compounds daily without calendar resets or new entry fees. Cash yield0; no terminal sale, extra slippage/impact or dollar inference expense. Token and latency costs are recorded separately.

## Primary endpoint

Primary: 1,244 paired returns from 2022-01-01 through 2025-05-28. The 364 returns in 2021 are warmup. Full trajectory:1,608 returns from2021-01-02 through2025-05-28.

`d_t = log(1+net_M1T0,t)-log(1+net_M0T0,t)`.

Report mean d_t and two-sided95% paired circular-block bootstrap interval:30-day blocks,5,000 resamples,seed16062. Positive/negative/inconclusive labels apply to this scoped contrast, not optimality or generic continuous learning. Freeze this endpoint before inference; do not select beta by historical profit.

The bootstrap conditions on realized closed-loop paths and does not regenerate learner decisions in resampled markets. One inference seed does not represent model-run uncertainty or independent market samples. Keep invalid-response/fallback days visible; an economic loss alone cannot stop progression.

## Secondary interaction and records

On aligned daily log growth, compute:

`interaction = (M1T1-M0T1)-(M1T0-M0T0)`.

Resample all four streams jointly in common day blocks. This is secondary, conditional and diagnostic. Reuse archived cells only under matched source/model/prompt/serving/runtime contracts. Do not combine corrected-method cells with legacy cells.

Record compounded/annualized growth, daily/annualized volatility, zero-rate Sharpe, maximum drawdown, daily loss CVaR95 with fractional tail mass, turnover, exposure and modeled costs. Use365.25 annualization, matching executed Stage6.2 reporting; its365 prose discrepancy remains historical.

Provide daily, monthly, yearly and common forecast Bull/Bear/Mix reports. Regime slices are noncontiguous conditional statistics, not tradable regime-only equity curves. Do not report stitched-regime drawdown. Undefined cash Sharpe and one-observation sample statistics remain undefined. Reporting windows never reset actual state.

External descriptive controls: cash, BTC B&H, daily-rebalanced50/50; static45 is exploratory. Passive controls have no RAMAS projection. No-memory is an internal ablation. No new ML or independent full-LLM baseline is implemented in this package.

Record eligible earlier-year memories, advice disagreements, desired-target disagreements, final BTC-weight disagreements and economic differences. Archive all three action previews, conditional action range, own wealth and memory age/counts. Distinguish abstention from zero influence and projection effects.

## Scientific scope and pre-call checks

Strict source/model/journal identity, finite data and ordered dates stop on mismatch. Record what is reconstructed separately from unresolved facts.

Known legacy clock: archived q follows the source's earlier decision/target convention, not a newly verified forecast for the next holding interval. Original fit-time evidence and feasible close-derived prompt/fill timing remain incomplete. Preserve the2024 imported reference seam; it can influence memory even while beta is fixed. Shared limitations are not automatically harmless.

Allowed scope: `LEGACY_MECHANISM_DIAGNOSTIC_NOT_REALISTIC_EXECUTION_OR_FRESH_OOS_CONFIRMATION`. No automatic claim of causal forecasting, live profitability or novelty. Historical dates can interact with Llama's pretrained knowledge. Model weights remain fixed.

If actions barely affect allocation, investigate that mechanism before attributing a small economic difference. If trades differ without established gain, report the scoped null or trade-off. A positive result motivates accumulated-experience controls; it does not validate every component. Any source/reference/freshness repair is a new version requiring matched corrected comparisons.
