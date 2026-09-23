# Stage 6.2 matched episodic-memory experiment

Protocol frozen for the new paired experiment on 2026-09-08, after inspecting Stage 6.1 historical results. This is a retrospective diagnostic protocol, not preregistration before first seeing these markets and not untouched out-of-sample confirmation.

## Research question and interpretation

Does access to retrieved completed episodes improve the return and risk of this RAMAS policy under the same adaptive-trust rule?

The estimand is the total effect of supplying episodic evidence, including later changes it causes in decisions, portfolio state and trust. Both arms still learn trust from completed outcomes. The no-memory arm is not a no-learning arm. A positive result would support this bounded claim for this dataset, runtime and policy; it would not prove that Llama becomes progressively optimal, that its weights learn, or that carrying many years of memory is better than a frozen or year-reset store. Those require later controls.

## Matched intervention

Both arms independently initialize in 2021 with beta 0.05 per regime, empty episodes and all-cash portfolios. Each generates its own Llama responses and stores its own completed outcomes. Memory receives the legacy nearest-same-regime retrieval (up to five episodes) and summary; no-memory receives an empty list and the legacy empty-summary marker. It retains a private performance ledger for the shared trust algorithm, but those episodes and their summary are never sent to its Llama prompt.

Common inputs: the same dated prices/close-derived features, router probabilities, pre-agent desired exposure and legacy pre-agent reference return. Common code: action schema, system prompt, memory similarity function, trust update formula, risk shield and proportional costs. Each arm has its own current exposure, beta, previews, simulated shadow outcome and resulting trust history. Divergent state is a consequence of independent decisions, not an uncontrolled input substitution.

Only Llama `llama3.3:70b` is allowed. Record Ollama version, model digest and stable model metadata before running. Both arms use temperature 0, seed 16061, context budget 8,192 and output budget 512. These explicit token budgets are shared Stage 6.2 transport settings; the old run did not pin both effective defaults. Consequently the old recorded action stream is not used as the treatment arm. Service order alternates each day. The server remains stateless between requests, with all decision context supplied explicitly. Fixed seeds do not guarantee bit-identical inference across hardware or different Ollama builds.

## Clock and scope

An action at decision time t may use completed outcomes with outcome date at or before t, and never the controlled return for t+1. Source audit checks the exact next-day clock and frozen source identity. Trade cost is charged on turnover from the arm's drifted pre-trade BTC weight. The same portfolio carries across every reporting boundary.

Return dates: 2021-01-02 through 2025-05-28. Primary evaluation: 2022-01-01 through 2025-05-28, following fresh 2021 warmup. Both arms use 1,608 decisions; 1,244 are post-2021. No annual economic stop and no year resets. 2015–2020 source history is not agent episodic training. No new sentiment or on-chain information is introduced.

## Accounting and outcomes

For final BTC weight x, drifted pre-trade weight p, next-day BTC return r and cost c=0.001:

`net_return = (1 - c * abs(x - p)) * (1 + x * r) - 1`

`next_pretrade = x * (1 + r) / (1 + x * r)`

Primary daily contrast: `d_t = log1p(memory_net_return_t) - log1p(no_memory_net_return_t)`. Report its post-2021 mean, cumulative log advantage, point estimates and a two-sided 95% percentile interval from 5,000 paired circular-block samples of length 30, seed 16062. The paired indices preserve contemporaneous common-market shocks in both streams. The interval is a descriptive conditional uncertainty estimate from the realized paths. It does not retrain/replay adaptive agents in each bootstrap sample, and stationarity under changing markets is limited.

If the interval lies above zero, label directional positive memory evidence; below zero, negative; otherwise inconclusive. Any invalid completed response in either arm, including warmup, prevents a clean memory-effect label and requests interface interpretation while retaining raw estimates. These labels are not p-values, guarantees or final scientific judgements. Do not switch the primary metric after inspecting this run.

Secondary outcomes: after-cost compounded return, Sharpe using 365-day annualization and zero cash yield, chronological maximum drawdown, daily 95% loss CVaR using exact fractional tail mass, exposure, turnover, cost, action disagreement, visible/cited episodes and trust dynamics. Report all years and regimes, including unfavorable slices. Year/regime findings are exploratory; correlated endpoints and multiple slices do not constitute independent confirmation. Progressively higher returns across years alone cannot establish learning because market difficulty changes.

## Baselines and regime slicing

Cash earns zero interest with zero trading cost; its Sharpe is undefined. B&H starts in cash, incurs one 10-bp entry at the common start and stays fully invested. Neither restarts annually. Fixed daily 45% BTC/55% cash is an explicitly exploratory exposure sensitivity: its weight was motivated by the earlier observed mean exposure. It is not a pre-specified independent benchmark discovered without seeing results.

Bull/Bear/Mix are decision-day router forecasts, checked against router argmax, not realized next-day market labels. Report conditional daily moments and additive log-growth contributions. Do not concatenate disconnected regime days and call the resulting drawdown the true portfolio drawdown. Sharpe and losses should be read with sample counts and conditional scope.

## Preserved limitations

1. The legacy trust rule can reuse its last 60 eligible non-abstain episodes in a new month without fresh evidence. Both arms retain it; report the evidence IDs/dates and any unchanged-evidence beta changes. A fresh-evidence-only rule must be a separate experiment.
2. The imported legacy reference has a 2024 entry-cost seam. Both arms retain the identical reference used in the completed Stage 6.1 run. This experiment does not claim that reference has been repaired; changing it would alter the policy being tested. Cash/B&H are constructed independently and continuously.
3. The original result did not independently pin every upstream base-manifest byte. The new audit records available current source identities and requires reproduction of the original features, costs, base streams and all 4,824 risk previews. This is behavioral compatibility evidence, not recovery of unknowable original bytes.
4. Each decision sees at most five retrieved episodes. Growing storage is not equivalent to effective lifelong learning or useful long-term consolidation.
5. Calendar dates and historical prices are shown to a pretrained model. Causal external inputs do not rule out pretraining knowledge of history. A later date-masked robustness study and genuinely untouched future interval remain necessary.
6. Model digest, options, software versions, traces and hashes aid reproducibility. They do not prove scientific validity, model training cutoff, hardware determinism or novelty.

## Why the project is worth testing, and what remains unproved

Stage 6.1 exhibits a meaningful risk-return trade-off versus B&H, but much of it may follow from lower BTC exposure. Over the full recorded period RAMAS returned 118.8188%, versus 117.9840% for an exploratory static 45/55 allocation; maximum drawdowns were 45.6124% and 45.0466%. Similar performance does not establish statistical equivalence, but it makes memory attribution an essential next test. The research is promising as a testable investigation; present evidence does not establish a distinct economic benefit from the agent or memory.

LLM trading agents with memory have already been studied, including [FinMem](https://arxiv.org/abs/2311.13743) and [FinAgent](https://arxiv.org/abs/2402.18485). The possible contribution here must be specified more narrowly around regime-conditioned trust, bounded exposure, the feedback mechanism and convincing empirical isolation. These two references are a limited novelty sanity check, not a systematic literature-gap audit or a directly comparable performance evaluation.

## Stop and restart semantics

Source mismatch, invalid decision clock, changed model/runtime identity, transport failure or repeated malformed responses are technical reasons to pause. Economic loss is not. Completed responses are saved before they affect state. Resume replays saved responses under identical inputs; no annual reset is introduced. A complete archive includes model identity, all calls, both state histories, raw contrast and secondary outcomes. A synthetic controlled-provider run cannot produce economic evidence.
