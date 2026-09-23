# RAMAS — Technical Report and Supplementary Reference

**Episodic memory in a bounded LLM Bitcoin allocator: system, data, experiments, results, reproduction.**

| | |
|---|---|
| Report version | 1.0, 2026-09-23 |
| Covers | every RAMAS experiment from the Stage 5 lineage (2026-09-01) to Stage 8 (2026-09-22) and the Llama-70B evidence consolidation |
| Written for | the people writing the manuscript's Methods and Supplementary Material, and anyone who wants to inspect, re-analyse or re-run the work |
| Paths | relative to the repository root unless they start with `/`. The pinned absolute root of the original machine is `/home/infonet/wahid/leader_router_fresh`; see §10.1 for how to satisfy it on another machine |
| Numbers | every figure below is read from an archived artifact or recomputed from an archived daily ledger with the frozen metrics module. Where a number was recomputed for this report rather than read from a shipped file, the text says so |

---

## Contents

1. [The work in one page](#1-the-work-in-one-page)
2. [System architecture](#2-system-architecture)
3. [Data](#3-data)
4. [Execution workflow and engineering guarantees](#4-execution-workflow-and-engineering-guarantees)
5. [Baselines, controls and metrics](#5-baselines-controls-and-metrics)
6. [The experiments, one by one](#6-the-experiments-one-by-one)
7. [Consolidated results](#7-consolidated-results)
8. [Observations, interpretation and claim boundaries](#8-observations-interpretation-and-claim-boundaries)
9. [Verification ledger](#9-verification-ledger)
10. [Reproduction guide](#10-reproduction-guide)
11. [Artifact and hash index](#11-artifact-and-hash-index)
12. [Appendices](#12-appendices) — glossary, formulas, supplementary-table recipes

---

## 1. The work in one page

**What RAMAS is.** A deterministic daily Bitcoin allocator (the *numerical core*, inherited unchanged from the RAMoE control: a three-regime posterior, a cash/Bitcoin policy blend, a CVaR/turnover risk projection and 10-bps cost accounting) is extended with a *bounded advisor*: Llama-3.3-70B, served locally, asked once per day for one of `BTC`, `CASH`, `ABSTAIN`. The vote is blended into the core's desired exposure with a small per-regime trust weight β (0.05 by default, never above 0.20) and then re-projected by the same risk layer that governs the core. An **episodic memory** shows the advisor its own completed past decisions in similar market states, each with a stored outcome label. A monthly **trust rule** can move β up or down from those same labels.

**The question.** Does episodic memory change what the advisor decides, do those changes survive blending and risk projection to reach the executed portfolio, and is the net effect after costs distinguishable from zero?

**The answer, in the order the evidence was produced.**

1. The advisor itself adds a large, positive, but statistically inconclusive amount of net return over the numerical core (+6.3 to +9.9 percentage points over 2022–2025 depending on the arm, +17.3 pp when the policy weights are also frozen; every 95% interval includes zero). It does this mainly by *holding more Bitcoin on average*, not by timing (§6.6, §6.11).
2. Memory versus no-memory, holding everything else fixed, was tested **nine times** (Stages 6.2, 6.3, 7 ×2, the 2×2×2 factorial ×4, Stage 8). Eight of nine intervals include zero; the one that does not favours **no**-memory and comes from a sensitivity diagnostic, not a closed loop. Pooled across the first eight realizations the mean effect is **+0.00097 bps/day** with four positive and four negative signs (§7.2).
3. The reason is measurable and does not depend on the model: the memory's retrieval **collapsed**. Because it prefilters to the same hard regime and ranks by a distance dominated by that regime, all retrieved episodes carry the *same* action on **85.9%** of days, so the evidence block can express a contrast between actions on only **13.5%** of days (§6.9.2, §7.5).
4. **Stage 8 repaired that defect** (balanced per-action retrieval, no regime prefilter; contrast expressible on **98.3%** of days) with the identical model, prompt, seed, trust and controller, and the null held: **−0.0123 bps/day, 95% CI [−0.0852, +0.0512]** (§6.9).
5. When memory *does* change a decision, most of the change is absorbed before execution: in the headline factorial pair, advice differed on 38 of 1,244 days, desired exposure on 37, executed exposure on 32; with fixed trust and the controller on, 18 of 30 differences merged at projection. Two independent runs of the identical no-memory configuration differed in the advisor's action on 2 of 1,608 days and in executed exposure on **none** (§7.4, §7.6).
6. Memory has one consistent behavioural signature: in 9 of 9 pairs it makes the model abstain about 1.8 pp more often, hold ~0.1 pp less Bitcoin, and run ~0.05 pp lower annual volatility and ~0.006 pp smaller 95% tail loss — negligible in size, no Sharpe improvement, and usually more turnover and fees (§7.3).
7. In a fresh-seed 2×2×2 factorial, the main effects on 2022–2025 net return were advisor **+6.64 pp**, controller **+3.65 pp**, adaptive-vs-fixed trust **+1.52 pp**, memory **+0.14 pp** — memory an order of magnitude below every other live component (§6.8; note the correction in §6.8.4).

**What this is not.** All results are retrospective, reused-history mechanism diagnostics on one asset and one model. 2024–2025 is not fresh out-of-sample; the regime posterior is a state estimate conditioned on prior-day information, not a validated next-day forecast; decisions use and fill at the same daily close. Every run records these as `claim_boundaries` and this report repeats them in §8.

---

## 2. System architecture

RAMAS is five components in a fixed daily order. Only the advisor and the memory are new; everything else is imported unchanged from the frozen numerical package and verified by hash on every run.

```
                 frozen router stream (q, expert proposals)          raw BTC/USD closes
                               │                                            │
                               ▼                                            ▼
 ┌──────────── 1. numerical core ────────────┐        ┌── causal scenario matrix (past-only windows) ──┐
 │  desired_core = q · W · e                 │        └────────────────────────┬───────────────────────┘
 │  W: per-regime policy weights, monthly    │                                 │
 └───────────────────┬───────────────────────┘                                 │
                     │ base exposure                                           │
      ┌──────────────┴──────────────┐                                          │
      │ 3. episodic memory          │  retrieved episodes + summary            │
      │    (completed episodes only)├──────────┐                               │
      └─────────────────────────────┘          ▼                               │
                        2. advisor (Llama-3.3-70B): BTC / CASH / ABSTAIN       │
                                               │ action                        │
      ┌────────────────────────────────────────┴──────────────┐                │
      │ 4. trust blend: x_desired = (1-β)·base + β·target      │   β per regime │
      └────────────────────────────────────────┬──────────────┘                │
                                               ▼                               ▼
      ┌──────── 5. controller: CVaR / grid / turnover projection from own drifted holdings ────────┐
      └────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                               ▼
                    execute at close[d]  →  earn return close[d]→close[d+1]  →  10-bps cost-before-return
                                               ▼
                    write completed episode (label) → memory; monthly → trust update
```

### 2.1 Numerical core (the "pre-agent RAMoE control")

Source package: `EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/artifacts/20260901T024730Z/base_source/LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_V1/` (28-file `MANIFEST.sha256`, manifest hash `cd672e46ae0874127f9495148641f19fb3c174228ffb7001b65b6903ecc40f24`; `config.json` hash `06b6516a190572d8f7edeb01404a817a931078aed9fe7f4599715cd97d12e6a3`). The RAMAS engine imports `src/accounting.py`, `src/risk.py`, `src/trust.py`, `src/router.py`, `src/simulator.py` from it and verifies the manifest before every run.

| element | value (from `config.json` / `04_CORE_AUDIT.json`) |
|---|---|
| regimes | `bear`, `bull`, `mix`; hard regime = argmax of the daily posterior q |
| regime posterior | `logreg_balanced_fixed` router, columns `prob_bear, prob_bull, prob_mix`; recorded contract "router decision d−1 → allocation decision d → return d+1" |
| expert registry | six slots: `cash, buy_and_hold, trend, volatility_target, drawdown_control, specialist_atp`; production admission = `cash` and `buy_and_hold` only; the other four are structurally masked (exactly zero weight) |
| effective experts | `exposure_cash` ≡ 0 and `exposure_buy_and_hold` ≡ 1 on every day, so the core is a **regime-weighted cash/Bitcoin scalar**, not a six-expert mixture |
| policy weights W | 3 regimes × 6 experts; initial admitted rows `[0.5, 0.5, 0, 0, 0, 0]`; updated once per completed return month from the *independent policy returns* (trust η 0.5, tail penalty 1.0, turnover penalty 0.1, exact posterior-weighted upper-tail estimator, minimum posterior mass 20, minimum effective sample size 20); **never** depends on the advisor or on the final portfolio |
| desired core exposure | `q · W · e` (row-wise), verified against the archived source to `3.3e-16` |
| risk projection | CVaR α 0.95 with limit 0.045 (daily loss), ambiguity quantile 0.9 across scenario models built from past-only windows [30, 90, 252] with 251 quantile samples (minimum history 252 days), exposure grid step 0.05 on [0, 1], maximum daily turnover 0.35; the feasible grid point nearest the desired exposure is chosen, ties → lower exposure |
| transaction cost | 10 bps of turnover, charged before the return: `net = (1 − 0.001·turnover)·(1 + x·r) − 1` |
| cash return | 0 |
| source boundary | the frozen stream has two periods (`corrected_pre2024`, 1,094 rows; `corrected_reused_oos`, 514 rows). The source's *independent* policy holdings reset at that boundary; W and every RAMAS portfolio **carry across** it (`W_resets_at_2024_source_boundary: false`) |

### 2.2 Advisor (Llama-3.3-70B)

| element | value |
|---|---|
| serving | Ollama 0.23.0, local `http://127.0.0.1:11434/api/chat`, model `llama3.3:70b`, digest `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa` (GGUF, 70.6B, Q4_K_M), serving-metadata hash `f731e05070648c6ae023793c866a34064f5d09cd34cd40b37529d50cca9e8d8d` |
| options | `temperature 0.0`, `num_ctx 8192`, `num_predict 512`, `seed 16061` (all runs except the factorial, which uses `seed 42` and `top_p 1.0`) |
| system prompt | `llama70b_memory_expert_v1_frozen` — hash `6b99d1922bf13d280e23c4f62bf82d3dbea2c2032f00be6208a83d8e5bb86783`; text in `RAMAS_STAGE6_4_COMPONENT_SUITE_V1/vendor/stage61/stage6lib/agent.py` (`SYSTEM_PROMPT`). It names the role ("bounded Llama expert"), the three actions and their meaning, forbids inventing memory IDs, and describes the blend/risk layer that follows. The factorial used `model_neutralized_frozen_v1` (hash `569897b2f09b09acda6d3b52645904f7eb7a179bb3751c46b79d51f091c8aedd`): one phrase changed, "bounded Llama expert" → "bounded language-model expert" |
| output contract | JSON schema enforced by Ollama structured output: `action ∈ {BTC, CASH, ABSTAIN}`, `confidence ∈ [0,1]`, `reason_codes` (≥1, from an 11-code enum: `BULLISH_ROUTER, BEARISH_ROUTER, MIXED_OR_UNCERTAIN, POSITIVE_MOMENTUM, NEGATIVE_MOMENTUM, HIGH_VOLATILITY, DRAWDOWN_RISK, TRANSITION_RISK, MEMORY_SUPPORT, MEMORY_WARNING, INSUFFICIENT_EVIDENCE`), `cited_memory_ids` (must be a subset of the episodes actually shown) |
| request payload | `task_type = historical_development_decision`; `objective`; `state` (16 allowlisted fields: `decision_date, target_return_date, prob_bear, prob_bull, prob_mix, hard_regime, base_ramoe_desired_exposure, return_1, return_7, return_30, return_90, realized_vol_30, drawdown_90, router_entropy, router_confidence, router_transition_l1`, plus `llama_trust_beta`); `memory` (retrieved episodes and their summary, §2.4); `safe_exposure_previews` for each of the three actions (blended desired exposure, risk-limited exposure, turnover, ambiguity CVaR); `constraints` (long-only, no leverage, invalid → ABSTAIN) |
| fail-closed | invalid JSON, schema violation, a cited ID that was not shown, or `done_reason = length` ⇒ the day is scored as `ABSTAIN` (i.e. the unchanged core); three consecutive invalid outputs or a valid rate below 0.98 stop the run |
| what the model never sees | the day's return, any future row, any episode whose outcome is not yet complete, the arm's wealth |

### 2.3 Trust blend

`stage6lib/trust.py::blend_exposure`:

```
target(BTC) = 1.0,  target(CASH) = 0.0,  ABSTAIN → x_desired = base
x_desired = (1 − β_regime) · base + β_regime · target
```

Two trust authorities were tested:

| authority | rule |
|---|---|
| **fixed** (`FixedTrust`) | β = 0.05 in every regime, forever |
| **adaptive** (`RegimeTrust.maybe_update`) | β starts at 0.05 per regime, bounds [0.00, 0.20], step 0.025. On the first decision of each new calendar month, for each regime: take the last 60 completed, *non-ABSTAIN* episodes of that regime; if fewer than 8, hold. Else let `mean` = mean stored label and `downside` = its 10th percentile. **Increase** if `mean > 1e-4` and `downside ≥ −0.03`; **decrease** if `mean < 0`; else hold. Every update is written to `trust_events.json` with the eligible episode IDs and whether fresh evidence arrived since the prior update |

Because ABSTAIN episodes are excluded, the trust rule only ever consumes labels from days on which the advisor took a directional view.

### 2.4 Episodic memory

Code: `RAMAS_STAGE6_4_COMPONENT_SUITE_V1/vendor/stage61/stage6lib/memory.py` (frozen design), `RAMAS_STAGE8_CORRECTED_MEMORY_V1/corrected_memory/{retrieval,labels}.py` (Stage 8 corrections).

**Episode record** (one per day, appended after the day's return is known): `episode_id, decision_date, return_date, hard_regime, state_vector, action, confidence, shadow_log_advantage_vs_ramoe, asset_return`.

**State vector** (9 dimensions, each divided by a fixed scale): `prob_bear, prob_bull, prob_mix, router_entropy` (scale 1), `router_transition_l1` (2), `return_7` (0.5), `return_30` (1), `realized_vol_30` (2), `drawdown_90` (1).

**Eligibility** (`stage63lib/engine.py::eligible_episodes`): an episode is visible on decision day *d* only if `return_date ≤ d` **and** `decision_date < d`. Episodes are appended after their return completes, so nothing about the current day or the future is ever retrievable. Every run's eligibility audit reports zero violations (§9).

**Frozen retrieval** (`EpisodicMemory.retrieve`): keep only eligible episodes whose `hard_regime` equals today's; rank by Euclidean distance between state vectors (ties by `episode_id`); show the nearest **5**.

**Evidence block shown to the model** (`evidence_summary`): the list of retrieved episodes (id, date, regime, action, confidence, label, asset return, distance) plus a per-action summary `{count, mean_log_advantage_vs_ramoe, positive_fraction}`; or the literal `NO_COMPLETED_SIMILAR_EPISODES`.

**Stored label** (frozen): `shadow_log_advantage_vs_ramoe = log1p(shadow_net) − log1p(reference_net)`, where the *shadow* portfolio executes the advisor's action at full authority (β = 1, ABSTAIN → β = 0) along its own holdings path, and the *reference* is the archived legacy RAMoE portfolio. This label is **action-conditional**: BTC scores positive on up days, CASH positive on down days; its correct-sign rate conditional on the action taken is 99.44% on BTC/CASH days (§6.9.4).

**Retrieval variants tested** (Stage 6.4): `expanding` (frozen design); `frozen2021` (only episodes with outcomes ≤ 2021-12-31); `current_year` (only episodes completed in the current calendar year); `pooled` (frozen ranking and budget but **without** the same-regime prefilter). Trust keeps learning from the full private ledger in every variant, and "no memory" removes *visibility only* — the private ledger still fills (1,607 episodes by the last day) and adaptive trust still learns from it.

**Stage 8 corrections** (both behind explicit spec fields, §6.9): `balanced` retrieval = the nearest **2 per action** (max 6 shown), prefilter removed (`regime_filter: none`); `counterfactual` label = `log1p(net(x_action)) − log1p(net(x_core))` with both exposures projected from the arm's **own** pre-trade holdings, ABSTAIN = exactly 0.

### 2.5 Controller (risk projection)

`stage63lib/legacy.py::project_action` → `src/risk.py::project_exposure` with the parameters of §2.1, applied to the *blended* desired exposure and the arm's *own drifted* pre-trade holdings (holdings drift with the asset return between days). "Controller off" (`risk: none`, factorial and `*_no_risk` arms) removes the CVaR constraint, the 0.05 grid and the 0.35 turnover cap together; long-only [0,1] and the 10-bps cost remain. The three safe-exposure previews in the prompt are recomputed the same way, so the model always sees what each action would actually execute to.

### 2.6 The daily loop (what `suite64/engine.py::run_arm` does for day *i*)

1. If a new calendar month has started: adaptive trust update from the eligible completed episodes (fixed trust: no-op).
2. Build `state` from the frozen stream row *i*, the arm's pre-trade exposure, the core's desired exposure and β for today's hard regime.
3. Retrieve memory evidence (per the arm's memory mode) from eligible episodes; record `retrieved_memory_count` and (Stage 8) `retrieved_action_diversity`.
4. Compute the three safe-exposure previews by projecting each action.
5. Build the payload; compute its digest; **journal** the request; call the model (or the deterministic rule, or return ABSTAIN for numerical arms); journal the raw response *before* scoring anything.
6. Validate the decision (fail-closed to ABSTAIN).
7. Blend, project, execute: `exposure_i`, `turnover_i`, `cost_fraction_i`.
8. Earn `asset_simple_return_i` (close[d] → close[d+1]); `net_i = (1 − cost_i)(1 + exposure_i·r_i) − 1`; update wealth; compute the shadow portfolio and the label.
9. Append the completed episode; drift holdings to the next pre-trade exposure `x·(1+r)/(1+x·r)`.
10. Every 25 days write a checkpoint; every arm ends with `RUN_COMPLETE.json` listing the hash of each output file.

### 2.7 Accounting and metric conventions (`suite64/metrics.py`, `METRIC_DEFINITIONS.md`)

| quantity | definition |
|---|---|
| headline | **net compounded return after trading costs** = `expm1(Σ log1p(net_i))` |
| additive form | daily net log return `log1p(net_i)`; contrasts are reported as mean daily difference in **bps** (×10⁴) |
| volatility, Sharpe | sample std (ddof = 1) of daily net simple returns × √365.25; Sharpe with rf = 0 |
| Sortino | mean / `sqrt(mean(min(net,0)²))` over **all** days, × √365.25 |
| maximum drawdown | on the wealth path including wealth = 1 before the first return |
| ES95 | exact fractional-weight mean of the worst 5% of daily simple returns, reported as a loss |
| CAGR | only when ≥ 365 calendar days |
| turnover, cost | `Σ turnover_i`; `Σ cost_fraction_i` (descriptive, not compounded drag) |
| paired uncertainty | **circular moving-block bootstrap** of the paired daily net-log-return difference: block 30 days, 5,000 resamples, `numpy.default_rng(16062)`, percentile 95% interval, **unadjusted** for multiple comparisons. Interpretation labels: `INCONCLUSIVE_NOT_EQUIVALENCE` when the interval covers zero |
| undefined ratios | cash Sharpe/Sortino/Calmar are blank, not zero |

The accounting identity `wealth_after = wealth_before · (1 + net)` and the cost formula were re-verified on every ledger in the project: maximum absolute error `≤ 2.2e-16` across 33,768 + 12,864 + 6,432 + 3,216 rows (§9).

---

## 3. Data

### 3.1 Raw market data

| item | value |
|---|---|
| file | `data/raw/full_data_set.csv` (pinned original path `/home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv`) |
| sha256 | `b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e` — every package verifies it before loading |
| shape | 3,804 daily rows, 187 columns; `Date` 2015-01-01 → 2025-05-31; `Close` 164.9 → 111,702.7 USD |
| content | daily BTC/USD OHLCV plus derived statistical features (entropy, kurtosis, MAD, …); RAMAS uses the **Close** series only, to build the causal scenario matrix for the risk projection (past-only windows of 30/90/252 days ending at the decision day; a 252-day minimum history is why the file must start well before 2021) |
| guard | `build_causal_market_features` / `build_causal_scenarios` slice `close[:index+1]` and assert at runtime that no later row is touched |

### 3.2 Frozen router / control stream (the numerical core's inputs)

`EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/artifacts/20260901T024730Z/results/{corrected_pre2024,corrected_reused_oos}/02_CORRECTED_INPUTS.csv`

| column | meaning |
|---|---|
| `decision_date`, `return_date` | the allocation day *d* and the day *d+1* whose close-to-close return is earned |
| `prob_bear, prob_bull, prob_mix` | the router posterior available on *d* |
| `canonical_asset_simple_return` | `close[d+1]/close[d] − 1` |
| `exposure_cash … exposure_specialist_atp` | the six expert proposals for *d* (cash ≡ 0, buy-and-hold ≡ 1; the other four are recorded but masked) |

| period | rows | decision dates | return dates | scientific label |
|---|---:|---|---|---|
| `corrected_pre2024` | 1,094 | 2021-01-01 → 2023-12-30 | 2021-01-02 → 2023-12-31 | development and walk-forward evidence |
| `corrected_reused_oos` | 514 | 2023-12-31 → 2025-05-27 | 2024-01-01 → 2025-05-28 | **reused** out-of-sample diagnostic, not untouched confirmation |
| total | **1,608** | | | |

Each run's `01_SOURCE_AUDIT.json` records: 1,608 archived requests and 1,608 response digests verified, 4,824 risk previews reconstructed, maximum accounting difference `1.96e-16`, maximum risk-preview difference `2.22e-16`, maximum state difference `0.0`.

### 3.3 Evaluation windows

| window | return dates | days | use |
|---|---|---:|---|
| FULL | 2021-01-02 → 2025-05-28 | 1,608 | every arm replays all of it; 2021 is the **burn-in** during which memory fills and trust first moves |
| **POST2021** (primary) | 2022-01-01 → 2025-05-28 | **1,244** | every headline contrast, table and bootstrap |
| yearly / monthly / regime | | | descriptive only; regime-conditional selections carry no annualized statistics |

Portfolios, memory and trust are **never** reset at year or period boundaries.

### 3.4 Known data-side limits (disclosed in every run contract)

- The router posterior "targets the allocation day, not the subsequent holding day" and its fit chronology is not independently verified (`03_CLOCK_REFERENCE_AUDIT.json`, `holding_period_target_alignment_verified: false`). Empirically it is a *degraded* signal, not look-ahead: it correlates +0.136 / +0.124 / +0.082 with the returns 3/2/1 days *before* the decision and +0.047 with the return it is scored on. Describe it as a regime **state estimate**, not a forecast.
- Decisions use `close[d]` and fill at `close[d]`; inference latency (median 6–8 s per call, §6.5) is not modelled.
- 2024–2025 was seen by the source pipeline before; it is reused diagnostic history.

---

## 4. Execution workflow and engineering guarantees

### 4.1 Pinned runtime

```
python 3.11.9   numpy 1.26.4   pandas 2.2.3        (/home/infonet/anaconda3/envs/wahid_test/bin/python)
Ollama 0.23.0   llama3.3:70b   digest a6eb4748…    temperature 0
```

The request digest that keys every journal entry includes the retrieval distances. A different numpy/BLAS build changes the last bit of a Euclidean norm, which changes the digest, which makes every archived call look "new". The system default `python3` on the original machine (3.11.7 / pandas 2.1.4 / MKL) does **not** reproduce the archives; Stage 8 and the consolidation refuse to start on the wrong interpreter (`assert_pinned_runtime`).

### 4.2 Durable journal and resume

Every model call is written as its own JSON file under `<arm>/calls/NNNNNN-<arm>.json` (request, request digest, raw response, latency, provider identity) **before** the decision is scored. A restart replays the journal from day 1: the engine recomputes each request, looks its digest up, and refuses to continue if the recomputed payload diverges from the archived one. Already-saved responses are never resampled; an in-flight response lost in a crash is the only thing that is re-requested. This is what makes the arms bit-reproducible and what let Stage 8 prove its engine is byte-faithful to the frozen design before spending GPU time (§6.9.1).

### 4.3 Gates before any model call

1. `sha256sum -c PACKAGE_MANIFEST.sha256` on the executable package (every launcher).
2. Unit tests (Stage 6.4: 31; factorial: 31 self-tests + 8; Stage 8: 7 checks; vendored Stage 6.1 agent: 17).
3. Source audit: raw-data hash, frozen-stream hashes, archived request/response digests, risk-preview and accounting reconstruction to `1e-16`.
4. Core audit: W reconstruction and desired-exposure equality with the archived control (`3.3e-16`).
5. Provider identity: model name, digest, Ollama version, options, serving metadata; re-checked every 100 calls in the factorial.
6. Semantic preflight (Stages 6, 6.1, 6.2, 6.3): 12 frozen decision contexts with unambiguous expected actions must pass at valid rate 1.0 and expected-action rate ≥ 0.9.

### 4.4 Stop rules

Technical or source failures stop **fail-closed** with evidence retained (`FAILURE.json`, `STOP_TECHNICAL_OR_SOURCE`). Economic loss never stops a run (`no_yearly_economic_hard_stop: true`) — losing years are findings, not aborts.

### 4.5 Output contract of a run directory

`00_*CONTRACT.json` (identity: config hash, package manifest hash, provider identity, runtime, claim boundaries) → `01_SOURCE_AUDIT.json` → `02_CORE_AUDIT.json` (or clock audit) → per-arm directories (`DAILY_LEDGER.csv`, `episodes.json`, `trust_events.json`, `ARM_STATE.json`, `PROGRESS.json`, `calls/`, `RUN_COMPLETE.json`) → consolidated `daily_ledger.csv` / `all_metrics.csv` / `COMPARISONS.json` / `*_TRANSMISSION.csv` → `FINAL_STATUS.json` → `RUN_COMPLETE.json` with the sha256 of every file it lists.

---

## 5. Baselines, controls and metrics

| arm / benchmark | what it is | why it is there |
|---|---|---|
| `cash` | 0% Bitcoin, 0 return | floor |
| `buy_hold` | 100% Bitcoin from day 1, one initial fee | external ceiling for return; shows the risk RAMAS gives up |
| `static50` | 50% Bitcoin rebalanced daily, own 10-bps costs | matched-exposure external benchmark for the ~45% average exposure of RAMAS |
| `STATIC45_REBALANCED_SECONDARY` (6.2/6.3 only) | 45% rebalanced daily | closer exposure match in the early runs |
| `numerical_only` | advisor forced to ABSTAIN, β = 0, memory none, controller on | the **pre-agent RAMoE control**: the core alone with continuous holdings |
| `numerical_no_risk`, `numerical_hard_allocator`, `numerical_uniform_allocator`, `numerical_frozen_W` | the core with one mechanism changed (controller off; q → one-hot argmax; q → [⅓,⅓,⅓]; W frozen at its initial rows) | isolates the allocator and controller without any LLM |
| `rule_fixed`, `rule_adaptive` | a **deterministic rule advisor** (`deterministic_action`: CASH if `prob_bear ≥ 0.60` or `drawdown_90 ≤ −0.25` or `realized_vol_30 ≥ 1.50`; BTC if `prob_bull ≥ 0.70` and `return_30 > 0` and `realized_vol_30 ≤ 1.10`; else ABSTAIN) blended and projected exactly like the LLM, with fixed or adaptive β | answers "is the *LLM* doing anything a transparent rule on the same information would not?" |
| `llm_* / *_no_memory` | the LLM advisor with memory visibility removed | the memory contrast |
| `llama_*` component arms | the full adaptive-memory system with one mechanism changed | attribution of the integrated result |

Metric conventions are in §2.7. The **primary** endpoint of every memory experiment is the paired mean daily net-log-return difference, memory minus no-memory, over POST2021, with the block bootstrap of §2.7; compounded-return gaps in percentage points are reported alongside as the economically readable form.

---

## 6. The experiments, one by one

### 6.0 Programme overview

Every completed run is listed. "Fresh calls" are model calls made in that run (each is journalled). Hashes are sha256; full values are in §11.

| # | stage | package / run id | completed (UTC) | model | fresh calls | question | outcome |
|---|---|---|---|---|---:|---|---|
| L1 | 5 | `RAMOE_STAGE5_SINGLE_LLM_AGENT_QWEN3_CODER_30B_V1` / `20260901T085240Z`, `20260901T085405Z` | 2026-09-01 | qwen3-coder:30b | 1,094 | can a single LLM agent set daily target exposure? | all 1,094 development actions failed contract validation → portfolio copied the control; `REVISE_AGENT_DEVELOPMENT_GATE_FAILED` |
| L2 | 5.1 | `RAMOE_STAGE5_1_SINGLE_LLM_AGENT_INTERFACE_REPAIR_V1` / `20260902T004332Z` | 2026-09-02 | qwen3-coder:30b | 1,094 (+30 preflight) | same, with a schema-enforced interface | interface valid; development gate failed |
| L3 | 5.2 | `RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1` / `20260902T045940Z` | 2026-09-02 | none (audit) | 0 | did the 5.1 agent add value over a transparent regime rule? | no: −0.088 terminal growth, −0.086 Sharpe, one-sided p 0.82; memory gate failed → `REDESIGN_AS_TOOL_USING_RESIDUAL_AGENT_NO_OOS` |
| L4 | 5.3 | `RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1` / `20260902T054019Z` | 2026-09-02 | qwen3-coder:30b | 298 | bounded tool-using residual (±0.25) agent, called on 149 uncertain days | 89 non-zero proposals, all zeroed by the evidence gate → identical to the rule; `STOP_AGENT_ECONOMIC_BRANCH_DEVELOPMENT_GATE_FAILED` |
| L5 | 5.4 | `RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1` / `20260902T063436Z` | 2026-09-02 | llama3.3:70b | 30 | does Llama-70B follow the 5.3 tool-citation contract better than Qwen? | JSON valid 100%, semantic accept 96.7%, but full-citation rate 0% → `LLAMA70B_SEMANTIC_INTERFACE_FAILED` |
| L6 | 5.5 | `RAMAS_STAGE5_5_EVENT_CONDITIONED_TRUST_AGENT_V1` / `20260902T082413Z` | 2026-09-02 | llama3.3:70b | preflight only | event-triggered agent that may propose a bounded, temporary change to one trust-matrix row | mechanism tests pass; economic replay **not run**: `MECHANISM_PASS_AWAIT_POINT_IN_TIME_EVENT_DATA` |
| L7 | 5.5.1 | `RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1` / `20260902T085752Z` | 2026-09-02 | llama3.3:70b | preflight only | two-round tool loop with a KEEP fallback | interface preflight failed (the model's safe `KEEP` answers cannot be told from weakness) → `REVISE` |
| 1 | 6 | `RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1` / `20260903T004539Z` | 2026-09-03 09:03 | llama3.3:70b | 364 (+12) | **the RAMAS design**: bounded BTC/CASH/ABSTAIN advisor + trust + memory, 2021 pilot | pipeline valid; terminal growth above control (+0.289 vs +0.270) but not above the deterministic controller, Sharpe/bootstrap checks fail → `REVISE_…_GATE_FAILED_NO_POST2021` |
| 2 | 6.1 | `RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1` / `20260907T012534Z` | 2026-09-08 04:04 | llama3.3:70b | 1,608 (+12) | continuous 2021–2025 prequential replay, no resets | complete; full-horizon growth 1.188 vs control 1.023; economic value vs control **not established**; needs matched ablations |
| 3 | 6.2 | `RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1` / `20260908T055052711433762Z` | 2026-09-08 12:29 | llama3.3:70b | 3,216 (+12) | memory vs no-memory, adaptive trust, both arms fresh | −0.0017 bps/day, CI [−0.1265, +0.0992]; inconclusive |
| 4 | 6.3 | `RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1` / `20260909T013716356200839Z` | 2026-09-09 08:14 | llama3.3:70b | 3,216 (+12) | memory vs no-memory, **fixed** β = 0.05 | +0.0369 bps/day, CI [−0.0340, +0.1221]; inconclusive |
| 5 | 6.4 | `RAMAS_STAGE6_4_COMPONENT_SUITE_V1` / `20260909T114202240674964Z` | 2026-09-10 13:24 | llama3.3:70b | 11,256 | 21-arm component map: benchmarks, rule advisor, allocator, controller, W, retrieval variants | complete; primary `fixed_memory − rule_fixed` +0.0048 bps/day [−0.0633, +0.0755]; 21 of 22 intervals cover zero |
| 6 | 7 | `RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1` / `20260911T053155161422662Z` | 2026-09-11 17:58 | llama3.3:70b | 6,432 | closed-loop replication of the fixed-β pair + a state-controlled sensitivity pair | closed loop +0.0543 [−0.0256, +0.1521]; state-controlled **−0.1075 [−0.2335, −0.0042]** (no-memory higher) |
| 7 | ISO | `RAMAS_FULL_PIPELINE_ISOLATION_V1` / `20260920T084300Z` | 2026-09-21 11:14 (llama cell) | llama3.3:70b, **seed 42**, neutral prompt | 12,864 | 2×2×2 factorial memory × trust × controller | main effects advisor +6.64, controller +3.65, trust +1.52, memory +0.14 pp; cross-model extension aborted fail-closed on glm-4.7-flash |
| 8 | 8 | `RAMAS_STAGE8_CORRECTED_MEMORY_V1` / `20260922T014916Z` | 2026-09-22 07:49 | llama3.3:70b | 3,216 | repaired retrieval (+ new label), fixed β, controller on | retrieval contrast 13.5% → 98.3%; **−0.0123 bps/day [−0.0852, +0.0512]**; inconclusive |
| C | — | `RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1` / `20260921T000000Z` | 2026-09-22 01:49 | none (read-only) | 0 | all 24 Llama-70B arms in one table; 8 contrasts; pooled effect; replication | bit-reproducible (7/7 outputs identical on re-run) |
| — | — | `RAMAS_MEMORY_CAUSAL_EXPERIMENT` (SMOKE only, 2026-09-15) | — | — | 0 | frozen-state, same-prompt memory intervention (4,976 calls planned) | **designed and smoke-tested, never executed** |

### 6.1 Stage 5 lineage (before the RAMAS design existed)

These seven runs are the audit trail that led to the bounded-advisor design. They tested LLM agents that *set exposure directly* (5, 5.1), a *residual* on top of a transparent rule (5.3), and an agent that edits the *trust matrix* (5.5, 5.5.1). None established incremental value; two failed on interface compliance. Their decision files are the record:

| stage | decision file | key figures |
|---|---|---|
| 5 | `EXPERIMENT_BRANCHES/RAMOE_STAGE5_SINGLE_LLM_AGENT_QWEN3_CODER_30B_V1/artifacts/20260901T085405Z/05_FINAL_DECISION.json` | undeclared reason codes, unseen citations and off-grid exposures invalidated every one of 1,094 actions |
| 5.1 | `…/RAMOE_STAGE5_1_SINGLE_LLM_AGENT_INTERFACE_REPAIR_V1/artifacts/20260902T004332Z/results/05_FINAL_DECISION.json` | `agent_interface_valid: true`, `pre2024_development_gate_passed: false` |
| 5.2 | `…/RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1/artifacts/20260902T045940Z/{06_BEHAVIOR_AND_MEMORY_AUDIT,08_INCREMENTAL_AGENT_GATE,09_FINAL_DECISION}.json` | the agent deviated from the static regime map on 74 of 1,094 days (35 better); agent − static: terminal growth −0.0878, Sharpe −0.0857, mean log advantage −5.2e-5/day, one-sided block-bootstrap p 0.82; lesson memory: 5 promoted, 505 quarantined, 5 rolled back, **0** active lessons changed an action |
| 5.3 | `…/RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1/artifacts/20260902T054019Z/results/{05_DEVELOPMENT_GATE,11_FINAL_DECISION}.json` | rule default exposure bear 0.25 / bull 0.50 / mix 0.25; agent triggered on 149 days (router transitioning or confidence < 0.80), 298 calls, 149 valid plans and decisions, 89 non-zero residual proposals — every one zeroed because the four required tools were not all executed and cited → 0 non-zero residual days; p = 1.0 |
| 5.4 | `…/RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1/artifacts/20260902T063436Z/03_SUMMARY.json` | 30 frozen contexts (1 eligible non-zero, 19 ineligible non-zero, 10 abstain): valid JSON 1.0, expected-action agreement 1.0, semantic accept 0.967, non-zero *fully cited* responses 0.0, the eligible case not accepted; 2,847 s |
| 5.5 | `…/RAMAS_STAGE5_5_EVENT_CONDITIONED_TRUST_AGENT_V1/artifacts/20260902T082413Z/03_FINAL_DECISION.json` | `interface_preflight_passed: true`, `economic_replay_run: false`, `point_in_time_event_data_gate_passed: false` |
| 5.5.1 | `…/RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1/artifacts/20260902T085752Z/03_FINAL_DECISION.json` | `interface_preflight_passed: false` |

Lesson carried into Stage 6: do not let the model output exposures or edit trust; give it one bounded vote, blend it with a small β, keep the risk layer as final authority, and separate *mechanism* tests from *economic* tests.

### 6.2 Stage 6 — the design, 2021 development pilot

- **Package** `RAMAS/code/RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1/` (`run_stage6.py`, `stage6lib/`); code archive sha `92cb044f…`; run tarball `f1174b96…`.
- **Design** as in §2 with adaptive trust and expanding memory, replayed over the first 364 rows (return dates 2021-01-02 → 2021-12-31) only; later rows were hashed but not read. Semantic preflight (12 contexts) passed first: `semantic_preflight/RUN_COMPLETE.json` → `CONTINUE_TO_2021_DEVELOPMENT_PILOT`.
- **Variants** (`development_2021/04_VARIANT_METRICS.csv`):

| variant | terminal growth | Sharpe | max DD | CVaR95 | mean exposure |
|---|---:|---:|---:|---:|---:|
| `ramas_base` (pre-agent control) | 0.2702 | 0.8383 | 0.2841 | 0.0422 | 0.4533 |
| `ramas_llama70b_adaptive_memory` | **0.2895** | 0.8822 | 0.2732 | 0.0415 | 0.4525 |
| `ramas_llama70b_frozen_trust` | 0.2882 | 0.8779 | 0.2731 | 0.0416 | 0.4552 |
| `deterministic_same_information` | 0.2913 | 0.8852 | 0.2732 | 0.0415 | 0.4540 |
| `llama70b_shadow_only` (β = 1) | 0.2313 | 0.7872 | 0.1812 | 0.0419 | 0.3345 |
| `buy_and_hold` | 0.5727 | 0.9637 | 0.5311 | 0.0903 | 1.0000 |

- **Gate** (`06_DEVELOPMENT_GATE.json`): passed terminal growth above control, drawdown and CVaR not worse, valid-action rate 1.0, 183 non-ABSTAIN actions; **failed** growth above the deterministic controller, Sharpe improvement, and positive one-sided bootstrap lower bound (mean log advantage 4.1e-5/day, p = 0.279, 5,000 blocks of 30, seed 16061). Decision `REVISE_STAGE6_2021_DEVELOPMENT_GATE_FAILED_NO_POST2021`.
- **Run**: `nohup bash RAMAS/code/RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1/run_server.sh > RAMAS_STAGE6_RUN.log 2>&1 &` then `tail -f RAMAS_STAGE6_RUN.log`; output `EXPERIMENT_BRANCHES/RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1/artifacts/<run_id>/{semantic_preflight,development_2021}/`.

### 6.3 Stage 6.1 — continuous 2021–2025 prequential replay

- **Package** `RAMAS/code/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/` (code `b253e921…`; run tarball `8a273d9f…`; config hash `dace37cf…`). The vendored copy under `RAMAS_STAGE6_4_COMPONENT_SUITE_V1/vendor/stage61/` is the agent, memory and trust code every later stage imports.
- **Design**: the Stage 6 agent, unchanged (its accepted config `ce47a92c…` and prompt `6b99d192…` are pinned), run over all 1,608 rows with `year_end_action: REPORT_AND_CONTINUE_WITHOUT_RESET`; 2021 is burn-in, 2022–2025 `REUSED_OOS_DIAGNOSTIC`. Internal control `pre_agent_ramoe`; external baselines deferred to Stage 6.4.
- **Results** (`continuous_2021_2025/04_FULL_HORIZON_METRICS.csv`, FULL 1,608 days):

| variant | terminal growth | Sharpe | max DD | CVaR95 | mean exposure | mean daily turnover |
|---|---:|---:|---:|---:|---:|---:|
| `ramas_full_continuous_llama70b_memory_trust` | **1.1882** | 0.7831 | 0.4561 | 0.0330 | 0.4504 | 0.0098 |
| `pre_agent_ramoe_internal_control` | 1.0231 | 0.7234 | 0.4660 | 0.0329 | 0.4432 | 0.0079 |
| `llama70b_frozen_agent_trust_ablation` | 1.1602 | 0.7777 | 0.4492 | 0.0325 | 0.4468 | 0.0096 |
| `deterministic_same_information_ablation` | 1.1541 | 0.7704 | 0.4570 | 0.0330 | 0.4507 | 0.0095 |
| `llama70b_unblended_shadow_diagnostic` (β = 1) | 1.7340 | 0.9623 | 0.2963 | 0.0336 | 0.3907 | 0.0296 |
| `buy_and_hold_external_reference` | 2.6676 | 0.7881 | 0.7664 | 0.0728 | 1.0000 | 0.0006 |

Yearly (`05_YEARLY_METRICS.csv`): 2021 RAMAS +0.2895 vs control +0.2702; 2022 −0.3458 vs −0.3538 (both Sharpe ≈ −1.41). Post-2021 compounded return of the RAMAS arm, recomputed by the consolidation: **69.687%** (Sharpe 0.7586, MDD 37.09%).
- **Decision** (`11_FINAL_RESEARCH_STATUS.json`): `CONTINUOUS_PIPELINE_COMPLETE_REQUIRES_SCIENTIFIC_INTERPRETATION`; `technical_pipeline_valid: true`; `full_horizon_economic_value_vs_pre_agent_ramoe_established: false` (single realization, no matched ablation, no uncertainty). The +16.5 pp gap over the control motivated the matched ablations that follow.
- **Run**: `nohup bash RAMAS/code/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/run_server.sh > RAMAS_STAGE6_1_RUN.log 2>&1 &`; output `EXPERIMENT_BRANCHES/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/artifacts/<run_id>/continuous_2021_2025/` (`03_DAILY_TRACE.csv` sha `c645ed50…`, `08_LLM_CALLS.jsonl` 1,608 lines, `09_MEMORY_EPISODES.jsonl`, `10_TRUST_STATE.json`).

### 6.4 Stage 6.2 — matched memory ablation, adaptive trust

- **Package** `RAMAS/code/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/` (code `1779f47b…`; package manifest `d8ee5ef5…`; config hash `64f8a64c…`; run tarball `e9296f2f…`).
- **Design**: two arms, `memory` and `no_memory`, both started fresh and empty on 2021-01-01, same model/seed/prompt, adaptive trust in both; `no_memory_disables: RETRIEVED_EPISODES_AND_THEIR_SUMMARY_ONLY` — the private ledger and the trust rule keep working. 12-case semantic preflight, then 3,216 fresh calls. Primary: `POST2021_MEAN_DAILY_LOG_RETURN_MEMORY_MINUS_NO_MEMORY`, block bootstrap seed 16062.
- **Results** (`06_PAIRED_MEMORY_CONTRAST.json`, `04_FULL_AND_YEARLY_METRICS.csv`):

| | memory | no-memory | Δ |
|---|---:|---:|---:|
| POST2021 net return | 67.7304% | 67.7655% | −0.035 pp |
| POST2021 annual vol / Sharpe | 24.354% / 0.7452 | 24.374% / 0.7450 | −0.020 pp / +0.0002 |
| POST2021 max DD / ES95 | 37.087% / 2.9233% | 37.087% / 2.9233% | 0 / 0 |
| FULL cumulative return | 116.30% | 116.34% | |
| paired contrast | **−0.001682 bps/day, 95% CI [−0.126484, +0.099171]**, 1,244 days | | `INCONCLUSIVE` |

Both arms: 1,608/1,608 valid actions, 0 fallback days. Status `MATCHED_MEMORY_EXPERIMENT_COMPLETE_REQUIRES_INTERPRETATION`.
- **Run**: `bash RAMAS/code/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/start.sh` (prints `LOG=` and `FOLLOW_LOG=RAMAS_STAGE6_2_RUN.log`); `tail -f RAMAS_STAGE6_2_RUN.log`; output `EXPERIMENT_BRANCHES/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/artifacts/<run_id>/` with `calls/NNNNNN-{memory,no_memory}.json`, `episodes/`, `trust/`.

### 6.5 Stage 6.3 — matched memory ablation, fixed trust β = 0.05

- **Package** `RAMAS/code/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/` (code `ebbfb637…`; run tarball `b44d65ca…`). Its `stage63lib/` is what Stage 6.4 later froze.
- **Design**: Stage 6.2 with `agent_trust_mode: FIXED_POSITIVE`, β = 0.05 in every regime. Fixed trust removes the trust rule as a second consumer of the memory label, so the two arms differ only in what the model can see.
- **Results** (`06_PAIRED_MEMORY_CONTRAST.json`, `04_FULL_AND_YEARLY_METRICS.csv`, `14_COMPUTATIONAL_DIAGNOSTICS.json`):

| | memory | no-memory | Δ |
|---|---:|---:|---:|
| POST2021 net return | 68.6247% | 67.8521% | +0.773 pp |
| POST2021 annual vol / Sharpe / Sortino | 23.926% / 0.7607 / 1.1321 | 23.959% / 0.7544 / 1.1221 | −0.033 pp / +0.0063 / +0.0100 |
| POST2021 max DD / ES95 | 36.253% / 2.8572% | 36.253% / 2.8619% | 0 / −0.0047 pp |
| FULL turnover / cost (initial-wealth units) | 15.406 / 0.0202 | 14.254 / 0.0186 | +1.152 / +0.0016 |
| prompt tokens / output tokens (1,608 calls) | 2,267,738 / 91,681 | 1,440,327 / 69,150 | |
| latency median / p95 | 7.79 s / 10.52 s | 6.01 s / 6.26 s | |
| paired contrast | **+0.036917 bps/day, 95% CI [−0.033995, +0.122050]** | | `INCONCLUSIVE` |
| memory × trust interaction (`13_MEMORY_TRUST_INTERACTION.json`): (adaptive gap) − (fixed gap) | −0.0386 bps/day, CI [−0.1846, +0.0897] | | secondary |

- **Run**: `bash RAMAS/code/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/start.sh`; `tail -f RAMAS_STAGE6_3_RUN.log`; output `EXPERIMENT_BRANCHES/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/artifacts/<run_id>/`.

### 6.6 Stage 6.4 — the component suite (21 arms, 22 contrasts)

- **Package** `RAMAS_STAGE6_4_COMPONENT_SUITE_V1/` at the repository root — the **frozen engine** (`run_suite.py`, `suite64/{engine,core_variants,metrics,archive}.py`, `stage63lib/`, `vendor/stage61/`); `PACKAGE_MANIFEST.sha256` hash `6a03c967673f6f2e98fd8134910a9a8231d0b9fb578bc1aa2615c9c18bf9da0b` (45 files); code archive `52698118…`; run tarball `c84940f5…`. `SCIENTIFIC_PROTOCOL.md` and `METRIC_DEFINITIONS.md` are the protocol and formula references.
- **Design** (`00_CONTRACT.json`): reuse the four archived Stage 6.2/6.3 trajectories (verified by hash: 3,244 and 3,249 artifacts) → recompute cash / B&H / static50 on the identical days → reconstruct the numerical core (W, desired exposures, previews) → run **14 new arms** in a fixed order, never stopping for losses. Seven need the model (7 × 1,608 = 11,256 calls); seven are deterministic. Primary contrast `fixed_memory − rule_fixed`; everything else is secondary/exploratory and reported in full. Scope label: `LEGACY_MECHANISM_DIAGNOSTIC_NOT_REALISTIC_EXECUTION_OR_FRESH_OOS_CONFIRMATION`.
- **The 14 new arms** (`advisor / core / memory / risk / trust`): `rule_fixed`, `rule_adaptive` (rule advisor, original core, fixed/adaptive β); `numerical_only` (advisor ABSTAIN); `numerical_hard_allocator`, `numerical_uniform_allocator`, `numerical_frozen_W`, `numerical_no_risk`; `llama_hard_allocator`, `llama_uniform_allocator`, `llama_frozen_W`, `llama_no_risk` (full adaptive-memory advisor with the corresponding core/controller change); `llama_memory_frozen2021`, `llama_memory_current_year`, `llama_memory_pooled` (retrieval variants, §2.4). `hard_allocator` replaces q by one-hot(argmax q) *only* inside q·W·e; `uniform_allocator` uses q = [⅓,⅓,⅓] there; `frozen_W` keeps the initial rows; in all three the prompt, regime selection and retrieval still see the original q.
- **Results — POST2021, 1,244 days** (`full_period_metrics.csv`, scope `POST2021`; return, vol, MDD, ES95 as fractions in the file, shown here in %):

| arm | net return % | ann. vol % | Sharpe | Sortino | max DD % | ES95 % | turnover Σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `buy_hold` | 133.2061 | 53.92 | 0.7306 | 1.0814 | 66.95 | 6.457 | 0.000 |
| `llama_frozen_W` | 77.3842 | 26.78 | 0.7621 | 1.1331 | 39.70 | 3.197 | 8.839 |
| `static50` | 71.7921 | 26.96 | 0.7240 | 1.0714 | 40.21 | 3.229 | 5.999 |
| `numerical_frozen_W` | 70.1940 | 26.35 | 0.7242 | 1.0690 | 40.28 | 3.166 | 6.374 |
| `llama_memory_frozen2021` | 69.9025 | 24.37 | 0.7603 | 1.1309 | 37.09 | 2.922 | 11.615 |
| `fixed_memory` (= 6.3 memory) | 68.6247 | 23.93 | 0.7607 | 1.1321 | 36.25 | 2.857 | 11.094 |
| `llama_memory_pooled` | 68.5632 | 24.34 | 0.7514 | 1.1167 | 37.09 | 2.919 | 11.293 |
| `rule_fixed` | 68.5234 | 23.95 | 0.7594 | 1.1299 | 36.20 | 2.862 | 10.620 |
| `llama_uniform_allocator` | 68.4750 | 24.07 | 0.7566 | 1.1252 | 36.36 | 2.882 | 10.353 |
| `llama_memory_current_year` | 68.4428 | 24.33 | 0.7508 | 1.1161 | 37.09 | 2.917 | 11.684 |
| `fixed_no_memory` (= 6.3 no-memory) | 67.8521 | 23.96 | 0.7544 | 1.1221 | 36.25 | 2.862 | 10.406 |
| `adaptive_no_memory` (= 6.2 no-memory) | 67.7655 | 24.37 | 0.7450 | 1.1066 | 37.09 | 2.923 | 11.375 |
| `adaptive_memory` (= 6.2 memory, **main method**) | 67.7304 | 24.35 | 0.7452 | 1.1069 | 37.09 | 2.923 | 11.380 |
| `llama_hard_allocator` | 67.7175 | 24.39 | 0.7444 | 1.1061 | 36.90 | 2.923 | 11.493 |
| `rule_adaptive` | 66.8173 | 24.34 | 0.7389 | 1.0951 | 37.03 | 2.929 | 11.143 |
| `llama_no_risk` | 66.3881 | 24.26 | 0.7373 | 1.0941 | 36.61 | 2.909 | 10.410 |
| `numerical_no_risk` | 60.4856 | 23.88 | 0.7009 | 1.0343 | 36.95 | 2.876 | 7.007 |
| `numerical_only` (**pre-agent control**) | 60.0460 | 23.87 | 0.6979 | 1.0270 | 38.23 | 2.881 | 8.132 |
| `numerical_uniform_allocator` | 59.9128 | 23.42 | 0.7056 | 1.0391 | 36.82 | 2.832 | 6.059 |
| `numerical_hard_allocator` | 59.6689 | 23.92 | 0.6939 | 1.0213 | 38.08 | 2.885 | 8.360 |
| `cash` | 0 | 0 | — | — | 0 | 0 | 0 |

FULL-period (1,608-day) figures are in the same file under scope `FULL` (e.g. `buy_hold` 266.76%, `static50` 133.08%, `adaptive_memory` 116.30%, `numerical_only` 103.30%).

- **All 22 paired contrasts** (`COMPARISONS.json`; POST2021; mean daily net-log difference A − B in bps/day with 95% block-bootstrap interval; gap = compounded-return gap in pp):

| role | A − B | bps/day | 95% CI | gap pp | reading |
|---|---|---:|---|---:|---|
| PRIMARY | `fixed_memory − rule_fixed` | +0.0048 | [−0.0633, +0.0755] | +0.10 | inconclusive |
| secondary | `adaptive_memory − rule_adaptive` | +0.0439 | [−0.1299, +0.2340] | +0.91 | inconclusive |
| secondary | `fixed_no_memory − rule_fixed` | −0.0321 | [−0.0745, +0.0037] | −0.67 | inconclusive |
| secondary | `adaptive_memory − numerical_only` | +0.3770 | [−0.1940, +1.0516] | +7.68 | inconclusive |
| secondary | `rule_adaptive − numerical_only` | +0.3331 | [−0.2262, +0.9608] | +6.77 | inconclusive |
| prior | `adaptive_memory − adaptive_no_memory` | −0.0017 | [−0.1265, +0.0992] | −0.04 | inconclusive |
| prior | `fixed_memory − fixed_no_memory` | +0.0369 | [−0.0340, +0.1221] | +0.77 | inconclusive |
| trust | `adaptive_memory − fixed_memory` | −0.0427 | [−0.5948, +0.5897] | −0.89 | inconclusive |
| core | `numerical_only − numerical_hard_allocator` | +0.0190 | [−0.1599, +0.1896] | +0.38 | inconclusive |
| core | `numerical_only − numerical_uniform_allocator` | +0.0067 | [−0.4901, +0.4921] | +0.13 | inconclusive |
| core | `numerical_only − numerical_frozen_W` | −0.4942 | [−1.2661, +0.2748] | −10.15 | inconclusive |
| core | `numerical_only − numerical_no_risk` | −0.0221 | [−0.2964, +0.2384] | −0.44 | inconclusive |
| component | `adaptive_memory − llama_hard_allocator` | +0.0006 | [−0.1946, +0.1868] | +0.01 | inconclusive |
| component | `adaptive_memory − llama_uniform_allocator` | −0.0356 | [−0.6114, +0.5172] | −0.74 | inconclusive |
| component | `adaptive_memory − llama_frozen_W` | −0.4498 | [−1.2966, +0.4853] | −9.65 | inconclusive |
| component | `adaptive_memory − llama_no_risk` | +0.0646 | [−0.2312, +0.3541] | +1.34 | inconclusive |
| component | `adaptive_memory − llama_memory_frozen2021` | −0.1034 | [−0.2263, +0.0007] | −2.17 | inconclusive |
| component | `adaptive_memory − llama_memory_current_year` | −0.0341 | [−0.1291, +0.0394] | −0.71 | inconclusive |
| component | `adaptive_memory − llama_memory_pooled` | −0.0398 | **[−0.1010, −0.0027]** | −0.83 | B higher (pooled retrieval) |
| external | `adaptive_memory − cash` | +4.1575 | [−3.1421, +11.5061] | +67.73 | inconclusive |
| external | `adaptive_memory − buy_hold` | −2.6492 | [−10.9759, +6.1926] | −65.48 | inconclusive |
| external | `adaptive_memory − static50` | −0.1923 | [−1.0705, +0.8569] | −4.06 | inconclusive |

- **Component transmission** (`COMPONENT_TRANSMISSION.csv`, regime `ALL`, POST2021; days on which the two arms differ in advisor action → desired exposure → executed exposure → net return): `adaptive_memory` vs `adaptive_no_memory` **32 → 12 → 8 → 14**; `fixed_memory` vs `fixed_no_memory` **33 → 33 → 13 → 25**; `adaptive_memory` vs `numerical_only` 691 → 484 → 276 → 304; `fixed_memory` vs `rule_fixed` 46 → 46 → 13 → 24; `adaptive_memory` vs `llama_memory_pooled` 15 → 6 → 5 → 9. (In adaptive pairs desired exposure can differ on days the action agrees, because each arm's β follows its own history.)
- **Decision** (`FINAL_STATUS.json`, `TEST_STATUS.json`): `COMPLETE`, 33,768 ledger rows; `all_component_claims_validated: false`; `continuous_learning_established: false`; the forecast-calibration and executable-clock questions are recorded as `UNRESOLVED`, the T/V/D/ATP deletion as a structural zero-weight invariance only.
- **Run**: see §10.4.1.

### 6.7 Stage 7 — sequential memory validation (closed loop + state-controlled)

- **Package** `RAMAS/code/RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/` (`run_stage7.py`; vendors its own copy of the Stage 6.4 package); code `1bb4ff69…`; run tarball `f6a63683…`. Provider identity identical to Stage 6.x (`03_PROVIDER_IDENTITY.json`, seed 16061).
- **Design**: four fresh arms, fixed β = 0.05, controller on. `closed_loop_memory` / `closed_loop_no_memory` replicate the Stage 6.3 pair independently (primary). `state_controlled_memory` / `state_controlled_no_memory` feed both arms the **same** current-state stream so that only memory visibility differs; their README labels this a *direct memory-conditioned action-sensitivity diagnostic, not a closed-loop performance estimate*. `MEMORY_ELIGIBILITY_AUDIT.json`: 1,608 episodes per arm, `no_memory_leak_rows: 0`, no violations.
- **Results** (`COMPARISONS.json`, `TRANSMISSION.csv`, POST2021):

| pair | memory % | no-memory % | bps/day | 95% CI | verdict | action ≠ → desired ≠ → executed ≠ | mean \|Δ exposure\| | days memory cited |
|---|---:|---:|---:|---|---|---|---:|---:|
| closed loop | 68.9907 | 67.8521 | +0.0543 | [−0.0256, +0.1521] | inconclusive | 35 → 35 → 11 | 0.00044 | 671 of 1,244 |
| state-controlled | 63.7490 | 65.9538 | **−0.1075** | **[−0.2335, −0.0042]** | no-memory higher | 86 → 86 → 52 | 0.00209 | 0 of 1,244 |

By regime (all 1,608 days): every closed-loop difference occurs in `bull` (75 action differences, 21 executed) or `bear` (9, 0 executed); none in `mix`. The state-controlled arms turn over far more (Σ 22.6 / 25.0 vs 10.9 / 10.4) because they follow the reference state path rather than their own holdings.

`closed_loop_no_memory` reproduces Stage 6.3 `fixed_no_memory` **bit-for-bit** in daily net return (§7.6). Status: `COMPLETE`, 6,432 rows, `economic_superiority_claim: false`, `clock_repair_required_before_confirmatory_claim: true`.
- **Run**: `bash RAMAS/code/RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/start.sh` (prints PID and LOG; pointer `RAMAS_STAGE7_LAST_LOG.txt`); `tail -f "$(cat RAMAS_STAGE7_LAST_LOG.txt)"`; one phase only with `RAMAS_STAGE7_MODE=closed_loop|state_controlled`; output `EXPERIMENT_BRANCHES/RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/artifacts/<run_id>/{closed_loop,state_controlled}/<arm>/`.

### 6.8 Isolation factorial — 2 × 2 × 2 on a fresh seed

- **Package** `RAMAS_FULL_PIPELINE_ISOLATION_V1/` (`RAMAS_FULL_PIPELINE_ISOLATION.py`, `RAMAS_VERIFY_RESULTS.py`, `self_test.py`, `config/experiment.json`); manifest `8bc1a6e013191bb3d6e0916073a5aac6e4160a1b1d9c273ea86b001623c652cf` (13 files); code archive `1ac9b814…`. It drives the frozen Stage 6.4 engine (expected archive `52698118…`, pipeline 1.12.3 `683b8495…`) — nothing is re-implemented.
- **Design**: for each model × sampling seed cell, 8 sequential arms `llm_{adaptive,fixed}_{memory,no_memory}_controller_{on,off}` over all 1,608 days (12,864 calls per cell), plus `numerical_only_controller_{on,off}` once (numerical seed 314159). Sampling **seed 42** (every earlier run used 16061); prompt `model_neutralized_frozen_v1`; `fail_on_any_invalid_output: true`; provider identity re-checked every 100 calls; bootstrap seed 20260919 reserved for the verifier. Run identity: config `58923886…`, preflight `4b1b42de…`. Started 2026-09-20T08:43:01Z; llama cell `CELL_COMPLETE` 2026-09-21T11:14:48Z (all 8 arms valid rate 1.0); the run then moved to `glm-4.7-flash` and stopped fail-closed at 11:15:08Z (`FAILURE.json`, `STOP_TECHNICAL_OR_SOURCE`, `economic_stop: false`). The independent verifier (`04_verify_results.sh`) was therefore **not** run on the archived cell; the numbers below were recomputed from the arm ledgers with the frozen metrics module (consolidation, §6.10) and re-checked for this report.
- **Arms, POST2021** (`cells/llama3.3_70b__seed_42/arms/*/DAILY_LEDGER.csv`; the two numerical controls are bit-identical to Stage 6.4's `numerical_only` and `numerical_no_risk`):

| arm | net return % | ann. vol % | Sharpe | Sortino | max DD % | ES95 % | mean BTC % | turnover Σ | fees % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm_adaptive_memory_controller_on` | 69.6156 | 24.36 | 0.7586 | 1.1281 | 37.09 | 2.918 | 44.90 | 10.921 | 1.092 |
| `llm_adaptive_no_memory_controller_on` | 68.9113 | 24.40 | 0.7528 | 1.1191 | 37.09 | 2.927 | 45.02 | 11.358 | 1.136 |
| `llm_fixed_memory_controller_on` | 68.3072 | 23.93 | 0.7584 | 1.1286 | 36.25 | 2.857 | 44.46 | 10.981 | 1.098 |
| `llm_fixed_no_memory_controller_on` | 68.0768 | 23.96 | 0.7561 | 1.1248 | 36.25 | 2.862 | 44.50 | 10.606 | 1.061 |
| `llm_adaptive_no_memory_controller_off` | 66.1410 | 24.39 | 0.7330 | 1.0878 | 36.54 | 2.925 | 45.11 | 10.072 | 1.007 |
| `llm_adaptive_memory_controller_off` | 65.9883 | 24.28 | 0.7341 | 1.0893 | 36.61 | 2.912 | 44.89 | 9.976 | 0.998 |
| `llm_fixed_no_memory_controller_off` | 64.2157 | 23.79 | 0.7311 | 1.0836 | 36.05 | 2.852 | 44.37 | 8.598 | 0.860 |
| `llm_fixed_memory_controller_off` | 63.9836 | 23.76 | 0.7300 | 1.0816 | 36.09 | 2.849 | 44.31 | 8.975 | 0.897 |
| `numerical_only_controller_off` | 60.4856 | 23.88 | 0.7009 | 1.0343 | 36.95 | 2.876 | 44.24 | 7.007 | 0.701 |
| `numerical_only_controller_on` | 60.0460 | 23.87 | 0.6979 | 1.0270 | 38.23 | 2.881 | 44.04 | 8.132 | 0.813 |

- **6.8.4 Main effects** (mean POST2021 compounded net return of the arms with the factor on minus those with it off; advisor = mean of the 8 LLM arms minus mean of the 2 numerical controls). Point estimates only — **no bootstrap intervals exist for these** because the run aborted before its verification step; compute them from the ledgers before they enter a table (recipe in §12.3).

| factor | main effect | in bps/day |
|---|---:|---:|
| advisor vs numerical-only | **+6.64 pp** | +0.326 |
| controller ON vs OFF | **+3.65 pp** | +0.176 |
| adaptive vs fixed trust | **+1.52 pp** | +0.073 |
| memory vs no-memory | **+0.14 pp** | +0.006 |

> **Correction.** Internal notes and the 2026-09-22 manuscript-update report quoted the memory main effect as **+0.07 pp**. Recomputing all four effects with one definition (above) gives **+0.1375 pp**; the other three reproduce exactly, so +0.07 was an arithmetic slip (half the true mean of the four pair gaps −0.153, +0.704, −0.232, +0.230). The conclusion is unchanged: memory is an order of magnitude below every other live component.

- **The four memory pairs** (`MEMORY_CONTRASTS.csv`; transmission recomputed for this report):

| pair | bps/day | 95% CI | gap pp | action ≠ → desired ≠ → executed ≠ (merged) | mean \|Δ exposure\| on acting days |
|---|---:|---|---:|---|---:|
| adaptive / controller on | +0.0334 | [−0.0602, +0.1407] | +0.704 | 38 → 37 → 32 (5) | 0.0578 |
| adaptive / controller off | −0.0074 | [−0.1053, +0.0836] | −0.153 | 40 → 167 → 167 (0) | 0.0173 |
| fixed / controller on | +0.0110 | [−0.0609, +0.0918] | +0.230 | 30 → 30 → 12 (**18**) | 0.0500 |
| fixed / controller off | −0.0114 | [−0.0566, +0.0335] | −0.232 | 29 → 29 → 29 (0) | 0.0257 |

With the controller off there is no projection, so desired = executed and nothing merges; with it on, 18 of 30 fixed-trust differences vanish at the grid. In adaptive pairs desired exposure differs on many agreeing days (167 vs 40) because the two β paths diverge after the first monthly update — these pair differences are *total-system* effects.
- **Cross-model extension — why it stopped.** `glm-4.7-flash` and `qwen3:8b` expose Ollama's `thinking` capability; `num_predict: 512` was consumed by `message.thinking`, leaving empty `content` with `done_reason: length`. `"think": false` fixes that (both then answer in 95–133 tokens), but glm-4.7-flash then **ignores the JSON-schema `enum`** on `reason_codes` (invented codes such as `PROB_BEAR_HIGH`), which the fail-closed validator rightly rejects. qwen3:8b passes the validator but is an 8B model (size confound) and is outside the frozen decision that permitted only Llama-70B. Model generality is therefore **not established**; the retrieval-collapse diagnostic (§7.5) is model-independent by construction.
- **Run**: see §10.4.2.

### 6.9 Stage 8 — corrected episodic memory

- **Package** `RAMAS_STAGE8_CORRECTED_MEMORY_V1/` (`RAMAS_CORRECTED_MEMORY.py` with subcommands `preflight | validate | diagnose | run`; `corrected_memory/{labels,retrieval,engine,replay}.py`; `config/experiment.json`; scripts `00`–`05`; `EXPERIMENT_CONTRACT.md`); manifest `4a8eaf0afd1701a83d01b8ffadb3fbd83e95edf9e1edb088233dae88e3fd16d3` (16 files). Imports the Stage 6.4 package read-only and verifies its manifest on every command.
- **6.9.1 Gate — engine byte-faithfulness** (`EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/validation/legacy_equivalence__llama_memory_current_year/LEGACY_EQUIVALENCE.json`, 2026-09-21T04:43Z): with the corrections switched off, the new engine replayed the archived Stage 6.4 arm `llama_memory_current_year` from its journal: **1,608 / 1,608 request digests matched, 52 columns compared, 0 mismatched, terminal wealth 2.174904923218857 in both, 0 model calls**. Status `PASS_ENGINE_REPRODUCES_ARCHIVED_ARM`. Nothing may run on the GPU unless this passes.
- **6.9.2 Diagnosis of the frozen memory** (`diagnostics/MEMORY_DIAGNOSTICS.json`, no model calls, measured on the 1,608 archived prompts of that arm):

| measurement | value |
|---|---|
| days with 0 / 1 / 2 / 3 distinct actions among the retrieved episodes | 10 / **1,381** / 217 / 0 |
| fraction of days on which the evidence block can contrast two actions | **13.50%** |
| correlation of the retrieved memory signal (advantage-weighted BTC vote) with the return it informs | −0.034 (p ≈ 0.28), 1,009 days with a non-zero signal |
| directional agreement of that signal with the realized return | 47.9% |
| fraction of stored labels that are positive | 49.4% |
| `credit_label_validity` block (pooled corr −0.36) | **withdrawn diagnostic**, see 6.9.4 — the file is kept as archived, its interpretation is not |

Cause (from `memory.py`): the same-regime prefilter plus a distance in which four of nine dimensions are near-deterministic functions of that regime returns near-duplicates of today's state, and the model's own past action in near-duplicate states is usually the same action.

- **6.9.3 The experiment** (`00_RUN_CONTRACT.json`, config `bb43811e…`, started 2026-09-22T01:49:28Z, completed 07:49:55Z, 3,216 fresh calls, valid rate 1.0). Two arms, fixed β = 0.05, controller on, model/digest/prompt/seed/serving identical to Stage 6.3 and Stage 7: `corrected_memory` (memory `expanding`, retrieval `balanced` 2 per action, `regime_filter: none`, label `counterfactual`) and `corrected_no_memory` (memory `none`, same label). Retrieval repair verified in the ledger (`retrieved_action_diversity`): days with 0 / 1 / 2 / 3 distinct actions **0.06% / 1.62% / 6.59% / 91.73%** → contrast expressible on **98.32%** of days (legacy 13.50%); median 6 episodes shown; the model cited ≥ 1 episode on 216 of 1,608 days (0.26 citations/day); the no-memory arm retrieved nothing on any day.

| POST2021 | `corrected_memory` | `corrected_no_memory` | Δ |
|---|---:|---:|---:|
| net return % | 67.5951 | 67.8521 | −0.2569 |
| annual volatility % | 23.8949 | 23.9586 | −0.0637 |
| Sharpe | 0.7539 | 0.7544 | −0.0005 |
| Sortino | 1.1207 | 1.1221 | −0.0014 |
| max drawdown % | 36.2533 | 36.2533 | 0 |
| ES95 daily loss % | 2.8572 | 2.8619 | −0.0047 |
| worst day % | −7.1284 | −7.1284 | 0 |
| mean BTC exposure % | 44.4132 | 44.5096 | −0.0965 |
| turnover Σ / fees % | 11.7121 / 1.1712 | 10.4062 / 1.0406 | +1.3059 / +0.1306 |
| action mix ABSTAIN / CASH / BTC | 559 / 378 / 307 | 533 / 372 / 339 | +26 / +6 / −32 |
| **primary contrast** (`COMPARISONS.json`) | **−0.012314 bps/day, 95% CI [−0.085197, +0.051225]**, 1,244 paired days | | `INCONCLUSIVE_NOT_EQUIVALENCE` |
| transmission | 42 disagreement days → 42 desired ≠ → **24** executed ≠ (18 merged); disagreements no-memory→memory: BTC→ABSTAIN 34, ABSTAIN→CASH 6, ABSTAIN→BTC 2; mean \|Δ executed\| on the 24 days exactly **0.0500** (one β step) | | |

`RUN_COMPLETE.json`: `corrected_label_validated_as_improving_decisions: false`. Two seams changed together, so Stage 8 cannot attribute between retrieval and label; the retrieval change is the one with a measured defect behind it.

- **6.9.4 Withdrawn diagnostic — the "credit-label" criticism.** Between Stage 7 and Stage 8 an analysis claimed the stored label was "anti-correlated with decision quality" (pooled corr −0.36 with the day's return) and that the counterfactual label correlated +0.67. That is invalid: the label is action-conditional (BTC positive on up days, CASH positive on down days), so pooling across actions mixes two sign conventions; the +0.67 measured a different quantity (a state-independent BTC-minus-CASH contrast). The fair test — correct-sign rate conditional on the action taken, on Stage 8's own ledger where both labels are recorded — gives: BTC days (n 397) archived 98.74% vs corrected 96.47%; CASH (501) 100.00% vs 98.40%; BTC+CASH (898) **99.44%** vs 97.55%; ABSTAIN days (710) archived label exactly zero on 0.99%, corrected 100%. The archived label is slightly *better* on directional days and its ABSTAIN-day values are near zero (median 0.00 bps). The trust rule excludes ABSTAIN episodes anyway. **Credit mis-specification is not a finding of this project and must not appear in the manuscript.** The `EXPERIMENT_CONTRACT.md` shipped with the package still states the original motivation; this section supersedes it.
- **Run**: see §10.4.3.

### 6.10 Llama-70B evidence consolidation

- **Package** `RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/` (`consolidate.py` sha `37df1610…`, `pooled.py`, `sources.json`); manifest `b28c5ebe2f897a7f8946c4978bbe4c10fd3c7877c3c09b55c1bcb37adf12605f`. Read-only; no model calls; refuses the wrong interpreter.
- **What it does**: reads every completed `llama3.3:70b` arm from its *own* run (three archived formats: single trace 6.1; paired trace 6.2/6.3; per-arm ledgers 6.4/7/ISO — Stage 6.4's republished 6.2/6.3 trajectories are counted once), verifies each arm's accounting (max error 2.2e-16), recomputes all metrics with `suite64.metrics`, bootstraps every memory-vs-no-memory pair with the Stage 6.4 contract, pools descriptively, and measures cross-run replication. Output `EXPERIMENT_BRANCHES/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/20260921T000000Z/`: `ALL_ARMS_DAILY.csv` (38,592 rows = 24 arms × 1,608), `ARM_INVENTORY.csv`, `ARM_METRICS.csv`, `MEMORY_CONTRASTS.csv` (8 rows), `POOLED_MEMORY_EFFECT.json`, `REPLICATION_TRANSMISSION.json`, `CROSS_RUN_REPLICATION.csv` (14 configuration groups), `CONSOLIDATED_REPORT.md`, `SOURCE_MANIFEST.json` (sha256 of all 22 inputs and 10 outputs), `REPRODUCE.sh`. Two independent re-runs produced byte-identical data outputs (7/7). Stage 8 is *not* among its inputs; §7.2 adds it by hand.
- **Run**: `/home/infonet/anaconda3/envs/wahid_test/bin/python RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/consolidate.py --output EXPERIMENT_BRANCHES/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/<new_run_id>` (≈ 1 minute; then `sha256sum -c`-style comparison of the new `SOURCE_MANIFEST.json` `outputs_sha256` against the archived one).

### 6.11 One-off read-only analyses (`RAMAS/scripts/` → `RAMAS/results/`)

None of these calls the model; all read archived ledgers or run tarballs.

| script → result | what it measures | headline |
|---|---|---|
| `RAMAS_NO_CONTROLLER_EXPERIMENT.py` → `ramas_no_controller_results/{no_controller_metrics.json,*_daily.csv}` | replays the 6.2 and 6.3 pairs executing the *desired* (blended, unprojected) exposure with 10-bps costs — a controller-bypass counterfactual | adaptive: memory 65.87% vs no-memory 66.23% (Δ −0.355 pp); fixed: 64.44% vs 64.50% (Δ −0.059 pp). Without the controller both memory gaps go slightly negative |
| → `ramas_numerical_core_controller_ablation.json` | the core alone with the controller on vs off (advisor ABSTAIN, β 0) | FULL wealth 2.0330 (on) vs 2.1498 (off); POST2021 60.05% vs 60.49%; turnover 12.31 vs 10.39; mean exposure 0.4433 vs 0.4527 — for the core alone the projection costs a little return |
| `RAMAS_LLAMA_ADVISOR_VALUE_BOOTSTRAP.py` → `ramas_llama_advisor_value_bootstrap.json` | each Stage 6.4 LLM arm minus `numerical_only`, POST2021, paired block bootstrap (5,000 × 30 d, seed 20260915), controller on | `llama_memory_current_year` +8.40 pp, +0.441 bps/day [−0.112, +1.122]; `llama_memory_frozen2021` +9.86 pp, +0.513 [−0.078, +1.232]; `llama_memory_pooled` +8.52 pp, +0.448 [−0.125, +1.144]; `llama_uniform_allocator` +8.43 pp; `llama_hard_allocator` +7.67 pp; `llama_no_risk` +6.34 pp; `llama_frozen_W` +17.34 pp, +1.028 [−0.024, +2.240] (confounded with the W change). Every interval covers zero; the advisor's gain comes with +0.8 to +5.6 pp more average Bitcoin exposure |
| `RAMAS_PURE_LLAMA_ADVISOR_ABLATION_V3.py` → `ramas_pure_llama_advisor_ablation_v3.json` | decomposition N (numerical only) → L (advisor, no memory) → LM (advisor + memory), controller on and off, from the 6.2/6.3 arms | adaptive: L − N **+7.72 pp** (+0.412 bps), LM − L −0.035 pp, LM − N +7.68 pp; fixed: L − N **+7.81 pp** (+0.388 bps), LM − L +0.773 pp (+0.035 bps), LM − N +8.58 pp. Controller off: L − N +5.74 / +4.02 pp, LM − L −0.355 / −0.059 pp. The advisor path carries essentially the whole effect; memory's increment is within ±0.8 pp either way |
| `RAMAS_EXISTING_CALL_PAIR_AUDIT.py` → `ramas_existing_call_pair_audit.json` | provider-contract uniformity of the 6.2 and 6.3 call journals | one provider contract variant per run, retrospective only |
| — → `ramas_memory_call_schema_report.json`, `ramas_stage64_trace_schema.json`, `ramas_stage62_stage63_trace_inventory.json` | schema/inventory of the archived calls and traces | descriptive; use them to locate columns |
| — → `RAMAS_STAGE6_1_BACK_CALCULATION.xlsx` | manual back-calculation of the Stage 6.1 accounting | reproduces the ledger by hand |

---

## 7. Consolidated results

Window POST2021 (return dates 2022-01-01 → 2025-05-28, 1,244 days) throughout; memory minus no-memory; block bootstrap as in §2.7.

### 7.1 All 24 Llama-70B arms across runs (`ARM_METRICS.csv`)

| run | arm | trust | memory | ctrl | seed | net ret % | vol % | Sharpe | Sortino | MDD % | ES95 % | BTC % | turnover | fees % |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.1 | `stage61_continuous` | adaptive | expanding | on | 16061 | 69.687 | 24.38 | 0.7586 | 1.1284 | 37.09 | 2.922 | 44.98 | 11.520 | 1.152 |
| 6.2 | `adaptive_memory` | adaptive | expanding | on | 16061 | 67.730 | 24.35 | 0.7452 | 1.1069 | 37.09 | 2.923 | 44.88 | 11.380 | 1.138 |
| 6.2 | `adaptive_no_memory` | adaptive | none | on | 16061 | 67.765 | 24.37 | 0.7450 | 1.1066 | 37.09 | 2.923 | 44.94 | 11.375 | 1.138 |
| 6.3 | `fixed_memory` | fixed | expanding | on | 16061 | 68.625 | 23.93 | 0.7607 | 1.1321 | 36.25 | 2.857 | 44.46 | 11.094 | 1.109 |
| 6.3 | `fixed_no_memory` | fixed | none | on | 16061 | 67.852 | 23.96 | 0.7544 | 1.1221 | 36.25 | 2.862 | 44.51 | 10.406 | 1.041 |
| 6.4 | `llama_frozen_W` | adaptive | expanding | on | 16061 | 77.384 | 26.78 | 0.7621 | 1.1331 | 39.70 | 3.197 | 49.60 | 8.839 | 0.884 |
| 6.4 | `llama_hard_allocator` | adaptive | expanding | on | 16061 | 67.717 | 24.39 | 0.7444 | 1.1061 | 36.90 | 2.923 | 44.98 | 11.493 | 1.149 |
| 6.4 | `llama_memory_current_year` | adaptive | current_year | on | 16061 | 68.443 | 24.33 | 0.7508 | 1.1161 | 37.09 | 2.917 | 44.86 | 11.684 | 1.168 |
| 6.4 | `llama_memory_frozen2021` | adaptive | frozen2021 | on | 16061 | 69.903 | 24.37 | 0.7603 | 1.1309 | 37.09 | 2.922 | 44.96 | 11.615 | 1.162 |
| 6.4 | `llama_memory_pooled` | adaptive | pooled | on | 16061 | 68.563 | 24.34 | 0.7514 | 1.1167 | 37.09 | 2.919 | 44.87 | 11.293 | 1.129 |
| 6.4 | `llama_no_risk` | adaptive | expanding | off | 16061 | 66.388 | 24.26 | 0.7373 | 1.0941 | 36.61 | 2.909 | 44.95 | 10.410 | 1.041 |
| 6.4 | `llama_uniform_allocator` | adaptive | expanding | on | 16061 | 68.475 | 24.07 | 0.7566 | 1.1252 | 36.36 | 2.882 | 44.21 | 10.353 | 1.035 |
| 7 | `closed_loop_memory` | fixed | expanding | on | 16061 | 68.991 | 23.93 | 0.7634 | 1.1363 | 36.25 | 2.857 | 44.47 | 10.948 | 1.095 |
| 7 | `closed_loop_no_memory` | fixed | none | on | 16061 | 67.852 | 23.96 | 0.7544 | 1.1221 | 36.25 | 2.862 | 44.51 | 10.406 | 1.041 |
| 7 | `state_controlled_memory` | fixed | expanding | on | 16061 | 63.749 | 23.82 | 0.7269 | 1.0796 | 36.34 | 2.852 | 44.28 | 22.615 | 2.262 |
| 7 | `state_controlled_no_memory` | fixed | none | on | 16061 | 65.954 | 23.95 | 0.7406 | 1.1011 | 36.34 | 2.863 | 44.49 | 24.987 | 2.499 |
| ISO | `llm_adaptive_memory_controller_on` | adaptive | expanding | on | 42 | 69.616 | 24.36 | 0.7586 | 1.1281 | 37.09 | 2.918 | 44.90 | 10.921 | 1.092 |
| ISO | `llm_adaptive_no_memory_controller_on` | adaptive | none | on | 42 | 68.911 | 24.40 | 0.7528 | 1.1191 | 37.09 | 2.927 | 45.02 | 11.358 | 1.136 |
| ISO | `llm_fixed_memory_controller_on` | fixed | expanding | on | 42 | 68.307 | 23.93 | 0.7584 | 1.1286 | 36.25 | 2.857 | 44.46 | 10.981 | 1.098 |
| ISO | `llm_fixed_no_memory_controller_on` | fixed | none | on | 42 | 68.077 | 23.96 | 0.7561 | 1.1248 | 36.25 | 2.862 | 44.50 | 10.606 | 1.061 |
| ISO | `llm_adaptive_no_memory_controller_off` | adaptive | none | off | 42 | 66.141 | 24.39 | 0.7330 | 1.0878 | 36.54 | 2.925 | 45.11 | 10.072 | 1.007 |
| ISO | `llm_adaptive_memory_controller_off` | adaptive | expanding | off | 42 | 65.988 | 24.28 | 0.7341 | 1.0893 | 36.61 | 2.912 | 44.89 | 9.976 | 0.998 |
| ISO | `llm_fixed_no_memory_controller_off` | fixed | none | off | 42 | 64.216 | 23.79 | 0.7311 | 1.0836 | 36.05 | 2.852 | 44.37 | 8.598 | 0.860 |
| ISO | `llm_fixed_memory_controller_off` | fixed | expanding | off | 42 | 63.984 | 23.76 | 0.7300 | 1.0816 | 36.09 | 2.849 | 44.31 | 8.975 | 0.897 |

Plus Stage 8 (§6.9.3): `corrected_memory` 67.595%, `corrected_no_memory` 67.852%. Reference points on the same days: `numerical_only` 60.046%, `static50` 71.792%, `buy_hold` 133.206%, `cash` 0.

Cross-run replication (`CROSS_RUN_REPLICATION.csv`): the configuration `fixed / expanding / on / original` was realized 4 times (6.3, 7 ×2, ISO): 63.75–68.99%, spread 5.24 pp (2.46 sd) — the low value is the state-controlled diagnostic; `adaptive / expanding / on / original` 3 times (6.1, 6.2, ISO): 67.73–69.69%, spread 1.96 pp; `fixed / none / on` 4 times: 65.95–68.08%. Averages across runs mix prompt variants (`frozen_llama_role` vs `model_neutralized_frozen_v1`) and seeds and are flagged `cross_run_average_is_matched: false`.

### 7.2 The nine realizations of the memory contrast

| # | stage | protocol | trust | ctrl | seed | prompt | bps/day | 95% CI | gap pp | verdict |
|---|---|---|---|---|---:|---|---:|---|---:|---|
| 1 | 6.2 | closed loop | adaptive | on | 16061 | frozen | −0.0017 | [−0.1265, +0.0992] | −0.035 | inconclusive |
| 2 | 6.3 | closed loop | fixed | on | 16061 | frozen | +0.0369 | [−0.0340, +0.1221] | +0.773 | inconclusive |
| 3 | 7 | closed loop | fixed | on | 16061 | frozen | +0.0543 | [−0.0256, +0.1521] | +1.139 | inconclusive |
| 4 | 7 | **state-controlled diagnostic** | fixed | on | 16061 | frozen | −0.1075 | [−0.2335, −0.0042] | −2.205 | **no-memory higher** |
| 5 | ISO | closed loop | adaptive | off | 42 | neutral | −0.0074 | [−0.1053, +0.0836] | −0.153 | inconclusive |
| 6 | ISO | closed loop | adaptive | on | 42 | neutral | +0.0334 | [−0.0602, +0.1407] | +0.704 | inconclusive |
| 7 | ISO | closed loop | fixed | off | 42 | neutral | −0.0114 | [−0.0566, +0.0335] | −0.232 | inconclusive |
| 8 | ISO | closed loop | fixed | on | 42 | neutral | +0.0110 | [−0.0609, +0.0918] | +0.230 | inconclusive |
| 9 | 8 | closed loop, **repaired retrieval** | fixed | on | 16061 | frozen | −0.0123 | [−0.0852, +0.0512] | −0.257 | inconclusive |

Pooled over rows 1–8 (`POOLED_MEMORY_EFFECT.json`): mean **+0.00097 bps/day**, sd across realizations 0.0497, **4 positive / 4 negative**, mean compounded gap +0.028 pp, 7 individually inconclusive, 1 significant (against memory). Adding row 9: 4 positive / 5 negative. Closed-loop rows only (1–3, 5–9): mean +0.0129 bps/day, 4/4, all inconclusive. **No pooled confidence interval is legitimate**: every row replays the same 1,244 market days, so these are repeated implementations, not independent samples; each row's own interval is the inferential statement. With 9 unadjusted intervals, ≈ 0.45 false positives are expected by chance; the one that appeared runs against memory.

The cleanest three-way comparison in the project is rows 2, 3 and 9 — same model, digest, prompt, seed, trust, controller and source stream; only retrieval (and label) differ: legacy retrieval +0.773 and +1.139 pp; repaired retrieval −0.257 pp.

### 7.3 Behavioural signature and risk (memory minus no-memory, 9 pairs)

| quantity | mean Δ | direction consistent in |
|---|---:|---|
| ABSTAIN rate | **+1.83 pp** | **9 / 9** (more abstention) |
| mean BTC exposure | **−0.098 pp** | **9 / 9** |
| annual volatility | **−0.054 pp** | **9 / 9** |
| ES95 daily tail loss | **−0.006 pp** | **9 / 9** |
| Sharpe | +0.0010 | 6 / 9 |
| Sortino | +0.0015 | 6 / 9 |
| max drawdown | +0.012 pp | 2 / 9 higher, 7 unchanged |
| total turnover | +0.043 | 6 / 9 (more trading) |
| fees | +0.004 pp | 6 / 9 |
| net return | −0.004 pp | 4 / 9 positive |

Per pair (pp except Sharpe/Sortino/turnover):

| pair | return | vol | Sharpe | Sortino | MDD | ES95 | turnover | fees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 6.2 adaptive/on | −0.035 | −0.020 | +0.0002 | +0.0003 | 0 | −0.0000 | +0.005 | +0.0005 |
| 6.3 fixed/on | +0.773 | −0.033 | +0.0063 | +0.0100 | 0 | −0.0047 | +0.688 | +0.0688 |
| 7 fixed/on | +1.139 | −0.031 | +0.0090 | +0.0142 | 0 | −0.0048 | +0.542 | +0.0542 |
| 7 fixed/on [state-controlled] | −2.205 | −0.132 | −0.0137 | −0.0215 | 0 | −0.0116 | −2.372 | −0.2372 |
| ISO adaptive/off | −0.153 | −0.112 | +0.0011 | +0.0015 | +0.071 | −0.0129 | −0.096 | −0.0096 |
| ISO adaptive/on | +0.704 | −0.037 | +0.0058 | +0.0090 | 0 | −0.0082 | −0.437 | −0.0437 |
| ISO fixed/off | −0.232 | −0.033 | −0.0011 | −0.0020 | +0.033 | −0.0027 | +0.376 | +0.0376 |
| ISO fixed/on | +0.230 | −0.029 | +0.0023 | +0.0038 | 0 | −0.0048 | +0.375 | +0.0375 |
| 8 corrected | −0.257 | −0.064 | −0.0005 | −0.0014 | 0 | −0.0047 | +1.306 | +0.1306 |

Reading: memory makes the model decline to take a view about 2 pp more often, so it holds slightly less Bitcoin and the path is very slightly smoother (0.054 pp on a base of ~24% volatility, i.e. ~0.2% relative). It does not improve risk-adjusted return and usually costs more in fees. Because all pairs share one market history, "9 of 9" is a consistent direction, not a sign test.

### 7.4 Decision transmission — how much of a changed vote reaches the portfolio

| pair | window | action ≠ | desired ≠ | executed ≠ | net return ≠ | source |
|---|---|---:|---:|---:|---:|---|
| 6.2 `adaptive_memory` vs `adaptive_no_memory` | POST2021 | 32 | 12 | 8 | 14 | `COMPONENT_TRANSMISSION.csv` |
| 6.3 `fixed_memory` vs `fixed_no_memory` | POST2021 | 33 | 33 | 13 | 25 | same |
| 7 closed loop | POST2021 | 35 | 35 | 11 | — | `TRANSMISSION.csv` |
| 7 state-controlled | POST2021 | 86 | 86 | 52 | — | same (82 of 86 are no-memory BTC vs memory ABSTAIN) |
| ISO adaptive / on | POST2021 | 38 | 37 | 32 | — | recomputed (§6.8) |
| ISO adaptive / off | POST2021 | 40 | 167 | 167 | — | recomputed |
| ISO fixed / on | POST2021 | 30 | 30 | 12 | — | recomputed |
| ISO fixed / off | POST2021 | 29 | 29 | 29 | — | recomputed |
| 8 corrected | POST2021 | 42 | 42 | 24 | — | recomputed (§6.9.3) |
| for scale: 6.4 `adaptive_memory` vs `numerical_only` | POST2021 | 691 | 484 | 276 | 304 | `COMPONENT_TRANSMISSION.csv` |

Memory changes the advisor's vote on 2–3% of days; with the controller on, roughly a third to two thirds of those changes are merged away at the 0.05 grid; when a change does execute it moves exposure by about one β step (0.05). Compare the advisor itself, which changes the vote against the numerical core on 56% of days and the executed exposure on 22%.

### 7.5 Retrieval collapse (legacy) and its repair (Stage 8)

| distinct actions among retrieved episodes | legacy (Stage 6.4 `llama_memory_current_year`, 1,608 prompts) | Stage 8 `corrected_memory` |
|---|---:|---:|
| 0 (nothing retrieved) | 0.62% (10) | 0.06% |
| 1 | **85.88%** (1,381) | 1.62% |
| 2 | 13.50% (217) | 6.59% |
| 3 | 0.00% (0) | **91.73%** |
| contrast expressible (≥ 2) | **13.50%** | **98.32%** |

Model-independent: it is a property of what is put into the prompt, measured before the model reads it. The five legacy retrieval ablations (expanding, frozen2021, current_year, pooled, and the allocator variants) all share the same-regime-prefilter/k-nearest contract and so did not span this defect; Stage 8 did.

### 7.6 Independent replication of an identical configuration (`REPLICATION_TRANSMISSION.json`)

| pair (separate runs, 1,608 fresh calls each) | days | action differs | executed exposure differs | net return differs |
|---|---:|---:|---:|---:|
| 6.3 `fixed_no_memory` vs 7 `closed_loop_no_memory` | 1,608 | **2** (2024-11-02 ABSTAIN/BTC at 0.45; 2025-03-29 CASH/ABSTAIN at 0.40) | **0** | **0** — daily net returns bit-identical |
| 6.3 `fixed_memory` vs 7 `closed_loop_memory` | 1,608 | 22 | 8 | 14 — **14 of 22 flips absorbed entirely** |

An action flip is the largest possible change in the advisor's output; where the executed exposure is unchanged, the β = 0.05 blend and the CVaR grid absorbed the advice completely. This is absorption measured with no analytic choices at all.

### 7.7 Factorial main effects (fresh seed 42, neutral prompt)

Advisor **+6.64 pp**, controller **+3.65 pp**, adaptive trust **+1.52 pp**, memory **+0.14 pp** (§6.8.4; point estimates, intervals to be computed). In daily terms +0.326 / +0.176 / +0.073 / +0.006 bps.

### 7.8 What the component map says about the rest of the system (Stage 6.4)

- **The advisor's value is exposure, not timing.** Every LLM arm holds 44–50% Bitcoin on average versus 44.0% for the core with the controller on, and every LLM-vs-core interval covers zero. `llama_frozen_W` (+17.3 pp over the core) simply holds 49.6% Bitcoin; its drawdown and tail loss are worse.
- **The deterministic rule is as good as the model.** `rule_fixed` 68.52% vs `fixed_memory` 68.62% vs `fixed_no_memory` 67.85%; primary contrast +0.0048 bps/day.
- **The regime posterior's softness barely matters** for the core (`hard_allocator` −0.38 pp, `uniform_allocator` −0.13 pp vs `numerical_only`); freezing W matters a lot (+10.1 pp) but mostly through higher exposure.
- **The controller costs the core a little return and buys a little drawdown**: core alone 60.05% (on) vs 60.49% (off) with MDD 38.2% vs 36.9%; but with the advisor present the controller *adds* +3.65 pp (factorial), because it merges away the advisor's more erratic votes.
- **Retrieval variant that beat the main method**: `llama_memory_pooled` (no regime prefilter) is the only Stage 6.4 interval that excludes zero (+0.83 pp over `adaptive_memory`), which pointed to the prefilter; Stage 8 removed it and found nothing.

---

## 8. Observations, interpretation and claim boundaries

### 8.1 Plain-language reading

1. **The system works as engineered.** Every arm reproduces bit-for-bit; accounting is exact to machine precision; no future information reaches a prompt; the model's fallible outputs are caught and neutralized. What follows is about the *idea*, not the plumbing.
2. **The advisor helps, but only because it leans long.** Adding the Llama vote raises 2022–2025 return by 6–10 pp over the numerical core in every configuration, with no interval excluding zero and with 0.8–5.6 pp more Bitcoin held. A transparent three-line rule on the same inputs does the same.
3. **Memory is a null result with a known cause.** Nine matched tests, eight inconclusive, one significant against; pooled +0.001 bps/day. The mechanism was starved: 86% of the time the memory showed the model five examples that all said the same thing.
4. **Fixing the mechanism did not revive the effect.** With the memory finally able to contrast actions on 98% of days, the result was −0.26 pp and inconclusive. So the retrieval defect explains why the channel carried no *information*, but removing it shows that this kind of memory — single-day outcomes of the model's own past votes, retrieved by the router's own features — has nothing the router did not already know.
5. **Whatever memory does is mostly absorbed.** A 5% trust weight and a 5%-step risk grid turn most changed votes into the same trade. Two independent runs of the same no-memory configuration differed on two votes and zero trades.
6. **Memory does change behaviour, in one small, consistent way**: more abstention, a hair less Bitcoin, a hair less volatility and tail loss, more fees. Report it as a behavioural finding, not a risk benefit.

### 8.2 Why the null is expected on first principles

The memory retrieves by the router's own state features and stores single-day outcomes. It therefore cannot know more than the router (whose posterior correlates ≈ 0.05 with the next-day return) and each stored outcome is dominated by daily noise. The advisor's influence is capped at β ≤ 0.2 (0.05 in most arms) and then grid-rounded. Any redesign would have to change what is *stored* (multi-day outcomes, aggregated statistics), what is *asked* (router reliability in this state, risk posture — not price direction) and how influence is *measured* (advice quality directly, before any portfolio run). That is a separate project; nothing here tests it.

### 8.3 Claim boundaries (recorded as `claim_boundaries` in every run contract)

| boundary | value | consequence for the text |
|---|---|---|
| `fresh_out_of_sample` | false | 2024–2025 is reused diagnostic history |
| `router_temporal_validity_resolved` | false | describe the posterior as a state estimate, not a forecast (§3.4) |
| `feasible_information_to_fill_clock_resolved` | false | decisions use and fill at the same close; latency unmodelled |
| `causal_memory_effect_from_closed_loop_factorial` | false | closed-loop pair gaps are total-system effects; the frozen-state intervention that would isolate memory was designed (`RAMAS_MEMORY_CAUSAL_EXPERIMENT`) but not run |
| `corrected_label_validated_as_improving_decisions` | false | Stage 8 does not validate the new label; its motivation is withdrawn |
| `continuous_learning_established` | false | every run |
| `economic_superiority_claim` | false | every run |
| model generality | not established | one model completed; §6.8 |

### 8.4 Must not be claimed

A pooled confidence interval across realizations · that memory was "removed" in no-memory arms (visibility only) · a six-expert mixture (it is a cash/B&H scalar) · that LLMs cannot use episodic memory (one design family tested) · that Stage 8 isolates retrieval (two seams changed) · the credit-label mis-specification (withdrawn, §6.9.4) · deployability · fresh out-of-sample performance · the best point estimate among 22 exploratory contrasts as a finding.

### 8.5 Open items

- Bootstrap intervals for the four factorial main effects (recipe §12.3).
- The independent verifier of the factorial (`04_verify_results.sh`) has not been run on the archived llama cell.
- `RAMAS_MEMORY_CAUSAL_EXPERIMENT` (frozen-state same-prompt intervention, 4,976 calls) is designed and smoke-tested only.
- The router's fit chronology and the executable clock remain as disclosed limitations, not repaired.

---

## 9. Verification ledger

Independently re-derived (not taken from run summaries):

| check | result |
|---|---|
| accounting identity and cost formula on every ledger row | max abs error ≤ 2.2e-16 (33,768 Stage 6.4 rows, 12,864 factorial, 6,432 Stage 7, 3,216 Stage 8, 38,592 consolidated) |
| next-day clock on every row | `return_date = decision_date + 1 day` everywhere |
| causal feature/scenario construction | `close[:index+1]` with runtime assertions |
| episode eligibility | 0 look-ahead episodes among 8,025 retrieved episode-days; 0 hallucinated citations (every cited id ∈ shown ids) |
| journal ↔ ledger | 11,256 Stage 6.4 calls, 6,432 Stage 7, 12,864 factorial, 3,216 Stage 8: 100% `llama3.3:70b`, digest `a6eb4748…`, valid rate 1.0 |
| bootstrap reproduction | all 22 Stage 6.4 intervals, Stage 7 and Stage 8 intervals reproduce bit-for-bit with seed 16062 |
| headline returns | every compounded return in §6–§7 reproduces exactly from its ledger |
| engine byte-faithfulness | Stage 8 `validate`: 1,608/1,608 digests, 52 columns, wealth Δ 0.0 |
| consolidation determinism | two independent runs, 7/7 data outputs byte-identical |
| package integrity | `sha256sum -c PACKAGE_MANIFEST.sha256`: 0 mismatches in all four executable packages; test suites 31 + 17 + 31 + 8 + 7 pass |
| numerical core reconstruction | W and desired exposure equal to the archived control to 3.3e-16; inactive experts exactly zero |

Structural facts confirmed from code: `exposure_cash ≡ 0`, `exposure_buy_and_hold ≡ 1`; the trust rule excludes ABSTAIN episodes; "no memory" leaves the private ledger and trust learning intact; W never depends on the advisor.

---

## 10. Reproduction guide

### 10.1 Environment and path pin

```bash
# interpreter (mandatory for bit-exact replay)
PY=/home/infonet/anaconda3/envs/wahid_test/bin/python      # python 3.11.9, numpy 1.26.4, pandas 2.2.3
$PY -c "import sys,numpy,pandas;print(sys.version.split()[0],numpy.__version__,pandas.__version__)"
# → 3.11.9 1.26.4 2.2.3

# the packages pin the original project root and raw-data path. On another machine either
sudo mkdir -p /home/infonet/wahid /home/infonet/wahid/projects/leader_fresh/full_data_set
sudo ln -s "$PWD" /home/infonet/wahid/leader_router_fresh
sudo ln -s "$PWD/data/raw/full_data_set.csv" /home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv
# or write clone-local configs (run identities will then differ from the archives):
python tools/localize_configs.py        # present in the public repository

# model serving, only for arms that call the model
ollama --version                        # 0.23.0
ollama show llama3.3:70b --modelfile | head -3 ; ollama list | grep llama3.3   # digest a6eb4748…
```

### 10.2 Integrity checks (no model, seconds)

```bash
cd /home/infonet/wahid/leader_router_fresh
sha256sum -c RAMAS_STAGE6_4_COMPONENT_SUITE_V1/PACKAGE_MANIFEST.sha256 | grep -vc ': OK$'      # → 0
sha256sum -c RAMAS_FULL_PIPELINE_ISOLATION_V1/PACKAGE_MANIFEST.sha256 | grep -vc ': OK$'       # → 0
sha256sum -c RAMAS_STAGE8_CORRECTED_MEMORY_V1/PACKAGE_MANIFEST.sha256 | grep -vc ': OK$'       # → 0
sha256sum -c RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/PACKAGE_MANIFEST.sha256 | grep -vc ': OK$'
sha256sum data/raw/full_data_set.csv          # b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e
( cd EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/artifacts/20260901T024730Z/base_source/LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_V1 && sha256sum -c MANIFEST.sha256 | grep -vc ': OK$' )
# every run directory: RUN_COMPLETE.json lists sha256 per file; verify one, e.g.
$PY - <<'EOF'
import json,hashlib,pathlib
run=pathlib.Path('EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts/20260922T014916Z')
rec=json.load(open(run/'RUN_COMPLETE.json')); bad=0
for rel,h in rec.get('artifact_sha256',{}).items():
    bad+= hashlib.sha256((run/rel).read_bytes()).hexdigest()!=h
print('mismatches:',bad)
EOF
```

### 10.3 Tests (no model, < 1 minute)

```bash
cd RAMAS_STAGE6_4_COMPONENT_SUITE_V1 && $PY -m pytest tests -q && (cd vendor/stage61 && $PY -m unittest discover -s tests) && cd ..
cd RAMAS_FULL_PIPELINE_ISOLATION_V1 && $PY self_test.py --config config/experiment.json && $PY -m pytest tests -q && cd ..
cd RAMAS_STAGE8_CORRECTED_MEMORY_V1 && $PY tests/test_corrections.py && cd ..
```

### 10.4 Running the experiments

Runtimes observed on the original server (one 70B model, sequential calls): Stage 6.2/6.3 ≈ 6.6 h each; Stage 6.4 ≈ 25.7 h; Stage 7 ≈ 12.4 h; factorial llama cell ≈ 26.5 h; Stage 8 ≈ 6.0 h. Median latency 5.8–6.0 s per no-memory call, 7.0–8.6 s per memory call. Resume is always safe: saved responses are never resampled.

#### 10.4.1 Stage 6.4 component suite

```bash
cd /home/infonet/wahid/leader_router_fresh
bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh                 # verifies manifest + tests, then launches run_server.sh in the background
#   prints PID=… LOG=…   and writes RAMAS_STAGE6_4_LAST_LOG.txt
tail -f "$(cat RAMAS_STAGE6_4_LAST_LOG.txt)"                     # watch; Ctrl-C stops watching, not the run
#   results → EXPERIMENT_BRANCHES/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/<run_id>/   (pointer: RAMAS_STAGE6_4_LAST_RUN.txt)
#   per arm: arms/<arm>/{DAILY_LEDGER.csv,episodes.json,trust_events.json,calls/,RUN_COMPLETE.json}
#   suite-level: daily_ledger.csv, all_metrics.csv, full_period_metrics.csv, COMPARISONS.json, COMPONENT_TRANSMISSION.csv, FINAL_STATUS.json
export RAMAS_RESUME_DIR="$(cat RAMAS_STAGE6_4_LAST_RUN.txt)"; bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh   # resume
RAMAS_SUITE_MODE=numeric bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh    # no model: benchmarks + 7 numerical arms only
RAMAS_SUITE_MODE=archive bash RAMAS_STAGE6_4_COMPONENT_SUITE_V1/start.sh    # no model: verify + re-report the archived arms
```
The archived run is `20260909T114202240674964Z`; the launcher's original log is kept as `RAMAS/logs/RAMAS_STAGE6_4_RUN.log` where the repository includes logs.

#### 10.4.2 Isolation factorial

```bash
export PATH=/home/infonet/anaconda3/envs/wahid_test/bin:$PATH    # the scripts call plain `python`
cd /home/infonet/wahid/leader_router_fresh/RAMAS_FULL_PIPELINE_ISOLATION_V1
./01_preflight.sh                                    # self-test, plan, preflight (no model)
#   → EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/{PREFLIGHT.json,RESOLVED_CONFIG.json}
./02_determinism_probe.sh                            # optional 4-model × 4-seed response-distinctness probe (was NOT run for the archived cell)
RAMAS_MODELS=llama3.3:70b RAMAS_SEEDS=42 \
  nohup ./03_run_experiment.sh > ../EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/console_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 &
tail -f ../EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/console_*.log     # lines: STAGE64_ARM=… PROGRESS=n/1608 VALID=n NEW_CALLS=… ; CELL=… ARM=… STATUS=COMPLETE
#   results → EXPERIMENT_BRANCHES/RAMAS_FULL_PIPELINE_ISOLATION_V1/artifacts/<run_id>/cells/<model>__seed_<s>/arms/<arm>/  and numerical_controls/
#   pointer  → /home/infonet/wahid/leader_router_fresh/RAMAS_FULL_PIPELINE_ISOLATION_LAST_RUN.txt
RAMAS_RUN_DIR="$(cat ../RAMAS_FULL_PIPELINE_ISOLATION_LAST_RUN.txt)" ./03_run_experiment.sh   # resume (same models/seeds/config)
./04_verify_results.sh                               # independent verifier → <run>/independent_verification/ (not yet run on the archive)
```
Archived run: `20260920T084300Z` (its `resume_console.log` is inside the run directory).

#### 10.4.3 Stage 8 corrected memory

```bash
cd /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE8_CORRECTED_MEMORY_V1
./00_check_safe_to_start.sh          # refuses if the factorial is running
./01_preflight.sh                    # no model → EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/PREFLIGHT/PREFLIGHT.json
./02_validate_legacy_equivalence.sh  # no model, ~5 min → …/validation/legacy_equivalence__llama_memory_current_year/LEGACY_EQUIVALENCE.json  (must say PASS_ENGINE_REPRODUCES_ARCHIVED_ARM)
./03_diagnose_memory.sh              # no model, seconds → …/diagnostics/MEMORY_DIAGNOSTICS.json
./04_run_corrected_pair.sh           # GPU, ~6 h; prints run dir + log; writes …/LAST_RUN.txt and …/LAST_LOG.txt
./05_monitor.sh -f                   # follow the live log (-1 once, -n 30 poll interval, -r <run_id> a specific run)
tail -f "$(cat ../EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/LAST_LOG.txt)"          # equivalent
#   results → EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts/<run_id>/{00_RUN_CONTRACT.json,01_SOURCE_AUDIT.json,02_CORE_AUDIT.json,arms/<arm>/…,ALL_DAILY_LEDGER.csv,COMPARISONS.json,RUN_COMPLETE.json,console.log}
RAMAS_STAGE8_RUN_ID=<run_id> ./04_run_corrected_pair.sh --resume        # resume
```
Archived run: `20260922T014916Z`. (A harmless `RUN_DIR: unbound variable` line may print when the launcher exits.)

#### 10.4.4 Stage 7, 6.2, 6.3, 6.1, 6 and the Stage 5 lineage (archived packages under `RAMAS/code/`)

```bash
cd /home/infonet/wahid/leader_router_fresh
bash RAMAS/code/RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/start.sh ; tail -f "$(cat RAMAS_STAGE7_LAST_LOG.txt)"
#   → EXPERIMENT_BRANCHES/RAMAS_STAGE7_SEQUENTIAL_MEMORY_VALIDATION_V1/artifacts/<run_id>/ ; RAMAS_STAGE7_MODE=closed_loop|state_controlled for one phase
bash RAMAS/code/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/start.sh ; tail -f RAMAS_STAGE6_3_RUN.log
bash RAMAS/code/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/start.sh   ; tail -f RAMAS_STAGE6_2_RUN.log
nohup bash RAMAS/code/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/run_server.sh > RAMAS_STAGE6_1_RUN.log 2>&1 & tail -f RAMAS_STAGE6_1_RUN.log
nohup bash RAMAS/code/RAMAS_STAGE6_LLAMA70B_MEMORY_AGENT_EXPERT_V1/run_server.sh   > RAMAS_STAGE6_RUN.log   2>&1 & tail -f RAMAS_STAGE6_RUN.log
bash RAMAS/code/RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1/run_server.sh         # no model; audits the newest Stage 5.1 archive
```
Each writes to `EXPERIMENT_BRANCHES/<package>/artifacts/<run_id>/`; the archived run ids are in §6.0. These packages expect the frozen inputs of §3 at the pinned paths and, for 6.2–7, the completed earlier runs they audit.

#### 10.4.5 Consolidation and one-off analyses (no model)

```bash
$PY RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/consolidate.py --output EXPERIMENT_BRANCHES/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/$(date -u +%Y%m%dT%H%M%SZ)
#   compare outputs_sha256 in the new SOURCE_MANIFEST.json with the archived 20260921T000000Z/SOURCE_MANIFEST.json → identical
$PY RAMAS/scripts/RAMAS_NO_CONTROLLER_EXPERIMENT.py --help
$PY RAMAS/scripts/RAMAS_LLAMA_ADVISOR_VALUE_BOOTSTRAP.py --help
$PY RAMAS/scripts/RAMAS_PURE_LLAMA_ADVISOR_ABLATION_V3.py --help
```

### 10.5 Reading a result without running anything

```bash
$PY - <<'EOF'
import pandas as pd, numpy as np, json
led=pd.read_csv('EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts/20260922T014916Z/ALL_DAILY_LEDGER.csv')
post=led[(led.return_date>='2022-01-01')&(led.return_date<='2025-05-28')]
for arm,g in post.groupby('arm'):
    r=g.net_return_after_trading_costs.values
    print(arm, f"net {100*np.expm1(np.log1p(r).sum()):.4f}%  vol {100*r.std(ddof=1)*np.sqrt(365.25):.4f}%  days {len(r)}")
print(json.load(open('EXPERIMENT_BRANCHES/RAMAS_STAGE8_CORRECTED_MEMORY_V1/artifacts/20260922T014916Z/COMPARISONS.json'))['contrasts'][0]['mean_daily_net_log_difference_bps'])
EOF
```

---

## 11. Artifact and hash index

### 11.1 Executable packages (repository root)

| package | manifest sha256 | files | code archive sha256 (`RAMAS/archives/*_CODE.tar.gz.sha256`) |
|---|---|---:|---|
| `RAMAS_STAGE6_4_COMPONENT_SUITE_V1` | `6a03c967673f6f2e98fd8134910a9a8231d0b9fb578bc1aa2615c9c18bf9da0b` | 45 | `52698118cbe19749250156a3b6aaa30caa06b75cef60c77e0a6908f99b2a67c3` |
| `RAMAS_FULL_PIPELINE_ISOLATION_V1` | `8bc1a6e013191bb3d6e0916073a5aac6e4160a1b1d9c273ea86b001623c652cf` | 13 | `1ac9b814b606f684b39b15abe86c251442e90e94647694206c9cd88c4b36e40b` |
| `RAMAS_STAGE8_CORRECTED_MEMORY_V1` | `4a8eaf0afd1701a83d01b8ffadb3fbd83e95edf9e1edb088233dae88e3fd16d3` | 16 | — |
| `RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1` | `b28c5ebe2f897a7f8946c4978bbe4c10fd3c7877c3c09b55c1bcb37adf12605f` | 4 | — |

### 11.2 Archived code packages (`RAMAS/code/`) and run tarballs (`EXPERIMENT_BRANCHES/<pkg>/artifacts/<run_id>.tar.gz.sha256`)

| stage | code archive sha256 | run id | run tarball sha256 |
|---|---|---|---|
| 5 (RAMoE lineage) | — | `20260901T085240Z`, `20260901T085405Z` | `1799dfe51ed7324f84c2857af13fb7dfdb94743bd6ecb952e57df22ee96778ea` |
| 5.1 | — | `20260902T004332Z` | `684290142357e65b6cebdc427ecf852bd027f5d1e6bcb2ebcddd5a911ef3c906` |
| 5.2 | `9d64a7cb26994f48d7b949efb5862de83733286d60d6090a106092ea419a542f` | `20260902T045940Z` | `59b42133d345105339cb3835a86016936b4d63df4138641b3a9797bd6b37c70d` |
| 5.3 | `116fb6fd2eb5a1ac4fffc469bc2e5ab80ed9594c22a922f7876d9a623f5a4190` | `20260902T054019Z` | `708cfec97a2182b539b87e39146937762fd9d8ef11a9697fd49d14775e28b24b` |
| 5.4 | `94811bccbad9fa4ce2dae9df5764bbf53737b4a9789a0dbee83f522eb239edc8` | `20260902T063436Z` | `55db17815820bc0604ce542ade7b7c639f02ec15d588be93494667ba97a8d8df` |
| 5.5 | `06afaf074b5a1e5cb93a2a5b322b35ed7cac2bb528194777f787efe46fcdf49c` | `20260902T082413Z` | `82c249595609431fde7a8209356b32ad1492791ed7632b4fd45d5427f806861b` |
| 5.5.1 | `15604a2e51064746948668f75838244440cf9fbda15c7451ff71819408556ee3` | `20260902T085752Z` | `46523afce0c546f26063e19b2e5a1a0c2c389eecd2d6b83fc00e6e716684e3ea` |
| 6 | `92cb044f643d6ea2087a9a27324147a61b00ea75a3c9f49d751e118d54f8b47b` | `20260903T004539Z` | `f1174b96b6a30dc1e398bb4ddfc6977ae2074198167266dab5ca99158b5c76cd` |
| 6.1 | `b253e9214a5bbc4383dc2a183a57abdcc92eeb4a0b2af0595f4ab567b8019060` | `20260907T012534Z` | `8a273d9fe338345bd4499b2c8222f1752a5432e6422c89632b2c11340fb5b9f6` |
| 6.2 | `1779f47bb061cda449b384a5b102e4508bdbd3e134f80f65412236a82c728397` | `20260908T055052711433762Z` | `e9296f2f28898c8694bcd86f49932a7d8408f689f512635a10fee0693660f37c` |
| 6.3 | `ebbfb637ece330b01d342217012a545f5701f1c1b052650bb8820b452425f2ea` | `20260909T013716356200839Z` | `b44d65ca34d41ad526ade62e7a62e0dcaffe2aa42191a228d5112083d67c6dea` |
| 6.4 | (see 11.1) | `20260909T114202240674964Z` | `c84940f55016eecdbd7479cbbf07346a08620119ee04b6e0aa69e8bd03b52426` |
| 7 | `1bb4ff6947a31e962465b567062deefb6a13054a82b0662f4ae37d4316915bb1` | `20260911T053155161422662Z` | `f6a636838d9fc30d4e76580e0ed665c338d4b830b7b9157b7dad2eb0775d04db` |
| ISO | (see 11.1) | `20260920T084300Z` | — (not tarred) |
| 8 | (see 11.1) | `20260922T014916Z` | — (not tarred) |

### 11.3 Run-identity hashes

| run | config sha256 | package manifest | notes |
|---|---|---|---|
| 6.1 | `dace37cf72659b667a917b671fcab75ddcee0307da054e5434044c2199203e34` | — | accepted Stage 6 config `ce47a92c518c98c9b3504c9c3e5bba2f4d9182a5a0e8ad9f73168197a13eee18`; `03_DAILY_TRACE.csv` `c645ed5014e0adf53c73fcedb9c6bdd18b234b067f2eb94c9ed4423e1f77d9b5`; complete `7f2caecb476d77394cc004f937e9bc774596f5b492cc6e6f52011954db4e7e93` |
| 6.2 | `64f8a64c2d40e2253ba2ed4c5b63be1952b4f6e074113d272b83673a5f579240` | `d8ee5ef548621ccb32ace9c5eaac5cbf6579f3393cadd3b1ed26ea82e297449c` | trace `da29884c0c44709c2cf72fa282ba69769621e9e78e1d0fcb5abb147b06dee3a3`; complete `679a8d7bd95048615de8f3740a1fa2f20162bc55f2e3072543b822c910bb2d31` |
| 6.3 | — | — | trace `91c5560294c3ca161f6196452dc7021a028b71a330503e9abba25ee4116af134`; complete `b1e5ac3b898de5b3616ffab76746ef6deee72da2e8015177d69168f0d9cb6cd0` |
| 6.4 | — | `6a03c967…` | source audit `4d43ba772eced5ef441b78ce813372c8ee2260c6c34e3c25fcc245280bf191a6`; `COMPARISONS.json` `5db18af8340d1a4682b3c506b840223b6d5d6813372c253bded86de3c7e85faa`; `daily_ledger.csv` see `RUN_COMPLETE.json` |
| ISO | `58923886fb1173e27cbacaf654412fc6e0b4354b5f7d1e0f9a43208e5b1887b1` | `8bc1a6e0…` | preflight `4b1b42de682ea08ff8102cbdb48967c2746ad6cac769965dfcd4557e7cf110d3`; neutral prompt `569897b2f09b09acda6d3b52645904f7eb7a179bb3751c46b79d51f091c8aedd` |
| 8 | `bb43811e1dcfdfcb92123ec5cc4e53d7869980906c9faebba414b2904ce8fbec` | `4a8eaf0a…` | imports Stage 6.4 manifest `6a03c967…` |
| consolidation | `consolidate.py` `37df1610744ed4aed4bff81988c26faa00a624dd02a9ce2dcf15709c36eb7953` | `b28c5ebe…` | outputs: `ALL_ARMS_DAILY.csv` `ac4b11f80f8cb34de813884f9f46297417f44359d3f3d9980a2340025cc76987`, `ARM_METRICS.csv` `2b664990b9fa37bfe5e0af8e5e55c22d0507988ca568f6cc9ee16d0be4a81292`, `MEMORY_CONTRASTS.csv` `33f5270f182ef543ee7e6eafacab06de7851b5b645f0f092bbfc83b411660887`, `POOLED_MEMORY_EFFECT.json` `6638346b6f4abd56da7b4f5ca0af6e7f7a8203238855a5be854ff93f0b53c528`, `CROSS_RUN_REPLICATION.csv` `d504ba0eb2b982ffec6de3922212ae3b5f26ddb0cab2d7c99854273341b0e260`, `ARM_INVENTORY.csv` `7e2806d1cc725bae7eda226226c1c5695db471837641f3c533ea5243dbedcf8f` |
| frozen inputs | core config `06b6516a190572d8f7edeb01404a817a931078aed9fe7f4599715cd97d12e6a3` | core manifest `cd672e46ae0874127f9495148641f19fb3c174228ffb7001b65b6903ecc40f24` | raw data `b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e`; frozen prompt `6b99d1922bf13d280e23c4f62bf82d3dbea2c2032f00be6208a83d8e5bb86783`; model digest `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa`; serving metadata `f731e05070648c6ae023793c866a34064f5d09cd34cd40b37529d50cca9e8d8d`; terminal W (pre-2024 period) `3c96a5bceb898dbc319c0fd5f7db03dfdb7191d65aa50a1fc4f59a93c37bcba0` |

Per-arm ledger hashes for all 24 consolidated arms are in `EXPERIMENT_BRANCHES/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/20260921T000000Z/SOURCE_MANIFEST.json` (`input_files_sha256`); per-file hashes of every run are in that run's `RUN_COMPLETE.json`.

---

## 12. Appendices

### 12.1 Glossary

| term | meaning here |
|---|---|
| numerical core / pre-agent RAMoE control | the deterministic allocator of §2.1, run with continuous holdings and the advisor forced to ABSTAIN |
| advisor | Llama-3.3-70B emitting BTC / CASH / ABSTAIN once per day |
| β, trust | the per-regime blend weight of the advisor's vote; fixed 0.05 or adaptive in [0, 0.2] |
| controller | the CVaR / 0.05-grid / 0.35-turnover projection applied to the blended exposure |
| episode | one completed day: state vector, action, stored label, realized asset return |
| label (`shadow_log_advantage_vs_ramoe`) | log-return advantage of the full-authority shadow of the action over the archived reference path |
| expanding / frozen2021 / current_year / pooled | memory visibility variants (§2.4) |
| balanced retrieval | Stage 8: nearest 2 completed episodes per action, no regime prefilter |
| closed loop | each arm evolves its own holdings, memory and trust from its own decisions |
| state-controlled | both arms see the same current-state stream; only memory visibility differs |
| transmission | counts of days on which two arms differ in action → desired exposure → executed exposure |
| merged | a desired-exposure difference that the controller projected to the same grid point |
| POST2021 | return dates 2022-01-01 → 2025-05-28, 1,244 days |
| inconclusive | the 95% block-bootstrap interval covers zero; not equivalence |

### 12.2 Formulas

```
gross_i     = exposure_i · r_i                              r_i = close[d+1]/close[d] − 1
net_i       = (1 − 0.001 · turnover_i) · (1 + gross_i) − 1
wealth_i    = wealth_{i−1} · (1 + net_i)
pretrade_{i+1} = exposure_i (1 + r_i) / (1 + exposure_i r_i)          (holdings drift)
turnover_i  = |exposure_i − pretrade_i|
net return  = expm1( Σ log1p(net_i) )
bps/day     = 1e4 · mean( log1p(net_i^A) − log1p(net_i^B) )            (paired, same days)
vol         = std(net_i, ddof=1) · sqrt(365.25);  Sharpe = mean(net_i)/std(net_i) · sqrt(365.25)
Sortino     = mean(net_i) / sqrt(mean(min(net_i,0)^2)) · sqrt(365.25)
ES95        = mean of the worst 5% of net_i (fractional weight on the boundary observation), as a loss
MDD         = max_t (1 − wealth_t / max_{s≤t} wealth_s), including wealth_0 = 1
blend       = (1 − β)·base + β·target,  target(BTC)=1, target(CASH)=0, ABSTAIN → base
block bootstrap: n=1244 paired differences; blocks of 30 consecutive days drawn circularly; ceil(1244/30) blocks per resample; 5,000 resamples; rng = numpy.default_rng(16062); CI = 2.5th and 97.5th percentiles of the resampled means
```

### 12.3 Supplementary-table recipes

| supplementary item | build from |
|---|---|
| S1 arm register (stage, run id, factor settings, seed, prompt, provenance) | `…/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/20260921T000000Z/ARM_INVENTORY.csv` + Stage 8 `00_RUN_CONTRACT.json` `arms` + Stage 6.4 `00_CONTRACT.json` `variants` |
| S2 per-arm metrics, all arms | `ARM_METRICS.csv` (24 LLM arms) + `…/RAMAS_STAGE6_4_COMPONENT_SUITE_V1/artifacts/20260909T114202240674964Z/full_period_metrics.csv` (scope POST2021; benchmarks, rule and numerical arms) + Stage 8 `ALL_DAILY_LEDGER.csv` via §10.5 |
| S3 all bootstrap contrasts | Stage 6.4 `COMPARISONS.json` (22), Stage 7 `COMPARISONS.json` (2), Stage 8 `COMPARISONS.json` (1), `MEMORY_CONTRASTS.csv` (8, recomputed); cite `COMPARISONS.json`, never the empty `paired_comparisons.json` |
| S4 transmission table | Stage 6.4 `COMPONENT_TRANSMISSION.csv` (regime `ALL`), Stage 7 `TRANSMISSION.csv` (`POST2021`), factorial and Stage 8 from ledgers: for a pair, restrict to POST2021, count `action_a ≠ action_b`, `desired_exposure_a ≠ desired_exposure_b`, `exposure_a ≠ exposure_b` (use `numpy.isclose`) |
| S5 retrieval diversity | Stage 8 `arms/corrected_memory/DAILY_LEDGER.csv` column `retrieved_action_diversity`; legacy from `…/diagnostics/MEMORY_DIAGNOSTICS.json` `retrieval_collapse` (or recount distinct `action` values in `request.memory.similar_completed_episodes` of each `…/arms/llama_memory_current_year/calls/*.json`) |
| S6 behavioural signature / risk deltas | per pair, POST2021 rows of both ledgers: ABSTAIN share of `action`, mean `exposure`, vol, ES95, Sharpe, Sortino, MDD, Σ `turnover`, Σ `cost_fraction`; memory minus no-memory |
| S7 factorial main-effect intervals (**to compute**) | for each factor, per day take the mean daily net log return of the "on" arms minus the "off" arms (advisor: 8 LLM arms vs 2 numerical controls), then apply the §12.2 block bootstrap to that daily series (seed 16062 or the package's 20260919; state which) |
| S8 identities | §11 of this report; `provider_identity` blocks in each run contract |
| S9 yearly and regime breakdowns | Stage 6.4 `yearly_metrics.csv`, `regime_metrics.csv`, `monthly_metrics.csv`; Stage 7 `year_regime_metrics.csv`; Stage 6.1 `05_YEARLY_METRICS.csv` (regime scopes carry no annualized statistics by convention) |
| S10 trust paths | `trust_events.json` per arm (adaptive arms): `effective_year/month, regime, old_beta, new_beta, direction, eligible_completed_episodes` |

### 12.4 Known traps for anyone re-analysing

1. Wrong interpreter → every journal digest misses; looks like a logic bug, is not.
2. `paired_comparisons.json` in the Stage 6.4 run is an empty list by design; the contrasts are in `COMPARISONS.json`.
3. Stage 6.4's consolidated `daily_ledger.csv` republishes the 6.2/6.3 trajectories under `adaptive_*` / `fixed_*`; do not count them twice when pooling.
4. `ARM_METRICS.csv` returns are in percent; `full_period_metrics.csv` returns, volatilities, drawdowns and ES are fractions.
5. Adaptive pairs differ in desired exposure on days the action agrees (β paths diverge); this is a total-system effect, not an error.
6. Regime-conditional selections are non-contiguous; never annualize them.
7. The `credit_label_validity` block in `MEMORY_DIAGNOSTICS.json` and the motivation paragraph in Stage 8's `EXPERIMENT_CONTRACT.md` describe a withdrawn diagnostic (§6.9.4).
8. Absolute paths inside archived JSON point at the original machine; they are provenance, not instructions.

*End of report.*
