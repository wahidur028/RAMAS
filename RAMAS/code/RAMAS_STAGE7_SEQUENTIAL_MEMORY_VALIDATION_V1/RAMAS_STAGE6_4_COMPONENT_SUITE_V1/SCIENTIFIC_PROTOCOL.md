# RAMAS Stage6.4: sequential component suite

Status: implemented experimental protocol; real new-arm results require the user's server. This does not change previous Stage6.2/6.3 experiments or promote a fixed-beta ablation to the main method.

## Why this suite

A profitable integrated portfolio does not identify which component caused the profit. This suite keeps one accounting definition and separates external benchmarks from internal component interventions. It executes all specified arms in a fixed sequence, saves each completed arm, and continues through economic losses. There is no favorable-2021 or “beat every metric” gate. Technical/source failures stop with evidence retained.

## Execution order and exact scope

| Step | Work | Interpretation |
|---|---|---|
| 1 | Verify all hashes in pinned completed Stage6.2 and6.3 runs; normalize their four real trajectories | Reuse existing evidence; zero new inference |
| 2 | Recompute cash, B&H and daily-rebalanced50/50 on identical holding days | External baselines with their own costs |
| 3 | Load exact raw BTC/source, reconstruct causal scenarios, compare original archived states/previews, reconstruct W | Numeric integrity and inherited-clock diagnostics |
| 4 | Existing rule at beta.05, existing rule with adaptive beta, numerical-only ABSTAIN | Llama-advisor and trust attribution controls |
| 5 | Numerical-only with hard aggregation q, uniform aggregation q, frozen initial W, no risk projection | Cheap allocation/risk mechanism effects |
| 6 | Full adaptive-memory Llama with hard aggregation q, uniform aggregation q, frozen initial W, no risk projection | Corresponding whole-pipeline interventions with new decisions and independent states |
| 7 | Full adaptive-memory Llama with frozen2021, current-year-only or pooled-regime retrieval | Accumulation, cross-year access and retrieval-selection interventions |
| 8 | Unified daily/monthly/yearly/regime statistics, action transmission and paired uncertainty | Interpret technical, economic, null and conditional findings |

Four archived arms plus three passive portfolios plus14newarms produce21portfolio records per date in full mode. Seven new arms call Llama; each has1,608daily decisions, maximum11,256new trading calls. No new architecture variants are selected from winning results. This is not an exhaustive factorial search of every component interaction.

**Adaptive-memory Stage6.2 is the main-method reference.** Existing fixed-memory/no-memory Stage6.3 addresses advice influence under fixed beta. Primary new comparison is fixed_memory versus rule_fixed. All other component and external comparisons are secondary/exploratory. Report the complete table, not only favorable cells. Multiple unadjusted95% intervals must not be sold as simultaneous95% coverage.

### Allocation router and policy weights

The actual active vector is Cash/B&H within a six-slot registry. Original W is reconstructed with the frozen source's independent policy evidence and monthly updates, and checked against all original desired exposures and terminal W matrices. Its initial admitted rows are [.5,.5,0,0,0,0].

Hard and uniform **allocator-only** interventions change q in qWe. Original q continues into W's historical update stream, the prompt, regime selection and retrieval. Hard q uses argmax with the recorded Bear/Bull/Mix index order. Uniform aggregation q=[1/3,1/3,1/3] has no prompt/argmax tie consequence because context q is unchanged. These tests identify the contribution of soft versus hard/unconditional aggregation, not whether the complete regime predictor is necessary or calibrated.

Frozen-W uses initial equal admitted Cash/B&H rows throughout. This differs from external static50 because RAMAS still applies risk and advice. Original W uses historical policy returns, not Llama outcomes, so the common W stream is a defined intervention input. W carries across the2024source boundary; independent policy holdings and the numerical reference have legacy resets. Those historical features are disclosed and preserved in reconstruction, not silently repaired.

The four inactive policies retain their actual proposals, but zero admitted weight gives exactly zero current contribution. The source audit verifies this mathematically. Their activation/retraining is a different policy-admission experiment and is not manufactured as a useful deletion ablation.

### Risk

“No risk” removes the projection block (CVaR constraint, exposure grid and turnover constraint together), while keeping0≤BTCweight≤1 and the same transaction charge. Its Llama previews are recalculated accordingly. This is not an isolated estimate of CVaR alone. Standard-risk arms recompute projection from their own drifted holdings and actual causal scenario matrix; archived three-action previews cannot substitute for new state.

### Memory and learning

All new arms begin with empty2021memory and independent wealth, holdings, shadow holdings and agent trust. No annual portfolio/trust reset. Only completed episodes are eligible.

- Expanding: original same-regime nearest-neighbor retrieval.
- Frozen2021: retrieve only eligible episodes whose outcome date≤2021-12-31; later decisions still update the private trust ledger.
- Currentyear: retrieve only eligible outcomes from the current decision year; never reset wealth/trust/private history.
- Pooled: remove only the same-regime prefilter, preserving distances, ordering, budget and true episode labels.

These new memory variants retain adaptive trust. Their contrast against the adaptive-memory main method is the **total system response** to a retrieval intervention, including subsequent changed actions, outcomes and trust. It is not a pure memory effect with future beta paths fixed. The existing Stage6.3 fixed-beta pair supplies a separate simpler memory-access contrast. More positive annual returns are not evidence of learning by themselves.

The model's pretrained parameters stay frozen. The historical shadow-log advantage reference is identical across arms, including the known seam. It is an internal feedback score, not the marginal contribution of the deployed low-beta action. The no-agent comparison uses a newly continuous portfolio, not the archived reset reference; this does not retroactively repair rewards already consumed by archived Llama.

## Reproducibility and boundaries

Same pinned Llama70B digest, prompt, Ollama serving metadata/options and Python/NumPy/pandas versions are required for real replay. New altered-context Llama arms use fresh inference; no cross-arm reuse of previously observed advice is performed. Each arm journals exact requests and responses before scoring outcomes. Restart reconstructs state from durable responses and refuses mismatches; already saved responses are not resampled. An unsaved in-flight response after a crash can require retry, as in Stage6.3.

The wrapper first verifies code/tests, then runs the suite sequentially. Completed arms have identity/hashes and are skipped on resume. An interruption leaves completed reports and call journals. Economic loss never stops execution. Invalid contracts are recorded as ABSTAIN; three consecutive invalid outputs or a valid-rate below.98 trigger a technical stop. A transport outage stops before scoring that decision.

**Still unresolved:** original forecast fitting/label availability, forecast target versus earned-return interval, executable close/data/inference/fill schedule, the reference/expert-holdings seam, and pretrained historical knowledge contamination. Source audits report what they can verify; they do not fabricate training provenance. Clock/reference correction requires matched corrected Llama/control reruns. None of the current dates are fresh OOS. Model-seed robustness, independent ML/full-LLM/literature baselines, and an empirical novelty claim require additional dedicated evidence. TEST_STATUS.json explicitly distinguishes complete, pending and unresolved questions.
