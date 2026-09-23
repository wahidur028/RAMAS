# Llama-70B consolidated evidence — RAMAS

Generated 2026-09-22T01:49:03+00:00 by `RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/consolidate.py`.
Read-only consolidation of every completed `llama3.3:70b` arm in the project.
No model calls were made. All numbers are recomputed from archived daily
trajectories with the frozen Stage 6.4 metrics module.

**Model** `llama3.3:70b`, digest `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa`
**Evaluation window** 2022-01-01 to 2025-05-28 (1,244 paired days per arm)
**Reporting standard** net return after trading costs, 10 bps proportional, annualization 365.25
**Runtime** python 3.11.9, numpy 1.26.4, pandas 2.2.3

## 1. What is included

| stage | run_id | arm | trust | memory | controller | core | sampling_seed | prompt_variant |
|---|---|---|---|---|---|---|---|---|
| 6.1 | 20260907T012534Z | stage61_continuous | adaptive | expanding | on | original | 16061 | frozen_llama_role |
| 6.2 | 20260908T055052711433762Z | adaptive_memory | adaptive | expanding | on | original | 16061 | frozen_llama_role |
| 6.2 | 20260908T055052711433762Z | adaptive_no_memory | adaptive | none | on | original | 16061 | frozen_llama_role |
| 6.3 | 20260909T013716356200839Z | fixed_memory | fixed | expanding | on | original | 16061 | frozen_llama_role |
| 6.3 | 20260909T013716356200839Z | fixed_no_memory | fixed | none | on | original | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_frozen_W | adaptive | expanding | on | frozen_W | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_hard_allocator | adaptive | expanding | on | hard_allocator | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_memory_current_year | adaptive | current_year | on | original | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_memory_frozen2021 | adaptive | frozen2021 | on | original | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_memory_pooled | adaptive | pooled | on | original | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_no_risk | adaptive | expanding | off | original | 16061 | frozen_llama_role |
| 6.4 | 20260909T114202240674964Z | llama_uniform_allocator | adaptive | expanding | on | uniform_allocator | 16061 | frozen_llama_role |
| 7 | 20260911T053155161422662Z | closed_loop_memory | fixed | expanding | on | original | 16061 | frozen_llama_role |
| 7 | 20260911T053155161422662Z | closed_loop_no_memory | fixed | none | on | original | 16061 | frozen_llama_role |
| 7 | 20260911T053155161422662Z | state_controlled_memory | fixed | expanding | on | original | 16061 | frozen_llama_role |
| 7 | 20260911T053155161422662Z | state_controlled_no_memory | fixed | none | on | original | 16061 | frozen_llama_role |
| ISO | 20260920T084300Z | llm_adaptive_memory_controller_off | adaptive | expanding | off | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_adaptive_memory_controller_on | adaptive | expanding | on | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_adaptive_no_memory_controller_off | adaptive | none | off | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_adaptive_no_memory_controller_on | adaptive | none | on | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_fixed_memory_controller_off | fixed | expanding | off | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_fixed_memory_controller_on | fixed | expanding | on | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_fixed_no_memory_controller_off | fixed | none | off | original | 42 | model_neutralized_frozen_v1 |
| ISO | 20260920T084300Z | llm_fixed_no_memory_controller_on | fixed | none | on | original | 42 | model_neutralized_frozen_v1 |

Total 24 arms, 38,592 daily rows. Accounting identity re-verified on
every arm; worst absolute error 2.22e-16.

## 2. Headline: the memory effect across 8 independent realizations

| stage | memory_arm | trust | controller | sampling_seed | mean_daily_net_log_difference_bps | ci95_lower_bps | ci95_upper_bps | net_compounded_gap_pp | interpretation |
|---|---|---|---|---|---|---|---|---|---|
| 6.2 | adaptive_memory | adaptive | on | 16061 | -0.0017 | -0.1265 | 0.0992 | -0.0351 | INCONCLUSIVE_NOT_EQUIVALENCE |
| 6.3 | fixed_memory | fixed | on | 16061 | 0.0369 | -0.0340 | 0.1221 | 0.7726 | INCONCLUSIVE_NOT_EQUIVALENCE |
| 7 | closed_loop_memory | fixed | on | 16061 | 0.0543 | -0.0256 | 0.1521 | 1.1387 | INCONCLUSIVE_NOT_EQUIVALENCE |
| 7 | state_controlled_memory | fixed | on | 16061 | -0.1075 | -0.2335 | -0.0042 | -2.2047 | B_HIGHER_ON_THIS_RESAMPLING_SPECIFICATION |
| ISO | llm_adaptive_memory_controller_off | adaptive | off | 42 | -0.0074 | -0.1053 | 0.0836 | -0.1527 | INCONCLUSIVE_NOT_EQUIVALENCE |
| ISO | llm_adaptive_memory_controller_on | adaptive | on | 42 | 0.0334 | -0.0602 | 0.1407 | 0.7043 | INCONCLUSIVE_NOT_EQUIVALENCE |
| ISO | llm_fixed_memory_controller_off | fixed | off | 42 | -0.0114 | -0.0566 | 0.0335 | -0.2321 | INCONCLUSIVE_NOT_EQUIVALENCE |
| ISO | llm_fixed_memory_controller_on | fixed | on | 42 | 0.0110 | -0.0609 | 0.0918 | 0.2303 | INCONCLUSIVE_NOT_EQUIVALENCE |

**Pooled across all 8 realizations: +0.000972 bps/day**
(4 positive, 4 negative; spread -0.1075 to +0.0543 bps/day).
Mean compounded gap +0.0277 pp.

> All realizations replay the same 1,244 days of market history. They are repeated implementations, not independent samples, so no pooled confidence interval is reported and the spread across realizations must not be read as a standard error.
>
> Eight unadjusted 95% intervals were computed. At nominal coverage roughly 0.4 false positives are expected, so a single significant interval is unremarkable. The one that excludes zero favours NO-memory, not memory.

## 3. Cross-run replication of matched configurations

| config_key | realizations | stages | seeds | mean_net_compounded_return_pct | min_pct | max_pct | spread_pp |
|---|---|---|---|---|---|---|---|
| trust=adaptive|memory=expanding|controller=off|core=original | 2 | 6.4,ISO | 16061,42 | 66.1882 | 65.9883 | 66.3881 | 0.3997 |
| trust=adaptive|memory=expanding|controller=on|core=original | 3 | 6.1,6.2,ISO | 16061,42 | 69.0110 | 67.7304 | 69.6870 | 1.9566 |
| trust=adaptive|memory=none|controller=on|core=original | 2 | 6.2,ISO | 16061,42 | 68.3384 | 67.7655 | 68.9113 | 1.1458 |
| trust=fixed|memory=expanding|controller=on|core=original | 4 | 6.3,7,7,ISO | 16061,42 | 67.4179 | 63.7490 | 68.9907 | 5.2417 |
| trust=fixed|memory=none|controller=on|core=original | 4 | 6.3,7,7,ISO | 16061,42 | 67.4337 | 65.9538 | 68.0768 | 2.1231 |

## 4. Full per-arm metrics

See `ARM_METRICS.csv`. Post-2021 net compounded return by arm:

| stage | arm | net_compounded_return_pct | annualized_sharpe_rf_zero | maximum_drawdown_pct | mean_btc_exposure_pct | turnover_sum |
|---|---|---|---|---|---|---|
| 6.4 | llama_frozen_W | 77.3842 | 0.7621 | 39.7029 | 49.5981 | 8.8391 |
| 6.4 | llama_memory_frozen2021 | 69.9025 | 0.7603 | 37.0866 | 44.9638 | 11.6154 |
| 6.1 | stage61_continuous | 69.6870 | 0.7586 | 37.0866 | 44.9839 | 11.5205 |
| ISO | llm_adaptive_memory_controller_on | 69.6156 | 0.7586 | 37.0866 | 44.9035 | 10.9206 |
| 7 | closed_loop_memory | 68.9907 | 0.7634 | 36.2533 | 44.4654 | 10.9480 |
| ISO | llm_adaptive_no_memory_controller_on | 68.9113 | 0.7528 | 37.0866 | 45.0201 | 11.3576 |
| 6.3 | fixed_memory | 68.6247 | 0.7607 | 36.2533 | 44.4574 | 11.0943 |
| 6.4 | llama_memory_pooled | 68.5632 | 0.7514 | 37.0866 | 44.8674 | 11.2929 |
| 6.4 | llama_uniform_allocator | 68.4750 | 0.7566 | 36.3614 | 44.2122 | 10.3528 |
| 6.4 | llama_memory_current_year | 68.4428 | 0.7508 | 37.0866 | 44.8553 | 11.6839 |
| ISO | llm_fixed_memory_controller_on | 68.3072 | 0.7584 | 36.2533 | 44.4614 | 10.9814 |
| ISO | llm_fixed_no_memory_controller_on | 68.0768 | 0.7561 | 36.2533 | 44.5016 | 10.6061 |
| 7 | closed_loop_no_memory | 67.8521 | 0.7544 | 36.2533 | 44.5096 | 10.4062 |
| 6.3 | fixed_no_memory | 67.8521 | 0.7544 | 36.2533 | 44.5096 | 10.4062 |
| 6.2 | adaptive_no_memory | 67.7655 | 0.7450 | 37.0866 | 44.9397 | 11.3750 |
| 6.2 | adaptive_memory | 67.7304 | 0.7452 | 37.0866 | 44.8834 | 11.3802 |
| 6.4 | llama_hard_allocator | 67.7175 | 0.7444 | 36.9019 | 44.9759 | 11.4928 |
| 6.4 | llama_no_risk | 66.3881 | 0.7373 | 36.6127 | 44.9528 | 10.4105 |
| ISO | llm_adaptive_no_memory_controller_off | 66.1410 | 0.7330 | 36.5422 | 45.1077 | 10.0723 |
| ISO | llm_adaptive_memory_controller_off | 65.9883 | 0.7341 | 36.6127 | 44.8940 | 9.9764 |
| 7 | state_controlled_no_memory | 65.9538 | 0.7406 | 36.3444 | 44.4936 | 24.9866 |
| ISO | llm_fixed_no_memory_controller_off | 64.2157 | 0.7311 | 36.0523 | 44.3685 | 8.5984 |
| ISO | llm_fixed_memory_controller_off | 63.9836 | 0.7300 | 36.0850 | 44.3130 | 8.9748 |
| 7 | state_controlled_memory | 63.7490 | 0.7269 | 36.3444 | 44.2846 | 22.6150 |

## 5. Reproduction

```bash
cd /home/infonet/wahid/leader_router_fresh/RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1
/home/infonet/anaconda3/envs/wahid_test/bin/python consolidate.py --output <dir>
```

`SOURCE_MANIFEST.json` records the SHA-256 of every input file consumed and every
output written, plus the pinned runtime. The pinned interpreter is required: a
different numpy BLAS build changes the last bit of derived quantities.

## 6. Claim boundaries

These remain unresolved and are not addressed by this consolidation:

- 2024-2025 is reused diagnostic history, not fresh out-of-sample.
- The regime posterior targets the allocation day, not the holding day, and its
  fit chronology is not independently verified.
- Decisions use close[d] and are filled at close[d]; inference latency and
  realistic fills are not modelled.
- The archived credit label (`shadow_log_advantage_vs_ramoe`) compares a
  full-authority shadow portfolio against a different holdings path.
