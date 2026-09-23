"""Descriptive pooling of repeated memory contrasts, plus the report writer.

The realizations share the SAME 1,244 days of market history, so they are not
independent samples. Pooling summarizes agreement across implementations; it
does NOT justify a narrower interval than the individual bootstraps.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


def pooled_memory_effect(contrasts: pd.DataFrame) -> dict:
    usable = contrasts[contrasts["mean_daily_net_log_difference_bps"].notna()].copy()
    usable["approx_se_bps"] = (usable["ci95_upper_bps"] - usable["ci95_lower_bps"]) / (2 * 1.959964)

    def summarize(frame: pd.DataFrame, label: str) -> dict:
        values = frame["mean_daily_net_log_difference_bps"].to_numpy(float)
        gaps = frame["net_compounded_gap_pp"].to_numpy(float)
        return {
            "subset": label,
            "realizations": int(len(values)),
            "mean_daily_net_log_difference_bps": float(values.mean()),
            "sd_across_realizations_bps": float(values.std(ddof=1)) if len(values) > 1 else None,
            "min_bps": float(values.min()), "max_bps": float(values.max()),
            "positive_realizations": int((values > 0).sum()),
            "negative_realizations": int((values < 0).sum()),
            "mean_compounded_gap_pp": float(gaps.mean()),
            "range_compounded_gap_pp": [float(gaps.min()), float(gaps.max())],
            "individually_inconclusive": int((frame["interpretation"] == "INCONCLUSIVE_NOT_EQUIVALENCE").sum()),
            "individually_significant": int((frame["interpretation"] != "INCONCLUSIVE_NOT_EQUIVALENCE").sum()),
            "significant_directions": sorted(set(
                frame.loc[frame["interpretation"] != "INCONCLUSIVE_NOT_EQUIVALENCE", "interpretation"])),
        }

    primary = usable[usable["note"].fillna("") == ""]
    result = {
        "endpoint": "MEAN_DAILY_NET_LOG_RETURN_DIFFERENCE_BPS_MEMORY_MINUS_NO_MEMORY",
        "evaluation_window": "2022-01-01..2025-05-28, 1,244 paired days per realization",
        "pooling_is_descriptive_not_meta_analytic": True,
        "why": ("All realizations replay the same 1,244 days of market history. They are repeated "
                "implementations, not independent samples, so no pooled confidence interval is "
                "reported and the spread across realizations must not be read as a standard error."),
        "all_realizations": summarize(usable, "all"),
        "closed_loop_only": summarize(primary, "closed_loop_protocols_only"),
        "by_realization": usable[[
            "stage", "run_id", "memory_arm", "trust", "controller", "sampling_seed",
            "prompt_variant", "note", "mean_daily_net_log_difference_bps",
            "ci95_lower_bps", "ci95_upper_bps", "net_compounded_gap_pp", "interpretation",
        ]].to_dict("records"),
        "multiplicity_note": ("Eight unadjusted 95% intervals were computed. At nominal coverage "
                              "roughly 0.4 false positives are expected, so a single significant "
                              "interval is unremarkable. The one that excludes zero favours "
                              "NO-memory, not memory."),
    }
    return result


REPORT_HEADER = """# Llama-70B consolidated evidence — RAMAS

Generated {created} by `RAMAS_LLAMA70B_EVIDENCE_CONSOLIDATION_V1/consolidate.py`.
Read-only consolidation of every completed `llama3.3:70b` arm in the project.
No model calls were made. All numbers are recomputed from archived daily
trajectories with the frozen Stage 6.4 metrics module.

**Model** `{model}`, digest `{digest}`
**Evaluation window** {start} to {end} ({rows:,} paired days per arm)
**Reporting standard** net return after trading costs, 10 bps proportional, annualization 365.25
**Runtime** python {py}, numpy {np}, pandas {pd}

## 1. What is included

{inventory_table}

Total {arms} arms, {daily_rows:,} daily rows. Accounting identity re-verified on
every arm; worst absolute error {worst_acc:.2e}.

## 2. Headline: the memory effect across {n_real} independent realizations

{contrast_table}

**Pooled across all {n_real} realizations: {pooled_mean:+.6f} bps/day**
({pos} positive, {neg} negative; spread {lo:+.4f} to {hi:+.4f} bps/day).
Mean compounded gap {mean_gap:+.4f} pp.

{pooling_caveat}

## 3. Cross-run replication of matched configurations

{replication_table}

## 4. Full per-arm metrics

See `ARM_METRICS.csv`. Post-2021 net compounded return by arm:

{arm_table}

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
"""
