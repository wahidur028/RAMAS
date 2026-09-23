# RAMAS suite results

Standard name: **net return after trading costs**. Log return is a separate additive metric.

Read METRIC_DEFINITIONS.md in the code package and TEST_STATUS.json here. PENDING/UNRESOLVED tests are not passes. The numerical/router controls are internal ablations. Cash, B&H and daily-rebalanced50/50 are external benchmarks.

The main method is adaptive-memory RAMAS. Fixed-beta results are attribution controls. Primary new contrast is fixed_memory minus rule_fixed; all other contrasts are secondary/exploratory. Intervals are pointwise, not adjusted for selecting among many tests. A larger return alone does not establish improvement in risk-adjusted performance.

This suite preserves inherited clocks/reference feedback for matched mechanism diagnostics. It does not certify executable fills, original router fitting chronology, fresh OOS, novelty or optimality. Changing the clock requires a separately versioned matched rerun.
