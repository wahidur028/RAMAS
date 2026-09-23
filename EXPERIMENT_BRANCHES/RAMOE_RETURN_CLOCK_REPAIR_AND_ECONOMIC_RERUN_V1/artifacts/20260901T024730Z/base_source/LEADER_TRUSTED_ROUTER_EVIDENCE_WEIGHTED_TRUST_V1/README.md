# LEADER Evidence-Weighted Trust Repair V1

This package repairs a mathematical defect in the Trusted Router's monthly regime-by-expert trust update.

The previous implementation selected the market-wide worst 5% of days and then conditioned that tail signal on each regime posterior. That is not regime-conditional CVaR. It also updated regime rows with almost no posterior evidence.

The repaired implementation:

- computes exact posterior-weighted CVaR separately for every regime and expert;
- requires posterior mass and effective sample size of at least 20, derived from the 5% tail probability;
- preserves unsupported trust rows exactly;
- records every supported and skipped update;
- compares the Router with constant exposure having the same mean BTC exposure;
- uses one fixed 30-day dependent block-bootstrap test;
- leaves all rejected specialists at exact zero weight.

## Server run

```bash
cd /home/infonet/wahid/leader_router_fresh
conda activate wahid_test

tar -xzf LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_V1_CODE.tar.gz

bash LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_V1/run_server.sh \
  /home/infonet/wahid/leader_router_fresh \
  2>&1 | tee LEADER_TRUSTED_ROUTER_EVIDENCE_WEIGHTED_TRUST_V1.log
```

`STATUS=PASS` means the repaired code ran correctly. The separate `ECONOMIC_GATE_PASSED` line determines whether the repaired Router earned economic acceptance. Specialist admission remains closed in either case.
