# RAMAS Stage 5.4: Llama-70B semantic-interface diagnostic

This package does **not** rerun the trading experiment. It tests whether
`llama3.3:70b` follows the Stage 5.3 decision contract better than
`qwen3-coder:30b` on 30 frozen decision contexts.

The deterministic sample contains:

- the single context with an eligible nonzero residual;
- 19 contexts where Qwen proposed an ineligible nonzero residual;
- 10 contexts where Qwen abstained.

The original prompt and permissive citation schema are retained so this is a
fair model-interface comparison. The diagnostic measures JSON validity,
agreement with tool eligibility, complete citations, and semantic acceptance.

## Run

```bash
cd /home/infonet/wahid/leader_router_fresh

sha256sum -c RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1_CODE.tar.gz

nohup bash RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1/run_server.sh \
  > RAMAS_STAGE5_4_LLAMA70B_RUN.log 2>&1 &

echo $!
tail -f RAMAS_STAGE5_4_LLAMA70B_RUN.log
```

If automatic Stage 5.3 discovery fails, provide the exact results directory:

```bash
export RAMAS_STAGE53_RESULTS=/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE5_3_BOUNDED_TOOL_USING_RESIDUAL_AGENT_V1/artifacts/20260902T054019Z/results
bash RAMAS_STAGE5_4_LLAMA70B_SEMANTIC_INTERFACE_DIAGNOSTIC_V1/run_server.sh
```

## Interpretation

`LLAMA70B_INTERFACE_BETTER_NO_ECONOMIC_CLAIM` means the larger model fixed the
semantic interface on these cases. It does not reverse the Stage 5.3 economic
failure. Any other decision means the larger model did not reliably solve the
interface problem.
