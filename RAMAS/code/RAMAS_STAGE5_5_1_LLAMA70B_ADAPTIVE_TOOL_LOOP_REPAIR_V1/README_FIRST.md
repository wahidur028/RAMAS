# RAMAS Stage 5.5.1: Llama-70B adaptive tool-loop repair

This package repairs the specific problem found in the Stage 5.5 result.

The old agent produced valid JSON, but it often proposed a trust action before
running every tool required for that action. The safety shield correctly
rejected those proposals, so zero interventions were executed.

The repaired loop is:

1. Llama plans and runs its first evidence tools.
2. Llama proposes a bounded trust action.
3. The runtime checks required evidence and the safety shield.
4. If evidence is missing, the runtime runs only the missing tools.
5. If the action is unsafe, the runtime returns the exact rejection reasons.
6. Llama gets one final correction round: confirm, reduce, change, or KEEP.
7. A second invalid or unsafe proposal is automatically changed to KEEP.

There are never more than two decision rounds. The agent still cannot predict
BTC exposure directly, create arbitrary weights, change the router, or bypass
the risk layer.

## Model lock

This experiment permits only:

```text
llama3.3:70b
```

The request context is limited to 16,384 tokens to reduce memory use. The model
is kept loaded for 15 minutes between calls. These settings do not change the
research contract.

## Run on the server

```bash
cd /home/infonet/wahid/leader_router_fresh

sha256sum -c \
  RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1_CODE.tar.gz.sha256

tar -xzf \
  RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1_CODE.tar.gz

nohup env \
  RAMAS_PROVIDER=ollama \
  RAMAS_MODEL=llama3.3:70b \
  bash RAMAS_STAGE5_5_1_LLAMA70B_ADAPTIVE_TOOL_LOOP_REPAIR_V1/run_server.sh \
  > RAMAS_STAGE5_5_1_LLAMA70B_RUN.log 2>&1 &

echo $!
```

Monitor the run:

```bash
tail -f \
  /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE5_5_1_LLAMA70B_RUN.log
```

Press `Ctrl+C` only after the final status is printed. This stops `tail`; it
does not stop the experiment.

## What counts as a mechanism pass

The 12-case interface gate now requires all of the following:

- at least 11 valid plans;
- at least 11 valid final decisions;
- at least 11 evidence-complete final actions;
- at least 11 shield-safe final actions;
- at least one accepted bounded intervention among three severe positive-path
  cases;
- no future return and no reused OOS data.

A successful mechanism-only run ends with:

```text
DECISION=MECHANISM_PASS_AWAIT_POINT_IN_TIME_EVENT_DATA
```

If the stronger interface gate fails, it ends with:

```text
DECISION=REVISE_LLAMA70B_ADAPTIVE_TOOL_LOOP_FAILED
```

Neither result proves higher profit or lower risk. Economic testing remains
blocked until a real, hashed, point-in-time event ledger passes the separate
data gate.

## Result files

At completion, the log prints exact paths named `RESULT_ARCHIVE` and
`RESULT_SHA256`. Upload both files for the next audit.

