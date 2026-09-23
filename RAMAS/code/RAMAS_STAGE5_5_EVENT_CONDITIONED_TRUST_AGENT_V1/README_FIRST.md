# RAMAS Stage 5.5: event-conditioned trust LLM-agent

This package implements the approved mechanism branch. It does not run a
profitability replay and does not open reused OOS data.

## What changed from Stage 5.3

Stage 5.3 asked the LLM-agent for a daily exposure residual and found no
incremental value. Stage 5.5 prohibits direct exposure output. The agent is
called only by uncertainty, transition, disagreement, risk, or verified-event
triggers. It investigates with tools and may propose a temporary bounded
change to one row of the regime–expert trust matrix.

Daily RAMoE execution remains deterministic. The agent cannot bypass the risk
shield. Tool provenance is attached automatically instead of asking the model
to copy citation identifiers.

## First server run

```bash
cd /home/infonet/wahid/leader_router_fresh

sha256sum -c RAMAS_STAGE5_5_EVENT_CONDITIONED_TRUST_AGENT_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_STAGE5_5_EVENT_CONDITIONED_TRUST_AGENT_V1_CODE.tar.gz

nohup env \
  RAMAS_PROVIDER=ollama \
  RAMAS_MODEL=qwen3-coder:30b \
  bash RAMAS_STAGE5_5_EVENT_CONDITIONED_TRUST_AGENT_V1/run_server.sh \
  > RAMAS_STAGE5_5_RUN.log 2>&1 &

echo $!
```

Monitor:

```bash
tail -f /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE5_5_RUN.log
```

The first run executes 12 synthetic interface cases. A correct run normally
ends with:

```text
DECISION=MECHANISM_PASS_AWAIT_POINT_IN_TIME_EVENT_DATA
```

That is the expected result, not a failed experiment. It means the software
mechanism passed but authentic event evidence has not been supplied.

## Run the event-data gate later

After creating a real ledger and manifest following `EVENT_DATA_SCHEMA.md`:

```bash
nohup env \
  RAMAS_PROVIDER=ollama \
  RAMAS_MODEL=qwen3-coder:30b \
  RAMAS_EVENT_CSV=/absolute/path/events.csv \
  RAMAS_EVENT_MANIFEST=/absolute/path/event_manifest.json \
  bash RAMAS_STAGE5_5_EVENT_CONDITIONED_TRUST_AGENT_V1/run_server.sh \
  > RAMAS_STAGE5_5_DATA_GATE.log 2>&1 &
```

A data-gate pass authorizes construction of the causal pre-2024 development
adapter. It still does not authorize an economic claim or reused-OOS replay.

## Expected runtime

The Ollama preflight makes 24 calls: one planning and one decision call for
each of 12 cases. Qwen3-Coder 30B is the default. Llama 70B should be used only
as a later robustness audit because the earlier diagnostic required roughly
95 seconds per response and did not solve the evidence problem.

