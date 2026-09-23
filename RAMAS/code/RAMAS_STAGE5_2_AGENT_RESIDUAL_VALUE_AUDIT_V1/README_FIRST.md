# RAMAS Stage 5.2 — agent residual value audit

This package does not call Ollama. It audits the completed Stage 5.1 result and
tests whether the LLM actions added anything beyond a transparent regime rule.

## Run on the original server

```bash
cd /home/infonet/wahid/leader_router_fresh

sha256sum -c RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1_CODE.tar.gz

bash RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1/run_server.sh
```

The runner automatically locates the newest Stage 5.1 result archive. To use an
explicit result archive:

```bash
bash RAMAS_STAGE5_2_AGENT_RESIDUAL_VALUE_AUDIT_V1/run_server.sh \
  /home/infonet/wahid/leader_router_fresh \
  /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMOE_STAGE5_1_SINGLE_LLM_AGENT_INTERFACE_REPAIR_V1/artifacts/20260902T004332Z.tar.gz
```

Expected runtime is minutes, not hours. `ollama ps` may remain empty because no
model is used.

## Read first after completion

- `09_FINAL_DECISION.json`
- `10_PLAIN_ENGLISH_REPORT.md`
- `08_INCREMENTAL_AGENT_GATE.json`
- `06_BEHAVIOR_AND_MEMORY_AUDIT.json`

Do not run another LLM experiment and do not open 2024–2025 until this audit has
been interpreted.
