# RAMAS Stage 6.1 — continuous Llama-70B memory-agent experiment

This package tests the main RAMAS pipeline as one uninterrupted causal process.
It resumes the cryptographically verified Stage 6 state after the completed 2021
return stream, then carries memory, Llama trust, and portfolio state through 2022,
2023, 2024, and 2025 without a year-boundary reset.

Important scientific roles:

- Proposed method: full continuous RAMAS with the bounded Llama-70B memory expert.
- Internal control: the pre-agent RAMoE stream. It is not the headline baseline.
- Descriptive external references in this run: buy-and-hold and cash-only.
- Later baseline phase: ML/RL and full-LLM portfolio methods after main-pipeline validation.

The year 2021 is burn-in evidence, not a hard economic acceptance gate. Each later
year is reported separately, but good or bad performance does not reset or stop the
learning process. Only a technical validity failure makes the package fail.

The broader project has already accessed 2024–2025. Those years are therefore
reused-OOS diagnostic evidence, not untouched confirmation.

## Server run

From `/home/infonet/wahid/leader_router_fresh`:

```bash
sha256sum -c RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1_CODE.tar.gz
bash RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/run_server.sh
```

The script automatically finds the newest verified Stage 6 `development_2021`
result. To select one explicitly:

```bash
export RAMAS_STAGE6_SEED_OUTPUT=/absolute/path/to/development_2021
bash RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/run_server.sh
```

Only `llama3.3:70b` is allowed. Qwen is disabled.
