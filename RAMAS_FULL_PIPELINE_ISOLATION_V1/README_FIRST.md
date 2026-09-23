# RAMAS full-pipeline isolation V1

## What this package does

This package runs the missing full sequential isolation study on the actual
frozen RAMAS Stage‑6.4 engine. It does not recreate the allocator, memory,
authority update, controller, or accounting from a verbal description.

For every selected model and LLM sampling seed it runs:

```text
memory / no memory
× adaptive / fixed authority
× controller ON / OFF
= 8 sequential LLM arms
```

It also runs numerical-only controller ON/OFF once. Every arm replays all
1,608 days so that the 2021 burn-in determines holdings, memory, and trust
before the 1,244-row 2022–2025 evaluation window is selected.

The independent verifier then reconstructs accounting and produces:

- arm-level return, Sharpe, volatility, MDD, ES95, exposure, turnover, and cost;
- memory, authority, controller, and LLM-versus-numerical contrasts;
- paired 30-day circular-block intervals;
- controller × advisor difference-in-differences;
- projection-only controller replays;
- state-matched advice-to-execution transmission counts;
- model-call, token, and latency totals;
- exact cross-seed response distinctness checks.

## Scientific labels that must not be changed

This is a **retrospective reused-history mechanism diagnostic**. It is not a
fresh out-of-sample test. It does not resolve the historical regime
predictor's point-in-time provenance, the feasible information-to-inference-
to-order-to-fill clock, or the 2024 reference seam.

Closed-loop memory ON/OFF paths can diverge after an early action difference.
Their final outcome difference is therefore a total-system effect, not a pure
same-state causal memory effect. The verifier reports same-state dates
separately. If the paper retains episodic memory as its central causal claim,
the separate frozen-state 4,976-call intervention is still required.

## Compute budget—do not ignore this

Each model × seed cell costs:

```text
1,608 days × 8 LLM arms = 12,864 calls
```

Examples:

| Matrix | Maximum calls |
|---|---:|
| 1 model × 1 seed | 12,864 |
| 3 models × 1 seed | 38,592 |
| 3 models × 4 seeds | 154,368 |
| 4 models × 4 seeds | 205,824 |

Running every seed at temperature zero without first showing that the server
actually changes its output would waste compute and create fake statistical
replication. The runner blocks that by default.

## Required server state

The default configuration expects:

```text
/home/infonet/wahid/leader_router_fresh
/home/infonet/wahid/leader_router_fresh/RAMAS_STAGE6_4_COMPONENT_SUITE_V1
/home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv
```

It also expects the pinned Stage‑6.1/6.2/6.3 and corrected numerical source
paths already named by Stage‑6.4 `source_config.json`. The preflight verifies
these read-only dependencies before inference.

For cross-model fairness, all new arms use the same reviewed prompt with one
declared lexical neutralization: `bounded Llama expert` becomes `bounded
language-model expert` for every model, including Llama. The task, evidence,
constraints, action set, and JSON schema are otherwise unchanged. Both the
source and effective prompt identities are recorded.

Use the original `wahid_test` environment:

```bash
cd /home/infonet/wahid/leader_router_fresh
conda activate wahid_test
```

## Step 1 — unpack and verify code

```bash
sha256sum -c RAMAS_FULL_PIPELINE_ISOLATION_V1_CODE.tar.gz.sha256
tar -xzf RAMAS_FULL_PIPELINE_ISOLATION_V1_CODE.tar.gz
cd RAMAS_FULL_PIPELINE_ISOLATION_V1
python self_test.py --config config/experiment.json
```

The self-test runs this package's unit/end-to-end tests and the upstream
Stage‑6.4 tests. It makes no model calls.

## Step 2 — read-only preflight

The safe default plans only one model and one seed, while preflight locks all
enabled model digests:

```bash
bash 01_preflight.sh
```

Inspect:

```text
.../RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/PREFLIGHT.json
.../RAMAS_FULL_PIPELINE_ISOLATION_V1/PREFLIGHT/RESOLVED_CONFIG.json
```

Do not continue if preflight fails.

## Step 3 — determinism/seed probe

The default probe uses all configured models and four seeds but only three
archived production prompts and two repeats per seed:

```bash
bash 02_determinism_probe.sh
```

Inspect each model's:

- `within_seed_exact_repeatability`;
- `across_seed_exact_equality`;
- `seed_realizations_distinct_on_probe`.

If a temperature-zero model is identical across seeds, use one seed for the
main experiment. Do not average duplicate runs and call them independent.

## Step 4 — staged full run

First run one seed across three deliberately different model settings:

```bash
export RAMAS_MODELS='llama3.3:70b,glm-4.7-flash,qwen3:8b'
export RAMAS_SEEDS='42'
bash 03_run_experiment.sh
```

This is already 38,592 maximum model calls. `qwen3-coder:30b` is available as
an additional model, not a reason to launch every combination automatically.

If the probe demonstrates real stochastic variation and the extra serving
budget is justified, expand seeds explicitly:

```bash
export RAMAS_MODELS='llama3.3:70b,glm-4.7-flash,qwen3:8b'
export RAMAS_SEEDS='42,2024,3407,7777'
bash 03_run_experiment.sh
```

If the process stops, resume the same identity:

```bash
export RAMAS_RUN_DIR="$(cat /home/infonet/wahid/leader_router_fresh/RAMAS_FULL_PIPELINE_ISOLATION_LAST_RUN.txt)"
export RAMAS_MODELS='the exact same comma-separated list'
export RAMAS_SEEDS='the exact same comma-separated list'
bash 03_run_experiment.sh
```

Durable responses are reused. A completed call is never resampled.

## Step 5 — independent reproduction and verification

```bash
bash 04_verify_results.sh
```

The verification directory contains:

| File | Purpose |
|---|---|
| `VERIFICATION_REPORT.json` | top-level integrity and interpretation contract |
| `01_VERIFIED_METRICS.csv` | independently recomputed per-arm metrics |
| `02_SEED_AGGREGATE_METRICS.csv` | descriptive mean ± SD across actual serving seeds |
| `03_PAIRED_CONTRASTS.csv` | deterministic paired metric differences |
| `04_PAIRED_BLOCK_BOOTSTRAP.csv` | 30-day circular-block intervals |
| `05_TRANSMISSION.csv` | all-date and state-matched advice/desired/executed changes |
| `06_CALL_COSTS.csv` | calls, tokens, and latency |
| `07_SEED_DISTINCTNESS.csv` | exact request/response/action equality across seeds |
| `08_ACCOUNTING_INTEGRITY.csv` | maximum reconstruction errors |
| `09_PROJECTION_ONLY_DAILY.csv` | controller-OFF replay of controller-ON advice sequences |

Only use results if both the experiment `RUN_COMPLETE.json` and verifier
`RUN_COMPLETE.json` exist and validate.

## Malformed outputs

The provider validates the model response before the Stage‑6.4 engine can
score it. A malformed response stops the arm and writes evidence. It is not
silently converted to ABSTAIN, and no heuristic decision replaces it.

## Recommended manuscript use

The decisive question is not which model earns the highest return. Report each
model separately and ask whether the response-to-execution attenuation pattern
persists. Lead with transmission and controller results; keep returns as
secondary descriptive outcomes. A null or negative component effect is a
valid result.
