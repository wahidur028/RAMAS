# RAMAS Stage 6.2: Does remembered experience help?

This package runs two independent Llama-70B agents through the same historical days. One receives remembered completed episodes; the other receives no retrieved episodes or episode summary. Both keep the same adaptive trust rule. It includes cash, BTC buy-and-hold, yearly and Bull/Bear/Mix reports.

The earlier cash/B&H comparison is complete. This is the next experiment to test the added value of episodic memory. It does not assume that memory will help, and it does not stop after a poor year.

## 1. What will run

| Item | Contract |
|---|---|
| Experiment | `RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1` |
| Model | Ollama `llama3.3:70b` only; Qwen is prohibited |
| Agent A | Receives up to five completed, similar, same-regime episodes |
| Agent B | Receives no episodes or episode summary; trust still adapts |
| Initialization | Both start with empty memory, initial beta 0.05 and their own all-cash portfolio in 2021 |
| Continuity | Each carries its own decisions, completed outcomes, trust and portfolio across years |
| Historical return dates | 2021-01-02 through 2025-05-28; 1,608 days per arm |
| Primary comparison | 2022-01-01 through 2025-05-28; 1,244 days after 2021 warmup |
| Expected fresh model calls | 12 shared semantic checks plus 2 × 1,608 daily decisions = 3,228 calls |
| Other comparisons | Cash, buy-and-hold; exploratory fixed 45% BTC/55% cash |
| Year and regime reports | Annual slices and decision-day Bull/Bear/Mix router forecasts |

The agents generate fresh decisions. Neither arm reuses the earlier Stage 6.1 agent decisions or its learned state. Historical results are read only to validate the common input and risk-projection reconstruction. Llama weights remain frozen: this experiment studies external memory under an adaptive trust mechanism.

## 2. Run on your server

Put the code archive and its `.sha256` file in `/home/infonet/wahid/leader_router_fresh`. Use your existing `wahid_test` environment. It needs Python 3.10+, NumPy, pandas, a running local Ollama service and the already-installed `llama3.3:70b` model. No model download is started automatically.

```bash
cd /home/infonet/wahid/leader_router_fresh
sha256sum -c RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1_CODE.tar.gz.sha256 &&
tar -xzf RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1_CODE.tar.gz &&
bash RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/start.sh
```

The chained commands stop if the archive checksum or extraction fails. The launcher prints the process ID and log path, then returns your terminal. Follow progress with:

```bash
tail -f /home/infonet/wahid/leader_router_fresh/RAMAS_STAGE6_2_RUN.log
```

`Ctrl+C` while following this log stops the log viewer. The background experiment continues. The number of model calls is larger than the previous single-agent run; elapsed time depends on your server. No runtime estimate has been measured locally.

The source check runs before model calls. The following existing artifacts must remain on the server:

```text
/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_1_LLAMA70B_CONTINUOUS_MEMORY_AGENT_EXPERT_V1/artifacts/20260907T012534Z/continuous_2021_2025

/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMOE_RETURN_CLOCK_REPAIR_AND_ECONOMIC_RERUN_V1/artifacts/20260901T024730Z
```

The raw dataset is discovered at either frozen legacy path:

```text
/home/infonet/wahid/projects/leader_fresh/full_data_set/full_data_set.csv
/home/infonet/wahid/leader_router_fresh/full_data_set/full_data_set.csv
```

If that exact file was moved, pass its current location using `RAMAS_FULL_DATASET=/absolute/path/full_data_set.csv bash .../start.sh`. The content must still match SHA-256 `b69f17a1233a58c3e0c7d6289fc5bf79173aae471a31074cf17cfffbc8198e7e`. A missing or changed source produces a specific error. Do not bypass the hash check or substitute a newly downloaded dataset.

## 3. If the run is interrupted

Wait until the old run has stopped, then restart the same result directory:

```bash
cd /home/infonet/wahid/leader_router_fresh
RAMAS_RESUME_DIR="$(cat RAMAS_STAGE6_2_LAST_RUN.txt)" \
bash RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/start.sh
```

Completed responses are saved separately for each arm and day. Resume verifies the exact code, configuration, source, Python/NumPy/pandas versions, Ollama version and model digest, then reconstructs state using those saved responses. A completed run produces no new calls on resume. Do not edit package files, update the model or change the environment midway through this experiment.

A transport error pauses the run before applying that day's outcome. A process death between model completion and durable response saving can require repeating that one unsaved call. Malformed completed responses are logged as `ABSTAIN`; three consecutive invalid responses in either arm stop progression for investigation. Sparse invalid responses are retained and reported. This is an interface validity rule, not an economic acceptance gate.

## 4. Result files to send back

The result directory is:

```text
/home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_STAGE6_2_LLAMA70B_MATCHED_MEMORY_ABLATION_V1/artifacts/<NEW_UTC_RUN_ID>
```

On completion, or a caught stop, the server wrapper prints `RESULT_ARCHIVE` and `RESULT_SHA256`. Send both files. The new run ID will differ from earlier runs.

| File inside the result | Meaning |
|---|---|
| `00_CONTRACT.json` | Frozen experiment and runtime/model identity |
| `01_SOURCE_AUDIT.json` | Source, hashes and input/risk reconstruction checks |
| `02_PREFLIGHT.json` | Semantic checks; these do not prove financial skill |
| `03_DAILY_TRACE.csv` | Both arms' decisions, exposures, trust and net returns |
| `04_FULL_AND_YEARLY_METRICS.csv` | Full, post-2021 and yearly performance, including cash/B&H |
| `05_FORECAST_REGIME_METRICS.csv` | Bull/Bear/Mix conditional performance |
| `06_PAIRED_MEMORY_CONTRAST.json` | Primary contrast and paired block-bootstrap uncertainty |
| `07_ACTION_AND_STATE_DIAGNOSTICS.csv` | Whether remembered evidence changes decisions or final exposure |
| `08_RESEARCH_INTERPRETATION.md` | Plain-language interpretation and limitations |
| `09_ANALYSIS_CONTRACT.json` | Metric and reporting conventions |
| `10_FINAL_STATUS.json` | Completion, validity and call counts |
| `calls/`, `episodes/`, `trust/` | Auditable responses and each arm's independent state history |
| `RUN_COMPLETE.json` | Hashes of completed artifacts |

`PASS` means the run completed its technical checks. Economic findings may favor memory, favor no-memory, show a risk-return trade-off or remain inconclusive. The software never declares that continuous learning has been proved merely because the run completed.

## 5. Local validation and limits

The release passed 26 automated tests and a synthetic 40-day paired replay. The synthetic provider is explicitly marked `CONTROLLED_NON_ECONOMIC`; it is not Llama and its risk projection is a test double. A restart check verifies that the completed replay is reused.

Real Ollama inference and integration with your complete upstream source files have **not** been run in the local preparation environment. The real source preflight reconstructs all 4,824 recorded legacy risk previews before new model calls; that integration check must run on your server.

Read `SCIENTIFIC_PROTOCOL.md` before interpreting results. The data are reused historical evidence, and 2025 is partial. This package neither establishes novelty nor promises profit.
