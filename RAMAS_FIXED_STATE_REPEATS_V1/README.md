# RAMAS evidence check and server experiment

The manuscript is unchanged. This package contains one focused new experiment and read-only scripts for the completed evidence.

**New experiment:** Llama 3.3 70B and Qwen3 8B, exposed/hidden retrieval, three separately served repeats, and 1,244 frozen states. This requires 14,928 planned calls, plus a separate 48-call pilot. Read `PROTOCOL.md` before running. The main runner uses the Python standard library only.

## 1. Extract and check

Place `RAMAS_FIXED_STATE_REPEATS_V1.zip` in `/home/infonet/wahid/leader_router_fresh`, then run:

```bash
cd /home/infonet/wahid/leader_router_fresh
unzip RAMAS_FIXED_STATE_REPEATS_V1.zip
cd RAMAS_FIXED_STATE_REPEATS_V1
export RAMAS_PYTHON=/home/infonet/anaconda3/envs/wahid_test/bin/python
"$RAMAS_PYTHON" runner.py check
"$RAMAS_PYTHON" runner.py preflight --output runs/preflight
```

Use a dedicated Ollama session at `http://127.0.0.1:11434`. The expected models must already be installed. Do not start concurrent GPU experiments or update model tags during this run. The preflight records the live configuration and makes no model calls. If it fails, send the generated `blocked.json`; do not replace a model automatically. Set `RAMAS_PYTHON` to another existing Python 3.10+ interpreter if the recorded environment path has changed.

## 2. Collect the missing existing records

```bash
"$RAMAS_PYTHON" collect_existing_metadata.py
```

Send `runs/existing_metadata.zip`. This collects the original independent-rerun contracts and the raw response journals for the rerun, factorial, and action-balanced comparisons. It reads only the three identified experiment runs. It makes no model calls. It reports missing paths instead of guessing. If the action-balanced run was moved, supply its exact directory with `--balanced-run /absolute/path/to/run`.

If separate transport-attempt or retry logs survive for these runs, also provide them. If they do not exist, say so. We can distinguish that missing provenance from the recorded valid-response results.

## 3. Run the separate pilot

```bash
bash launch.sh pilot runs/pilot
tail -f runs/pilot/server.log
```

The background launcher survives terminal disconnection. Press Ctrl-C to stop following the log; this does not stop the background run. On completion:

```bash
"$RAMAS_PYTHON" runner.py status --output runs/pilot
"$RAMAS_PYTHON" runner.py export --output runs/pilot
```

The pilot must finish with `COMPLETE_ALL_VALID` and 48 valid responses. Inspect `runs/pilot/analysis/summary.json` for the runtime estimate. It is an estimate from the live pilot, not a guaranteed duration. If blocked or invalid, send `runs/pilot_results.zip` before changing the configuration. If the pilot passes, continue directly to the main run.

## 4. Run the main experiment

```bash
bash launch.sh run runs/main runs/pilot
tail -f runs/main/server.log
```

After interruption, use **the same command and directories**. Do not start a new run directory to replace inconvenient or failed observations. File locking prevents two writers. A saved valid or invalid response is never queried again. An interrupted request without a saved response is marked missing on resume and is not resubmitted. Missing responses will remain visible in the analysis.

Check progress, regenerate analysis, and export after the run finishes:

```bash
"$RAMAS_PYTHON" runner.py status --output runs/main
"$RAMAS_PYTHON" runner.py analyze --output runs/main
"$RAMAS_PYTHON" runner.py export --output runs/main
```

Send **`runs/main_results.zip`**, **`runs/pilot_results.zip`**, and **`runs/existing_metadata.zip`**. If the run pauses, export and send the partial run instead of deleting records. Do not edit the frozen package files. Changes to models, prompts, or options require a separately versioned protocol.

## Outputs to inspect

| File inside each run | Purpose |
|---|---|
| `contract.json` | Input, model, serving, prompt, hardware, and schedule identity |
| `calls/*.json` | Durable request/response records, validity, timings, and checksums |
| `progress.json`, `complete.json`, `blocked.json` | Progress, completion, or interruption evidence; an old blocked file remains part of the history |
| `analysis/validity_and_distributions.csv` | Validity and action distributions, with explicit denominators |
| `analysis/paired_summary.csv` | Within-condition repeat variation first, then between-condition transmission |
| `analysis/repeat_ranges.csv` | Descriptive repeat ranges on complete states |
| `analysis/paired_state_results.csv` | Traceable state-level comparisons |
| `analysis/summary.json` | Completion, valid-state counts, scope, and pilot timing estimate |

An undefined transmission rate is blank in CSV and `null` in JSON. It is not a zero transmission result. No financial-performance result is produced by this experiment.

## Existing evidence: already reproduced locally

`audit_output/comparison_inventory.csv` provides all eight closed-loop contrasts and the separate fixed-state diagnostic. The eight closed-loop intervals were reproduced from the supplied daily ledgers. All span zero. Their descriptive mean abstention difference is +1.276 percentage points. Their mean exposure difference is -0.08438 points. The fixed-state diagnostic has +6.270 and -0.2090 points, respectively.

The original controller and retrieval audits are complete. You do not need to rerun them on the server to start the new experiment. To reproduce them independently, use the existing research environment with NumPy and pandas:

```bash
"$RAMAS_PYTHON" reproduce_existing.py
```

The recorded versions are NumPy 1.26.4 and pandas 2.2.3. The supplied frozen controller source is unchanged. The reproduction checks raw principal responses, pairing, retrieval pools, five controller settings, and the existing interval inventory. It overwrites only the generated `audit_output/` files. No model calls are made.

The package's local tests use artificial provider responses in temporary directories. They verify scheduling, validation, integrity, and resume behavior. They supply no empirical model results. Run them with:

```bash
"$RAMAS_PYTHON" -m unittest discover -s tests -v
```

Keep this package for evidence collection. Final manuscript wording will be revised only after the returned results and missing provenance are assessed and you ask for the update.
