# RAMAS state-controlled episodic-memory experiment

This closes the largest unresolved experiment in the manuscript:

```text
1,244 frozen decision states
× adaptive authority / fixed authority
× retrieved memory OFF / ON
= 4,976 LLaMA calls
```

The purpose is not to ask LLaMA for a hidden chain of thought. It measures
whether exposing the same state's retrieved episodes changes observable advice:
action, confidence, reason codes, or cited memory IDs. It then measures how
those outputs change the downstream target and performance under a controller-OFF
counterfactual. Exact controller-ON results are joined from the project's
deterministic controller ledger.

The script is stdlib-only and uses the frozen `llama3.3:70b` model through the
local Ollama HTTP API. It never writes to the episodic-memory store.

## Important limitation

The archived `03_DAILY_TRACE.csv` is not enough to construct this experiment.
It contains outcomes and summary fields, but it does not necessarily contain the
complete decision-time feature payload and retrieved episode text needed to
prove that OFF and ON prompts differ only in memory. The current server checkout
must export a frozen state snapshot first.

Do not fabricate this snapshot from a trace CSV. Export it from the same
Stage6/RAMAS code path that constructs the advisor request, immediately before
the LLaMA call, with memory writes disabled.

## Required state snapshot format

Use JSONL (one object per line), with exactly 1,244 rows. The field names below
are required unless marked optional:

```json
{
  "state_id": "2022-01-03",
  "decision_date": "2022-01-03",
  "return_date": "2022-01-04",
  "hard_regime": "bull",
  "core_desired_exposure": 0.5,
  "beta_adaptive": 0.20,
  "beta_fixed": 0.05,
  "asset_simple_return": 0.012,
  "initial_pretrade_exposure": 0.35,
  "action_targets": {
    "BTC": 1.0,
    "CASH": 0.0,
    "ABSTAIN": 0.5
  },
  "state_view": {
    "router_probabilities": {"Bear": 0.1, "Bull": 0.75, "Mix": 0.15},
    "router_uncertainty": 0.2,
    "portfolio": {"pretrade_exposure": 0.35, "drawdown": 0.03},
    "features": {"return_1d": 0.004, "return_7d": 0.021, "volatility_20d": 0.18}
  },
  "retrieved_memory": [
    {
      "memory_id": "episode_20211220_bull_001",
      "episode_date": "2021-12-20",
      "regime": "bull",
      "outcome": {"net_return": 0.008},
      "text": "Bull regime with moderate uncertainty: the prior target remained profitable but turnover was costly."
    }
  ]
}
```

Rules enforced by `make-manifest` and `preflight`:

- one unique state per decision date;
- every memory episode date is strictly before its decision date;
- no memory/retrieval/future-return fields inside `state_view`;
- `return_date` and `asset_simple_return` are metadata only and are never put in
  the LLaMA prompt;
- four complete cells per state;
- the adaptive and fixed rows have identical prompts for each memory condition;
- the OFF prompt contains none of the retrieved memory IDs;
- the ON prompt contains all retrieved memory IDs;
- memory IDs are logged for auditing but are not visible to the OFF model call.

The exporter should also record a hash of the frozen source row and, ideally, a
hash of the memory database snapshot. The runner stores those hashes in the
manifest but cannot invent them.

## Commands on the server

Copy the Python runner, schema, and this protocol file from the package into a
working directory on the server.
Then use the exact Python environment that owns the RAMAS code.

### 1. Build and validate the full manifest without calling LLaMA

```bash
cd /home/infonet/wahid/leader_router_fresh

python -m py_compile RAMAS_MEMORY_CAUSAL_EXPERIMENT.py

python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py make-manifest \
  --states /ABSOLUTE/PATH/state_snapshots.jsonl \
  --output /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --memory-snapshot-sha256 "$(awk '{print $1}' /ABSOLUTE/PATH/MEMORY_SNAPSHOT_SHA256.txt)"

python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py preflight \
  --manifest /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl
```

The first command must report `states: 1244` and `calls: 4976`. The second
command must report zero authority prompt identity failures. If either command
fails, do not run the model.

The `--memory-snapshot-sha256` argument expects the literal SHA-256 string, not
the path to a checksum file. The command above reads the first field from a
standard checksum file. The equivalent explicit form is:

```bash
MEMORY_SHA=$(awk '{print $1}' /ABSOLUTE/PATH/MEMORY_SNAPSHOT_SHA256.txt)
python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py make-manifest \
  --states /ABSOLUTE/PATH/state_snapshots.jsonl \
  --output /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --memory-snapshot-sha256 "$MEMORY_SHA"
```

### 2. Check the model and run a paired 20-state pilot

The pilot is for engineering only. It uses 20 complete states × 4 cells = 80
calls and writes into the same output directory that can later be resumed.

```bash
python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py run \
  --manifest /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --output /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/\
RAMAS_MEMORY_CAUSAL_EXPERIMENT/artifacts/PILOT \
  --dry-run

python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py run \
  --manifest /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --output /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/\
RAMAS_MEMORY_CAUSAL_EXPERIMENT/artifacts/PILOT \
  --max-states 20
```

The actual shell command should use a single line for `--output`; the line
break above is only for readability. The runner uses:

```text
model       llama3.3:70b
temperature 0
top_p       1
seed        20260915
format      json
```

It checks `/api/tags`, retries provider/parser failures, appends one durable
JSONL record per attempt's final result, and can resume after interruption.

### 3. Run all 4,976 calls

Use a new durable output directory for the final run, or resume the pilot
directory after confirming the pilot is healthy:

```bash
python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py run \
  --manifest /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --output /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_MEMORY_CAUSAL_EXPERIMENT/artifacts/FINAL
```

If the process stops, run exactly the same command again. Successful call IDs
are skipped; failed IDs are retried. `RUN_COMPLETE.json` is written only after
all 4,976 call IDs have a successful validated response.

### 4. Analyze the compact evidence

```bash
python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py analyze \
  --manifest /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --calls /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_MEMORY_CAUSAL_EXPERIMENT/artifacts/FINAL/02_CALLS.jsonl \
  --output /home/infonet/wahid/leader_router_fresh/EXPERIMENT_BRANCHES/RAMAS_MEMORY_CAUSAL_EXPERIMENT/artifacts/FINAL/MEMORY_EXPERIMENT_ANALYSIS.json
```

This writes a compact `MEMORY_PAIRED_EFFECTS.jsonl` table and a JSON summary.
The primary output is `memory_reasoning_effect_by_authority`, not the return
difference.

## Exact controller-ON join

The call runner does not reimplement the project's risk controller, because
silently replacing the locked deterministic controller would invalidate the
comparison. After each validated advisor output is passed through the exact
RAMAS deterministic controller, export one JSONL row per call:

```json
{
  "call_id": "0001_2022-01-03_adaptive_memory_off",
  "controller_on_exposure": 0.35,
  "controller_on_turnover": 0.0,
  "controller_on_net_return": 0.0042,
  "controller_version": "1.12.3",
  "controller_sha256": "...",
  "risk_inputs_sha256": "..."
}
```

The `controller_on_*` fields must be generated from the same call's blended
advisor target and the same pretrade state. No realized outcome may be written
back to memory during this replay. Pass the ledger with:

```bash
python RAMAS_MEMORY_CAUSAL_EXPERIMENT.py analyze \
  --manifest /ABSOLUTE/PATH/ramas_memory_experiment_manifest.jsonl \
  --calls /ABSOLUTE/PATH/FINAL/02_CALLS.jsonl \
  --controller-ledger /ABSOLUTE/PATH/controller_on_ledger.jsonl \
  --output /ABSOLUTE/PATH/FINAL/MEMORY_EXPERIMENT_ANALYSIS.json
```

Without this ledger, the script still answers whether memory changes the
observable LLaMA output and reports a separately-labelled controller-OFF direct
target replay. It does not claim an exact controller-ON economic result.

## How to interpret the result

1. Check `successful_calls == 4976` and zero missing calls.
2. Check `same_prompt_repeatability`. Adaptive/fixed calls have the same prompt;
   their disagreement is the empirical provider/GPU noise floor.
3. Check `memory_reasoning_effect_by_authority[*].primary_informative_memory_pairs`;
   pairs with `retrieved_memory_count == 0` are reported but excluded from the
   primary memory test.
4. A memory reasoning effect requires observable ON/OFF differences in action,
   confidence, or reason codes above that repeatability floor. A return
   difference alone cannot establish reasoning change.
5. A memory reasoning effect and a memory economic benefit are separate claims.
   The memory can change advice without improving returns, or improve returns
   through a small number of decisions without changing most outputs.
6. The fixed/adaptive authority comparison is downstream value analysis. It is
   not evidence that authority caused the LLaMA to reason differently, because
   the LLaMA prompts are intentionally identical across authority rows.
