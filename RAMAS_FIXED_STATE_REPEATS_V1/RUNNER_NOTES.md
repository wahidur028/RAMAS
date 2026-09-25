# Server-side notes for RAMAS_FIXED_STATE_REPEATS_V1 (2026-09-24)

These notes record what was done on the experiment server beyond the files that arrived in the
archive. Read together with `UPLOAD_TRUNCATION_NOTE.md` and `PROTOCOL.md`.

## What arrived, what did not
The archive was truncated in transfer (17,707,010 bytes, no central directory). All 5,242
delivered members verify against the shipped `MANIFEST.sha256`; 1,251 entries never arrived,
among them `runner.py`, `tests/test_runner.py` and the three `reproduce_*.py` scripts. Every
data, contract, controller and analysis file needed to run the protocol did arrive and was
verified byte-for-byte against the original archives on this server (state banks = the archived
Stage 6.3 / 6.2 request payloads; `frozen_controller/src` = the frozen numerical core;
`recorded/*` = the archived ledgers and journals).

## Files written on the server
| file | purpose |
|---|---|
| `runner.py` | implements `PROTOCOL.md` v1 with the standard library only and serves the interface the shipped `analyze.py` / `launch.sh` expect (`MODELS`, `CONDITIONS`, `TOL`, `canonical`, `atomic`, `load_inputs_cached`, `load_run`) |
| `tests/test_runner.py` | 7 tests with an artificial provider: schedule completeness/balance/identity across models, hidden request differs only in `memory` and exposed payload equals the bank, contract validation, checksummed records + tamper detection, interruption/resume without resampling (transport failure and crash marker), consecutive-invalid pause, analysis with undefined transmission |
| `smoke_other_models.py` | 4 real calls (2 states × exposed/hidden) per additional installed model; not part of the protocol |
| `launch_chain.sh` | resume-safe chain: pilot gate → main run → analyze/export → optional extension pilot/run on additional models |
| `monitor.sh` | one-screen status of every run directory; `-f` follows the active log |

## Decisions taken while implementing the protocol
1. **Pilot = 48 calls = one repeat** (12 states × 2 models × 2 conditions), exactly as PROTOCOL.md
   states. Pilot states: four per regime at the 1/8, 3/8, 5/8, 7/8 positions of that regime's
   date-ordered sequence (indices 47, 166, 261, 410, 449, 608, 695, 814, 884, 950, 1032, 1102).
2. **Schedule**: per repeat one shuffled state order shared by both models (`random.Random`
   seeded from 16061 + protocol version + repeat), exposed-first/hidden-first balanced 622/622,
   both conditions consecutive per state, model order Llama/Qwen, Qwen/Llama, Llama/Qwen.
   The schedule is rebuilt from the contract on every resume and hash-checked.
3. **Request body**: the frozen provider's format (`format` = decision JSON schema, system prompt
   = neutral prompt `569897b2…`, user message = `json.dumps(payload, sort_keys=True)`), options
   `{temperature 0, top_p 1.0, seed 16061, num_ctx 8192, num_predict 512}`; `think: false` for
   `qwen3:8b`, no `think` key for `llama3.3:70b`. The exposed payload is asserted byte-equivalent
   to the state bank on every call; the hidden payload differs only in `memory`.
4. **Records**: one JSON per call with the full request, provider response, validation result,
   timings and a record checksum; a pending marker precedes every request. On resume, saved
   responses are reused (valid or not), a leftover pending marker becomes an
   `INTERRUPTED_UNKNOWN_OUTCOME` record and is never resubmitted, and a transport failure is
   recorded as invalid and pauses the run.
5. **Pause rules**: 3 consecutive invalid responses for a model, or a valid rate below 98% once a
   model has ≥ 50 recorded calls, or any identity change → `blocked.json` and exit code 3.
6. **Identity**: Ollama version, model digest, details, capabilities, template hash and the
   `/api/show` metadata are pinned in the contract and re-checked at start, at every model
   change, every 100 scheduled calls and at completion. **Ollama 0.23.0 returns the `parameters`
   block (and the generated `modelfile`) with its lines in a random order on each call**, which
   made a naive hash unstable and blocked the first pilot launch before any model call; the
   runner therefore hashes the parameter lines in sorted order. The Llama digest must equal the
   pinned `a6eb4748…`; the Qwen digest is recorded at preflight and must match between pilot and
   main run.
7. **Model unloading**: the model just used is unloaded (`keep_alive: 0`) at every model change
   and at completion.

## Smoke test of the other installed models (`runs/model_smoke/`, 4 calls each, 2026-09-24)
| model | valid | mean latency | verdict |
|---|---:|---:|---|
| qwen3:8b (protocol) | 4/4 | 3.3 s | protocol second model |
| qwen3-coder:30b | 4/4 | 6.8 s | candidate for an extension run |
| deepseek-r1:8b (`think: false`) | 4/4 | 2.7 s | candidate |
| gemma4 (8B, `think: false`) | 4/4 | 4.1 s | candidate |
| mistral 7B | 3/4 | 2.4 s | **excluded**: cited a memory id that was not shown (contract violation) |
| glm-4.7-flash | 0/4 | 6.7 s | **excluded**: invents `reason_codes` outside the enum / non-JSON — same failure as in the isolation run |

Extension runs (if executed) are labelled `extension_v1_1_additional_models`, use their own
pilot and contract, and are reported separately from the protocol's Llama/Qwen result. Their
`think` flag follows the model's advertised capability (false when it advertises thinking).

## Main run result (2026-09-25T01:13Z) and extension decision
Main protocol run: 14,928/14,928 recorded, **14,919 valid**; Llama 7,464/7,464 valid (7.3 s/call),
Qwen 7,455/7,464 (2.1 s/call; 9 exposed-condition citation-contract violations: 4 duplicate ids,
5 unseen ids — recorded as invalid, never resampled). Status `COMPLETE_WITH_INVALID_OR_MISSING`
because of those 9; the 98 % valid-rate rule was never approached (99.88 % for Qwen).

Extension pilot (qwen3-coder:30b, deepseek-r1:8b, gemma4; 72 calls): 71/72 valid — one
`deepseek-r1:8b` hidden-state response hit `num_predict` (`done_reason: length`) even with
`think: false`. The all-valid pilot gate therefore refused the three-model extension. Decision:
**drop deepseek-r1:8b** (a 1/24 truncation rate would also threaten the 98 % rule over 7,464
calls) and run the extension on the two models that passed 24/24, `qwen3-coder:30b` and
`gemma4:latest`, via `launch_extension.sh` (own pilot `runs/extension_pilot_2models`, run
`runs/extension`). The failed three-model pilot is kept as `runs/extension_pilot`.

## Extension result (2026-09-25T11:20Z)
`runs/extension` (qwen3-coder:30b + gemma4, 14,928 calls): 14,923 valid — gemma4 7,464/7,464 (2.2 s),
qwen3-coder 7,459/7,464 (2.4 s; 5 exposed-condition unseen-citation violations). Between contexts
(exposed minus hidden, per repeat): qwen3-coder advice differs on 128/126/133 states, executed on
69/73/71, abstention **+8.9/+8.9/+9.3 pp**, exposure −0.28 pp; gemma4 41/43/41 → 21/21/19,
abstention **+1.05 pp**, exposure −0.08 pp. Within-condition repeat disagreement: qwen3-coder
exposed 3–5 / hidden 18–24 states; gemma4 0–4. Cross-minus-within: qwen3-coder 0.094 / 0.051,
gemma4 0.032 / 0.016. Full tables: `RAMAS/docs/RAMAS_TECHNICAL_REPORT.md` §6.12 (monorepo) /
`docs/RAMAS_TECHNICAL_REPORT.md` §6.12 (public RAMAS repo).
