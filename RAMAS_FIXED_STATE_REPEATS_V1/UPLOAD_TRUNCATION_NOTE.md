# Upload truncation note (2026-09-24)

`RAMAS_FIXED_STATE_REPEATS_V1.zip` as received on the experiment server was **truncated**
(17,707,010 bytes; no end-of-central-directory record; sha256
`3341a80c8d33c797e0b2b4a839d80faef50cee45fbab482d8f64f9abbaf77932`). The 5,242 members whose
data was present were recovered from their local headers, each decompressed to its full declared
size, and every one matches the hash in the shipped `MANIFEST.sha256`.

**1,251 manifest entries never arrived** (listed in `MANIFEST_ENTRIES_NOT_RECEIVED.txt`):

- `runner.py`, `tests/test_runner.py`
- `reproduce_existing.py`, `reproduce_controller_retrieval.py`, `reproduce_response_audit.py`
- `recorded/raw_fixed/contract.json` and `recorded/raw_fixed/calls/000986-no_memory.json` …
  `001608-no_memory.json` (copies of Stage 6.3 journals; the originals are in
  `EXPERIMENT_BRANCHES/RAMAS_STAGE6_3_LLAMA70B_FIXED_TRUST_MEMORY_ABLATION_V1/artifacts/20260909T013716356200839Z/calls/`)

Because the runner did not arrive, `runner.py` and `tests/test_runner.py` in this directory were
**written on the server** to implement `PROTOCOL.md` exactly and to serve the interface that the
shipped `analyze.py` and `launch.sh` expect (`MODELS`, `CONDITIONS`, `TOL`, `canonical`, `atomic`,
`load_inputs_cached`, `load_run`). They are not the archive author's files; their hashes are
therefore not in `MANIFEST.sha256`. `runner.py check` reports this explicitly.

`audit_output/` (10 files) was present in the archive but is not listed in its manifest.
