# Fixed-state repeated-serving protocol, version 1

This is a prospective specification for new calls on recorded historical states. It is not a new out-of-sample financial evaluation. No manuscript file is modified by this package.

## Question and design

Does the advice-to-execution pattern also appear with a second model configuration? How much do separately served responses vary when the complete request remains unchanged?

| Element | Frozen specification |
|---|---|
| State anchor | All 1,244 requests from the original fixed-authority retrieval-exposed trajectory |
| Reporting window | Return dates 2022-01-01 through 2025-05-28; corresponding decision dates 2021-12-31 through 2025-05-27 |
| Model configurations | `llama3.3:70b` and `qwen3:8b` |
| Conditions | Recorded retrieval payload exposed; episode list and its summary jointly hidden |
| Repeats | Three separate API calls per model, condition, and state |
| Main call count | 2 × 2 × 3 × 1,244 = 14,928 planned calls |
| Engineering pilot | 12 states, four per regime, spaced through that regime's date sequence; 48 separate calls, excluded from the main analysis |
| Controller | Fixed authority 0.05 and original projection, using independently reconstructed frozen previews |
| Shared explicit options | Temperature 0, seed 16061, context 8192, maximum output 512 |
| Qwen thinking | Disabled with `think: false`; Llama uses its nonreasoning interface |
| Output | The existing four-field action contract and its permitted reason labels |

The Qwen choice is grounded in the project record: it was installed and previously returned valid structured output with thinking disabled. Its live availability is checked, not assumed. The model sizes differ. This is a comparison of two deployed model configurations. It does not isolate architecture, family, parameter count, or quantization effects.

The original Llama weight digest must match `a6eb4748fd2990ad2952b2335a95a7f952d1a06119a0aa6a2df6cd052a93a3fa`. The Qwen digest, both chat templates, model metadata, Ollama version, and runtime hardware are recorded before the pilot. The main run must match the pilot identities. No model is downloaded or substituted automatically. The current server version is recorded and frozen for this experiment; it is not assumed to match the historical server.

## Inputs and intervention

The exposed request is copied byte-equivalently at the JSON-value level from the frozen state bank. The hidden request changes only `memory` to:

```json
{"similar_completed_episodes": [], "summary": "NO_COMPLETED_SIMILAR_EPISODES"}
```

Numerical state, pretrade holdings, action previews, instructions, and every other request field remain equal between conditions. The same anchor is used for both conditions and both models. The independently evolved historical hidden trajectory is not used as the new hidden state bank. Within a condition, the complete request body is identical across the three repeats. No timestamp or repeat identifier is inserted into model-visible content.

Both models use the same neutral system prompt. Its sole change from the original prompt is `bounded Llama expert` → `bounded language-model expert`. Its SHA-256 is `569897b2f09b09acda6d3b52645904f7eb7a179bb3751c46b79d51f091c8aedd`, matching the existing neutral prompt variant. All calls, including Llama calls, are fresh. The old seed-42 results are not treated as repeats of this protocol.

The original request field names are preserved for fidelity. They are source-format identifiers, not proposed manuscript terminology. In particular, the legacy request mentions a failure-to-abstention rule. This runner does **not** apply that fallback: malformed, truncated, or missing responses are invalid observations with no action or exposure assigned.

The intervention jointly removes episode content and the summary. It changes prompt length. It does not isolate episode semantics from length, formatting, or summary effects. The shared episodes came from the historical Llama trajectory. They are not Qwen's naturally evolved memory. No histories or holdings evolve during this test.

## Serving and scheduling

The seed is held fixed across all three repeats. These are repeated requests under a fixed serving recipe, not three independent sampling seeds. Zero observed variation is an admissible result. Temperature is not raised to manufacture variation.

Each repeat uses a deterministic shuffled state order. Exposed-first and hidden-first order is balanced at 622 states each. Both conditions are queried consecutively for each state. The same ordering is used for both models. Models run sequentially to fit the recorded two-GPU server. The model order is Llama/Qwen, Qwen/Llama, Llama/Qwen across repeats. This is partial order balancing; three repeats cannot fully balance two model positions.

Use a dedicated Ollama session for this experiment. Do not change model tags, templates, server version, or scheduling options during a run. Identities are checked on start, at model changes, every 100 scheduled calls, and at completion. These checks detect observed drift; they cannot certify an unobserved transient server change between checks. Model-specific native chat templates and token counts are saved. The runner unloads only the model it has just used when changing model blocks or finishing.

Every completed response is saved with its full request, provider response, validation result, timing, and checksums. Resume reuses all saved responses, including invalid ones. Invalid responses are never silently converted to abstention or resampled. A transport failure pauses execution. It remains a missing response if the same run is resumed. An interrupted pending request is recorded as having an unknown outcome and is not submitted again. Therefore, failures can reduce the number of valid responses below 14,928; the exported denominators expose this loss.

The pilot must contain all 48 valid responses before the main run can start. The runner pauses after three consecutive invalid responses for a model, or a valid rate below 98% once that model has at least 50 recorded calls. A failed pilot or systematic format problem requires review and a separately versioned recipe. Do not edit this protocol in place or reroll failed responses until they pass.

## Analysis, in order

1. Report response completion and validity by model, condition, and repeat. Keep valid abstention separate from malformed, truncated, and missing output.
2. Report within-condition action and execution disagreement for repeat pairs (1,2), (1,3), and (2,3). These are stability diagnostics under unchanged requests.
3. For each matched repeat, compare exposed and hidden valid actions. Report abstention differences, advice disagreements, desired-exposure disagreements, executed-exposure disagreements, and mean signed exposure differences.
4. Report conditional transmission as executed disagreements divided by advice disagreements. With zero advice disagreements, report **undefined**, represented by JSON `null` and empty CSV cells. Do not replace this with zero.
5. Show results on available valid pairs and on the common subset with all six responses valid for that model. Show regime breakdowns as descriptive checks.
6. Report the three repeat-specific values and their range. A secondary analysis averages all nine cross-context pairings and each condition's three within-context repeat pairings within a state, then weights complete states equally. Its cross-minus-within contrast is descriptive. It is not a purified causal effect of semantic memory.

Repeated calls, pair combinations, and market dates are dependent. The script reports descriptive repeat ranges. It makes no significance, independence, equivalence, or statistical-power claim. Three repeats are a limited stability check. Model-dependent patterns, zero disagreements, and complete suppression are all reportable outcomes.

The controller mapping is already determined at each anchor state. New actions are mapped through its reconstructed original previews. No new controller sweep is needed. There is no portfolio compounding, financial-performance comparison, predictor validation, or additional trading baseline in the new experiment.

## Existing evidence and remaining provenance

`recorded/` contains the source ledgers and principal raw response records. `audit_output/` contains regenerated exports. `comparison_inventory.csv` has eight independently queried closed-loop comparisons and a separate fixed-state row. The four factorial comparisons include freshly queried conditions with projection active or bypassed. They are not fixed-upstream projection-bypass replays. Such replays are excluded from the eight-comparison count.

The two principal experiments have 6,432 raw decision records across their full histories. Their checksums, action contract, stopping status, and ledger alignment are rechecked. The 1,244 reporting pairs are selected by **target return date**, including the 2021-12-31 decision for the 2022-01-01 return. The earlier 1,243-pair count incorrectly filtered by decision date.

For the remaining comparisons, supplied ledgers mark all retained decisions valid. Their raw responses and the original rerun serving contract are requested by `collect_existing_metadata.py`. Until those are returned, the package does not claim an independent schema revalidation of every nonprincipal raw response. Accepted-response records also do not prove that no unrecorded transport retries occurred. Separate attempt logs may be unavailable; if so, that limitation must remain explicit.

## Exact implemented projection

The unchanged source is in `frozen_controller/src/risk.py` and `core.py`. The code uses `TOL = 1e-12` for turnover and risk feasibility. A point is feasible when turnover is at most the turnover ceiling plus `TOL` and ambiguity CVaR is at most its ceiling plus `TOL`.

For both minimum-risk fallback candidates and nearest-distance candidates, NumPy is called as `np.isclose(values, minimum, atol=1e-14)`. In the recorded NumPy 1.26.4 environment, the default relative tolerance is `1e-5`; `equal_nan=False`. The exact finite-value predicate is:

```text
abs(value - minimum) <= 1e-14 + 1e-5 * abs(minimum)
```

First select points satisfying both feasibility conditions. If that set is empty, restrict to turnover-feasible points, find their minimum ambiguity CVaR, and retain points close to that minimum under the predicate above. Then find minimum absolute distance to the desired exposure, retain distances close to that minimum under the same predicate, and choose the lowest exposure. An empty turnover-feasible set raises an error. The fallback preserves the turnover constraint but can exceed the CVaR ceiling.

The grid uses `floor(1 / step + TOL)`, multiplies integer indices by the step, retains points no larger than `1 + TOL`, appends 1, clips to [0,1], and removes duplicates. The exported-exposure comparison threshold `1e-10` is an audit threshold; it is not the projector's feasibility or tie tolerance.

The five-setting reproduction is a check of existing results. Its maximum original-preview residual is approximately `9.02e-17`. It makes no model calls and does not regenerate portfolios. No discrepancy requiring another sensitivity sweep was found.

## Primary API references

- Ollama chat API: https://docs.ollama.com/api/chat
- Ollama thinking option: https://docs.ollama.com/capabilities/thinking

Live compatibility remains subject to the pilot. Successful local checks are not presented as a successful server experiment.
