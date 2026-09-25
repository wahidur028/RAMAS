# RAMAS evidence assessment — 24 September 2026

**The manuscript has not been edited.** The latest review supports a targeted revision. One focused new experiment is useful. The other remaining tasks use existing results or concern wording and reproducibility.

## Reviewer comments checked against the evidence

| Comment | Assessment | Action before manuscript revision |
|---|---|---|
| Broaden the advisor evidence and measure fixed-request serving variation | Valid. The records contain a completed Llama factorial, but no completed comparable second-family repeated-call study. Changing both prompt and seed in the earlier factorial does not supply fixed-request repeats. | Run the supplied Llama/Qwen fixed-state experiment. |
| Give an auditable inventory of eight closed-loop comparisons | Valid reporting concern. All eight estimates and confidence intervals were reproduced from the supplied daily ledgers. | Use the generated comparison inventory when the supplement is next revised. |
| Separate the fixed-state diagnostic from closed-loop averages | Valid. They have different estimands. | Use the separate summaries below. No new calls are needed. |
| State exact projection tolerances and fallback order | Valid reproducibility concern. The frozen source and preview reconstruction agree. | The exact implementation is documented in the supplied protocol. No further sweep is needed. |
| Replace stronger behavioral language with observed associations | Valid. Several passages still say “increases,” “consistently increased,” or “behaviorally active.” | Queue a small wording change. Preserve the observed direction without attributing the entire difference to retrieval content. |
| Correct the Causal Agent Replay citation and comparison | Valid for the reviewed PDF. Its citation renders as `[?]`. The supplied source has a citation key and the earlier revised bibliography has the entry, so source/PDF build consistency also needs attention. The conceptual comparison needs tightening. | Rebuild the matched final files later. Describe CAR accurately; no CAR experiment is needed. |
| Supply reproducible response, retrieval, and controller audits | Mostly available. The principal raw responses and central deterministic audits were independently reproduced. The other comparisons currently have accepted-response validity flags in ledgers, with incomplete raw provenance in the supplied package. | Collect the missing existing journals and rerun contract using the supplied script. |
| Redraw figures or repeat the controller sensitivity | Not justified by this review. The reviewer accepts both revised figures and the existing sensitivity study. | Keep them. |
| Add assets, trading baselines, or a broad new factorial | Not needed for the present conditional diagnostic claims. | Keep the experiment scope narrow. |

A second model is a reviewer recommendation for stronger empirical support. It is not an IEEE TAI submission rule or a guarantee of acceptance. It also cannot turn finite-action enumeration into a new control-theory result.

## Results verified from existing data

The main reporting window is **2022-01-01 through 2025-05-28**, selected by target return date.

| Quantity | Eight closed-loop comparisons | Separate fixed-state diagnostic |
|---|---:|---:|
| Valid-abstention difference, exposed minus hidden | Mean **+1.2761 percentage points**; range +0.4823 to +2.0900 | **+6.2701 percentage points** |
| Mean exposure difference, exposed minus hidden | Mean **−0.08438 percentage points**; range −0.21366 to −0.04019 | **−0.20900 percentage points** |
| Existing outcome intervals containing zero | **8 of 8** | Separate one-step estimand; excluded from the closed-loop count |

These are descriptive averages over configurations. They are not independent replications or a pooled causal effect. The direction agrees across all nine recorded comparisons. Intervals containing zero do not establish equivalence.

The complete inventory includes the configuration, serving seed and identifier, prompt identifier and hash, date window, behavioral differences, outcome estimate, and interval. The maximum difference between the reproduced outcome estimates/intervals and their source exports was approximately **1.04 × 10⁻¹⁴ basis points**. Fixed-upstream projection-bypass replays are excluded from the eight closed-loop comparisons.

The principal raw-response audit covers **6,432** full-history decision records. Their request, response, and record hashes agree. All retained raw outputs pass the decision contract and end normally. Their actions match the ledgers. The reporting subsets have **1,244 complete pairs** in each principal comparison. The previously omitted pair is the **2021-12-31 decision / 2022-01-01 return**. It was lost by applying the return-window cutoff to decision dates.

The response ledgers for all nine comparisons report zero invalid decisions in their retained reporting records. For the nonprincipal comparisons, this is a ledger-level verification until their raw journals are supplied. Neither retained valid records nor zero fallback counts prove that all transport attempts succeeded.

The retrieval reconstruction reproduces the selected episode IDs, order, actions, and distances. The controller reproduction checks **37,320 projections** across the existing five settings and both fixed-authority state banks. The original-preview residual is approximately **9.02 × 10⁻¹⁷**. The completed sensitivity outputs match the supplied exports. This reproduction generated no model calls and no new portfolio trajectories.

## The one proposed new experiment

Use the same 1,244 recorded fixed-authority states for every condition. Compare retrieval exposed and hidden with **Llama 3.3 70B** and **Qwen3 8B**. Make three fresh, separately served repeats per state, model, and condition. This requires **14,928 planned calls**, plus a **48-call engineering pilot** excluded from the main analysis.

The models use the same neutral instructions. Qwen thinking is disabled, following the recorded successful structured-output check. Their sizes differ, so this tests two model configurations; it does not isolate model-family effects. The live model digests, templates, and server version are checked and saved. No automatic download or substitution is performed.

The full numerical state, holdings, authority 0.05, original action previews, prompt, and episode content are frozen. The hidden condition changes only the episode list and its summary. All three repeats use the same request and seed. They estimate observed serving stability under that recipe, not independent-seed variability.

Report within-condition repeat variation first. Then report valid action distributions, abstention differences, between-condition advice disagreements, actual-pair transmission, and projected exposure differences. If advice never differs, conditional transmission is **undefined**, not zero. The code gives state-level records, repeat-specific estimates, and descriptive ranges. It does not treat 14,928 calls as independent market observations.

The new test is not required to reproduce 13/33, show a positive financial result, or favor either model. Different patterns remain informative. It cannot isolate episode semantics from prompt length and summary removal. It does not evaluate Qwen's own evolved memory. **No new portfolio-performance claim should be computed from these fixed states.**

## Information still required from your server

1. **The new pilot and main-run results.** Return the two ZIP files produced by the runner. If the pilot fails or the main run pauses, return the partial export before changing settings.
2. **Existing raw-response provenance.** Run `collect_existing_metadata.py` and return `existing_metadata.zip`. It collects the original independent-rerun serving contract and the response journals for the rerun, factorial, and action-balanced comparisons. These are existing records, not additional experiments. The collector reports missing paths and does not invent replacements.
3. **Any surviving separate transport-attempt logs.** If none exist, say so. We will state that full attempt-level verification is unavailable. We will not infer zero retries from accepted outputs.

The current server's availability and Qwen model digest cannot be verified from historical project memory alone. The pilot resolves those questions. The package has been checked locally, but **no live Llama or Qwen experiment has been run here**.

## Changes reserved for the later manuscript update

- Add the unified supplementary inventory and separate the two estimands in the behavioral summary.
- Add the exact numerical projection predicate and fallback order.
- Use consistent associative wording for the recorded behavioral differences.
- Correct the CAR comparison and resolve the citation in the compiled PDF.
- Add the new repeated-serving results only after inspecting the actual returned evidence.

CAR resamples decisions under an unchanged policy and runs stochastic continuations to estimate outcome distributions. Its paper discusses the difficulty of separating direct effects from downstream resampling. RAMAS enumerates a deterministic downstream controller at fixed recorded states. This is the relevant distinction; CAR should not be described as supplying a clean serving-noise control.

Primary sources: [CAR, arXiv:2606.08275v1](https://arxiv.org/html/2606.08275v1), [Ollama chat API](https://docs.ollama.com/api/chat), and [Ollama thinking controls](https://docs.ollama.com/capabilities/thinking). The empirical checks above use the supplied project artifacts, not these external sources.
