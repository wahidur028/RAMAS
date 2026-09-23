# RAMAS full-pipeline isolation contract

## Decision

The experiment tests whether the observed advice-to-execution attenuation is
stable across LLMs, serving realizations, memory access, authority mode, and
the deterministic projection controller. It is not a search for the most
profitable arm.

## Factorial

Each enabled model and sampling seed runs eight independent sequential arms:

| Factor | Levels |
|---|---|
| Episodic retrieval | expanding completed memory; no retrieved memory |
| Authority | adaptive regime-specific beta; fixed beta = 0.05 |
| Controller | ON; OFF |

Two model-free numerical controls are run once:

- numerical-only/controller ON;
- numerical-only/controller OFF.

Controller OFF removes CVaR projection, exposure-grid rounding, and turnover
limiting together. It retains long-only/no-leverage exposure bounds and the
same proportional transaction-cost rule.

## Estimands

1. Memory total-system effect within model, seed, authority, and controller.
2. Authority total-system effect within model, seed, memory, and controller.
3. Controller total-system effect within model, seed, memory, and authority.
4. Advisor value relative to the matched numerical-only arm.
5. Controller × advisor difference-in-differences.
6. Projection-only effect: replay each controller-ON desired-exposure sequence
   without projection through the complete 2021 burn-in before selecting the
   2022–2025 evaluation window.

Closed-loop pairs need not retain the same later state. Therefore their
differences are total-system effects. The verifier separately counts dates on
which decision-time states match; only those dates support pair-specific
advice-change claims.

## Seeds

LLM sampling seeds, the numerical provenance seed, and the bootstrap seed are
separate. A temperature-zero determinism probe precedes the full run. If seed
changes produce identical responses, repeated seeds are not independent
evidence and the runner refuses the redundant matrix unless explicitly
overridden.

The historical system prompt is held fixed except that its role label
"bounded Llama expert" is replaced for every model, including Llama, by
"bounded language-model expert." This declared one-phrase neutralization
removes a model-identity confound without changing the task, evidence,
constraints, action space, or output schema.

## Evidence classification

This is a retrospective reused-history mechanism diagnostic. It does not
resolve the regime predictor's point-in-time provenance, the feasible
information-to-inference-to-order-to-fill clock, the 2024 reference seam, or
fresh out-of-sample performance. Those labels must remain in the manuscript.

## Stop rules

- Stop on source, schema, digest, model-identity, or accounting mismatch.
- Stop before scoring a malformed model response; never convert it to an
  ABSTAIN trade.
- Do not stop for weak or negative economic results.
- Never overwrite a completed arm or resample a durable response.
