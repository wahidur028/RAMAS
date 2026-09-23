# Frozen Llama-70B agent contract

- Model: `llama3.3:70b` only.
- Actions: `BTC`, `CASH`, or `ABSTAIN`.
- The agent is a bounded expert, not the router, risk layer, or entire portfolio manager.
- `ABSTAIN` preserves the pre-agent RAMoE desired exposure exactly before risk projection.
- `BTC` and `CASH` are blended using regime-specific beta in `[0.00, 0.20]`.
- The unchanged risk layer remains final authority.
- Invalid output or provider failure becomes `ABSTAIN` and is logged.
- A memory citation is valid only if that completed episode was included in the prompt.
- The current next-day return and all future data are forbidden from the prompt.
- An episode enters memory only after its return is complete.
- Trust changes only at a new month using completed, non-abstaining shadow outcomes.
- Memory, trust, and portfolio state continue across calendar years without reset.
- Human parameter or architecture changes after 2022 is opened invalidate the run.
