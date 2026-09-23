# Agent contract

- Model: `llama3.3:70b` only.
- Output keys: `action`, `confidence`, `reason_codes`, `cited_memory_ids`.
- Actions: `BTC`, `CASH`, `ABSTAIN`.
- Invalid JSON, timeout, unknown action, unknown reason, or unseen memory citation becomes `ABSTAIN`.
- The agent cannot change the router, original RAMAS trust, transaction cost, or risk layer.
- The agent never receives the target next-day return in its request.
- Memory contains only completed earlier episodes.
- Trust is separate by hard regime, starts at 0.05, and is bounded to `[0.00, 0.20]`.
