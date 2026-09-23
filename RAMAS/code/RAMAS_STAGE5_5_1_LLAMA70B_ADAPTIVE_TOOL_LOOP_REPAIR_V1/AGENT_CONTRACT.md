# Frozen LLM-agent contract

## What the agent may do

The agent may choose an ordered subset of six tools:

1. `inspect_current_state`
2. `get_verified_events`
3. `inspect_expert_risk`
4. `retrieve_similar_episodes`
5. `simulate_trust_candidates`
6. `inspect_active_intervention`

After seeing tool results, it may select one frozen trust operator, one regime
row, one strength, and one horizon. It may abstain through `KEEP`.

## What the agent may not do

- Output BTC exposure.
- Output portfolio weights.
- Change the router or expert models.
- Modify the accounting or return clock.
- Bypass the deterministic risk shield.
- Read an event not available by the decision timestamp.
- Treat its own unsupported explanation as evidence.
- activate more than one intervention.

## Decision response

```json
{
  "operator": "KEEP",
  "target_regime": "bear",
  "strength": 0.0,
  "horizon_days": 7,
  "confidence": 0.5,
  "hypothesis": "",
  "invalidation_condition": ""
}
```

Allowed operators are `KEEP`, `DEFENSIVE_SHRINK`, `RISK_AWARE_REWEIGHT`, and
`ROLLBACK`. Allowed strengths are 0, 0.05, and 0.10. Allowed horizons are 3, 7,
and 14 days.

## Automatic provenance

Every tool result receives a deterministic identifier based on the tool name
and canonical result hash. The runtime attaches these identifiers to the
decision trace. The model does not manually copy them. Operator-specific tool
coverage remains mandatory and is checked by code.

## Bounded correction round

The first proposal is a draft. If its operator-specific evidence is incomplete,
the runtime executes only the missing required tools. If the draft violates a
safety limit, the runtime reports the exact shield reasons. Llama then receives
all available tool results and one structured correction request.

The second proposal must use an evidence-complete operator and a shield-safe
simulated candidate. It may reduce the strength or choose `KEEP`. There is no
third decision round.

## Failure behavior

An invalid plan, invalid decision, provider failure, or unsafe final proposal
results in unchanged trust. The single correction round never loosens the
contract; a repeated failure becomes `KEEP`.
