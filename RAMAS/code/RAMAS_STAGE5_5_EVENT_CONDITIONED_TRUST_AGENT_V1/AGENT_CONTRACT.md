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

## Failure behavior

Invalid plan, invalid decision, provider failure, missing tools, active
intervention, or safety rejection results in unchanged trust. There is no
retry that loosens the contract.

