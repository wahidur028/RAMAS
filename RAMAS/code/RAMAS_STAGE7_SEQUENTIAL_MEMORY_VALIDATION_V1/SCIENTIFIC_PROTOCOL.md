# RAMAS Stage 7 scientific protocol

## Primary question

For a repeated BTC/cash decision process, does giving a frozen Llama-70B advisor
access to **completed, point-in-time eligible episodes** change later bounded
decisions and improve the paired daily net-log-return process relative to the
same advisor with memory retrieval disabled?

## Arms and intervention

The closed-loop pair starts from empty memory, identical initial trust, wealth,
holdings and model serving identity. Both arms continue through 2021–2025 with
no annual reset. The only intended advisor-input intervention is:

| Arm | Llama | beta | retrieval | state evolution |
|---|---|---:|---|---|
| `closed_loop_memory` | yes | 0.05 fixed | same-regime completed episodes | own actions/outcomes |
| `closed_loop_no_memory` | yes | 0.05 fixed | none | own actions/outcomes |

Fixed beta removes the confounding of adaptive trust. The earlier adaptive-beta
memory result remains a separate total-system contrast.

The state-controlled pair uses the deterministic source pre-trade stream for
both arms. It asks whether the same current state plus different visible
memory produces different advice. It is not used as the primary economic claim.

## Pre-registered endpoints

1. Technical: every Llama response validates the exact action schema; every
   retrieved episode satisfies `episode.return_date <= decision_date` and
   `episode.decision_date < decision_date`; the no-memory arm has zero retrieved
   or cited memory.
2. Transmission: action, desired exposure and final exposure difference counts,
   overall and by Bull/Bear/Mix and calendar year.
3. Primary economic endpoint: paired mean daily **net log return after trading
   costs**, memory minus no-memory, from 2022-01-01 onward.
4. Secondary risk endpoints: compounded net return, annualized volatility,
   maximum drawdown, daily ES95, turnover, mean BTC exposure and regime/year
   conditional contributions.

Uncertainty uses a circular 30-day moving-block bootstrap with 5,000 draws and
seed 16062. The interval is pointwise and not multiplicity-adjusted. If it
contains zero, the result is inconclusive—not evidence of no possible effect.

## Interpretation rules

- If memory changes no actions or final exposures, the current prompt/risk
  contract did not transmit memory into portfolio behaviour.
- If memory changes actions but not final exposures, the projection or beta
  layer is suppressing the language intervention.
- If final exposures change but the paired interval crosses zero, adaptation is
  technically present but economic usefulness is not established.
- If the interval excludes zero but source timing, pretraining provenance or
  multiple-test controls fail, report it as a diagnostic association, not a
  confirmatory learning claim.
- B&H, cash and static 50/50 remain contextual financial controls; this package
  does not define success as beating them.

## Known boundary

The package reuses the Stage 6.4 source adapter and therefore preserves its
legacy clock/reference seam. It is a matched mechanism validation, not a new
out-of-sample confirmation. A confirmatory result requires the same four-arm
design rerun after a separately hashed clock/fill repair and a date-masked
training-provenance audit.
