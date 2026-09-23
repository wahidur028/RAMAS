from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import REGIMES, MarketState
from .trust import TrustMatrix, btc_exposure, row_l1, validate_matrix


@dataclass(frozen=True)
class ShieldResult:
    accepted: bool
    reasons: tuple[str, ...]
    current_exposure: float
    proposed_exposure: float
    target_row_l1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "current_exposure": self.current_exposure,
            "proposed_exposure": self.proposed_exposure,
            "target_row_l1": self.target_row_l1,
        }


def evaluate_shield(
    state: MarketState,
    current: TrustMatrix,
    proposed: TrustMatrix,
    *,
    target_regime: str,
    config: dict[str, Any],
    active_intervention: bool,
    evidence_complete: bool,
) -> ShieldResult:
    current = validate_matrix(current)
    proposed = validate_matrix(proposed)
    reasons: list[str] = []
    current_exposure = btc_exposure(state, current)
    proposed_exposure = btc_exposure(state, proposed)
    movement = row_l1(current[target_regime], proposed[target_regime])
    if active_intervention:
        reasons.append("ACTIVE_INTERVENTION_EXISTS")
    if not evidence_complete:
        reasons.append("OPERATOR_EVIDENCE_INCOMPLETE")
    if movement > float(config["maximum_target_row_l1_change"]) + 1e-12:
        reasons.append("TRUST_ROW_CHANGE_TOO_LARGE")
    if abs(proposed_exposure - current_exposure) > float(config["maximum_btc_exposure_change"]) + 1e-12:
        reasons.append("BTC_EXPOSURE_CHANGE_TOO_LARGE")
    if not float(config["minimum_btc_exposure"]) <= proposed_exposure <= float(config["maximum_btc_exposure"]):
        reasons.append("BTC_EXPOSURE_OUT_OF_BOUNDS")
    changed_rows = [
        regime for regime in REGIMES if row_l1(current[regime], proposed[regime]) > 1e-12
    ]
    if len(changed_rows) > 1 or (changed_rows and changed_rows[0] != target_regime):
        reasons.append("MORE_THAN_ONE_TRUST_ROW_CHANGED")
    return ShieldResult(not reasons, tuple(reasons), current_exposure, proposed_exposure, movement)

