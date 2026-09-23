from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np


REGIMES = ("bear", "bull", "mix")


@dataclass
class RegimeTrust:
    config: dict[str, Any]
    beta: dict[str, float] = field(init=False)
    last_month: tuple[int, int] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        initial = float(self.config["initial_beta"])
        self.beta = {regime: initial for regime in REGIMES}

    def value(self, regime: str) -> float:
        return float(self.beta[regime])

    def maybe_update(self, decision_date: date, completed: list[dict[str, Any]]) -> None:
        month = (decision_date.year, decision_date.month)
        if self.last_month is None:
            self.last_month = month
            return
        if month == self.last_month:
            return
        old_month = self.last_month
        self.last_month = month
        minimum = int(self.config["minimum_completed_episodes_per_regime"])
        lookback = int(self.config["lookback_episodes_per_regime"])
        step = float(self.config["monthly_step"])
        lower = float(self.config["minimum_beta"])
        upper = float(self.config["maximum_beta"])
        increase = float(self.config["minimum_mean_log_advantage_to_increase"])
        decrease = float(self.config["maximum_mean_log_advantage_to_decrease"])
        for regime in REGIMES:
            eligible = [
                item for item in completed
                if item["hard_regime"] == regime and item["action"] != "ABSTAIN"
            ][-lookback:]
            old = float(self.beta[regime])
            mean = None
            direction = "HOLD_INSUFFICIENT"
            new = old
            if len(eligible) >= minimum:
                values = np.asarray(
                    [item["shadow_log_advantage_vs_ramoe"] for item in eligible],
                    dtype=float,
                )
                mean = float(np.mean(values))
                downside = float(np.quantile(values, 0.10))
                if mean > increase and downside >= -0.03:
                    new = min(upper, old + step)
                    direction = "INCREASE"
                elif mean < decrease:
                    new = max(lower, old - step)
                    direction = "DECREASE"
                else:
                    direction = "HOLD"
            self.beta[regime] = float(new)
            self.events.append(
                {
                    "effective_year": month[0],
                    "effective_month": month[1],
                    "evidence_through_year": old_month[0],
                    "evidence_through_month": old_month[1],
                    "regime": regime,
                    "eligible_completed_episodes": len(eligible),
                    "mean_shadow_log_advantage": mean,
                    "old_beta": old,
                    "new_beta": float(new),
                    "direction": direction,
                }
            )


def blend_exposure(base_exposure: float, action: str, beta: float) -> float:
    if action == "ABSTAIN":
        return float(base_exposure)
    target = 1.0 if action == "BTC" else 0.0
    return float((1.0 - beta) * base_exposure + beta * target)
