"""Per-episode credit labels.

The label is the value written next to a stored episode.  It is the only
outcome signal the agent ever sees about its own past decisions, and in the
frozen design it is also the signal that drives the adaptive trust rule.  It
therefore has to measure decision quality and nothing else.
"""
from __future__ import annotations

import math
from typing import Any


ACTIONS = ("BTC", "CASH", "ABSTAIN")


class LabelError(ValueError):
    pass


def legacy_label(shadow_net: float, reference_net: float) -> float:
    """The frozen Stage 6.4 label: full-authority shadow minus archived legacy.

    Preserved verbatim so legacy mode reproduces the archived arms exactly and
    so both labels can be recorded side by side for audit.
    """
    for value in (shadow_net, reference_net):
        if not math.isfinite(value) or value <= -1.0:
            raise LabelError("Label inputs must be finite and greater than -100%")
    return math.log1p(shadow_net) - math.log1p(reference_net)


def counterfactual_label(
    *,
    action: str,
    action_exposure: float,
    core_exposure: float,
    pretrade: float,
    asset_return: float,
    cost_rate: float,
    accounting: Any,
) -> float:
    """Advantage of the taken action over the unchanged core, same holdings path.

    ``action_exposure`` is the exposure that would have been executed for this
    action at full advisory authority; ``core_exposure`` is the exposure the
    unchanged numerical core would have executed on the same day.  Both are
    projected from the SAME ``pretrade`` holdings, so the turnover and cost
    terms are directly comparable and no cross-portfolio drift enters.

    ABSTAIN is the baseline by construction and scores exactly zero: abstaining
    is following the core, so it adds nothing and is credited with nothing.
    """
    if action not in ACTIONS:
        raise LabelError(f"Unknown action: {action!r}")
    if action == "ABSTAIN":
        return 0.0
    for name, value in (("action_exposure", action_exposure), ("core_exposure", core_exposure),
                        ("pretrade", pretrade)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise LabelError(f"{name} must be a finite exposure inside [0,1]")
    acted = accounting.net_return(action_exposure, pretrade, asset_return, cost_rate)
    base = accounting.net_return(core_exposure, pretrade, asset_return, cost_rate)
    return math.log1p(float(acted)) - math.log1p(float(base))


def build_label(
    mode: str,
    *,
    action: str,
    shadow_net: float,
    reference_net: float,
    action_exposure: float,
    core_exposure: float,
    pretrade: float,
    asset_return: float,
    cost_rate: float,
    accounting: Any,
) -> tuple[float, float, float]:
    """Return (active_label, legacy_value, counterfactual_value).

    Both values are always computed and recorded.  Only ``active_label`` reaches
    episodic retrieval and the trust rule, so a run is never ambiguous about
    which signal drove its behaviour.
    """
    legacy_value = legacy_label(shadow_net, reference_net)
    counterfactual_value = counterfactual_label(
        action=action, action_exposure=action_exposure, core_exposure=core_exposure,
        pretrade=pretrade, asset_return=asset_return, cost_rate=cost_rate, accounting=accounting,
    )
    if mode == "legacy":
        active = legacy_value
    elif mode == "counterfactual":
        active = counterfactual_value
    else:
        raise LabelError(f"Unknown label mode: {mode!r}")
    if not math.isfinite(active):
        raise LabelError("Active label is not finite")
    return active, legacy_value, counterfactual_value
