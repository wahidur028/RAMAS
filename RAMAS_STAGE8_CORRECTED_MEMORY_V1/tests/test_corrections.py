"""Unit checks for the two corrected seams. Run with the pinned interpreter:

    /home/infonet/anaconda3/envs/wahid_test/bin/python tests/test_corrections.py
"""
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, "/home/infonet/wahid/leader_router_fresh/RAMAS_STAGE6_4_COMPONENT_SUITE_V1")

from corrected_memory import labels, retrieval  # noqa: E402

FEATURES = ("prob_bear", "prob_bull", "prob_mix", "router_entropy", "router_transition_l1",
            "return_7", "return_30", "realized_vol_30", "drawdown_90")


class Accounting:
    @staticmethod
    def net_return(exposure, pretrade, asset, cost):
        return (1 - cost * abs(exposure - pretrade)) * (1 + exposure * asset) - 1


def test_label_signs():
    up = dict(pretrade=0.5, asset_return=0.02, cost_rate=0.001, accounting=Accounting)
    btc = labels.counterfactual_label(action="BTC", action_exposure=1.0, core_exposure=0.5, **up)
    cash = labels.counterfactual_label(action="CASH", action_exposure=0.0, core_exposure=0.5, **up)
    assert btc > 0, "raising exposure on an up day must score positive"
    assert cash < 0, "cutting exposure on an up day must score negative"
    down = dict(up, asset_return=-0.02)
    assert labels.counterfactual_label(action="BTC", action_exposure=1.0, core_exposure=0.5, **down) < 0
    assert labels.counterfactual_label(action="CASH", action_exposure=0.0, core_exposure=0.5, **down) > 0


def test_abstain_is_exactly_zero():
    value = labels.counterfactual_label(
        action="ABSTAIN", action_exposure=0.5, core_exposure=0.5,
        pretrade=0.3, asset_return=0.05, cost_rate=0.001, accounting=Accounting)
    assert value == 0.0, "abstaining is following the core and must add exactly nothing"


def test_label_rejects_bad_input():
    for bad in ({"action_exposure": 1.5}, {"core_exposure": -0.1}, {"pretrade": 2.0}):
        kwargs = dict(action="BTC", action_exposure=1.0, core_exposure=0.5, pretrade=0.5,
                      asset_return=0.01, cost_rate=0.001, accounting=Accounting)
        kwargs.update(bad)
        try:
            labels.counterfactual_label(**kwargs)
        except labels.LabelError:
            continue
        raise AssertionError(f"expected LabelError for {bad}")


def _episode(index, action, regime, offset):
    return dict(episode_id=f"episode-{index:06d}", decision_date="2021-01-01",
                return_date="2021-01-02", hard_regime=regime, action=action, confidence=0.5,
                shadow_log_advantage_vs_ramoe=0.001 * offset, asset_return=0.01,
                state_vector=[0.1 * offset] * 9)


def _state(regime="bull"):
    state = {name: 0.0 for name in FEATURES}
    state["hard_regime"] = regime
    return state


def test_balanced_retrieval_surfaces_every_action():
    episodes = [_episode(0, "BTC", "bull", 0), _episode(1, "CASH", "bull", 1),
                _episode(2, "ABSTAIN", "bull", 2), _episode(3, "BTC", "bear", 3),
                _episode(4, "CASH", "bull", 4)]
    got = retrieval.balanced_retrieve(episodes, _state(), per_action=1, regime_filter="same")
    assert {e["action"] for e in got} == {"BTC", "CASH", "ABSTAIN"}
    assert retrieval.action_diversity(retrieval.summarize(got)) == 3


def test_regime_filter_none_widens_the_pool():
    episodes = [_episode(0, "BTC", "bear", 0), _episode(1, "CASH", "bull", 1)]
    same = retrieval.balanced_retrieve(episodes, _state("bull"), per_action=1, regime_filter="same")
    pooled = retrieval.balanced_retrieve(episodes, _state("bull"), per_action=1, regime_filter="none")
    assert {e["action"] for e in same} == {"CASH"}
    assert {e["action"] for e in pooled} == {"BTC", "CASH"}


def test_empty_memory_reports_the_frozen_sentinel():
    assert retrieval.summarize([])["summary"] == retrieval.NO_EVIDENCE
    assert retrieval.action_diversity({"summary": retrieval.NO_EVIDENCE}) == 0


def test_retrieval_is_deterministic_under_distance_ties():
    tied = [_episode(5, "BTC", "bull", 7), _episode(2, "BTC", "bull", 7)]
    first = retrieval.balanced_retrieve(tied, _state(), per_action=2, regime_filter="same")
    second = retrieval.balanced_retrieve(list(reversed(tied)), _state(), per_action=2, regime_filter="same")
    assert [e["episode_id"] for e in first] == [e["episode_id"] for e in second]


if __name__ == "__main__":
    passed = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"  PASS {name}")
            passed += 1
    print(f"\n{passed} checks passed")
