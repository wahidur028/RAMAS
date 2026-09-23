"""Episodic retrieval strategies.

The frozen design filters completed episodes to the current hard regime and
returns the k nearest overall.  Measured on the archived Stage 6.4 prompts, all
retrieved episodes carry the SAME action on 85.9% of days, so the evidence block
can express a contrast between candidate actions on only 13.5% of days.

``balanced`` retrieval instead takes the k nearest episodes PER ACTION, so a
contrast is available whenever the history contains the actions at all.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from stage63lib.legacy import state_vector  # re-export; also puts vendor/stage61 on sys.path

ACTIONS = ("BTC", "CASH", "ABSTAIN")
EPISODE_FIELDS = (
    "episode_id", "decision_date", "hard_regime", "action", "confidence",
    "shadow_log_advantage_vs_ramoe", "asset_return",
)
NO_EVIDENCE = "NO_COMPLETED_SIMILAR_EPISODES"


class RetrievalError(ValueError):
    pass


def _present(episode: dict[str, Any], distance: float) -> dict[str, Any]:
    output = {key: episode[key] for key in EPISODE_FIELDS}
    output["distance"] = float(distance)
    return output


def _scored(episodes: Iterable[dict[str, Any]], query: np.ndarray) -> list[tuple[float, dict[str, Any]]]:
    scored = []
    for episode in episodes:
        vector = np.asarray(episode["state_vector"], dtype=float)
        if vector.shape != query.shape:
            raise RetrievalError("Stored state vector has a different width than the query")
        scored.append((float(np.linalg.norm(query - vector)), episode))
    scored.sort(key=lambda item: (item[0], item[1]["episode_id"]))
    return scored


def balanced_retrieve(
    episodes: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    per_action: int,
    regime_filter: str = "same",
) -> list[dict[str, Any]]:
    """Return up to ``per_action`` nearest completed episodes for each action."""
    if per_action <= 0:
        return []
    if regime_filter not in {"same", "none"}:
        raise RetrievalError(f"Unknown regime_filter: {regime_filter!r}")
    query = state_vector(state)
    pool = episodes
    if regime_filter == "same":
        pool = [e for e in episodes if e["hard_regime"] == state["hard_regime"]]
    selected: list[tuple[float, dict[str, Any]]] = []
    for action in ACTIONS:
        candidates = _scored((e for e in pool if e["action"] == action), query)
        selected.extend(candidates[:per_action])
    selected.sort(key=lambda item: (item[0], item[1]["episode_id"]))
    return [_present(episode, distance) for distance, episode in selected]


def summarize(retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieved episodes by action, matching the frozen block shape."""
    if not retrieved:
        return {"similar_completed_episodes": [], "summary": NO_EVIDENCE}
    by_action: dict[str, list[float]] = {}
    for episode in retrieved:
        by_action.setdefault(episode["action"], []).append(
            float(episode["shadow_log_advantage_vs_ramoe"])
        )
    summary = {
        action: {
            "count": len(values),
            "mean_log_advantage_vs_ramoe": float(np.mean(values)),
            "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
        }
        for action, values in sorted(by_action.items())
    }
    return {"similar_completed_episodes": retrieved, "summary": summary}


def action_diversity(evidence: dict[str, Any]) -> int:
    """Number of distinct actions present in an evidence block (0 when empty).

    Recorded per day so retrieval collapse is measured directly rather than
    inferred from the economic result.
    """
    summary = evidence.get("summary")
    return len(summary) if isinstance(summary, dict) else 0
