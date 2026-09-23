from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


FEATURES = (
    "prob_bear",
    "prob_bull",
    "prob_mix",
    "router_entropy",
    "router_transition_l1",
    "return_7",
    "return_30",
    "realized_vol_30",
    "drawdown_90",
)


def state_vector(state: dict[str, Any]) -> np.ndarray:
    scales = {
        "router_transition_l1": 2.0,
        "return_7": 0.5,
        "return_30": 1.0,
        "realized_vol_30": 2.0,
        "drawdown_90": 1.0,
    }
    return np.asarray(
        [float(state[name]) / scales.get(name, 1.0) for name in FEATURES],
        dtype=float,
    )


@dataclass
class EpisodicMemory:
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def retrieve(self, state: dict[str, Any], maximum: int) -> list[dict[str, Any]]:
        if maximum <= 0:
            return []
        query = state_vector(state)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for episode in self.episodes:
            if episode["hard_regime"] != state["hard_regime"]:
                continue
            vector = np.asarray(episode["state_vector"], dtype=float)
            distance = float(np.linalg.norm(query - vector))
            candidates.append((distance, episode))
        candidates.sort(key=lambda item: (item[0], item[1]["episode_id"]))
        output = []
        for distance, episode in candidates[:maximum]:
            output.append(
                {
                    "episode_id": episode["episode_id"],
                    "decision_date": episode["decision_date"],
                    "hard_regime": episode["hard_regime"],
                    "action": episode["action"],
                    "confidence": episode["confidence"],
                    "shadow_log_advantage_vs_ramoe": episode["shadow_log_advantage_vs_ramoe"],
                    "asset_return": episode["asset_return"],
                    "distance": distance,
                }
            )
        return output

    def add_completed(self, episode: dict[str, Any]) -> None:
        required = {
            "episode_id",
            "decision_date",
            "return_date",
            "hard_regime",
            "state_vector",
            "action",
            "confidence",
            "shadow_log_advantage_vs_ramoe",
            "asset_return",
        }
        missing = required - set(episode)
        if missing:
            raise ValueError(f"memory episode missing fields={sorted(missing)}")
        self.episodes.append(dict(episode))

    def evidence_summary(self, state: dict[str, Any], maximum: int) -> dict[str, Any]:
        episodes = self.retrieve(state, maximum)
        if not episodes:
            return {"similar_completed_episodes": [], "summary": "NO_COMPLETED_SIMILAR_EPISODES"}
        by_action: dict[str, list[float]] = {}
        for episode in episodes:
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
        return {"similar_completed_episodes": episodes, "summary": summary}
