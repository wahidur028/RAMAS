from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass
class MemoryEpisode:
    episode_id: str
    context_key: str
    decision_time_utc: str
    operator: str
    horizon_days: int
    agent_net_log_return: float
    baseline_net_log_return: float
    agent_max_drawdown: float
    baseline_max_drawdown: float
    extra_turnover: float
    score: float
    completed: bool = True

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def counterfactual_score(
    *,
    agent_net_log_return: float,
    baseline_net_log_return: float,
    agent_max_drawdown: float,
    baseline_max_drawdown: float,
    extra_turnover: float,
    downside_penalty: float,
    turnover_penalty: float,
) -> float:
    extra_drawdown = max(0.0, abs(float(agent_max_drawdown)) - abs(float(baseline_max_drawdown)))
    return (
        float(agent_net_log_return)
        - float(baseline_net_log_return)
        - downside_penalty * extra_drawdown
        - turnover_penalty * max(0.0, float(extra_turnover))
    )


def circular_block_lower_bound(
    values: list[float], *, block_length: int, resamples: int, alpha: float, seed: int
) -> float:
    if not values:
        return float("-inf")
    length = min(max(1, int(block_length)), len(values))
    blocks_needed = math.ceil(len(values) / length)
    rng = random.Random(int(seed))
    boot: list[float] = []
    for _ in range(int(resamples)):
        selected: list[float] = []
        for _ in range(blocks_needed):
            start = rng.randrange(len(values))
            selected.extend(values[(start + offset) % len(values)] for offset in range(length))
        boot.append(mean(selected[: len(values)]))
    boot.sort()
    position = max(0, min(len(boot) - 1, int(math.floor(alpha * (len(boot) - 1)))))
    return float(boot[position])


@dataclass
class MemoryStore:
    config: dict[str, Any]
    episodes: list[MemoryEpisode] = field(default_factory=list)
    lessons: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_completed_episode(self, episode: MemoryEpisode) -> None:
        if not episode.completed:
            raise ValueError("only completed counterfactual episodes enter memory")
        if any(item.episode_id == episode.episode_id for item in self.episodes):
            raise ValueError("memory episode identifier is not unique")
        self.episodes.append(episode)

    def retrieve(self, context_key: str, limit: int) -> list[dict[str, Any]]:
        matches = [item for item in self.episodes if item.context_key == context_key and item.completed]
        return [item.as_dict() for item in matches[-int(limit) :]]

    def review(self, context_key: str) -> dict[str, Any]:
        matches = [item for item in self.episodes if item.context_key == context_key and item.completed]
        minimum = int(self.config["minimum_completed_episodes_for_promotion"])
        scores = [item.score for item in matches]
        first, second = scores[: len(scores) // 2], scores[len(scores) // 2 :]
        two_blocks = bool(first and second and mean(first) > 0.0 and mean(second) > 0.0)
        lower = circular_block_lower_bound(
            scores,
            block_length=int(self.config["bootstrap_block_length"]),
            resamples=int(self.config["bootstrap_resamples"]),
            alpha=float(self.config["bootstrap_alpha"]),
            seed=int(self.config["bootstrap_seed"]),
        )
        eligible = len(matches) >= minimum and mean(scores) > 0.0 and lower > 0.0 and two_blocks
        previous = self.lessons.get(context_key)
        if eligible:
            status = "ACTIVE"
        elif previous and previous.get("status") == "ACTIVE":
            status = "ROLLED_BACK"
        else:
            status = "QUARANTINED"
        lesson = {
            "context_key": context_key,
            "status": status,
            "completed_episodes": len(matches),
            "mean_counterfactual_score": mean(scores) if scores else None,
            "bootstrap_lower_bound": lower if scores else None,
            "two_positive_chronological_blocks": two_blocks,
        }
        self.lessons[context_key] = lesson
        return lesson


def make_context_key(regime: str, trigger_reasons: list[str], event_categories: list[str]) -> str:
    trigger = "+".join(sorted(set(trigger_reasons))) or "NO_TRIGGER"
    category = "+".join(sorted(set(event_categories))) or "NO_EVENT"
    return f"{regime}|{trigger}|{category}"

