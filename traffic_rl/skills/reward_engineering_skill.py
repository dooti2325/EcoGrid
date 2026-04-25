from __future__ import annotations

from traffic_rl.reward.reward_engine import RewardEngine, RewardWeights


def build_reward_engine(weights: dict | None = None) -> RewardEngine:
    if not weights:
        return RewardEngine()
    return RewardEngine(weights=RewardWeights(**weights))
