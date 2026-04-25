from __future__ import annotations

from traffic_rl.evaluation.evaluator import evaluate_fixed_controller


def run_fixed_baseline(env_config: dict, episodes: int = 20, switch_interval: int = 5) -> dict[str, float]:
    return evaluate_fixed_controller(env_config=env_config, episodes=episodes, switch_interval=switch_interval)
