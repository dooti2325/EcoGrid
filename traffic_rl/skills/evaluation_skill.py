from __future__ import annotations

from traffic_rl.evaluation.evaluator import compare_policies, evaluate_agent


def evaluate_system(agent, env_config: dict, baseline_metrics: dict, episodes: int = 20) -> dict:
    rl_metrics = evaluate_agent(agent=agent, env_config=env_config, episodes=episodes)
    comparison = compare_policies(baseline_metrics, rl_metrics)
    return {"rl": rl_metrics, "comparison": comparison}
