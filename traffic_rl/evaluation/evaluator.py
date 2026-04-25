from __future__ import annotations

from statistics import mean
from typing import Callable

from traffic_rl.agent.dqn_agent import DQNAgent
from traffic_rl.baseline.fixed_time_controller import FixedTimeController
from traffic_rl.env.traffic_env import TrafficEnv


PolicyFn = Callable[[list[float], int], int]


def run_episode(env: TrafficEnv, policy: PolicyFn) -> dict[str, float]:
    state = env.reset()
    done = False
    step_idx = 0

    rewards = []
    queues = []
    waits = []
    throughputs = []
    ambulance_cleared_count = 0

    while not done:
        action = policy(state.tolist(), step_idx)
        state, reward, done, info = env.step(action)

        rewards.append(float(reward))
        queues.append(float(info["queue_sum"]))
        waits.append(float(info["waiting_sum"]))
        throughputs.append(float(info["throughput"]))
        ambulance_cleared_count += int(bool(info["ambulance_cleared"]))
        step_idx += 1

    return {
        "reward": float(sum(rewards)),
        "avg_queue_length": float(mean(queues) if queues else 0.0),
        "avg_waiting_time": float(mean(waits) if waits else 0.0),
        "throughput": float(sum(throughputs)),
        "ambulance_clearances": float(ambulance_cleared_count),
    }


def evaluate_fixed_controller(
    *,
    env_config: dict,
    episodes: int = 20,
    switch_interval: int = 5,
) -> dict[str, float]:
    controller = FixedTimeController(switch_interval=switch_interval)

    def policy(_state: list[float], step: int) -> int:
        return controller.action_for_step(step)

    episode_metrics = []
    for _ in range(episodes):
        env = TrafficEnv(config=env_config)
        episode_metrics.append(run_episode(env, policy))

    return _aggregate(episode_metrics)


def evaluate_agent(
    *,
    agent: DQNAgent,
    env_config: dict,
    episodes: int = 20,
) -> dict[str, float]:
    def policy(state: list[float], _step: int) -> int:
        return agent.select_action(state, epsilon=0.0)

    episode_metrics = []
    for _ in range(episodes):
        env = TrafficEnv(config=env_config)
        episode_metrics.append(run_episode(env, policy))

    return _aggregate(episode_metrics)


def _aggregate(episode_metrics: list[dict[str, float]]) -> dict[str, float]:
    keys = episode_metrics[0].keys()
    return {k: float(mean(m[k] for m in episode_metrics)) for k in keys}


def compare_policies(baseline: dict[str, float], rl: dict[str, float]) -> dict[str, float]:
    def pct_improve(lower_is_better_key: str) -> float:
        b = baseline[lower_is_better_key]
        r = rl[lower_is_better_key]
        if b == 0:
            return 0.0
        return float(((b - r) / b) * 100.0)

    def pct_gain(higher_is_better_key: str) -> float:
        b = baseline[higher_is_better_key]
        r = rl[higher_is_better_key]
        if b == 0:
            return 0.0
        return float(((r - b) / b) * 100.0)

    return {
        "waiting_time_improvement_pct": pct_improve("avg_waiting_time"),
        "queue_length_improvement_pct": pct_improve("avg_queue_length"),
        "throughput_gain_pct": pct_gain("throughput"),
        "ambulance_clearance_gain_pct": pct_gain("ambulance_clearances"),
    }


def collect_trajectory(env: TrafficEnv, policy: PolicyFn) -> dict[str, list[float]]:
    state = env.reset()
    done = False
    step_idx = 0

    trace = {
        "reward": [],
        "queue_sum": [],
        "waiting_sum": [],
        "phase": [],
    }

    while not done:
        action = policy(state.tolist(), step_idx)
        state, reward, done, info = env.step(action)
        trace["reward"].append(float(reward))
        trace["queue_sum"].append(float(info["queue_sum"]))
        trace["waiting_sum"].append(float(info["waiting_sum"]))
        trace["phase"].append(float(info["phase"]))
        step_idx += 1

    return trace
