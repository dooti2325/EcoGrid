from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_rl.agent.dqn_agent import DQNAgent
from traffic_rl.env.traffic_env import TrafficEnv


@dataclass
class TrainingConfig:
    episodes: int = 120
    max_steps: int = 120
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    batch_size: int = 64
    buffer_size: int = 50000
    target_sync_interval: int = 10
    seed: int = 42


def _epsilon_for_episode(config: TrainingConfig, episode_idx: int) -> float:
    decayed = config.epsilon_start * (config.epsilon_decay**episode_idx)
    return float(max(config.epsilon_end, decayed))


def train_dqn(env: TrafficEnv, config: TrainingConfig | None = None) -> tuple[DQNAgent, dict[str, list[float]]]:
    cfg = config or TrainingConfig()

    agent = DQNAgent(
        state_dim=10,
        action_dim=3,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    history: dict[str, list[float]] = {
        "episode_reward": [],
        "avg_queue": [],
        "avg_wait": [],
        "throughput": [],
        "loss": [],
        "epsilon": [],
    }

    for episode in range(cfg.episodes):
        state = env.reset()
        epsilon = _epsilon_for_episode(cfg, episode)

        total_reward = 0.0
        step_queue = []
        step_wait = []
        total_throughput = 0

        for _step in range(min(cfg.max_steps, env.config.max_steps)):
            action = agent.select_action(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            if loss is not None:
                history["loss"].append(loss)

            total_reward += reward
            step_queue.append(float(info["queue_sum"]))
            step_wait.append(float(info["waiting_sum"]))
            total_throughput += int(info["throughput"])

            state = next_state
            if done:
                break

        if (episode + 1) % cfg.target_sync_interval == 0:
            agent.sync_target()

        history["episode_reward"].append(float(total_reward))
        history["avg_queue"].append(float(np.mean(step_queue) if step_queue else 0.0))
        history["avg_wait"].append(float(np.mean(step_wait) if step_wait else 0.0))
        history["throughput"].append(float(total_throughput))
        history["epsilon"].append(epsilon)

    return agent, history
