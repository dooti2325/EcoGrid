from __future__ import annotations

from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.training.trainer import TrainingConfig, train_dqn


def train_agent(env_config: dict, config: TrainingConfig | None = None):
    env = TrafficEnv(config=env_config)
    return train_dqn(env=env, config=config)
