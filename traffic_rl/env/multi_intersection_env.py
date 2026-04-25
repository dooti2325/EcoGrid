from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic_rl.env.traffic_env import TrafficEnv


@dataclass
class MultiStepResult:
    states: list[np.ndarray]
    rewards: list[float]
    dones: list[bool]
    infos: list[dict]


class MultiIntersectionEnv:
    """Bonus decentralized multi-intersection simulator."""

    def __init__(
        self,
        num_intersections: int = 2,
        env_config: dict | None = None,
        transfer_ratio: float = 0.2,
    ) -> None:
        if num_intersections <= 0:
            raise ValueError("num_intersections must be > 0")

        self.transfer_ratio = transfer_ratio
        self.intersections = [TrafficEnv(config=env_config or {}) for _ in range(num_intersections)]

    def reset(self) -> list[np.ndarray]:
        return [env.reset() for env in self.intersections]

    def step(self, actions: list[int]) -> tuple[list[np.ndarray], list[float], list[bool], list[dict]]:
        if len(actions) != len(self.intersections):
            raise ValueError("actions length must match number of intersections")

        states = []
        rewards = []
        dones = []
        infos = []

        upstream_flow = 0
        for idx, (env, action) in enumerate(zip(self.intersections, actions)):
            if idx > 0 and upstream_flow > 0:
                extra = int(round(upstream_flow * self.transfer_ratio))
                env._apply_arrivals([extra, 0, 0, 0])

            state, reward, done, info = env.step(action)
            states.append(state)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
            upstream_flow = int(info.get("throughput", 0))

        return states, rewards, dones, infos
