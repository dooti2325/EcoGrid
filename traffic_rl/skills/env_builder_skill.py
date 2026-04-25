from __future__ import annotations

from traffic_rl.env.traffic_env import TrafficEnv


def build_environment(config: dict | None = None) -> TrafficEnv:
    return TrafficEnv(config=config or {})
