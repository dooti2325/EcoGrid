from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from traffic_rl.reward.reward_engine import RewardEngine


@dataclass
class EnvConfig:
    max_steps: int = 200
    service_rate: int = 2
    arrival_mode: str = "stochastic"
    arrival_sequence: list[list[int]] | None = None
    seed: int = 42
    ambulance_spawn_prob: float = 0.05
    ambulance_lane: int = 0
    lane_bias: tuple[float, float, float, float] = (1.3, 0.8, 1.1, 0.6)
    peak_rates: tuple[float, float, float, float] = (3.2, 2.0, 2.8, 1.4)
    offpeak_rates: tuple[float, float, float, float] = (1.2, 0.8, 1.0, 0.6)
    peak_duration: int = 20
    cycle_duration: int = 40


class TrafficEnv:
    """Deterministic and fast OpenEnv-style traffic control environment."""

    def __init__(self, config: dict | None = None, reward_engine: RewardEngine | None = None) -> None:
        raw = config or {}
        self.config = EnvConfig(**{**EnvConfig().__dict__, **raw})
        self.reward_engine = reward_engine or RewardEngine()

        self.rng = np.random.default_rng(self.config.seed)
        self.step_count = 0
        self.phase = 0  # 0: NS green, 1: EW green
        self.lane_queues: list[deque[tuple[str, int]]] = [deque() for _ in range(4)]
        self.ambulance_active = False

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.phase = 0
        self.lane_queues = [deque() for _ in range(4)]
        self.ambulance_active = False
        return self.observation()

    def observation(self) -> np.ndarray:
        queue_lengths = [len(q) for q in self.lane_queues]
        waiting_loads = [sum(wait for _, wait in q) for q in self.lane_queues]
        ambulance_flag = 1.0 if self.ambulance_active else 0.0
        obs = np.asarray(
            queue_lengths + waiting_loads + [float(self.phase), ambulance_flag],
            dtype=np.float32,
        )
        return obs

    def compute_reward(
        self,
        *,
        throughput: int,
        switched: bool,
        ambulance_cleared: bool,
    ) -> float:
        queue_sum = float(sum(len(q) for q in self.lane_queues))
        waiting_sum = float(sum(sum(wait for _, wait in q) for q in self.lane_queues))
        ambulance_wait = float(self._ambulance_wait())
        return self.reward_engine.compute(
            queue_sum=queue_sum,
            waiting_sum=waiting_sum,
            throughput=float(throughput),
            switched=switched,
            ambulance_wait=ambulance_wait,
            ambulance_cleared=ambulance_cleared,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if action not in (0, 1, 2):
            raise ValueError("action must be 0 (hold), 1 (NS), or 2 (EW)")

        previous_phase = self.phase
        if action == 1:
            self.phase = 0
        elif action == 2:
            self.phase = 1
        switched = self.phase != previous_phase

        arrivals = self._next_arrivals()
        self._apply_arrivals(arrivals)
        self._maybe_spawn_ambulance()

        throughput, ambulance_cleared = self._serve_current_phase()
        self._increment_wait_times()

        reward = self.compute_reward(
            throughput=throughput,
            switched=switched,
            ambulance_cleared=ambulance_cleared,
        )

        self.step_count += 1
        done = self.step_count >= self.config.max_steps

        obs = self.observation()
        info = {
            "throughput": throughput,
            "queue_sum": int(sum(obs[:4])),
            "waiting_sum": float(np.sum(obs[4:8])),
            "phase": int(self.phase),
            "switched": switched,
            "ambulance_cleared": ambulance_cleared,
            "arrivals": arrivals,
        }
        return obs, float(reward), bool(done), info

    def _next_arrivals(self) -> list[int]:
        if self.config.arrival_mode == "deterministic" and self.config.arrival_sequence:
            idx = self.step_count % len(self.config.arrival_sequence)
            vals = self.config.arrival_sequence[idx]
            return [max(0, int(v)) for v in vals]

        phase_pos = self.step_count % self.config.cycle_duration
        is_peak = phase_pos < self.config.peak_duration
        base = self.config.peak_rates if is_peak else self.config.offpeak_rates
        rates = np.asarray(base, dtype=np.float32) * np.asarray(self.config.lane_bias, dtype=np.float32)
        arrivals = self.rng.poisson(rates).astype(int).tolist()
        return [max(0, int(v)) for v in arrivals]

    def _apply_arrivals(self, arrivals: list[int]) -> None:
        for lane_idx, count in enumerate(arrivals):
            for _ in range(max(0, count)):
                self.lane_queues[lane_idx].append(("car", 0))

    def _maybe_spawn_ambulance(self) -> None:
        if self.ambulance_active:
            return

        if self.rng.random() < self.config.ambulance_spawn_prob:
            lane = int(self.config.ambulance_lane)
            self.lane_queues[lane].append(("ambulance", 0))
            self.ambulance_active = True

    def _serve_current_phase(self) -> tuple[int, bool]:
        green_lanes = (0, 2) if self.phase == 0 else (1, 3)
        throughput = 0
        ambulance_cleared = False

        for lane in green_lanes:
            for _ in range(self.config.service_rate):
                if not self.lane_queues[lane]:
                    break
                vehicle_type, _wait = self.lane_queues[lane].popleft()
                throughput += 1
                if vehicle_type == "ambulance":
                    ambulance_cleared = True
                    self.ambulance_active = False

        return throughput, ambulance_cleared

    def _increment_wait_times(self) -> None:
        for lane_idx, lane_q in enumerate(self.lane_queues):
            updated = deque((vehicle_type, wait + 1) for vehicle_type, wait in lane_q)
            self.lane_queues[lane_idx] = updated

    def _ambulance_wait(self) -> int:
        if not self.ambulance_active:
            return 0
        for lane in self.lane_queues:
            for vehicle_type, wait in lane:
                if vehicle_type == "ambulance":
                    return wait
        return 0
