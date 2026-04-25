from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    queue_weight: float = 1.0
    waiting_weight: float = 0.35
    flow_weight: float = 2.0
    congestion_weight: float = 0.5
    emergency_bonus: float = 40.0
    ambulance_wait_penalty: float = 3.0
    switch_penalty: float = 1.5
    congestion_threshold: float = 25.0


class RewardEngine:
    """Dense multi-objective reward for adaptive signal control."""

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self.weights = weights or RewardWeights()

    def compute(
        self,
        *,
        queue_sum: float,
        waiting_sum: float,
        throughput: float,
        switched: bool,
        ambulance_wait: float,
        ambulance_cleared: bool,
    ) -> float:
        w = self.weights

        base_cost = (w.queue_weight * queue_sum) + (w.waiting_weight * waiting_sum)
        base_reward = -base_cost

        flow_bonus = w.flow_weight * throughput
        congestion_penalty = w.congestion_weight * max(queue_sum - w.congestion_threshold, 0.0)

        emergency_term = -w.ambulance_wait_penalty * ambulance_wait
        if ambulance_cleared:
            emergency_term += w.emergency_bonus

        switch_term = -w.switch_penalty if switched else 0.0

        total_reward = base_reward + flow_bonus - congestion_penalty + emergency_term + switch_term
        return float(total_reward)
