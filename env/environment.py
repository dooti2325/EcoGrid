"""
EcoGrid-OpenEnv — Core Environment

Implements the OpenEnv interface: reset(), step(), state().
All state transitions are deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional

from models.schemas import GridState, GridAction, StepResult, TaskConfig
from env.dynamics import (
    solar_output,
    wind_output,
    demand_curve,
    update_battery,
    compute_blackout_risk,
    carbon_emission,
    compute_supply,
    compute_price_signal,
    compute_grid_stability,
)
from env.reward import compute_reward


# ── Task Configurations ──────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": TaskConfig(
        name="easy",
        task_id="basic_grid_balance",
        episode_length=48,
        noise_level=0.05,
        carbon_budget=1000.0,
        battery_capacity=0.0,      # No battery in easy mode
        demand_volatility=0.2,
        carbon_strict=False,
        volatility_multiplier=1.0,
        description="Stable solar, flat demand, no battery. Goal: minimise cost.",
    ),
    "medium": TaskConfig(
        name="medium",
        task_id="renewable_variability",
        episode_length=96,
        noise_level=0.15,
        carbon_budget=800.0,
        battery_capacity=0.3,      # Small battery
        demand_volatility=1.0,
        carbon_strict=False,
        volatility_multiplier=1.0,
        description="Noisy solar+wind, demand spikes, small battery. Goal: avoid blackouts.",
    ),
    "hard": TaskConfig(
        name="hard",
        task_id="carbon_constrained",
        episode_length=96,
        noise_level=0.15,
        carbon_budget=500.0,       # Strict carbon cap
        battery_capacity=0.2,      # Limited storage
        demand_volatility=1.5,
        carbon_strict=True,        # Episode ends on overrun
        volatility_multiplier=2.0, # 2× noise on renewables
        description="Strict carbon cap, high volatility, limited storage. Episode ends on overrun.",
    ),
}


class EcoGridEnv:
    """OpenEnv-compliant RL environment for sustainable energy grid management.

    An agent controls energy distribution across renewable sources (solar + wind),
    fossil fuels, and battery storage to meet variable demand while minimising
    cost and carbon emissions.

    Usage:
        env = EcoGridEnv()
        state = env.reset(task="easy", seed=42)
        while True:
            action = agent.decide(state)
            result = env.step(action)
            if result.done:
                break
            state = result.observation
    """

    def __init__(self) -> None:
        """Initialise environment (no state until reset is called)."""
        self._state: Optional[GridState] = None
        self._task_config: Optional[TaskConfig] = None
        self._rng: Optional[np.random.Generator] = None
        self._step_count: int = 0
        self._done: bool = True
        self._episode_log: list[StepResult] = []
        self._previous_wind: float = 0.4
        self._previous_stability: float = 0.9

    def reset(
        self,
        task: Literal["easy", "medium", "hard"] = "easy",
        seed: int = 42,
    ) -> GridState:
        """Reset environment to initial state for a new episode.

        Args:
            task: Difficulty level ("easy", "medium", or "hard").
            seed: Random seed for deterministic behaviour.

        Returns:
            Initial GridState observation.
        """
        self._task_config = TASK_CONFIGS[task]
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._done = False
        self._episode_log = []
        self._previous_wind = 0.4
        self._previous_stability = 0.9

        # Generate initial state
        config = self._task_config
        effective_noise = config.noise_level * config.volatility_multiplier

        solar = solar_output(0, effective_noise, self._rng)
        wind = wind_output(0, self._previous_wind, effective_noise, self._rng)
        self._previous_wind = wind

        demand = demand_curve(
            0, 80.0, config.demand_volatility, self._rng
        )
        price = compute_price_signal(0, demand, demand, self._rng)

        self._state = GridState(
            demand=round(demand, 2),
            solar_capacity=round(solar, 4),
            wind_capacity=round(wind, 4),
            battery_level=0.5 if config.battery_capacity > 0 else 0.0,
            grid_stability=0.9,
            carbon_budget_remaining=config.carbon_budget,
            price_signal=round(price, 2),
            time_step=0,
        )
        return self._state

    def step(self, action: GridAction) -> StepResult:
        """Execute one timestep of the environment.

        Args:
            action: Agent's energy distribution decision.

        Returns:
            StepResult containing new observation, reward, done flag, and info dict.

        Raises:
            RuntimeError: If step() is called before reset() or after episode ends.
        """
        if self._state is None or self._done:
            raise RuntimeError(
                "Cannot call step() before reset() or after episode has ended."
            )

        config = self._task_config
        assert config is not None
        assert self._rng is not None

        self._step_count += 1
        prev_state = self._state
        effective_noise = config.noise_level * config.volatility_multiplier

        # ── Compute next-step environment dynamics ──
        solar = solar_output(self._step_count, effective_noise, self._rng)
        wind = wind_output(
            self._step_count, self._previous_wind, effective_noise, self._rng
        )
        self._previous_wind = wind

        demand = demand_curve(
            self._step_count, 80.0, config.demand_volatility, self._rng
        )

        # ── Compute supply from agent's action ──
        renewable_supply, fossil_supply, battery_supply, total_supply = compute_supply(
            action.renewable_ratio,
            action.fossil_ratio,
            action.battery_action,
            prev_state.solar_capacity,
            prev_state.wind_capacity,
            prev_state.battery_level,
            config.battery_capacity,
            prev_state.demand,
        )

        # ── Update battery ──
        new_battery = update_battery(
            prev_state.battery_level,
            action.battery_action,
            config.battery_capacity,
        )

        # ── Compute blackout risk ──
        blackout = compute_blackout_risk(prev_state.demand, total_supply)

        # ── Compute carbon emissions ──
        emissions = carbon_emission(action.fossil_ratio, prev_state.demand)
        new_carbon = prev_state.carbon_budget_remaining - emissions

        # ── Compute grid stability ──
        stability = compute_grid_stability(
            prev_state.demand,
            total_supply,
            action.renewable_ratio,
            self._previous_stability,
        )
        self._previous_stability = stability

        # ── Compute price signal ──
        price = compute_price_signal(
            self._step_count, demand, total_supply, self._rng
        )

        # ── Build next state ──
        next_state = GridState(
            demand=round(demand, 2),
            solar_capacity=round(solar, 4),
            wind_capacity=round(wind, 4),
            battery_level=round(new_battery, 4),
            grid_stability=round(stability, 4),
            carbon_budget_remaining=round(new_carbon, 2),
            price_signal=round(price, 2),
            time_step=self._step_count,
        )

        # ── Compute reward ──
        reward, breakdown = compute_reward(
            prev_state, action, next_state, config.model_dump()
        )

        # ── Check termination conditions ──
        done = False
        termination_reason = ""

        if self._step_count >= config.episode_length:
            done = True
            termination_reason = "episode_complete"

        if config.carbon_strict and new_carbon < 0:
            done = True
            termination_reason = "carbon_budget_exceeded"

        # ── Build result ──
        info = {
            "reward_breakdown": breakdown,
            "renewable_supply_mwh": round(renewable_supply, 2),
            "fossil_supply_mwh": round(fossil_supply, 2),
            "battery_supply_mwh": round(battery_supply, 2),
            "total_supply_mwh": round(total_supply, 2),
            "blackout_risk": round(blackout, 4),
            "carbon_emitted_step": round(emissions, 2),
            "termination_reason": termination_reason,
        }

        result = StepResult(
            observation=next_state,
            reward=reward,
            done=done,
            info=info,
        )

        self._state = next_state
        self._done = done
        self._episode_log.append(result)

        return result

    def state(self) -> GridState:
        """Return current grid state.

        Returns:
            Current GridState observation.

        Raises:
            RuntimeError: If called before reset().
        """
        if self._state is None:
            raise RuntimeError("No state available. Call reset() first.")
        return self._state

    def get_task_config(self, task: str) -> dict:
        """Get configuration for a specific task.

        Args:
            task: Task name ("easy", "medium", or "hard").

        Returns:
            Task configuration as a dictionary.
        """
        return TASK_CONFIGS[task].model_dump()

    def get_episode_log(self) -> list[StepResult]:
        """Return the log of all step results for the current episode.

        Returns:
            List of StepResult objects from the episode.
        """
        return list(self._episode_log)

    @property
    def is_done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done

    @property
    def current_step(self) -> int:
        """Current step count in the episode."""
        return self._step_count
