"""
EcoGrid-OpenEnv — Reward Function

Computes a dense, multi-objective scalar reward in [0, 1] plus a breakdown dict.
This design provides a continuous signal across the full trajectory, satisfying
the Mercor sub-theme by rewarding nuanced, multi-objective balancing.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple

from models.schemas import GridState, GridAction
from env.dynamics import compute_supply, compute_blackout_risk, carbon_emission


def compute_reward(
    state: GridState,
    action: GridAction,
    next_state: GridState,
    task_config: Dict[str, Any],
    actual_supply: Tuple[float, float, float, float, float] | None = None,
    actual_blackout_risk: float | None = None,
    actual_emissions: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute dense reward for a single timestep.

    Args:
        state: Grid state before the action.
        action: Agent's action.
        next_state: Grid state after the action (contains new demand/stability).
        task_config: Configuration dict for the current task.

    Returns:
        Tuple of (scalar_reward_in_0_1, breakdown_dict).
    """
    # ── 1. Base Components ──

    # Recompute supply to get exact fossil/renewable usage for this step
    if actual_supply is None:
        (
            renewable_supply,
            fossil_supply,
            battery_supply,
            total_supply,
            effective_fossil_ratio,
        ) = compute_supply(
            action.renewable_ratio,
            action.fossil_ratio,
            action.battery_action,
            state.solar_capacity,
            state.wind_capacity,
            state.battery_level,
            task_config["battery_capacity"],
            state.demand,
        )
    else:
        (
            renewable_supply,
            fossil_supply,
            battery_supply,
            total_supply,
            effective_fossil_ratio,
        ) = actual_supply

    # Cost Score (Weight: 0.30)
    # Fossil fuels are expensive. Grid purchases (for unmet demand) are very expensive.
    # We normalise cost based on a "worst case" where 100% of demand is met by expensive grid.
    if state.demand > 0:
        fossil_cost = fossil_supply * 100.0  # Assumed $100/MWh for fossil
        unmet_demand = max(0.0, state.demand - total_supply)
        grid_cost = unmet_demand * state.price_signal * 2.0  # Emergency grid buy is expensive
        total_cost = fossil_cost + grid_cost
        max_expected_cost = state.demand * 300.0 * 2.0  # Max price * 2
        cost_normalised = min(1.0, total_cost / max(1.0, max_expected_cost))
        cost_score = 1.0 - cost_normalised
    else:
        cost_score = 1.0

    # Carbon Score (Weight: 0.30)
    # Penalise carbon emissions relative to total demand.
    emissions = (
        actual_emissions
        if actual_emissions is not None
        else carbon_emission(effective_fossil_ratio, state.demand)
    )
    if state.demand > 0:
        # Normalise by worst case (100% fossil generation)
        worst_case_emissions = carbon_emission(1.0, state.demand)
        carbon_normalised = emissions / worst_case_emissions if worst_case_emissions > 0 else 0.0
        carbon_score = 1.0 - carbon_normalised
    else:
        carbon_score = 1.0

    # Stability Score (Weight: 0.25)
    # Directly inversely proportional to blackout risk
    blackout_risk = (
        actual_blackout_risk
        if actual_blackout_risk is not None
        else compute_blackout_risk(state.demand, total_supply)
    )
    stability_score = 1.0 - blackout_risk

    # Renewable Bonus (Weight: 0.15)
    # Reward high renewable usage, but ONLY if stability is maintained
    actual_renewable_ratio = renewable_supply / state.demand if state.demand > 0 else 1.0
    renewable_bonus = actual_renewable_ratio * stability_score

    # ── 2. Weighted Sum ──
    weighted_sum = (
        0.30 * cost_score +
        0.30 * carbon_score +
        0.25 * stability_score +
        0.15 * renewable_bonus
    )

    # ── 3. Penalties ──
    penalties = 0.0
    
    # Heavy penalty for significant blackouts (> 20% unmet demand)
    if blackout_risk > 0.20:
        penalties += 0.5
        
    # Carbon overrun penalty (only applies if budget drops below zero)
    if task_config.get("carbon_strict", False) and next_state.carbon_budget_remaining < 0:
        penalties += 0.8

    # ── 4. Final Calculation ──
    final_reward = float(np.clip(weighted_sum - penalties, 0.001, 0.999))

    breakdown = {
        "cost_score": float(cost_score),
        "carbon_score": float(carbon_score),
        "stability_score": float(stability_score),
        "renewable_bonus": float(renewable_bonus),
        "penalty": float(penalties),
        "final_reward": final_reward,
    }

    return final_reward, breakdown
