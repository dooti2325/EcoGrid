"""
EcoGrid-OpenEnv — Deterministic Physics Helpers

All functions are pure, stateless, and deterministic given a seeded RNG.
They use lightweight numpy math for speed (>10,000 episodes/second).
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def solar_output(time_step: int, noise_level: float, rng: Generator) -> float:
    """Compute solar generation capacity for a given timestep.

    Uses a sinusoidal day/night cycle (period = 24 steps) with Gaussian noise.
    Peak solar at step 12 (noon), zero at step 0 and 24 (midnight).

    Args:
        time_step: Current simulation step (maps to hour of day via modulo 24).
        noise_level: Standard deviation of Gaussian noise (0 = deterministic).
        rng: Seeded numpy random generator for reproducibility.

    Returns:
        Solar capacity as a float in [0, 1].
    """
    hour = time_step % 24
    # Sinusoidal curve: peaks at hour 12, trough at 0/24
    base = max(0.0, np.sin(np.pi * hour / 24.0))
    noise = rng.normal(0, noise_level) if noise_level > 0 else 0.0
    return float(np.clip(base + noise, 0.0, 1.0))


def wind_output(
    time_step: int,
    previous_wind: float,
    noise_level: float,
    rng: Generator,
) -> float:
    """Compute wind generation capacity using a smooth random walk.

    Wind is modelled as a mean-reverting random walk around 0.4,
    providing realistic temporal correlation.

    Args:
        time_step: Current simulation step (used for base variation).
        previous_wind: Wind capacity from the previous step.
        noise_level: Scale of random walk step.
        rng: Seeded numpy random generator.

    Returns:
        Wind capacity as a float in [0, 1].
    """
    # Mean-reverting around 0.4 with slow sinusoidal drift
    mean = 0.4 + 0.1 * np.sin(2 * np.pi * time_step / 48.0)
    # Random walk step with mean reversion
    reversion_strength = 0.1
    step_noise = rng.normal(0, noise_level * 0.15) if noise_level > 0 else 0.0
    new_wind = (
        previous_wind
        + reversion_strength * (mean - previous_wind)
        + step_noise
    )
    return float(np.clip(new_wind, 0.0, 1.0))


def demand_curve(
    time_step: int,
    base_demand: float,
    volatility: float,
    rng: Generator,
) -> float:
    """Compute energy demand for a given timestep.

    Realistic demand profile with morning and evening peaks,
    plus random spikes based on volatility.

    Args:
        time_step: Current simulation step.
        base_demand: Base demand level in MWh (typically 80).
        volatility: Probability multiplier for demand spikes.
        rng: Seeded numpy random generator.

    Returns:
        Demand in MWh, clamped to [0, 200].
    """
    hour = time_step % 24

    # Morning peak (around hour 8)
    morning = 30.0 * np.exp(-((hour - 8) ** 2) / 8.0)
    # Evening peak (around hour 18)
    evening = 40.0 * np.exp(-((hour - 18) ** 2) / 8.0)

    # Weekly operational cycle (business-day effect)
    day_of_week = (time_step // 24) % 7
    weekday_multiplier = 1.0 if day_of_week < 5 else 0.92

    # Correlated weather/event noise on demand.
    # Deterministic under the seeded RNG.
    stochastic_component = rng.normal(0, 4.0 * max(0.2, volatility))

    # Random spike (occurs with probability proportional to volatility)
    spike = 0.0
    if volatility > 0 and rng.random() < 0.05 * volatility:
        spike = rng.uniform(10, 40) * volatility

    demand = (base_demand + morning + evening + stochastic_component + spike) * weekday_multiplier
    return float(np.clip(demand, 0.0, 200.0))


def update_battery(
    level: float,
    action: float,
    capacity: float,
    charge_rate: float = 0.15,
    charge_efficiency: float = 0.94,
    discharge_efficiency: float = 0.94,
) -> float:
    """Update battery state of charge.

    Args:
        level: Current battery level [0, 1].
        action: Charge/discharge action [-1, +1].
        capacity: Battery capacity (0 = disabled, 1 = full capacity).
        charge_rate: Max charge/discharge per step.

    Returns:
        New battery level, clamped to [0, 1].
    """
    if capacity <= 0:
        return 0.0
    if action >= 0:
        delta = action * charge_rate * capacity * charge_efficiency
    else:
        # Discharging removes more SoC than delivered energy because of losses.
        eff = max(discharge_efficiency, 1e-6)
        delta = action * charge_rate * capacity / eff
    return float(np.clip(level + delta, 0.0, 1.0))


def compute_blackout_risk(demand: float, supply: float) -> float:
    """Compute blackout risk based on supply-demand gap.

    Args:
        demand: Current demand in MWh.
        supply: Total supply from all sources in MWh.

    Returns:
        Blackout risk as float in [0, 1]. 0 = no risk, 1 = total blackout.
    """
    if demand <= 0:
        return 0.0
    gap = max(0.0, demand - supply)
    risk = gap / demand
    return float(np.clip(risk, 0.0, 1.0))


def carbon_emission(
    fossil_ratio: float,
    demand: float,
    emission_factor: float = 0.5,
) -> float:
    """Compute carbon emissions from fossil fuel usage.

    Args:
        fossil_ratio: Fraction of demand met by fossil fuels [0, 1].
        demand: Current demand in MWh.
        emission_factor: kg CO₂ per MWh of fossil generation.

    Returns:
        Carbon emissions in kgCO₂.
    """
    return float(fossil_ratio * demand * emission_factor)


def compute_supply(
    action_renewable: float,
    action_fossil: float,
    battery_action: float,
    solar_cap: float,
    wind_cap: float,
    battery_level: float,
    battery_capacity: float,
    demand: float,
    previous_fossil_ratio: float | None = None,
    fossil_ramp_limit: float | None = None,
    discharge_efficiency: float = 0.94,
) -> tuple[float, float, float, float, float]:
    """Compute total energy supply from all sources.

    Args:
        action_renewable: Fraction allocated to renewables.
        action_fossil: Fraction allocated to fossil.
        battery_action: Battery charge/discharge action.
        solar_cap: Current solar capacity [0, 1].
        wind_cap: Current wind capacity [0, 1].
        battery_level: Current battery level [0, 1].
        battery_capacity: Battery storage capacity.
        demand: Current demand in MWh.

    Returns:
        Tuple of (renewable_supply, fossil_supply, battery_supply, total_supply) in MWh.
    """
    # Renewable supply is limited by actual capacity
    avg_renewable_cap = (solar_cap + wind_cap) / 2.0
    renewable_supply = action_renewable * demand * min(1.0, avg_renewable_cap / max(action_renewable, 0.01))

    # Fossil ramp-rate constraints emulate thermal plant limitations.
    effective_fossil_ratio = action_fossil
    if previous_fossil_ratio is not None and fossil_ramp_limit is not None:
        lower = max(0.0, previous_fossil_ratio - fossil_ramp_limit)
        upper = min(1.0, previous_fossil_ratio + fossil_ramp_limit)
        effective_fossil_ratio = float(np.clip(action_fossil, lower, upper))

    # Fossil supply (dispatchable but ramp-limited when configured)
    fossil_supply = effective_fossil_ratio * demand

    # Battery can supplement supply when discharging
    battery_supply = 0.0
    if battery_action < 0 and battery_capacity > 0:
        # Discharging: supply is proportional to discharge rate and level
        battery_supply = (
            abs(battery_action)
            * battery_level
            * battery_capacity
            * demand
            * 0.2
            * discharge_efficiency
        )

    total = renewable_supply + fossil_supply + battery_supply
    return (
        float(renewable_supply),
        float(fossil_supply),
        float(battery_supply),
        float(total),
        float(effective_fossil_ratio),
    )


def compute_price_signal(
    time_step: int,
    demand: float,
    supply: float,
    rng: Generator,
) -> float:
    """Compute spot electricity price based on supply-demand dynamics.

    Args:
        time_step: Current timestep.
        demand: Demand in MWh.
        supply: Total supply in MWh.
        rng: Seeded random generator.

    Returns:
        Price signal in $/MWh, clamped to [0, 300].
    """
    # Base price follows demand pattern
    base_price = 50.0 + (demand / 200.0) * 100.0

    # Scarcity premium when supply < demand
    if supply < demand and demand > 0:
        scarcity = (demand - supply) / demand
        base_price += scarcity * 150.0

    # Small random noise
    noise = rng.normal(0, 5.0)
    return float(np.clip(base_price + noise, 0.0, 300.0))


def compute_grid_stability(
    demand: float,
    supply: float,
    renewable_ratio: float,
    previous_stability: float,
) -> float:
    """Compute grid stability metric.

    Stability is affected by:
    - Supply-demand balance (main factor)
    - High renewable penetration (slight instability due to variability)
    - Momentum from previous stability (grid has inertia)

    Args:
        demand: Current demand.
        supply: Total supply.
        renewable_ratio: Fraction of supply from renewables.
        previous_stability: Stability from previous step.

    Returns:
        Grid stability in [0, 1].
    """
    # Balance factor: how well supply matches demand
    if demand > 0:
        balance = 1.0 - abs(demand - supply) / demand
    else:
        balance = 1.0
    balance = max(0.0, balance)

    # Renewable variability penalty (mild)
    variability_penalty = renewable_ratio * 0.05

    # Current stability
    current = balance - variability_penalty

    # Momentum: 70% current, 30% previous
    stability = 0.7 * current + 0.3 * previous_stability
    return float(np.clip(stability, 0.0, 1.0))
