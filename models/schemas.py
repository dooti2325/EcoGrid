"""
EcoGrid-OpenEnv — Pydantic v2 Typed Models

All data structures for state, action, step results, task scoring,
and task configuration. These models enforce type safety and validation
across the entire environment pipeline.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, Literal


class GridState(BaseModel):
    """Observable state of the energy grid at a given timestep.

    All capacity/level fields are normalised to [0, 1] ratios.
    Demand and price are in physical units (MWh, $/MWh).
    """

    demand: float = Field(
        ...,
        ge=0,
        le=200,
        description="Energy demand this timestep in MWh",
    )
    solar_capacity: float = Field(
        ...,
        ge=0,
        le=1,
        description="Available solar generation capacity (0 = night, 1 = peak sun)",
    )
    wind_capacity: float = Field(
        ...,
        ge=0,
        le=1,
        description="Available wind generation capacity (0 = calm, 1 = strong wind)",
    )
    battery_level: float = Field(
        ...,
        ge=0,
        le=1,
        description="Battery state of charge (0 = empty, 1 = full)",
    )
    grid_stability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Grid frequency stability indicator (1 = perfectly stable)",
    )
    carbon_budget_remaining: float = Field(
        ...,
        ge=-100,  # Allow slight overrun for detection
        le=1000,
        description="Remaining carbon emission budget in kgCO2",
    )
    price_signal: float = Field(
        ...,
        ge=0,
        le=300,
        description="Current spot electricity price in $/MWh",
    )
    time_step: int = Field(
        ...,
        ge=0,
        le=96,
        description="Current simulation timestep (0-indexed)",
    )


class GridAction(BaseModel):
    """Agent's energy distribution decision.

    renewable_ratio + fossil_ratio must be <= 1.0.
    The remainder represents unmet demand (blackout risk).
    battery_action: positive = charge, negative = discharge.
    """

    renewable_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of demand to meet with renewable sources",
    )
    fossil_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of demand to meet with fossil fuels",
    )
    battery_action: float = Field(
        ...,
        ge=-1,
        le=1,
        description="Battery action: -1 (full discharge) to +1 (full charge)",
    )

    @model_validator(mode="after")
    def validate_ratios_sum(self) -> "GridAction":
        """Ensure renewable + fossil ratios do not exceed 1.0."""
        total = self.renewable_ratio + self.fossil_ratio
        if total > 1.0 + 1e-6:  # Small epsilon for floating point
            raise ValueError(
                f"renewable_ratio ({self.renewable_ratio}) + "
                f"fossil_ratio ({self.fossil_ratio}) = {total} > 1.0"
            )
        return self


class StepResult(BaseModel):
    """Result returned by EcoGridEnv.step()."""

    observation: GridState = Field(
        ...,
        description="New grid state after the action",
    )
    reward: float = Field(
        ...,
        ge=0,
        le=1,
        description="Scalar reward for this step, normalised to [0, 1]",
    )
    done: bool = Field(
        ...,
        description="Whether the episode has ended",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary information (reward breakdown, warnings, etc.)",
    )


class TaskScore(BaseModel):
    """Final score produced by a task grader."""

    task_name: str = Field(
        ...,
        description="Identifier of the task that was graded",
    )
    score: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall score between 0 (failure) and 1 (perfect)",
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component score breakdown",
    )


class TaskConfig(BaseModel):
    """Configuration for a specific task difficulty level."""

    name: str = Field(..., description="Task name: easy, medium, or hard")
    task_id: str = Field(..., description="Task identifier (e.g. basic_grid_balance)")
    episode_length: int = Field(..., ge=1, description="Number of steps per episode")
    noise_level: float = Field(
        ..., ge=0, le=1, description="Noise multiplier for renewable generation"
    )
    carbon_budget: float = Field(
        ..., ge=0, description="Initial carbon budget in kgCO2"
    )
    battery_capacity: float = Field(
        ..., ge=0, le=1, description="Battery storage capacity (0 = disabled)"
    )
    demand_volatility: float = Field(
        ..., ge=0, description="Demand spike probability multiplier"
    )
    carbon_strict: bool = Field(
        default=False,
        description="If True, episode terminates on carbon budget overrun",
    )
    volatility_multiplier: float = Field(
        default=1.0,
        ge=0,
        description="Multiplier for renewable noise (2x for hard task)",
    )
    description: str = Field(default="", description="Human-readable task description")
