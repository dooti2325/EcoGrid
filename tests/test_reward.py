import pytest

from env.reward import compute_reward
from models.schemas import GridState, GridAction

def get_dummy_state(demand=100.0, solar=0.5, wind=0.5, battery=0.5, carbon=1000.0):
    return GridState(
        demand=demand,
        solar_capacity=solar,
        wind_capacity=wind,
        battery_level=battery,
        grid_stability=0.9,
        carbon_budget_remaining=carbon,
        price_signal=100.0,
        time_step=1
    )

def test_compute_reward_bounds():
    state = get_dummy_state()
    next_state = get_dummy_state()
    action = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
    task_config = {"battery_capacity": 0.5, "carbon_strict": False}
    
    reward, breakdown = compute_reward(state, action, next_state, task_config)
    
    assert 0.0 <= reward <= 1.0
    assert "cost_score" in breakdown
    assert "carbon_score" in breakdown
    assert "stability_score" in breakdown

def test_blackout_penalty():
    state = get_dummy_state(demand=100.0)
    next_state = get_dummy_state()
    # Meet 0% of demand
    action = GridAction(renewable_ratio=0.0, fossil_ratio=0.0, battery_action=0.0)
    task_config = {"battery_capacity": 0.5, "carbon_strict": False}
    
    reward, breakdown = compute_reward(state, action, next_state, task_config)
    
    assert breakdown["stability_score"] == 0.0
    assert breakdown["penalty"] >= 0.5 # Blackout penalty
    assert reward == 0.001 # Clipped to the environment's minimum positive reward

def test_carbon_penalty():
    state = get_dummy_state()
    # Next state has negative carbon budget
    next_state = get_dummy_state(carbon=-10.0)
    action = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
    task_config = {"battery_capacity": 0.5, "carbon_strict": True}
    
    reward, breakdown = compute_reward(state, action, next_state, task_config)
    
    assert breakdown["penalty"] >= 0.8
    assert reward == 0.001
