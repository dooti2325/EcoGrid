import pytest
import numpy as np

from env.environment import EcoGridEnv
from models.schemas import GridAction

def test_environment_initialization():
    env = EcoGridEnv()
    assert env.is_done is True  # Before reset
    assert env._state is None

def test_environment_reset():
    env = EcoGridEnv()
    state = env.reset(task="easy", seed=42)
    
    assert state is not None
    assert state.time_step == 0
    assert 0 <= state.demand <= 200
    assert 0 <= state.solar_capacity <= 1
    assert 0 <= state.wind_capacity <= 1
    assert env.is_done is False
    assert env.current_step == 0

def test_environment_step():
    env = EcoGridEnv()
    env.reset(task="easy", seed=42)
    
    action = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
    result = env.step(action)
    
    assert result.observation.time_step == 1
    assert 0 <= result.reward <= 1
    assert result.done is False
    assert "reward_breakdown" in result.info
    assert env.current_step == 1

def test_determinism():
    env1 = EcoGridEnv()
    state1 = env1.reset(task="medium", seed=100)
    
    env2 = EcoGridEnv()
    state2 = env2.reset(task="medium", seed=100)
    
    assert state1.model_dump() == state2.model_dump()
    
    action = GridAction(renewable_ratio=0.8, fossil_ratio=0.2, battery_action=1.0)
    
    res1 = env1.step(action)
    res2 = env2.step(action)
    
    assert res1.observation.model_dump() == res2.observation.model_dump()
    assert res1.reward == res2.reward

def test_episode_termination():
    env = EcoGridEnv()
    env.reset(task="easy", seed=42) # easy is 48 steps
    
    action = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
    
    for _ in range(47):
        result = env.step(action)
        assert result.done is False
        
    result = env.step(action) # Step 48
    assert result.done is True
    assert result.info["termination_reason"] == "episode_complete"

def test_carbon_overrun_termination():
    env = EcoGridEnv()
    # hard task is carbon_strict
    state = env.reset(task="hard", seed=42) 
    
    # Force high emissions to blow the budget quickly
    action = GridAction(renewable_ratio=0.0, fossil_ratio=1.0, battery_action=0.0)
    
    done = False
    for _ in range(96):
        result = env.step(action)
        if result.done:
            done = True
            assert result.info["termination_reason"] == "carbon_budget_exceeded"
            break
            
    assert done is True
