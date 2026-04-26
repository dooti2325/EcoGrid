import pytest

from env.tasks import BasicGridBalanceGrader, RenewableVariabilityGrader, CarbonConstrainedGrader
from models.schemas import StepResult, GridState

def get_dummy_log(num_steps=10, reward=0.8, blackout=0.0, carbon_term=False):
    log = []
    for i in range(num_steps):
        state = GridState(
            demand=100.0, solar_capacity=0.5, wind_capacity=0.5, 
            battery_level=0.5, grid_stability=0.9, 
            carbon_budget_remaining=100.0 if not carbon_term else -10.0,
            price_signal=50.0, time_step=i
        )
        info = {
            "reward_breakdown": {
                "cost_score": 0.8,
                "carbon_score": 0.8,
                "stability_score": 1.0 - blackout,
                "renewable_bonus": 0.5,
                "penalty": 0.0
            },
            "renewable_supply_mwh": 50.0,
            "blackout_risk": blackout,
            "termination_reason": "carbon_budget_exceeded" if carbon_term else ""
        }
        res = StepResult(observation=state, reward=reward, done=False, info=info)
        log.append(res)
    return log

def test_basic_grader():
    log = get_dummy_log()
    score = BasicGridBalanceGrader.grade(log)
    assert 0.0 <= score.score <= 1.0
    assert score.task_name == "easy"

def test_renewable_grader_blackout_penalty():
    # 3 blackouts should halve the score
    log = get_dummy_log(num_steps=5, blackout=0.5) 
    score = RenewableVariabilityGrader.grade(log)
    assert score.score < 0.5 # Heavily penalized

def test_carbon_grader_fatal():
    log = get_dummy_log(carbon_term=True)
    score = CarbonConstrainedGrader.grade(log)
    assert score.score == 0.001
    assert score.breakdown["fatal_error"] == 1.0
