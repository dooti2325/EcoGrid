"""
EcoGrid-OpenEnv — Task Graders

Deterministic graders that evaluate full episode logs to produce a final TaskScore.
Each grader corresponds to a specific difficulty level.
"""

from __future__ import annotations

import numpy as np

from models.schemas import StepResult, TaskScore


class BasicGridBalanceGrader:
    """Grader for the 'easy' task (basic_grid_balance).
    
    Checks: cost efficiency, no major blackouts, renewable usage.
    Score = total_reward / max_possible_reward
    """
    
    @staticmethod
    def grade(episode_log: list[StepResult]) -> TaskScore:
        if not episode_log:
            return TaskScore(task_name="easy", score=0.0, breakdown={})
            
        total_steps = len(episode_log)
        total_reward = sum(step.reward for step in episode_log)
        
        # Calculate specific metrics
        avg_renewable = np.mean([
            step.info.get("renewable_supply_mwh", 0) / max(step.observation.demand, 1.0)
            for step in episode_log
        ])
        
        max_blackout = max(step.info.get("blackout_risk", 0.0) for step in episode_log)
        avg_cost_score = np.mean([step.info["reward_breakdown"]["cost_score"] for step in episode_log])
        
        # Base score is just the average reward (since reward is in [0,1])
        base_score = total_reward / total_steps
        
        # Apply hard constraints for the task definition
        # If there was a major blackout, cap the score
        if max_blackout > 0.2:
            base_score = min(base_score, 0.4)
            
        # If cost reduction targets weren't met (avg cost score < 0.7 means high costs)
        if avg_cost_score < 0.7:
            base_score = min(base_score, 0.6)
            
        final_score = float(np.clip(base_score, 0.0, 1.0))
        
        breakdown = {
            "avg_reward": float(total_reward / total_steps),
            "avg_renewable_ratio": float(avg_renewable),
            "max_blackout_risk": float(max_blackout),
            "avg_cost_score": float(avg_cost_score),
        }
        
        return TaskScore(
            task_name="easy",
            score=final_score,
            breakdown=breakdown,
        )


class RenewableVariabilityGrader:
    """Grader for the 'medium' task (renewable_variability).
    
    Score = 0.4*(renewable_score) + 0.4*(stability_score) + 0.2*(cost_score)
    Checks: blackout frequency, renewable utilisation.
    """
    
    @staticmethod
    def grade(episode_log: list[StepResult]) -> TaskScore:
        if not episode_log:
            return TaskScore(task_name="medium", score=0.0, breakdown={})
            
        # Extract averages from the reward breakdowns
        avg_renewable = np.mean([step.info["reward_breakdown"]["renewable_bonus"] for step in episode_log])
        avg_stability = np.mean([step.info["reward_breakdown"]["stability_score"] for step in episode_log])
        avg_cost = np.mean([step.info["reward_breakdown"]["cost_score"] for step in episode_log])
        
        # Count major blackout events
        blackout_events = sum(1 for step in episode_log if step.info.get("blackout_risk", 0.0) > 0.1)
        
        # Composite score
        base_score = 0.4 * avg_renewable + 0.4 * avg_stability + 0.2 * avg_cost
        
        # Penalise frequent blackouts
        if blackout_events >= 3:
            base_score *= 0.5  # Heavy penalty for failing core objective
            
        final_score = float(np.clip(base_score, 0.0, 1.0))
        
        breakdown = {
            "renewable_component": float(avg_renewable),
            "stability_component": float(avg_stability),
            "cost_component": float(avg_cost),
            "blackout_events": float(blackout_events),
        }
        
        return TaskScore(
            task_name="medium",
            score=final_score,
            breakdown=breakdown,
        )


class CarbonConstrainedGrader:
    """Grader for the 'hard' task (carbon_constrained).
    
    Score = 0 if carbon_budget_remaining < 0 at any step.
    Otherwise: 0.5*(carbon_score) + 0.3*(cost_score) + 0.2*(stability_score)
    """
    
    @staticmethod
    def grade(episode_log: list[StepResult]) -> TaskScore:
        if not episode_log:
            return TaskScore(task_name="hard", score=0.0, breakdown={})
            
        # Check fatal condition first
        min_carbon_budget = min(step.observation.carbon_budget_remaining for step in episode_log)
        
        # Check termination reason
        carbon_failure = any(
            step.info.get("termination_reason") == "carbon_budget_exceeded" 
            for step in episode_log
        )
        
        if min_carbon_budget < 0 or carbon_failure:
            return TaskScore(
                task_name="hard",
                score=0.0,
                breakdown={
                    "fatal_error": 1.0,
                    "min_carbon_budget": float(min_carbon_budget)
                }
            )
            
        # If survived, calculate composite score
        avg_carbon = np.mean([step.info["reward_breakdown"]["carbon_score"] for step in episode_log])
        avg_cost = np.mean([step.info["reward_breakdown"]["cost_score"] for step in episode_log])
        avg_stability = np.mean([step.info["reward_breakdown"]["stability_score"] for step in episode_log])
        
        base_score = 0.5 * avg_carbon + 0.3 * avg_cost + 0.2 * avg_stability
        
        # Ensure they actually maintained stability (didn't just turn off power to save carbon)
        min_stability = min(step.observation.grid_stability for step in episode_log)
        if min_stability < 0.7:
            base_score = min(base_score, 0.4)
            
        final_score = float(np.clip(base_score, 0.0, 1.0))
        
        breakdown = {
            "carbon_component": float(avg_carbon),
            "cost_component": float(avg_cost),
            "stability_component": float(avg_stability),
            "min_stability": float(min_stability),
        }
        
        return TaskScore(
            task_name="hard",
            score=final_score,
            breakdown=breakdown,
        )
