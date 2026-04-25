"""
EcoGrid-OpenEnv — Baseline Inference Script

Runs a full episode of the environment using either a smart heuristic agent
(for reliable baselines) or an LLM (OpenAI) to demonstrate reasoning capabilities.
"""

import argparse
import json
import os
import time
from typing import Literal

# Try importing openai, handle gracefully if not installed
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from env.environment import EcoGridEnv
from env.tasks import BasicGridBalanceGrader, RenewableVariabilityGrader, CarbonConstrainedGrader
from models.schemas import GridAction, GridState


def heuristic_agent(state: GridState, task_name: str) -> GridAction:
    """A hardcoded baseline agent that performs reasonably well."""
    # Always max out renewables available
    avg_renewable_cap = (state.solar_capacity + state.wind_capacity) / 2.0
    
    # Try to meet demand with renewables first
    if state.demand > 0:
        renewable_ratio = min(1.0, avg_renewable_cap / max(0.01, state.demand/100))
        renewable_ratio = min(renewable_ratio, 1.0)
    else:
        renewable_ratio = 1.0
        
    # Fill remaining with fossil if necessary, but keep a small buffer
    fossil_ratio = max(0.0, 1.0 - renewable_ratio)
    
    # In hard mode, conserve carbon budget if it's getting low
    if task_name == "hard" and state.carbon_budget_remaining < 200:
        fossil_ratio = min(fossil_ratio, 0.4)  # Take the blackout risk to save carbon
        
    # Total can't exceed 1.0
    total = renewable_ratio + fossil_ratio
    if total > 1.0:
        if renewable_ratio > fossil_ratio:
            fossil_ratio = 1.0 - renewable_ratio
        else:
            renewable_ratio = 1.0 - fossil_ratio
            
    # Simple battery logic
    battery_action = 0.0
    if state.demand > 100 and state.battery_level > 0.2:
        battery_action = -0.8  # Discharge during high demand
    elif state.demand < 60 and state.battery_level < 0.8:
        battery_action = 0.8   # Charge during low demand
        
    return GridAction(
        renewable_ratio=round(renewable_ratio, 3),
        fossil_ratio=round(fossil_ratio, 3),
        battery_action=round(battery_action, 3)
    )


def llm_agent(state: GridState, task_name: str, client: "OpenAI") -> GridAction:
    """An agent that uses an LLM to make decisions via Chain-of-Thought."""
    
    prompt = f"""
You are an expert energy grid operator managing a power grid.
Your goal is to balance renewable energy, fossil fuels, and battery storage to meet demand while minimising cost and carbon emissions.

CURRENT STATE:
{state.model_dump_json(indent=2)}

TASK: {task_name}
CONSTRAINTS: 
- renewable_ratio + fossil_ratio <= 1.0
- battery_action must be between -1.0 (discharge) and 1.0 (charge)
- Grid stability target: >= 0.7
- Carbon budget remaining: {state.carbon_budget_remaining} kg CO2

Reason step-by-step internally about the best strategy, considering the current demand, available renewable capacity, and carbon budget.
Then, output ONLY a valid JSON object matching this schema, with no markdown fences:
{{
  "renewable_ratio": float,
  "fossil_ratio": float,
  "battery_action": float
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using best model as requested
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up markdown if model ignored instructions
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        data = json.loads(content)
        return GridAction(**data)
        
    except Exception as e:
        print(f"LLM Error: {e}. Falling back to heuristic.")
        return heuristic_agent(state, task_name)


def main():
    parser = argparse.ArgumentParser(description="EcoGrid-OpenEnv Baseline Inference")
    parser.add_argument("--task", type=str, choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent", type=str, choices=["heuristic", "llm"], default="heuristic")
    args = parser.parse_args()
    
    if args.agent == "llm" and not HAS_OPENAI:
        print("Error: openai package not installed. Run: pip install openai")
        return
        
    if args.agent == "llm" and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set. Falling back to heuristic.")
        args.agent = "heuristic"
        
    client = OpenAI() if args.agent == "llm" else None
    
    # Initialize environment
    print(f"Initializing EcoGridEnv for task: {args.task} (seed={args.seed})")
    env = EcoGridEnv()
    state = env.reset(task=args.task, seed=args.seed)
    
    start_time = time.time()
    total_reward = 0.0
    
    print("\nStarting episode...")
    print(f"{'Step':<5} | {'Demand':<8} | {'Renw Ratio':<10} | {'Foss Ratio':<10} | {'Blackout':<8} | {'Reward':<6}")
    print("-" * 65)
    
    while not env.is_done:
        if args.agent == "llm":
            action = llm_agent(state, args.task, client)
        else:
            action = heuristic_agent(state, args.task)
            
        try:
            result = env.step(action)
        except ValueError as e:
            # Handle constraint violations (e.g. ratios > 1.0)
            print(f"Action constraint violation: {e}. Falling back to safe action.")
            safe_action = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
            result = env.step(safe_action)
            
        state = result.observation
        total_reward += result.reward
        
        # Print progress every 10 steps or at the end
        if env.current_step % 10 == 0 or env.is_done:
            print(f"{env.current_step:<5} | "
                  f"{state.demand:<8.1f} | "
                  f"{action.renewable_ratio:<10.2f} | "
                  f"{action.fossil_ratio:<10.2f} | "
                  f"{result.info.get('blackout_risk', 0.0):<8.2f} | "
                  f"{result.reward:<6.2f}")

    elapsed = time.time() - start_time
    
    # Grade the episode
    log = env.get_episode_log()
    if args.task == "easy":
        score = BasicGridBalanceGrader.grade(log)
    elif args.task == "medium":
        score = RenewableVariabilityGrader.grade(log)
    else:
        score = CarbonConstrainedGrader.grade(log)
        
    print("\n" + "="*50)
    print("EPISODE COMPLETE")
    print("="*50)
    print(f"Task: {args.task}")
    print(f"Agent: {args.agent}")
    print(f"Steps: {env.current_step}")
    print(f"Time: {elapsed:.2f}s ({env.current_step/elapsed:.0f} steps/sec)")
    if log:
        print(f"Termination: {log[-1].info.get('termination_reason', 'unknown')}")
        
    print("\nFINAL SCORE:")
    print(f"{score.score * 100:.1f} / 100.0")
    print("\nScore Breakdown:")
    for k, v in score.breakdown.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
