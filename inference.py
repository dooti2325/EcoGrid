"""Inference Script - EcoGrid OpenEnv
================================================
MANDATORY environment variables (injected by the validator):
    API_BASE_URL        The LiteLLM proxy endpoint.
    HF_TOKEN            Your API key for the proxy.
    MODEL_NAME          The model identifier to use for inference.

STDOUT FORMAT (exact - do not deviate):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

from env.environment import EcoGridEnv
from env.tasks import BasicGridBalanceGrader, RenewableVariabilityGrader, CarbonConstrainedGrader
from models.schemas import GridAction, GridState

# -------------------------------------------------------------------
# MANDATORY: read from injected environment variables — no hardcoding.
# The validator checks that all LLM calls flow through API_BASE_URL.
# -------------------------------------------------------------------
API_BASE_URL: str = os.environ["API_BASE_URL"]
API_KEY: str = os.environ["API_KEY"]
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")
BENCHMARK: str = os.environ.get("BENCHMARK", "eco-grid-openenv")

SUCCESS_SCORE_THRESHOLD = 0.5
TASKS = ["easy", "medium", "hard"]

# Single shared client — always routed through the injected proxy URL.
_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    timeout=30.0,
    max_retries=1,
)


# ---------------------------------------------------------------------------
# Logging helpers — exact format required by the validator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Fallback policy
# ---------------------------------------------------------------------------

def _fallback_action(task_name: str, state: GridState) -> GridAction:
    """A safe fallback agent that performs reasonably well."""
    avg_renewable_cap = (state.solar_capacity + state.wind_capacity) / 2.0
    
    if state.demand > 0:
        renewable_ratio = min(1.0, avg_renewable_cap / max(0.01, state.demand/100))
        renewable_ratio = min(renewable_ratio, 1.0)
    else:
        renewable_ratio = 1.0
        
    fossil_ratio = max(0.0, 1.0 - renewable_ratio)
    
    if task_name == "hard" and state.carbon_budget_remaining < 200:
        fossil_ratio = min(fossil_ratio, 0.4)
        
    total = renewable_ratio + fossil_ratio
    if total > 1.0:
        if renewable_ratio > fossil_ratio:
            fossil_ratio = 1.0 - renewable_ratio
        else:
            renewable_ratio = 1.0 - fossil_ratio
            
    battery_action = 0.0
    if state.demand > 100 and state.battery_level > 0.2:
        battery_action = -0.8
    elif state.demand < 60 and state.battery_level < 0.8:
        battery_action = 0.8
        
    return GridAction(
        renewable_ratio=round(renewable_ratio, 3),
        fossil_ratio=round(fossil_ratio, 3),
        battery_action=round(battery_action, 3)
    )


# ---------------------------------------------------------------------------
# LLM call — ALWAYS goes through the injected proxy (API_BASE_URL / _client)
# ---------------------------------------------------------------------------

def get_action_from_llm(state: GridState, task_name: str) -> GridAction:
    """Call the LLM via the injected proxy to choose a grid action."""
    preferred = _fallback_action(task_name, state)
    
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
    
    # This call MUST reach the proxy
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
        stream=False,
    )
    
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    data = json.loads(content)
    return GridAction(**data)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> None:
    if not os.environ.get("API_BASE_URL"):
        print("Warning: API_BASE_URL not set. Defaulting to localhost:8000.", file=sys.stderr, flush=True)

    for task_name in TASKS:
        env = EcoGridEnv()
        env.reset(seed=42, task=task_name)

        rewards: List[float] = []
        steps_taken = 0
        success = False
        score = 0.001
        done = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            step = 1
            while not done:
                state = env.state()

                error: Optional[str] = None
                try:
                    action = get_action_from_llm(state, task_name)
                    # Create string representation for logging
                    action_str = json.dumps({
                        "ren": action.renewable_ratio,
                        "fos": action.fossil_ratio,
                        "bat": action.battery_action
                    })
                except Exception as exc:
                    action = _fallback_action(task_name, state)
                    action_str = json.dumps({
                        "ren": action.renewable_ratio,
                        "fos": action.fossil_ratio,
                        "bat": action.battery_action
                    })
                    error = f"llm_error:{type(exc).__name__}"

                try:
                    result = env.step(action)
                    reward = result.reward
                    done = result.done
                except Exception as exc:
                    reward = 0.0
                    done = True
                    error = str(exc)

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                step += 1

            # Grade the episode
            log = env.get_episode_log()
            if task_name == "easy":
                grader_result = BasicGridBalanceGrader.grade(log)
            elif task_name == "medium":
                grader_result = RenewableVariabilityGrader.grade(log)
            else:
                grader_result = CarbonConstrainedGrader.grade(log)
                
            score = float(grader_result.score)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            print(f"Fatal error in task {task_name}: {exc}", file=sys.stderr, flush=True)
            success = False
            score = 0.001

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    run_inference()
