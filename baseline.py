"""
EcoGrid-OpenEnv — Baseline Inference Script

Runs a full episode of the environment using either a smart heuristic agent
(for reliable baselines) or an LLM (OpenAI) to demonstrate reasoning capabilities.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

HAS_LITELLM = None

from env.environment import EcoGridEnv
from env.tasks import BasicGridBalanceGrader, RenewableVariabilityGrader, CarbonConstrainedGrader
from env.action_utils import safe_grid_action
from models.schemas import GridAction, GridState

_trained_model = None
_trained_tokenizer = None
TASK_EPISODE_LENGTH = {"easy": 48, "medium": 96, "hard": 96}
FOSSIL_EMISSION_FACTOR = 0.5
LORA_DIR = Path(os.environ.get("LORA_ADAPTER_DIR", str(Path(__file__).resolve().parent / "lora_adapter"))).resolve()


def _get_litellm():
    """Lazily import litellm to avoid startup-time network side effects."""
    global HAS_LITELLM
    if HAS_LITELLM is False:
        return None
    try:
        import litellm

        HAS_LITELLM = True
        return litellm
    except ImportError:
        HAS_LITELLM = False
        return None

def load_trained_model():
    """Lazily load the LoRA model if available."""
    global _trained_model, _trained_tokenizer
    if _trained_model is not None:
        return _trained_model, _trained_tokenizer
        
    adapter_config_path = LORA_DIR / "adapter_config.json"
    if not adapter_config_path.exists():
        return None, None
        
    print(f"Loading LoRA adapter from {LORA_DIR} ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch
        
        with adapter_config_path.open("r", encoding="utf-8") as f:
            peft_config = json.load(f)
        base_model_name = peft_config.get("base_model_name_or_path", "unsloth/Qwen2.5-1.5B-Instruct")
        
        _trained_tokenizer = AutoTokenizer.from_pretrained(str(LORA_DIR))
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quant_config,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="cpu",
                torch_dtype=torch.float32
            )
            
        _trained_model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
        print("LoRA successfully loaded!")
        return _trained_model, _trained_tokenizer
    except Exception as e:
        print(f"Failed to load LoRA: {e}")
        return None, None

def local_llm_agent(state: GridState, task_name: str) -> GridAction:
    """Agent that runs inference using the locally trained LoRA."""
    model, tokenizer = load_trained_model()
    if model is None:
        print("No LoRA found, falling back to heuristic.")
        return heuristic_agent(state, task_name)
        
    state_json = state.model_dump_json(indent=2)
    prompt = f"You are an expert energy grid operator.\nYour goal is to balance renewable energy, fossil fuels, and battery storage to meet demand while minimising cost and carbon emissions.\n\nCURRENT STATE:\n{state_json}\n\nTASK: {task_name}\nCONSTRAINTS: \n- renewable_ratio + fossil_ratio <= 1.0\n- battery_action must be between -1.0 (discharge) and 1.0 (charge)\n\nOutput ONLY a valid JSON object:\n{{\n  \"renewable_ratio\": float,\n  \"fossil_ratio\": float,\n  \"battery_action\": float\n}}"
    
    try:
        # Standard chat template formatting
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        outputs = model.generate(inputs, max_new_tokens=100, temperature=0.1, do_sample=True)
        content = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
        
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]
            
        data = json.loads(content)
        return safe_grid_action(
            renewable_ratio=data.get("renewable_ratio", 0.5),
            fossil_ratio=data.get("fossil_ratio", 0.5),
            battery_action=data.get("battery_action", 0.0),
        )
    except Exception as e:
        print(f"Local LLM Error: {e}. Falling back to heuristic.")
        return heuristic_agent(state, task_name)



def _constraint_aware_hard_controller(state: GridState) -> GridAction:
    """Hard-mode controller that enforces carbon budget pacing."""
    remaining_steps = max(1, TASK_EPISODE_LENGTH["hard"] - state.time_step)
    avg_renewable_cap = (state.solar_capacity + state.wind_capacity) / 2.0
    avg_renewable_cap = min(1.0, max(0.0, avg_renewable_cap))

    # Budget-aware fossil cap:
    # carbon_per_step = fossil_ratio * demand * emission_factor
    # => fossil_ratio <= carbon_budget_remaining / (remaining_steps * demand * emission_factor)
    if state.demand > 0:
        budget_fossil_cap = state.carbon_budget_remaining / (
            remaining_steps * state.demand * FOSSIL_EMISSION_FACTOR
        )
    else:
        budget_fossil_cap = 0.0

    # Keep a safety margin to avoid late-episode budget collapse.
    budget_fossil_cap = max(0.0, min(0.14, budget_fossil_cap * 0.92))
    future_floor = remaining_steps * max(state.demand, 1.0) * FOSSIL_EMISSION_FACTOR * 0.08
    if state.grid_stability < 0.75 and state.carbon_budget_remaining > future_floor:
        budget_fossil_cap = min(0.18, budget_fossil_cap + 0.03)

    renewable_ratio = min(0.9, max(0.62, avg_renewable_cap + 0.12))
    fossil_ratio = min(max(0.02, 1.0 - renewable_ratio), budget_fossil_cap)

    # Battery dispatch policy:
    # - discharge on high demand or low stability
    # - charge when demand is light and stability is healthy
    if (state.demand > 100 or state.grid_stability < 0.8) and state.battery_level > 0.12:
        battery_action = -0.9
    elif state.demand < 78 and state.battery_level < 0.7 and avg_renewable_cap > 0.4:
        battery_action = 0.6
    else:
        battery_action = 0.0

    return safe_grid_action(
        renewable_ratio=renewable_ratio,
        fossil_ratio=fossil_ratio,
        battery_action=battery_action,
    )


def heuristic_agent(state: GridState, task_name: str) -> GridAction:
    """Constraint-aware baseline agent with strict action validity guarantees."""
    if task_name == "hard":
        return _constraint_aware_hard_controller(state)

    avg_renewable_cap = (state.solar_capacity + state.wind_capacity) / 2.0
    avg_renewable_cap = min(1.0, max(0.0, avg_renewable_cap))
    renewable_ratio = min(0.95, max(0.05, avg_renewable_cap))
    fossil_ratio = max(0.0, 1.0 - avg_renewable_cap)

    if task_name == "medium" and (state.grid_stability < 0.8 or state.demand > 105):
        fossil_ratio = min(1.0, fossil_ratio + 0.05)

    if state.demand > 100 and state.battery_level > 0.2:
        battery_action = -0.9
        if task_name == "medium":
            fossil_ratio = max(0.0, fossil_ratio - 0.05)
    elif state.demand < 70 and state.battery_level < 0.8 and avg_renewable_cap > 0.5:
        battery_action = 0.7
        if task_name == "medium":
            fossil_ratio = min(1.0, fossil_ratio + 0.03)
    else:
        battery_action = 0.0

    return safe_grid_action(
        renewable_ratio=renewable_ratio,
        fossil_ratio=fossil_ratio,
        battery_action=battery_action,
    )


def llm_agent(state: GridState, task_name: str) -> GridAction:
    """An agent that uses an LLM to make decisions via Chain-of-Thought."""
    litellm = _get_litellm()
    if litellm is None:
        return heuristic_agent(state, task_name)
    
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
        response = litellm.completion(
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
        return safe_grid_action(
            renewable_ratio=data.get("renewable_ratio", 0.5),
            fossil_ratio=data.get("fossil_ratio", 0.5),
            battery_action=data.get("battery_action", 0.0),
        )
        
    except Exception as e:
        print(f"LLM Error: {e}. Falling back to heuristic.")
        return heuristic_agent(state, task_name)


def main():
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

    console = Console()
    
    parser = argparse.ArgumentParser(description="EcoGrid-OpenEnv Baseline Inference")
    parser.add_argument("--task", type=str, choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent", type=str, choices=["heuristic", "llm"], default="heuristic")
    args = parser.parse_args()
    
    if args.agent == "llm" and _get_litellm() is None:
        console.print("[bold red]Error:[/bold red] litellm package not installed. Run: pip install litellm")
        return
        
    if args.agent == "llm" and not os.environ.get("OPENAI_API_KEY"):
        console.print("[bold yellow]Warning:[/bold yellow] OPENAI_API_KEY environment variable not set. Falling back to heuristic.")
        args.agent = "heuristic"
        
    # Initialize environment
    console.print(f"[bold blue]Initializing EcoGridEnv for task:[/bold blue] {args.task} (seed={args.seed})")
    env = EcoGridEnv()
    state = env.reset(task=args.task, seed=args.seed)
    
    start_time = time.time()
    total_reward = 0.0
    
    console.print("\n[bold green]Starting episode...[/bold green]")
    
    table = Table(title="Live Grid Simulation", show_header=True, header_style="bold magenta")
    table.add_column("Step", style="dim", width=6)
    table.add_column("Demand", justify="right")
    table.add_column("Renw Ratio", justify="right")
    table.add_column("Foss Ratio", justify="right")
    table.add_column("Blackout", justify="right")
    table.add_column("Reward", justify="right", style="green")
    
    episode_length = env.get_task_config(args.task)["episode_length"]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        sim_task = progress.add_task("[cyan]Simulating grid...", total=episode_length)
        
        while not env.is_done:
            if args.agent == "llm":
                action = llm_agent(state, args.task)
            else:
                action = heuristic_agent(state, args.task)
                
            try:
                result = env.step(action)
            except ValueError as e:
                console.print(f"[bold yellow]Action constraint violation:[/bold yellow] {e}. Falling back to safe action.")
                safe_action = GridAction(renewable_ratio=0.5, fossil_ratio=0.5, battery_action=0.0)
                result = env.step(safe_action)
                
            state = result.observation
            total_reward += result.reward
            
            # Print progress every 10 steps or at the end
            if env.current_step % 10 == 0 or env.is_done:
                table.add_row(
                    str(env.current_step),
                    f"{state.demand:.1f}",
                    f"{action.renewable_ratio:.2f}",
                    f"{action.fossil_ratio:.2f}",
                    f"{result.info.get('blackout_risk', 0.0):.2f}",
                    f"{result.reward:.2f}"
                )
                
            progress.update(sim_task, advance=1)
            time.sleep(0.01) # slight delay to render progress smoothly for small baselines

    console.print(table)
    elapsed = time.time() - start_time
    
    # Grade the episode
    log = env.get_episode_log()
    if args.task == "easy":
        score = BasicGridBalanceGrader.grade(log)
    elif args.task == "medium":
        score = RenewableVariabilityGrader.grade(log)
    else:
        score = CarbonConstrainedGrader.grade(log)
        
    console.print("\n[bold]==================================================[/bold]")
    console.print("[bold cyan]EPISODE COMPLETE[/bold cyan]")
    console.print("[bold]==================================================[/bold]")
    console.print(f"Task: [bold]{args.task}[/bold]")
    console.print(f"Agent: [bold]{args.agent}[/bold]")
    console.print(f"Steps: {env.current_step}")
    console.print(f"Time: {elapsed:.2f}s ({env.current_step/elapsed:.0f} steps/sec)")
    if log:
        console.print(f"Termination: [bold]{log[-1].info.get('termination_reason', 'unknown')}[/bold]")
        
    console.print("\n[bold green]FINAL SCORE:[/bold green]")
    console.print(f"[bold text]{score.score * 100:.1f} / 100.0[/bold text]")
    console.print("\n[bold]Score Breakdown:[/bold]")
    
    breakdown_table = Table(show_header=False, box=None)
    breakdown_table.add_column("Metric", style="cyan")
    breakdown_table.add_column("Value", justify="right")
    
    for k, v in score.breakdown.items():
        breakdown_table.add_row(k.replace('_', ' ').title(), f"{v:.4f}")
    
    console.print(breakdown_table)

if __name__ == "__main__":
    main()
