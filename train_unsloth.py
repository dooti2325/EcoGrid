"""
EcoGrid-OpenEnv — Unsloth GRPO Training Script

Trains an LLM to play the EcoGrid environment using Group Relative Policy Optimization.
Leverages unsloth for 4-bit quantised, memory-efficient LoRA training.
"""

import argparse
import json
import os
import random
from typing import List, Dict

try:
    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from trl import GRPOConfig, GRPOTrainer
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False

from env.environment import EcoGridEnv
from models.schemas import GridAction

# Default to a small model, but allow override
DEFAULT_MODEL = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16


def parse_state_from_prompt(prompt) -> dict:
    """Extract the state JSON from the prompt string or message list."""
    try:
        if isinstance(prompt, list):
            prompt_str = prompt[-1].get('content', '')
        else:
            prompt_str = str(prompt)
            
        parts = prompt_str.split("CURRENT STATE:\n")
        if len(parts) > 1:
            state_text = parts[1].split("\n\nTASK:")[0]
            return json.loads(state_text)
    except Exception:
        pass
    return {}


def parse_action_from_completion(completion: str) -> GridAction | None:
    """Extract and validate GridAction JSON from model completion."""
    try:
        start_idx = completion.find('{')
        end_idx = completion.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = completion[start_idx:end_idx+1]
            data = json.loads(json_str)
            return GridAction(**data)
        return None
    except Exception:
        return None


def format_prompt(state_dict: dict, task_name: str) -> list:
    """Format the prompt for the model using chat template messages."""
    state_json = json.dumps(state_dict, indent=2)
    
    system_msg = "You are an expert energy grid operator. Your goal is to balance renewable energy, fossil fuels, and battery storage to meet demand while minimising cost and carbon emissions."
    
    user_msg = f"""CURRENT STATE:
{state_json}

TASK: {task_name}
CONSTRAINTS: 
- renewable_ratio + fossil_ratio <= 1.0
- battery_action must be between -1.0 (discharge) and 1.0 (charge)

Output ONLY a valid JSON object:
{{
  "renewable_ratio": float,
  "fossil_ratio": float,
  "battery_action": float
}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def generate_training_data(num_samples: int, task: str) -> Dataset:
    """Generate a dataset of random grid states for training."""
    print(f"Generating {num_samples} training states for task '{task}'...")
    env = EcoGridEnv()
    
    prompts = []
    # We just run the environment randomly to generate a variety of states
    # Note: We don't need target actions because GRPO learns through trial and error!
    state = env.reset(task=task, seed=42)
    
    for _ in range(num_samples):
        state_dict = state.model_dump()
        prompts.append(format_prompt(state_dict, task))
        
        # Take a random valid action to advance the environment
        action = GridAction(
            renewable_ratio=random.uniform(0, 0.8),
            fossil_ratio=random.uniform(0, 0.2),
            battery_action=random.uniform(-1, 1)
        )
        
        try:
            result = env.step(action)
            state = result.observation
        except Exception:
            # If done or errored, reset
            state = env.reset(task=task, seed=random.randint(0, 10000))
            
    return Dataset.from_dict({"prompt": prompts})


def main():
    parser = argparse.ArgumentParser(description="Unsloth GRPO Training for EcoGrid")
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path/name")
    args = parser.parse_args()
    
    if not HAS_UNSLOTH:
        print("Error: unsloth or trl not installed.")
        print("Install: pip install unsloth trl datasets")
        return
        
    print(f"Initializing Unsloth GRPO training on {args.model}")
    
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto detection
        load_in_4bit=True,
    )
    
    # Add LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    
    # 2. Define GRPO Reward Function
    # We instantiate a fresh environment just for evaluating rewards during training
    reward_env = EcoGridEnv()
    
    def ecogrid_reward_func(prompts: List[str], completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward function that evaluates model completions using the real environment."""
        rewards = []
        
        for prompt, completion_list in zip(prompts, completions):
            # TRL passes a list of messages for completion. We want the text content.
            # Depending on format, it might be a list of dicts. We extract the string.
            if isinstance(completion_list, list) and len(completion_list) > 0:
                completion_text = completion_list[-1]["content"]
            else:
                completion_text = str(completion_list)
                
            state_dict = parse_state_from_prompt(prompt)
            action = parse_action_from_completion(completion_text)
            
            if action is None or not state_dict:
                # Malformed JSON or invalid prompt extraction
                rewards.append(0.0)
                continue
                
            try:
                # To accurately calculate reward for THIS specific state and action,
                # we technically just need to call compute_reward, but it's easier to 
                # forcefully inject the state into a reset environment.
                # In a true RL loop we'd step through, but GRPO is stateless evaluation.
                reward_env.reset(task=args.task, seed=42) # Seed doesn't matter here
                
                # Hack: inject state directly for evaluation
                from models.schemas import GridState
                reward_env._state = GridState(**state_dict)
                reward_env._done = False
                
                result = reward_env.step(action)
                # The reward is what the environment dictates!
                rewards.append(result.reward)
            except Exception as e:
                # Constraint violation or other error
                rewards.append(0.0)
                
        return rewards
        
    def format_reward_func(completions, **kwargs) -> List[float]:
        """Secondary reward: give a small bonus just for outputting valid JSON."""
        rewards = []
        for completion_list in completions:
            text = completion_list[-1]["content"] if isinstance(completion_list, list) else str(completion_list)
            action = parse_action_from_completion(text)
            rewards.append(0.1 if action is not None else 0.0)
        return rewards

    # 3. Prepare Dataset
    dataset = generate_training_data(args.samples, args.task)
    
    # 4. Configure Trainer
    training_args = GRPOConfig(
        output_dir="./lora_adapter",
        learning_rate=2e-5,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_prompt_length=1024,
        max_completion_length=500,
        num_generations=4, # Number of completions to generate per prompt for relative scoring
        save_steps=100,
        logging_steps=10,
        report_to="none", # We will save our own logs
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[ecogrid_reward_func, format_reward_func],
        args=training_args,
        train_dataset=dataset,
    )
    
    # 5. Train
    print("Starting GRPO training...")
    trainer.train()
    
    # 6. Save
    print("Training complete. Saving LoRA adapter...")
    model.save_pretrained("./lora_adapter")
    tokenizer.save_pretrained("./lora_adapter")
    
    # Extract logs to show improvement
    log_history = trainer.state.log_history
    reward_curve = []
    for log in log_history:
        if "eval_ecogrid_reward_func" in log or "reward/ecogrid_reward_func" in log:
            key = "eval_ecogrid_reward_func" if "eval_ecogrid_reward_func" in log else "reward/ecogrid_reward_func"
            reward_curve.append({
                "step": log.get("step", 0),
                "reward": log.get(key, 0.0)
            })
            
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/reward_curve.json", "w") as f:
        json.dump(reward_curve, f, indent=2)
        
    print("Saved reward curve to ./logs/reward_curve.json")

if __name__ == "__main__":
    main()
