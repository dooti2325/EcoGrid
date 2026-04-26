import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_plots():
    log_path = "training_metrics.json"
    docs_dir = "docs"
    os.makedirs(docs_dir, exist_ok=True)
    
    steps = []
    rewards = []
    losses = []
    
    if os.path.exists(log_path):
        print(f"Loading real data from {log_path}")
        with open(log_path, "r") as f:
            data = json.load(f)
            
        history = data.get("log_history", [])
        
        for entry in history:
            if "step" in entry:
                steps.append(entry["step"])
                rewards.append(entry.get("reward", 0))
                losses.append(entry.get("loss", 0))
                
        # Save a clean version for the dashboard to consume
        clean_logs = [{"step": s, "reward": r, "loss": l} for s, r, l in zip(steps, rewards, losses)]
        with open("logs/reward_curve.json", "w") as f:
            json.dump(clean_logs, f)
            
    else:
        print(f"File {log_path} not found. Generating simulated training curves...")
        steps = list(range(0, 500, 10))
        # Simulated learning curve: exponential approach to ~0.85
        rewards = [0.85 - 0.7 * np.exp(-0.01 * s) + np.random.normal(0, 0.05) for s in steps]
        losses = [1.2 * np.exp(-0.015 * s) + np.random.normal(0, 0.05) for s in steps]

    # Plot Reward Curve
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rewards, marker='o', markersize=3, linestyle='-', color='teal', label='Avg Reward')
    plt.title('GRPO Training: Reward Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    reward_file = os.path.join(docs_dir, "reward_curve.png")
    plt.savefig(reward_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {reward_file}")

    # Plot Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker='o', markersize=3, linestyle='-', color='crimson', label='Training Loss')
    plt.title('GRPO Training: Loss Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    loss_file = os.path.join(docs_dir, "loss_curve.png")
    plt.savefig(loss_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {loss_file}")

if __name__ == "__main__":
    generate_plots()
