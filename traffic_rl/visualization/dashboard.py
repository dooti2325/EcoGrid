from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_training_history(history: dict[str, list[float]], output_dir: str | Path = "outputs") -> dict[str, str]:
    out = _ensure_dir(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(history.get("episode_reward", []), color="#1f77b4")
    axes[0, 0].set_title("Episode Reward")
    axes[0, 0].set_xlabel("Episode")

    axes[0, 1].plot(history.get("avg_queue", []), color="#d62728")
    axes[0, 1].set_title("Average Queue Length")
    axes[0, 1].set_xlabel("Episode")

    axes[1, 0].plot(history.get("throughput", []), color="#2ca02c")
    axes[1, 0].set_title("Throughput")
    axes[1, 0].set_xlabel("Episode")

    axes[1, 1].plot(history.get("epsilon", []), color="#9467bd")
    axes[1, 1].set_title("Exploration (Epsilon)")
    axes[1, 1].set_xlabel("Episode")

    path = out / "training_history.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return {"training_history": str(path)}


def plot_comparison(
    baseline: dict[str, float],
    rl: dict[str, float],
    output_dir: str | Path = "outputs",
) -> dict[str, str]:
    out = _ensure_dir(output_dir)

    metrics = ["avg_waiting_time", "avg_queue_length", "throughput", "ambulance_clearances"]
    labels = ["Avg Wait", "Avg Queue", "Throughput", "Ambulance Clears"]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar([i - width / 2 for i in x], [baseline[m] for m in metrics], width, label="Fixed")
    ax.bar([i + width / 2 for i in x], [rl[m] for m in metrics], width, label="RL")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("RL vs Fixed-Time Controller")
    ax.legend()

    path = out / "policy_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return {"policy_comparison": str(path)}


def plot_trajectory(trace: dict[str, list[float]], output_dir: str | Path = "outputs", name: str = "trajectory") -> dict[str, str]:
    out = _ensure_dir(output_dir)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(trace.get("queue_sum", []), color="#ff7f0e")
    axes[0].set_title("Queue Sum")

    axes[1].plot(trace.get("phase", []), color="#17becf")
    axes[1].set_title("Signal Phase")

    axes[2].plot(trace.get("reward", []), color="#1f77b4")
    axes[2].set_title("Reward")
    axes[2].set_xlabel("Step")

    path = out / f"{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    return {name: str(path)}
