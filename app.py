from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr

from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.evaluation.evaluator import (
    compare_policies,
    evaluate_agent,
    evaluate_fixed_controller,
)
from traffic_rl.training.trainer import TrainingConfig, train_dqn
from traffic_rl.visualization.dashboard import plot_comparison, plot_training_history


def _default_env_config(ambulance_prob: float, seed: int) -> dict:
    return {
        "max_steps": 120,
        "arrival_mode": "stochastic",
        "lane_bias": (1.6, 0.8, 1.4, 0.6),
        "peak_rates": (3.8, 2.2, 3.4, 1.5),
        "offpeak_rates": (1.4, 0.9, 1.2, 0.7),
        "peak_duration": 35,
        "cycle_duration": 60,
        "service_rate": 2,
        "ambulance_spawn_prob": ambulance_prob,
        "seed": seed,
    }


def run_experiment(train_episodes: int, eval_episodes: int, ambulance_prob: float, seed: int):
    env_config = _default_env_config(ambulance_prob=ambulance_prob, seed=seed)

    baseline_metrics = evaluate_fixed_controller(
        env_config=env_config,
        episodes=eval_episodes,
        switch_interval=5,
    )

    train_env = TrafficEnv(config=env_config)
    cfg = TrainingConfig(
        episodes=train_episodes,
        max_steps=env_config["max_steps"],
        batch_size=64,
        target_sync_interval=10,
        epsilon_decay=0.97,
        seed=seed,
    )

    agent, history = train_dqn(env=train_env, config=cfg)
    rl_metrics = evaluate_agent(agent=agent, env_config=env_config, episodes=eval_episodes)
    improvement = compare_policies(baseline_metrics, rl_metrics)

    run_dir = Path("outputs") / "space_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_plot = plot_training_history(history, output_dir=run_dir)["training_history"]
    compare_plot = plot_comparison(baseline_metrics, rl_metrics, output_dir=run_dir)["policy_comparison"]

    result = {
        "baseline": baseline_metrics,
        "rl": rl_metrics,
        "improvement": improvement,
        "config": {
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "ambulance_prob": ambulance_prob,
            "seed": seed,
        },
    }

    summary_md = (
        "## Run Summary\n"
        f"- Waiting time improvement: **{improvement['waiting_time_improvement_pct']:.2f}%**\n"
        f"- Queue length improvement: **{improvement['queue_length_improvement_pct']:.2f}%**\n"
        f"- Throughput gain: **{improvement['throughput_gain_pct']:.2f}%**\n"
        f"- Ambulance clearance gain: **{improvement['ambulance_clearance_gain_pct']:.2f}%**"
    )

    return result, summary_md, train_plot, compare_plot


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RL-Based Adaptive Traffic Intelligence") as demo:
        gr.Markdown(
            """
# ?? RL-Based Adaptive Traffic Intelligence System
Train and compare a DQN traffic controller against a fixed-time baseline.
This demo optimizes waiting time, queue length, throughput, and emergency handling.
            """
        )

        with gr.Row():
            train_episodes = gr.Slider(20, 200, value=70, step=10, label="Training Episodes")
            eval_episodes = gr.Slider(5, 50, value=20, step=5, label="Evaluation Episodes")
        with gr.Row():
            ambulance_prob = gr.Slider(0.0, 0.4, value=0.08, step=0.01, label="Ambulance Spawn Probability")
            seed = gr.Number(value=42, precision=0, label="Random Seed")

        run_btn = gr.Button("Run RL vs Baseline", variant="primary")

        metrics_json = gr.JSON(label="Metrics")
        summary = gr.Markdown()
        train_img = gr.Image(label="Training Trends")
        compare_img = gr.Image(label="Policy Comparison")

        run_btn.click(
            fn=run_experiment,
            inputs=[train_episodes, eval_episodes, ambulance_prob, seed],
            outputs=[metrics_json, summary, train_img, compare_img],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
