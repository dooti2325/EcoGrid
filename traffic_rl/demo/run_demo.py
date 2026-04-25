from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_rl.baseline.fixed_time_controller import FixedTimeController
from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.evaluation.evaluator import (
    collect_trajectory,
    compare_policies,
    evaluate_agent,
    evaluate_fixed_controller,
)
from traffic_rl.training.trainer import TrainingConfig, train_dqn
from traffic_rl.visualization.dashboard import plot_comparison, plot_training_history, plot_trajectory


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-Based Adaptive Traffic Intelligence Demo")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
        "max_steps": 120,
        "arrival_mode": "stochastic",
        "lane_bias": (1.6, 0.8, 1.4, 0.6),
        "peak_rates": (3.8, 2.2, 3.4, 1.5),
        "offpeak_rates": (1.4, 0.9, 1.2, 0.7),
        "peak_duration": 35,
        "cycle_duration": 60,
        "service_rate": 2,
        "ambulance_spawn_prob": 0.08,
        "seed": 42,
    }

    print("1) Running fixed-time baseline...")
    baseline_metrics = evaluate_fixed_controller(
        env_config=env_config,
        episodes=args.eval_episodes,
        switch_interval=5,
    )

    print("2) Training RL (DQN) agent...")
    train_env = TrafficEnv(config=env_config)
    training_config = TrainingConfig(
        episodes=args.episodes,
        max_steps=env_config["max_steps"],
        batch_size=64,
        target_sync_interval=10,
        epsilon_decay=0.97,
    )
    agent, history = train_dqn(train_env, training_config)

    print("3) Evaluating RL agent...")
    rl_metrics = evaluate_agent(agent=agent, env_config=env_config, episodes=args.eval_episodes)
    comparison = compare_policies(baseline_metrics, rl_metrics)

    ambulance_env = {
        **env_config,
        "ambulance_spawn_prob": 0.25,
        "seed": 99,
    }
    baseline_amb = evaluate_fixed_controller(env_config=ambulance_env, episodes=10, switch_interval=5)
    rl_amb = evaluate_agent(agent=agent, env_config=ambulance_env, episodes=10)

    _print_metrics("Fixed-Time Baseline", baseline_metrics)
    _print_metrics("RL Agent", rl_metrics)
    _print_metrics("Improvement (RL vs Fixed)", comparison)

    print("\n4) Ambulance Priority Scenario")
    print(f"fixed ambulance_clearances: {baseline_amb['ambulance_clearances']:.3f}")
    print(f"rl ambulance_clearances: {rl_amb['ambulance_clearances']:.3f}")

    print("\n5) Generating visualizations...")
    saved = {}
    saved.update(plot_training_history(history, output_dir=output_dir))
    saved.update(plot_comparison(baseline_metrics, rl_metrics, output_dir=output_dir))

    fixed_controller = FixedTimeController(switch_interval=5)
    fixed_trace = collect_trajectory(TrafficEnv(config=env_config), lambda _s, step: fixed_controller.action_for_step(step))
    rl_trace = collect_trajectory(TrafficEnv(config=env_config), lambda state, _step: agent.select_action(state, epsilon=0.0))

    saved.update(plot_trajectory(fixed_trace, output_dir=output_dir, name="fixed_trajectory"))
    saved.update(plot_trajectory(rl_trace, output_dir=output_dir, name="rl_trajectory"))

    summary = {
        "baseline": baseline_metrics,
        "rl": rl_metrics,
        "comparison": comparison,
        "ambulance_baseline": baseline_amb,
        "ambulance_rl": rl_amb,
        "artifacts": saved,
    }

    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDemo completed.")
    print(f"Metrics summary: {summary_path}")
    for name, path in saved.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
