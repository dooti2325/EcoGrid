from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.training.trainer import TrainingConfig, train_dqn
from traffic_rl.evaluation.evaluator import evaluate_agent, evaluate_fixed_controller


def test_training_loop_returns_history():
    env = TrafficEnv(
        config={
            "max_steps": 20,
            "arrival_mode": "deterministic",
            "arrival_sequence": [[1, 1, 1, 1]],
            "ambulance_spawn_prob": 0.0,
            "seed": 11,
        }
    )
    cfg = TrainingConfig(episodes=4, max_steps=20, batch_size=8, target_sync_interval=2)

    agent, history = train_dqn(env=env, config=cfg)

    assert len(history["episode_reward"]) == 4
    assert len(history["avg_queue"]) == 4
    assert agent.action_dim == 3


def test_evaluation_returns_metrics():
    env_config = {
        "max_steps": 20,
        "arrival_mode": "deterministic",
        "arrival_sequence": [[2, 1, 2, 1]],
        "ambulance_spawn_prob": 0.0,
        "seed": 12,
    }
    env = TrafficEnv(config=env_config)
    cfg = TrainingConfig(episodes=3, max_steps=20, batch_size=8, target_sync_interval=2)
    agent, _ = train_dqn(env=env, config=cfg)

    fixed_metrics = evaluate_fixed_controller(env_config=env_config, episodes=2, switch_interval=2)
    rl_metrics = evaluate_agent(agent=agent, env_config=env_config, episodes=2)

    assert fixed_metrics["avg_waiting_time"] >= 0
    assert rl_metrics["avg_queue_length"] >= 0
    assert "throughput" in fixed_metrics
    assert "throughput" in rl_metrics
