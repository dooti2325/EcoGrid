import json
from pathlib import Path

import torch

from traffic_rl.env.traffic_env import TrafficEnv
from traffic_rl.training.trainer import TrainingConfig, train_dqn
from traffic_rl.evaluation.evaluator import evaluate_agent

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

cfg = TrainingConfig(episodes=80, max_steps=120, batch_size=64, target_sync_interval=10, epsilon_decay=0.97)
env = TrafficEnv(config=env_config)
agent, history = train_dqn(env, cfg)

metrics = evaluate_agent(agent=agent, env_config=env_config, episodes=20)

artifacts = Path("artifacts")
artifacts.mkdir(exist_ok=True)
torch.save(agent.q_network.state_dict(), artifacts / "dqn_state_dict.pt")
(artifacts / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
(artifacts / "model_config.json").write_text(json.dumps(env_config, indent=2), encoding="utf-8")
print(json.dumps({"metrics": metrics, "artifact_dir": str(artifacts)}, indent=2))
