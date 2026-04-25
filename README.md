---
language: en
tags:
- reinforcement-learning
- dqn
- traffic-control
- pytorch
license: mit
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# RL-Based Adaptive Traffic Intelligence System

A modular, production-style Reinforcement Learning project for adaptive traffic signal control.

## Problem Statement
Traffic control is treated as a sequential decision-making problem. An agent selects signal actions each step to:
- minimize waiting time
- minimize queue length
- maximize throughput
- prioritize emergency vehicles

## Hugging Face Model Artifacts
- `artifacts/dqn_state_dict.pt` (trained DQN weights)
- `artifacts/model_config.json` (environment/training config snapshot)
- `artifacts/model_metrics.json` (evaluation snapshot)

## Highlights
- OpenEnv-style deterministic environment (`TrafficEnv`)
- Dense multi-component reward (no sparse-only objective)
- Fixed-time baseline controller for mandatory comparison
- Lightweight PyTorch DQN (replay buffer, epsilon-greedy, target network)
- Realistic traffic dynamics (stochastic arrivals, lane imbalance, peak/off-peak)
- Emergency vehicle priority handling
- Bonus multi-intersection decentralized simulator
- Visualization + end-to-end demo pipeline
- Skill-style modular wrappers for orchestration

## Architecture
- `traffic_rl/env`: core environment + multi-intersection extension
- `traffic_rl/reward`: reward engineering logic
- `traffic_rl/baseline`: fixed-time control baseline
- `traffic_rl/agent`: DQN and replay buffer
- `traffic_rl/training`: training loop
- `traffic_rl/evaluation`: metrics + policy comparison
- `traffic_rl/visualization`: dashboard plots
- `traffic_rl/demo`: demo pipeline runner
- `traffic_rl/skills`: modular reusable wrappers

## State, Action, Reward
### State
`[Q1,Q2,Q3,Q4,W1,W2,W3,W4,phase,ambulance_flag]`

### Action Space
- `0`: hold current phase
- `1`: switch/set NS green phase
- `2`: switch/set EW green phase

### Reward
`R_total = R_base + flow_bonus - congestion_penalty + emergency_bonus - switch_penalty`

Where:
- `R_base = - (queue_length + waiting_time)`
- `flow_bonus` rewards throughput
- `congestion_penalty` penalizes overloaded states
- `emergency_bonus` rewards ambulance clearance and penalizes ambulance delay
- `switch_penalty` discourages unstable phase flapping

## Quick Start
Use the project-local virtual environment.

```powershell
# run tests
.\.venv\Scripts\python.exe -m pytest

# run demo pipeline
.\.venv\Scripts\python.exe run_demo.py --episodes 70 --eval-episodes 20 --output-dir outputs
```

## Load Saved Weights
```python
import torch
from traffic_rl.agent.dqn_agent import DQNAgent

agent = DQNAgent(state_dim=10, action_dim=3)
state_dict = torch.load("artifacts/dqn_state_dict.pt", map_location="cpu")
agent.q_network.load_state_dict(state_dict)
agent.sync_target()
```

## Demo Pipeline
The demo performs:
1. fixed baseline simulation
2. RL training
3. RL evaluation
4. emergency-priority scenario
5. metrics + plots export

Artifacts are saved to `outputs/`:
- `metrics_summary.json`
- `training_history.png`
- `policy_comparison.png`
- `fixed_trajectory.png`
- `rl_trajectory.png`

## Measured Results (Demo Run)
From `outputs/metrics_summary.json`:

### Baseline vs RL
- Waiting time improvement: **9.82%**
- Queue length improvement: **1.64%**
- Throughput gain: **7.62%**

### Emergency Scenario
- Baseline ambulance clearances: **1.0**
- RL ambulance clearances: **2.0**

## Skill-Modular Components
Reusable orchestration wrappers:
- `env_builder_skill.py`
- `reward_engineering_skill.py`
- `baseline_skill.py`
- `dqn_training_skill.py`
- `evaluation_skill.py`
- `visualization_skill.py`

## Test Coverage
TDD tests include:
- Environment reset/step contracts, deterministic behavior, non-negative state
- Reward monotonicity and penalty/bonus triggers
- Baseline phase alternation logic
- DQN action validity and Q-value output shape
- Training/evaluation smoke tests
- Multi-intersection step contract
