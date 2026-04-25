---
title: RL Traffic Intelligence
emoji: ??
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
---

# RL-Based Adaptive Traffic Intelligence System

A modular reinforcement learning project for adaptive traffic signal control.

## What this Space does
- Runs a fixed-time baseline traffic controller
- Trains a lightweight DQN agent
- Compares baseline vs RL on:
  - waiting time
  - queue length
  - throughput
  - emergency handling
- Produces visual plots for training trends and policy comparison

## Local run
```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe run_demo.py --episodes 70 --eval-episodes 20 --output-dir outputs
```

## Project modules
- `traffic_rl/env` - environment and traffic dynamics
- `traffic_rl/reward` - reward engineering
- `traffic_rl/baseline` - fixed-time baseline
- `traffic_rl/agent` - DQN + replay buffer
- `traffic_rl/training` - training loop
- `traffic_rl/evaluation` - metric comparison
- `traffic_rl/visualization` - plotting dashboard
- `traffic_rl/skills` - modular wrappers
