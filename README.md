---
title: EcoGrid OpenEnv
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---
# ⚡ EcoGrid-OpenEnv
![Hackathon](https://img.shields.io/badge/Scaler_SoT_×_Meta_PyTorch-Finale-blue)

**EcoGrid-OpenEnv** is a production-grade Reinforcement Learning environment for the OpenEnv framework. It simulates sustainable energy grid management where an AI agent must balance renewable sources, fossil fuels, and battery storage to meet demand while minimising cost and carbon emissions.

Built for the **Theme #3: World Modeling** track (Mercor Sub-theme).

📖 **[Read the Project Writeup / Blog](BLOG.md)**


---

## 🌍 The Problem

Modern power grids are facing unprecedented volatility. The transition to renewable energy introduces extreme supply variance (the sun doesn't always shine, the wind doesn't always blow), while electrification of transport causes unpredictable demand spikes. 

Grid operators must solve a continuous, multi-objective optimization problem:
1. **Prevent Blackouts:** Meet demand perfectly.
2. **Minimise Cost:** Avoid expensive fossil fuels and spot-market emergency purchases.
3. **Cut Emissions:** Stay within strict carbon budgets.

This environment models that exact problem as a Reinforcement Learning Markov Decision Process (MDP).

---

## 🏗️ Environment Design

### State Space (Observation)
The agent receives a rich, dense state vector at every step:
```text
┌───────────────────────┐
│ GridState             │
│ ├─ demand (MWh)       │ ─> Varies wildly (morning/evening peaks)
│ ├─ solar_capacity     │ ─> Predictable daytime curve + cloud noise
│ ├─ wind_capacity      │ ─> Mean-reverting random walk + noise
│ ├─ battery_level      │ ─> State of charge [0,1]
│ ├─ grid_stability     │ ─> Momentum-based frequency indicator
│ ├─ carbon_budget      │ ─> Remaining kgCO₂
│ ├─ price_signal       │ ─> Surges when supply < demand
│ └─ time_step          │
└───────────────────────┘
```

### Action Space
At each step, the agent outputs a continuous action vector:
```text
┌───────────────────────┐
│ GridAction            │
│ ├─ renewable_ratio    │ ─> [0, 1] Fraction of demand met by renewables
│ ├─ fossil_ratio       │ ─> [0, 1] Fraction of demand met by fossil
│ └─ battery_action     │ ─> [-1, 1] Discharge(-1) to Charge(+1)
└───────────────────────┘
```
*Constraint: `renewable_ratio + fossil_ratio <= 1.0`*

---

## 📈 Reward Function

The reward is a dense scalar in `[0, 1]` calculated at every step. This provides immediate, continuous feedback to the agent, making it highly trainable via algorithms like GRPO or PPO.

| Component | Weight | Description |
|-----------|--------|-------------|
| **Cost Savings** | 0.30 | `1 - normalised(fossil_cost + grid_cost)` |
| **Carbon Score** | 0.30 | `1 - normalised(carbon_emission)` |
| **Stability** | 0.25 | `1 - blackout_risk` |
| **Green Bonus** | 0.15 | `renewable_ratio * stability_score` |

**Penalties:**
- `-0.5` applied if >20% of demand is unmet (blackout).
- `-0.8` applied if carbon budget is exceeded (Hard task only).

---

## 🎯 Tasks

| Task | Difficulty | Episode | Conditions | Goal | Grader |
|------|------------|---------|------------|------|--------|
| `easy` | 1 | 48 steps | Stable solar, flat demand, no battery | Minimise Cost | `BasicGridBalance` |
| `medium` | 2 | 96 steps | Noisy renewables, demand spikes, small battery | Avoid Blackouts | `RenewableVariability` |
| `hard` | 3 | 96 steps | Strict carbon cap, 2x noise, limited storage | Survive within Carbon Cap | `CarbonConstrained` |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/yourusername/eco-grid-openenv.git
cd eco-grid-openenv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify OpenEnv Compatibility
```bash
openenv validate openenv.yaml
# Output: ✅ Environment spec valid.
```

### 3. Run Baseline Agents
Run the deterministic heuristic baseline:
```bash
python baseline.py --task easy --agent heuristic
python baseline.py --task hard --agent heuristic
```

Run the LLM (OpenAI) agent:
```bash
export OPENAI_API_KEY="sk-..."
python baseline.py --task medium --agent llm
```

---

## 🧠 Training with Unsloth (GRPO)

We provide a full training pipeline using Unsloth and Hugging Face `trl` to train a small LLM (`Qwen2.5-1.5B-Instruct`) to play the environment. 

The training script uses the environment itself as the reward function for **Group Relative Policy Optimization (GRPO)**.

```bash
# Requires unsloth, trl, torch
python train_unsloth.py --task hard --epochs 3 --samples 500
```
This saves a LoRA adapter to `./lora_adapter/` and reward curves to `./logs/`.

---

## 📊 Baseline Scores

*Averaged over 5 random seeds.*

| Task | Random Agent | Heuristic Agent | Trained LLM (Expected) |
|------|-------------|-----------------|------------------------|
| `easy` | 0.21 | 0.72 | 0.81 |
| `medium` | 0.16 | 0.58 | 0.75 |
| `hard` | 0.00 (fail) | 0.41 | 0.62 |

---

## 🖥️ Live Dashboard (Hugging Face Space)

We've deployed an interactive Streamlit dashboard allowing you to run episodes and visualize live grid state, reward curves, and carbon emissions.

**[View the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Loosebag/EcoGrid)**

### Local Docker Build
```bash
docker build -t ecogrid .
docker run -p 7860:7860 ecogrid
# Open http://localhost:7860
```
