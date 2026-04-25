# EcoGrid-OpenEnv — Technical Specification

## 1. Overview

EcoGrid-OpenEnv is a production-grade Reinforcement Learning environment for the
OpenEnv framework that simulates sustainable energy grid management. An agent
controls energy distribution across renewable sources (solar + wind), fossil fuels,
and battery storage to meet variable demand while minimising cost and carbon emissions.

**Author:** Team DD  
**Hackathon:** Scaler School of Technology × Meta PyTorch Finale  
**Theme:** #3 — World Modeling (Professional Tasks) + Mercor Sub-theme

---

## 2. File Tree

```
eco-grid-openenv/
├── env/
│   ├── __init__.py           # Package init, re-exports EcoGridEnv
│   ├── environment.py        # Core EcoGridEnv class (reset/step/state)
│   ├── dynamics.py           # Deterministic physics helpers (solar/wind/demand/battery)
│   ├── reward.py             # Multi-objective reward function
│   └── tasks.py              # 3 task graders (easy/medium/hard)
├── models/
│   ├── __init__.py           # Package init, re-exports all schemas
│   └── schemas.py            # Pydantic v2 typed models
├── tests/
│   ├── test_environment.py   # Environment unit tests
│   ├── test_reward.py        # Reward function tests
│   └── test_graders.py       # Grader determinism tests
├── docs/
│   └── SPEC.md               # This file
├── logs/                     # Runtime logs (gitignored)
├── lora_adapter/             # Trained LoRA weights (gitignored)
├── openenv.yaml              # OpenEnv specification file
├── baseline.py               # Heuristic + LLM inference script
├── train_unsloth.py          # Unsloth GRPO training script
├── app.py                    # Streamlit dashboard for HF Spaces
├── Dockerfile                # Multi-stage production Dockerfile
├── docker-compose.yml        # Local dev compose config
├── requirements.txt          # Pinned Python dependencies
├── README.md                 # Full hackathon README
├── BLOG.md                   # HuggingFace blog post (≤400 words)
└── .gitignore                # Ignore logs/, lora_adapter/, __pycache__
```

---

## 3. State Space

| Field | Type | Range | Unit | Description |
|-------|------|-------|------|-------------|
| demand | float | [0, 200] | MWh | Energy demand this timestep |
| solar_capacity | float | [0, 1] | ratio | Available solar generation capacity |
| wind_capacity | float | [0, 1] | ratio | Available wind generation capacity |
| battery_level | float | [0, 1] | ratio | Battery state of charge |
| grid_stability | float | [0, 1] | ratio | Grid frequency stability indicator |
| carbon_budget_remaining | float | [0, 1000] | kgCO₂ | Remaining carbon emission budget |
| price_signal | float | [0, 300] | $/MWh | Current spot electricity price |
| time_step | int | [0, 96] | step | Current simulation timestep |

---

## 4. Action Space

| Field | Type | Range | Constraint | Description |
|-------|------|-------|-----------|-------------|
| renewable_ratio | float | [0, 1] | sum ≤ 1.0 | Fraction of demand met by renewables |
| fossil_ratio | float | [0, 1] | sum ≤ 1.0 | Fraction of demand met by fossil fuels |
| battery_action | float | [-1, 1] | — | Charge (+1) or discharge (-1) battery |

**Constraint:** `renewable_ratio + fossil_ratio ≤ 1.0`  
The remaining fraction `1 - renewable_ratio - fossil_ratio` is unmet demand (blackout risk).

---

## 5. Reward Function

### Terms (summing to [0, 1])

| Term | Weight | Formula |
|------|--------|---------|
| cost_savings | 0.30 | `1 - normalised(fossil_cost + grid_cost)` |
| carbon_score | 0.30 | `1 - normalised(carbon_emission)` |
| stability_score | 0.25 | `1 - blackout_risk` |
| renewable_bonus | 0.15 | `renewable_ratio × (1 - blackout_risk)` |

### Penalties (subtracted before clip)

| Penalty | Trigger | Value |
|---------|---------|-------|
| blackout_penalty | demand unmet > 20% | -0.5 |
| carbon_overrun | carbon_budget < 0 (hard only) | -0.8 |

**Final:** `clip(weighted_sum - penalties, 0, 1)`

---

## 6. Task Specifications

### Easy: BasicGridBalance
- **Episode length:** 48 steps
- **Conditions:** Predictable solar (low noise), stable demand, no battery
- **Objective:** Minimise cost
- **Grader criteria:** avg_cost_reduction > 30%, no blackouts, avg_renewable_ratio > 0.5
- **Score:** total_reward / max_possible_reward

### Medium: RenewableVariability
- **Episode length:** 96 steps
- **Conditions:** Noisy solar+wind, demand spikes, small battery (capacity=0.3)
- **Objective:** Avoid blackouts while maintaining renewable usage
- **Grader criteria:** blackout_episodes < 3, avg_renewable_ratio > 0.6
- **Score:** 0.4×(renewable) + 0.4×(stability) + 0.2×(cost)

### Hard: CarbonConstrained
- **Episode length:** 96 steps
- **Conditions:** Strict carbon cap (500 kgCO₂), high volatility (2× noise), limited storage
- **Objective:** Never exceed carbon cap, maintain stability > 0.7
- **Grader criteria:** carbon_budget ≥ 0 at all steps, grid_stability > 0.7
- **Score:** 0 if any carbon overrun; else 0.5×(carbon) + 0.3×(cost) + 0.2×(stability)

---

## 7. Physics Model

All functions are **deterministic** given a seed (use `np.random.default_rng(seed)`).

### Solar Output
```
solar = 0.5 + 0.5 × sin(2π × time_step / 24) + noise × N(0, noise_level)
clamped to [0, 1]
```

### Wind Output
```
wind = random_walk(previous_wind, step_size=0.05) + noise × N(0, noise_level)
clamped to [0, 1]
```

### Demand Curve
```
base = 80 MWh
morning_peak = 30 × exp(-((t - 8)² / 8))
evening_peak = 40 × exp(-((t - 18)² / 8))
spike = random_spike_probability × spike_magnitude
demand = base + morning_peak + evening_peak + spike
```

### Battery
```
new_level = clamp(level + action × charge_rate, 0, 1)
```

### Carbon Emission
```
emission = fossil_ratio × demand × emission_factor (kg CO₂/MWh)
```

---

## 8. Mercor Sub-theme

The reward function is designed so that **more tokens of agent reasoning correspond
to better reward scaling**. Specifically:
- Actions that consider multiple objectives (cost + carbon + stability) score higher
- Chain-of-thought prompts in baseline.py encourage structured reasoning
- GRPO training rewards longer, more considered outputs with higher task scores

---

## 9. Performance Targets

| Metric | Target |
|--------|--------|
| Episodes/second (numpy sim) | > 10,000 |
| Random baseline score (easy) | 0.15–0.25 |
| Heuristic baseline score (easy) | 0.68–0.74 |
| Trained agent score (easy) | 0.75–0.85 |
| Seed determinism | 100% reproducible |
