# What if an AI had to manage a city's power grid?

The transition to renewable energy is the defining engineering challenge of our generation. But it introduces a massive new problem for power grids: **volatility**. 

The sun doesn't always shine. The wind doesn't always blow. Yet, when a hospital needs power or a million commuters plug in their EVs at 6 PM, the grid must deliver immediately. If supply doesn't perfectly match demand, the frequency drops, and rolling blackouts begin.

Currently, human operators manage this by spinning up expensive, carbon-heavy fossil fuel "peaker plants" to cover the gaps. 

But what if an AI could balance it perfectly?

### Enter EcoGrid-OpenEnv

For the **Scaler School of Technology × Meta PyTorch Hackathon**, we built **EcoGrid-OpenEnv**, a production-grade Reinforcement Learning environment designed to train agents to solve this exact problem.

Built on Meta's OpenEnv framework, EcoGrid places an AI agent in the control room. At every timestep, the agent observes the current demand, the available solar/wind capacity, the spot price of electricity, and a strict remaining carbon budget. 

It then outputs a continuous action:
- How much demand to meet with renewables?
- How much fossil fuel to burn?
- Should we charge the battery with excess sun, or discharge it to cover a spike?

### The Hard Task: Carbon Constrained

We didn't want to build a toy game. We designed the reward function to force multi-objective optimization: minimising cost, maximising grid stability, and adhering to a strict carbon cap. 

In our "Hard" task, the agent is given a highly volatile weather forecast and a hard carbon limit. If the budget drops below zero, the episode terminates instantly with a massive penalty. 

### Training with GRPO

Because the state and action spaces are complex, standard PPO often struggles to explore effectively. We implemented a training pipeline using **Unsloth** and **TRL**, leveraging **Group Relative Policy Optimization (GRPO)**.

Instead of a learned critic model, we use the EcoGrid environment itself as the deterministic reward function. We prompt a 1.5B parameter model with the grid state, ask it to output its reasoning (Chain-of-Thought) followed by a JSON action. GRPO rewards the agent when its reasoning leads to a stable, low-carbon grid.

### See it in Action

We've deployed a live interactive dashboard where you can watch random agents, smart heuristics, and trained models battle the grid volatility in real-time.

[Check out the interactive EcoGrid Dashboard on Hugging Face Spaces](#)

*Built by Team DD.*
