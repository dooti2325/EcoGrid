from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50000,
        batch_size: int = 64,
        seed: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32)
        actions = torch.as_tensor([t.action for t in transitions], dtype=torch.int64).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([t.next_state for t in transitions]), dtype=torch.float32)
        dones = torch.as_tensor([t.done for t in transitions], dtype=torch.float32).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def sync_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
