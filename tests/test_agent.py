import numpy as np
import torch

from traffic_rl.agent.dqn_agent import DQNAgent


def test_dqn_valid_action_output():
    agent = DQNAgent(state_dim=10, action_dim=3, seed=7)
    state = np.zeros(10, dtype=np.float32)

    action = agent.select_action(state, epsilon=0.0)

    assert isinstance(action, int)
    assert 0 <= action < 3


def test_dqn_q_value_shape():
    agent = DQNAgent(state_dim=10, action_dim=3, seed=7)
    batch = torch.zeros((4, 10), dtype=torch.float32)

    q_values = agent.q_network(batch)

    assert tuple(q_values.shape) == (4, 3)
