import numpy as np

from traffic_rl.env.traffic_env import TrafficEnv


def make_env(**overrides):
    config = {
        "max_steps": 20,
        "arrival_mode": "deterministic",
        "arrival_sequence": [
            [2, 1, 0, 0],
            [1, 0, 2, 0],
            [0, 1, 1, 0],
            [1, 1, 0, 2],
        ],
        "service_rate": 2,
        "seed": 123,
    }
    config.update(overrides)
    return TrafficEnv(config=config)


def test_reset_returns_valid_state():
    env = make_env()
    state = env.reset()

    assert isinstance(state, np.ndarray)
    assert state.shape == (10,)
    assert np.all(state[:8] >= 0)
    assert state[8] in (0, 1)


def test_step_updates_queues_and_non_negative():
    env = make_env(arrival_sequence=[[1, 1, 0, 0]], service_rate=1)
    env.reset()

    next_state, reward, done, info = env.step(1)

    assert next_state.shape == (10,)
    assert np.all(next_state[:8] >= 0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "throughput" in info


def test_environment_deterministic_transitions():
    env1 = make_env()
    env2 = make_env()

    s1 = env1.reset()
    s2 = env2.reset()
    assert np.allclose(s1, s2)

    actions = [1, 0, 2, 0, 1]
    for a in actions:
        ns1, r1, d1, i1 = env1.step(a)
        ns2, r2, d2, i2 = env2.step(a)
        assert np.allclose(ns1, ns2)
        assert r1 == r2
        assert d1 == d2
        assert i1["throughput"] == i2["throughput"]


def test_no_negative_values_over_rollout():
    env = make_env()
    env.reset()
    for _ in range(10):
        state, _, done, _ = env.step(0)
        assert np.all(state[:8] >= 0)
        if done:
            break
