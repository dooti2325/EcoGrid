from traffic_rl.env.multi_intersection_env import MultiIntersectionEnv


def test_multi_intersection_step_contract():
    env = MultiIntersectionEnv(num_intersections=2, env_config={"max_steps": 10, "ambulance_spawn_prob": 0.0})

    states = env.reset()
    assert len(states) == 2
    assert states[0].shape == (10,)

    next_states, rewards, dones, infos = env.step([1, 2])
    assert len(next_states) == 2
    assert len(rewards) == 2
    assert len(dones) == 2
    assert isinstance(infos[0], dict)
