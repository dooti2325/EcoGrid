from traffic_rl.reward.reward_engine import RewardEngine


def test_higher_congestion_yields_lower_reward():
    reward_engine = RewardEngine()

    low = reward_engine.compute(
        queue_sum=5,
        waiting_sum=8,
        throughput=4,
        switched=False,
        ambulance_wait=0,
        ambulance_cleared=False,
    )
    high = reward_engine.compute(
        queue_sum=20,
        waiting_sum=30,
        throughput=4,
        switched=False,
        ambulance_wait=0,
        ambulance_cleared=False,
    )

    assert high < low


def test_ambulance_priority_bonus_applies():
    reward_engine = RewardEngine()

    no_clear = reward_engine.compute(
        queue_sum=10,
        waiting_sum=10,
        throughput=2,
        switched=False,
        ambulance_wait=3,
        ambulance_cleared=False,
    )
    cleared = reward_engine.compute(
        queue_sum=10,
        waiting_sum=10,
        throughput=2,
        switched=False,
        ambulance_wait=0,
        ambulance_cleared=True,
    )

    assert cleared > no_clear


def test_switch_penalty_triggers():
    reward_engine = RewardEngine()

    keep_phase = reward_engine.compute(
        queue_sum=8,
        waiting_sum=8,
        throughput=3,
        switched=False,
        ambulance_wait=0,
        ambulance_cleared=False,
    )
    switched = reward_engine.compute(
        queue_sum=8,
        waiting_sum=8,
        throughput=3,
        switched=True,
        ambulance_wait=0,
        ambulance_cleared=False,
    )

    assert switched < keep_phase
