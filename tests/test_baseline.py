from traffic_rl.baseline.fixed_time_controller import FixedTimeController


def test_fixed_signal_alternates_correctly():
    controller = FixedTimeController(switch_interval=2)
    phases = [controller.phase_for_step(step) for step in range(8)]

    assert phases == [0, 0, 1, 1, 0, 0, 1, 1]
