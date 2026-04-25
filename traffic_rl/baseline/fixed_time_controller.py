from __future__ import annotations


class FixedTimeController:
    """Simple fixed-time phase alternator used as benchmark."""

    def __init__(self, switch_interval: int = 5) -> None:
        if switch_interval <= 0:
            raise ValueError("switch_interval must be positive")
        self.switch_interval = switch_interval

    def phase_for_step(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        return (step // self.switch_interval) % 2

    def action_for_step(self, step: int) -> int:
        phase = self.phase_for_step(step)
        return 1 if phase == 0 else 2
