from __future__ import annotations

from traffic_rl.visualization.dashboard import plot_comparison, plot_training_history, plot_trajectory


def render_dashboard(history: dict, baseline: dict, rl: dict, fixed_trace: dict, rl_trace: dict, output_dir: str = "outputs") -> dict[str, str]:
    saved = {}
    saved.update(plot_training_history(history, output_dir=output_dir))
    saved.update(plot_comparison(baseline, rl, output_dir=output_dir))
    saved.update(plot_trajectory(fixed_trace, output_dir=output_dir, name="fixed_trajectory"))
    saved.update(plot_trajectory(rl_trace, output_dir=output_dir, name="rl_trajectory"))
    return saved
