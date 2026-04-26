"""Deterministic benchmark runner for EcoGrid policies."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean

from baseline import heuristic_agent
from env.environment import EcoGridEnv
from env.tasks import (
    BasicGridBalanceGrader,
    CarbonConstrainedGrader,
    RenewableVariabilityGrader,
)
from models.schemas import GridAction


TASKS = ("easy", "medium", "hard")
AGENTS = ("random", "heuristic")

# Historical reference numbers from pre-fix evaluation snapshot.
REFERENCE_MEAN = {
    "easy": {"random": 0.269, "heuristic": 0.748},
    "medium": {"random": 0.251, "heuristic": 0.376},
    "hard": {"random": 0.001, "heuristic": 0.001},
}


def grade_episode(task: str, episode_log):
    if task == "easy":
        return BasicGridBalanceGrader.grade(episode_log).score
    if task == "medium":
        return RenewableVariabilityGrader.grade(episode_log).score
    return CarbonConstrainedGrader.grade(episode_log).score


def choose_action(agent: str, task: str, state, rng: random.Random) -> GridAction:
    if agent == "heuristic":
        return heuristic_agent(state, task)

    renewable_ratio = rng.random()
    fossil_ratio = rng.random() * (1.0 - renewable_ratio)
    battery_action = rng.uniform(-1.0, 1.0)
    return GridAction(
        renewable_ratio=renewable_ratio,
        fossil_ratio=fossil_ratio,
        battery_action=battery_action,
    )


def run_episode(task: str, agent: str, seed: int) -> float:
    rng = random.Random(seed)
    env = EcoGridEnv()
    state = env.reset(task=task, seed=seed)
    while not env.is_done:
        action = choose_action(agent, task, state, rng)
        result = env.step(action)
        state = result.observation
    return grade_episode(task, env.get_episode_log())


def run_benchmarks(seeds: list[int]) -> dict:
    out = {
        "metadata": {"seeds": seeds, "agents": list(AGENTS), "tasks": list(TASKS)},
        "results": {},
    }
    for task in TASKS:
        out["results"][task] = {}
        for agent in AGENTS:
            scores = [run_episode(task, agent, seed) for seed in seeds]
            avg = float(mean(scores))
            ref = REFERENCE_MEAN[task][agent]
            out["results"][task][agent] = {
                "scores": [round(x, 6) for x in scores],
                "mean": round(avg, 6),
                "reference_mean": ref,
                "delta_vs_reference": round(avg - ref, 6),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="Run EcoGrid reproducible benchmark suite.")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated integer seeds")
    parser.add_argument(
        "--out",
        default="logs/benchmark_results.json",
        help="Path to save benchmark results JSON",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    results = run_benchmarks(seeds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved benchmark report to {out_path}")


if __name__ == "__main__":
    main()
