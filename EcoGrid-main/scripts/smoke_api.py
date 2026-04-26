"""Deployment smoke checks for EcoGrid OpenEnv server."""

from __future__ import annotations

import argparse
import json
import time

import requests


def check(base_url: str):
    t0 = time.perf_counter()
    health = requests.get(f"{base_url}/health", timeout=5)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0
    health.raise_for_status()

    reset = requests.post(f"{base_url}/reset", json={"task": "easy", "seed": 42}, timeout=10)
    reset.raise_for_status()

    step = requests.post(
        f"{base_url}/step",
        json={"renewable_ratio": 0.6, "fossil_ratio": 0.35, "battery_action": 0.0},
        timeout=10,
    )
    step.raise_for_status()

    return {
        "health_status": health.status_code,
        "reset_status": reset.status_code,
        "step_status": step.status_code,
        "cold_start_ms": round(cold_start_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Run API smoke checks against EcoGrid server.")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    result = check(args.base_url.rstrip("/"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
