---
title: EcoGrid OpenEnv
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# EcoGrid OpenEnv

Production-ready RL environment and API for sustainable grid control.

This repo now ships with:
- Stable OpenEnv API server (`/health`, `/reset`, `/step`, `/schema`)
- Deterministic environment behavior for fixed seeds
- Action safety guards that prevent invalid action sums after rounding
- Hardened deployment path for Hugging Face Spaces (Docker + `uv.lock`)
- Reproducible benchmark script with before/after comparison output

## 1) What The Agent Controls

At each step the agent picks:
- `renewable_ratio` in `[0, 1]`
- `fossil_ratio` in `[0, 1]`
- `battery_action` in `[-1, 1]`

Safety constraint:
- `renewable_ratio + fossil_ratio <= 1.0` (enforced with normalization guards)

## 2) Runtime Modes

- Dashboard mode (HF deployment): `streamlit run app.py`
- API mode (local/service): `python -m server.app`

HF Space Docker deployment serves the Streamlit dashboard.

## 3) Install

### Runtime only
```bash
pip install -r requirements.txt
```

### Runtime + training stack
```bash
pip install -r requirements-train.txt
```

## 4) Reproducible Benchmarks

Run:
```bash
python scripts/benchmark.py --seeds 1,2,3,4,5 --out logs/benchmark_results.json
```

Current post-fix means (5 seeds):
- easy: random `0.2721`, heuristic `0.7595`
- medium: random `0.2545`, heuristic `0.7847`
- hard: random `0.0010`, heuristic `0.4000`

Reference pre-fix means used for delta tracking:
- easy: random `0.269`, heuristic `0.748`
- medium: random `0.251`, heuristic `0.376`
- hard: random `0.001`, heuristic `0.001`

## 5) API Smoke Test

Start server:
```bash
python -m server.app
```

Then:
```bash
python scripts/smoke_api.py --base-url http://127.0.0.1:7860
```

This checks:
- `GET /health`
- `POST /reset`
- `POST /step` (direct action payload compatibility)

## 6) Test Suite

```bash
pytest -q
```

Coverage includes:
- environment unit tests
- reward/grader tests
- action normalization regression tests
- API smoke/integration tests
- reproducibility checks (same seed => same trajectory)

## 7) Hugging Face Space Deployment

### Lock strategy
- Runtime dependencies live in `pyproject.toml` default deps.
- Heavy training deps are optional (`[project.optional-dependencies].train`).
- `uv.lock` is committed and used with `--frozen`.

Regenerate lock file:
```bash
uv lock
```

### Docker (Space) build path
The committed `Dockerfile` does:
1. `uv sync --frozen --no-dev --no-install-project`
2. copy source
3. `uv sync --frozen --no-dev`
4. run `/opt/venv/bin/python -m server.app`

Healthcheck:
- container-level health probe hits `http://127.0.0.1:7860/health`

## 8) Minimal Deployment Checklist

- `uv lock` succeeds locally
- `uv sync --frozen --no-dev` succeeds locally
- `pytest -q` passes
- `python scripts/smoke_api.py` passes against local server
- Docker build succeeds
- Space runtime reports healthy and serves `/health`, `/schema`, `/docs`

## 9) Project Structure

- `env/` core environment dynamics, rewards, action safety helpers
- `server/` OpenEnv/FastAPI serving layer
- `baseline.py` heuristic + optional LLM baseline runner
- `train_unsloth.py` deterministic training pipeline and metric logging
- `scripts/benchmark.py` reproducible benchmark runner
- `scripts/smoke_api.py` deployment smoke test runner
