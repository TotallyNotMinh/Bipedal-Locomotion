# Repository Guidelines

## Project Structure & Module Organization

- `src/envs/` contains terrain generation, curriculum progression, history buffers, and the unified G1 environment.
- `configs/terrain.yaml` controls terrain geometry and curriculum parameters. `scripts/` contains training, benchmarking, visualization, and depth-map utilities.
- `tests/` contains Python tests; `assets/` holds documentation images. Experiment outputs belong in `results/`, which is ignored by Git.
- `README.md` describes the environment and API; `KAGGLE_GUIDE.md` covers the GPU setup and `PLAN-1.md` documents the research plan.

## Build, Test, and Development Commands

Run commands from the repository root so `src` imports resolve. Install the lightweight test dependencies with `python -m pip install numpy pyyaml pytest`.

- `python -m pytest tests/` runs the test suite; use `python -m pytest tests/test_curriculum.py` for a focused run.
- `python scripts/view_terrain_mujoco.py --category rolling_hills --difficulty 1.0` opens the terrain viewer when MuJoCo and a graphical environment are available.
- `bash setup_kaggle.sh` installs the GPU training stack for Kaggle. See `KAGGLE_GUIDE.md` before running `python scripts/benchmark_gpu.py` or `python scripts/train_kaggle.py`.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for modules, functions, and variables, and `PascalCase` for classes. Use descriptive names, type hints, and concise docstrings where useful. Keep terrain category names consistent with `configs/terrain.yaml`. No repository-wide formatter or linter configuration is present.

## Testing Guidelines

Tests use pytest-style `test_*.py` files and `test_*` functions with NumPy assertions. Add focused tests for behavior changes, especially deterministic terrain generation, curriculum transitions, history causality, and uphill constraints. Use explicit seeds for reproducibility. No coverage threshold is configured.

## Commit & Pull Request Guidelines

Recent commits use concise conventional prefixes such as `feat:` and `fix(readme):`; follow that pattern with an imperative summary. For pull requests, describe the behavior and motivation, list the commands run and their results, and link any relevant issue. Include images when terrain visualizations or other visible output changes.
