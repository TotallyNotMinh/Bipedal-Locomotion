"""Evaluation script for trained Model 2 (and M1) checkpoints.

Supports Phase 6 & 6A benchmarks from PLAN-1.md:
- `nominal`: baseline flat/task locomotion
- `friction_switch`: mid-episode contact friction switch (e.g., 1.0 -> 0.2 at step 250)
- `held_out_friction`: fixed unseen friction coefficient throughout the episode
- `push`: mid-episode external base velocity impulse
- `terrain`: evaluation on singular terrains from configs/terrain.yaml

Usage examples:
  python scripts/eval_m2.py --config configs/larger_mlp.yaml --checkpoint results/run/checkpoints --scenario nominal
  python scripts/eval_m2.py --config configs/larger_mlp.yaml --checkpoint results/run/checkpoints --scenario friction_switch --switched-friction 0.2
  python scripts/eval_m2.py --config configs/larger_mlp.yaml --checkpoint results/run/checkpoints --scenario push --push-vx 0.6 --push-vy 0.4
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.config_loader import load_training_config
from src.training.ppo_runner import (
    _ensure_jax_brax_compat,
    _to_json_safe,
    evaluate_policy_rollouts,
)

_ensure_jax_brax_compat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Model 2 / Model 1 PPO checkpoints.")
    parser.add_argument("--config", type=str, default="configs/larger_mlp.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path or checkpoints/ dir")
    parser.add_argument(
        "--scenario",
        type=str,
        default="nominal",
        choices=["nominal", "friction_switch", "held_out_friction", "push", "terrain"],
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stochastic", action="store_true", help="Sample stochastic policy actions")
    parser.add_argument("--switched-friction", type=float, default=0.2)
    parser.add_argument("--switch-step", type=int, default=250)
    parser.add_argument("--push-step", type=int, default=300)
    parser.add_argument("--push-vx", type=float, default=0.6)
    parser.add_argument("--push-vy", type=float, default=0.4)
    parser.add_argument(
        "--terrain-category",
        type=str,
        default=None,
        choices=["flat", "gentle_slopes", "rolling_hills", "rough_terrain", "steep_slopes_obstacles"],
    )
    parser.add_argument("--terrain-difficulty", type=float, default=None)
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_training_config(args.config)

    if args.scenario == "terrain":
        cfg.setdefault("env", {})["terrain_mode"] = "singular"

    results = evaluate_policy_rollouts(
        config=cfg,
        checkpoint_path=args.checkpoint,
        scenario=args.scenario,
        num_episodes=args.num_episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        terrain_category=args.terrain_category,
        terrain_difficulty=args.terrain_difficulty,
        switched_friction=args.switched_friction,
        switch_step=args.switch_step,
        push_step=args.push_step,
        push_vel_xy=(args.push_vx, args.push_vy),
    )

    print("\n================ Evaluation Summary ================")
    for key in (
        "scenario",
        "model_id",
        "num_episodes",
        "mean_return",
        "std_return",
        "mean_length",
        "success_rate",
        "fall_rate",
        "mean_vel_error",
        "mean_immediate_post_error",
        "mean_integrated_post_error",
        "mean_recovery_time_steps",
        "recovery_rate",
        "censored_no_recovery_count",
    ):
        print(f"  {key:<28}: {results.get(key)}")
    print("====================================================\n")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(_to_json_safe(results), f, indent=2)
        print(f"[✓] Saved evaluation JSON to: {out}")


if __name__ == "__main__":
    main()
