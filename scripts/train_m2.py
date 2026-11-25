"""Kaggle Dual-T4 & Local CLI Entrypoint for Training Model 2 (Larger PPO MLP)
and Shared Baseline Model 1.

Usage examples:
  # 1. Inspect parameter matching & config on CPU (no GPU required)
  python scripts/train_m2.py --config configs/larger_mlp.yaml --inspect-only

  # 2. Train Model 2 on Kaggle Dual-T4 (using shared baseline setup)
  python scripts/train_m2.py --config configs/larger_mlp.yaml

  # 3. Quick smoke test on Kaggle before full 50M run
  python scripts/train_m2.py --config configs/larger_mlp.yaml --num-timesteps 2000000 --num-evals 4

  # 4. Train Model 1 baseline with the exact same pipeline/config
  python scripts/train_m2.py --config configs/ppo_baseline.yaml
"""

import argparse
import os
from pathlib import Path
import sys

# Configure XLA memory behavior prior to JAX import
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training.config_loader import load_training_config
from src.training.ppo_runner import inspect_model_setup, run_ppo_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Unitree G1 Model 2 (Capacity-Matched Larger MLP) via Brax PPO."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/larger_mlp.yaml",
        help="Path to YAML config (defaults to Model 2 config inheriting ppo_baseline.yaml)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--num-timesteps", type=int, default=None, help="Override total env steps")
    parser.add_argument("--num-envs", type=int, default=None, help="Override parallel envs")
    parser.add_argument("--num-evals", type=int, default=None, help="Override evaluation count")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override PPO learning rate")
    parser.add_argument(
        "--impl",
        type=str,
        default=None,
        choices=["jax", "warp"],
        help="Override MJX physics backend (default: jax for Kaggle T4)",
    )
    parser.add_argument(
        "--terrain-mode",
        type=str,
        default=None,
        choices=["playground", "singular"],
        help="Use upstream Playground task or inject heightfield from configs/terrain.yaml",
    )
    parser.add_argument(
        "--terrain-category",
        type=str,
        default=None,
        choices=["flat", "gentle_slopes", "rolling_hills", "rough_terrain", "steep_slopes_obstacles"],
        help="Singular terrain category (when --terrain-mode singular)",
    )
    parser.add_argument(
        "--terrain-difficulty",
        type=float,
        default=None,
        help="Singular terrain difficulty d in [0.0, 1.0]",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=None,
        help="Reference M3 history length k to match M2 parameter count against",
    )
    parser.add_argument(
        "--no-auto-match",
        action="store_true",
        help="Disable automatic parameter matching and use explicit policy_hidden_layer_sizes from YAML",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicitly override policy hidden layer sizes (e.g. --hidden-sizes 768 512 256)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override results output directory")
    parser.add_argument("--restore", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only inspect resolved architecture and parameter comparison table without training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_training_config(args.config)

    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.num_timesteps is not None:
        cfg.setdefault("ppo", {})["num_timesteps"] = args.num_timesteps
    if args.num_envs is not None:
        cfg.setdefault("ppo", {})["num_envs"] = args.num_envs
    if args.num_evals is not None:
        cfg.setdefault("ppo", {})["num_evals"] = args.num_evals
    if args.learning_rate is not None:
        cfg.setdefault("ppo", {})["learning_rate"] = args.learning_rate
    if args.impl is not None:
        cfg.setdefault("env", {})["impl"] = args.impl
    if args.terrain_mode is not None:
        cfg.setdefault("env", {})["terrain_mode"] = args.terrain_mode
    if args.terrain_category is not None:
        cfg.setdefault("env", {})["terrain_category"] = args.terrain_category
    if args.terrain_difficulty is not None:
        cfg.setdefault("env", {})["terrain_difficulty"] = args.terrain_difficulty
    if args.history_len is not None:
        cfg.setdefault("network", {}).setdefault("capacity_matching", {})["history_length"] = args.history_len
    if args.no_auto_match:
        cfg.setdefault("network", {}).setdefault("capacity_matching", {})["auto_match"] = False
    if args.hidden_sizes is not None:
        cfg.setdefault("network", {})["policy_hidden_layer_sizes"] = list(args.hidden_sizes)
        cfg.setdefault("network", {}).setdefault("capacity_matching", {})["auto_match"] = False

    if args.inspect_only:
        _, summary = inspect_model_setup(cfg)
        print(summary)
        return

    run_ppo_training(
        config=cfg,
        output_dir=args.output_dir,
        restore_checkpoint_path=args.restore,
    )


if __name__ == "__main__":
    main()
