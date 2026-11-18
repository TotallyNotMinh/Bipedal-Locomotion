"""Kaggle Dual-T4 Training Entrypoint for Unitree G1 Locomotion.

Optimized for 2x NVIDIA Tesla T4 (16GB VRAM each).
Supports:
- Model selection: M1 (Baseline), M2 (Large MLP), M3 (History Stacking), M4 (Encoder)
- Procedural multi-terrain heightfield integration
- Multi-device sharded rollouts via JAX
"""

import os
import sys
import argparse
from pathlib import Path

# Configure XLA memory management before JAX imports
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None
import numpy as np

from src.envs.terrain import TerrainGenerator
from src.envs.terrain_utils import export_heightfield_to_mujoco_asset
from src.envs.g1_terrain_env import UnifiedG1TerrainEnv, G1EnvConfig


def main():
    parser = argparse.ArgumentParser(description="Train Unitree G1 Locomotion on Kaggle Dual-T4")
    parser.add_argument("--model", type=str, default="M3", choices=["M1", "M2", "M3", "M4"],
                        help="Policy architecture: M1, M2, M3, or M4")
    parser.add_argument("--category", type=str, default=None,
                        choices=["flat", "gentle_slopes", "rolling_hills", "rough_terrain", "steep_slopes_obstacles"],
                        help="Singular terrain category for training")
    parser.add_argument("--difficulty", type=float, default=0.0,
                        help="Initial curriculum difficulty d in [0, 1]")
    parser.add_argument("--terrain-config", type=str, default="configs/terrain.yaml",
                        help="Path to terrain configuration YAML")
    parser.add_argument("--num-envs", type=int, default=4096,
                        help="Total parallel environments (sharded across GPUs)")
    parser.add_argument("--history-len", type=int, default=5,
                        help="Number of history timesteps (k)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master PRNG seed")
    parser.add_argument("--output-dir", type=str, default="results/checkpoints",
                        help="Checkpoint and logs output directory")
    args = parser.parse_args()

    print("==================================================================")
    print(" Unitree G1 Adaptive Locomotion Training (Kaggle Dual-T4)")
    print("==================================================================")

    # 1. Device Inspection
    if jax is not None:
        devices = jax.devices()
        print(f"[+] Active JAX Devices ({len(devices)}):")
        for i, dev in enumerate(devices):
            print(f"    GPU {i}: {dev.device_kind} (platform={dev.platform}, id={dev.id})")

        if len(devices) >= 2:
            print(f"[✓] Dual-device sharding enabled: {args.num_envs // 2} envs per T4 GPU.")
        else:
            print(f"[!] Running on {len(devices)} device(s): {args.num_envs} envs per device.")
    else:
        print("[!] JAX not installed in current Python environment. Running in CPU/NumPy fallback mode.")

    # 2. Build Terrain and Export Heightfield
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[+] Generating singular terrain environment...")
    env_cfg = G1EnvConfig(
        model_id=args.model,
        category=args.category,
        difficulty=args.difficulty,
        terrain_config_path=args.terrain_config,
        num_envs=args.num_envs,
        history_length=args.history_len,
    )
    env = UnifiedG1TerrainEnv(env_cfg)

    hfield_bin, hfield_attrs = export_heightfield_to_mujoco_asset(
        arena=env.terrain_arena,
        output_dir=out_dir / "terrain_assets",
        asset_name=f"g1_{env.terrain_arena.category}_hfield",
    )
    print(f"[✓] Singular terrain environment generated:")
    print(f"    Category:   {env.terrain_arena.category}")
    print(f"    Difficulty: d = {env.terrain_arena.difficulty:.2f}")
    print(f"    Dimensions: {env.terrain_arena.size_x:.1f}m x {env.terrain_arena.size_y:.1f}m")
    print(f"    Grid:       {hfield_attrs['nrow']} x {hfield_attrs['ncol']} points")
    print(f"    Asset:      {hfield_bin}")

    # 3. Environment & Observation Setup
    initial_obs = env.reset()
    if isinstance(initial_obs, tuple):
        print(f"[✓] Model: {args.model} (Encoder Mode) - Obs: {initial_obs[0].shape}, Hist: {initial_obs[1][0].shape}")
    else:
        print(f"[✓] Model: {args.model} - Input shape: {initial_obs.shape}")

    print("\n[+] Ready for PPO rollouts and policy training!")
    print("==================================================================")


if __name__ == "__main__":
    main()
