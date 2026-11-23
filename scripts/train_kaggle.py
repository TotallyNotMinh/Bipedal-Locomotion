"""Kaggle Dual-T4 Training Entrypoint for Unitree G1 Locomotion.

Optimized for 2x NVIDIA Tesla T4 (16GB VRAM each).
Supports:
- Model selection: M1 (Baseline), M2 (Large MLP), M3 (History Stacking), M4 (Encoder)
- Procedural multi-terrain heightfield integration
- Multi-device sharded rollouts via JAX
"""

import argparse
import os
from pathlib import Path
import sys

# Support both the documented script command and module execution.
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))


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
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Total parallel environments (sharded across GPUs)")
    parser.add_argument("--history-len", type=int,
                        help="Number of history timesteps (M1: 0, other models default to 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master PRNG seed")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Checkpoint and logs output directory")
    parser.add_argument("--num-timesteps", type=int, help="Exact M1 or M3 environment step budget")
    parser.add_argument("--smoke", action="store_true", help="Run two PPO optimizer updates (pilot default: ten)")
    args = parser.parse_args()
    if args.history_len is None:
        args.history_len = 0 if args.model == "M1" else 5
    if args.model in ("M1", "M3"):
        if args.model == "M1" and args.history_len != 0:
            parser.error("M1 uses only the current observation (--history-len 0)")
        if args.model == "M3" and args.history_len < 1:
            parser.error("M3 history length must be positive")
        if args.category is not None or args.difficulty != 0 or Path(args.terrain_config).resolve() != ROOT / "configs/terrain.yaml":
            parser.error(f"{args.model} uses the established terrain configuration and curriculum")
        if any(v is not None and v <= 0 for v in (args.num_envs, args.num_timesteps)):
            parser.error("Environment count and step budget must be positive")
        if args.smoke and args.num_timesteps is not None:
            parser.error("Choose either --smoke or an explicit --num-timesteps")
        kind = "smoke" if args.smoke else "pilot"
        args.output_dir = args.output_dir or f"results/{args.model.lower()}_{kind}"
        try:
            from src.training.m3_pilot import run
        except ImportError as exc:
            parser.exit(2, f"{args.model} needs the configured Playground/Brax runtime: {exc}\n")
        run(args)
        # Release the trainer's GPU allocations before verifying in a fresh interpreter.
        os.execv(sys.executable, [sys.executable, "-m", "src.training.m3_pilot", args.output_dir])

    if args.num_timesteps is not None or args.smoke:
        parser.error("--num-timesteps and --smoke apply to M1 and M3")
    os.environ.update(XLA_PYTHON_CLIENT_PREALLOCATE="false",
                      XLA_PYTHON_CLIENT_MEM_FRACTION="0.85", TF_FORCE_GPU_ALLOW_GROWTH="true")
    args.num_envs = args.num_envs or 4096
    args.output_dir = args.output_dir or "results/checkpoints"
    try:
        import jax
    except ImportError:
        jax = None
    from src.envs.terrain_utils import export_heightfield_to_mujoco_asset
    from src.envs.g1_terrain_env import UnifiedG1TerrainEnv, G1EnvConfig


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
