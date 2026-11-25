"""Record an MP4 demo video of the Unitree G1 humanoid walking with trained Model 2 policy."""

import argparse
import glob
import os
from pathlib import Path
import sys

# Configure headless rendering before MuJoCo import
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from src.training.config_loader import load_training_config
from src.training.env_builder import build_g1_environment
from src.models.mlp_policy import build_model_spec_from_config, make_model_network_factory
from src.training.ppo_runner import (
    _ensure_jax_brax_compat,
    _load_policy_inference_fn,
    _resolve_latest_checkpoint,
)


def record_demo_video(
    config_path: str = "configs/larger_mlp.yaml",
    checkpoint_dir: str = None,
    output_path: str = "demo_m2_walking.mp4",
    num_steps: int = 500,
    camera_name: str = "track",
    fps: int = 30,
) -> str:
    """Executes a rollout with the trained policy and renders an MP4 video."""
    import jax
    import mujoco
    from mujoco import mjx
    try:
        import imageio
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio", "imageio-ffmpeg"])
        import imageio

    _ensure_jax_brax_compat()
    cfg = load_training_config(config_path)

    # 1. Resolve checkpoint directory
    if checkpoint_dir is None:
        run_dirs = sorted(glob.glob("results/m2_official/g1_m2_larger_mlp_*"))
        if not run_dirs:
            raise FileNotFoundError("Chưa thấy thư mục run trong results/m2_official!")
        checkpoint_dir = os.path.join(run_dirs[-1], "checkpoints")

    ckpt = _resolve_latest_checkpoint(checkpoint_dir)
    print(f"[+] Sử dụng checkpoint: {ckpt}")

    # 2. Build environment and load policy
    env, _, _ = build_g1_environment(cfg)
    obs_size = env.observation_size
    action_size = int(env.action_size)
    has_dict_obs = isinstance(obs_size, dict)

    spec = build_model_spec_from_config(cfg, obs_size, action_size)
    network_factory = make_model_network_factory(spec, has_dict_obs=has_dict_obs)
    policy_fn = _load_policy_inference_fn(
        checkpoint_path=ckpt,
        network_factory=network_factory,
        obs_size=obs_size,
        action_size=action_size,
        deterministic=True,
    )

    jit_policy = jax.jit(policy_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # 3. Rollout episode trajectory
    print(f"[+] Đang chạy mô phỏng {num_steps} bước...")
    key = jax.random.PRNGKey(42)
    key, reset_key = jax.random.split(key)
    state = jit_reset(reset_key)

    trajectory = [state.data]
    total_reward = 0.0

    for t in range(num_steps):
        key, act_key = jax.random.split(key)
        action, _ = jit_policy(state.obs, act_key)
        state = jit_step(state, action)
        trajectory.append(state.data)
        total_reward += float(state.reward)

        if bool(state.done):
            print(f"[!] Robot ngã hoặc kết thúc episode tại bước {t + 1}/{num_steps}")
            break

    print(f"[✓] Đã thu thập {len(trajectory)} khung hình mô phỏng. Tổng reward: {total_reward:.2f}")

    # 4. Render RGB frames via MuJoCo
    print("[+] Đang render khung hình sang video...")
    mj_model = getattr(env, "mj_model", getattr(env, "_mj_model", None))
    if mj_model is None:
        raise AttributeError("Không tìm thấy mj_model trên environment!")

    try:
        renderer = mujoco.Renderer(mj_model, height=480, width=640)
    except Exception:
        os.environ["MUJOCO_GL"] = "osmesa"
        renderer = mujoco.Renderer(mj_model, height=480, width=640)

    # Detect camera
    cam_id = -1
    if camera_name:
        try:
            cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        except Exception:
            cam_id = -1

    if cam_id == -1 and mj_model.ncam > 0:
        cam_id = 0

    mj_data = mujoco.MjData(mj_model)
    frames = []

    for d in trajectory:
        try:
            mjx.get_data_into(result=mj_data, model=mj_model, data=d)
        except Exception:
            mj_data.qpos[:] = np.asarray(d.qpos)
            mj_data.qvel[:] = np.asarray(d.qvel)
            mujoco.mj_forward(mj_model, mj_data)

        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)

        rgb = renderer.render()
        frames.append(rgb)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(str(out_file), frames, fps=fps, macro_block_size=None)
    print(f"[✓] Video đã được lưu thành công tại: {out_file} ({len(frames)} frames @ {fps} fps)")
    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render G1 demo video from trained checkpoint.")
    parser.add_argument("--config", type=str, default="configs/larger_mlp.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="demo_m2_walking.mp4")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera", type=str, default="track")
    args = parser.parse_args()

    record_demo_video(
        config_path=args.config,
        checkpoint_dir=args.checkpoint,
        output_path=args.output,
        num_steps=args.steps,
        camera_name=args.camera,
        fps=args.fps,
    )
