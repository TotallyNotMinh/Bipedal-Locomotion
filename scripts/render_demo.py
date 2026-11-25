"""Render video demo từ checkpoint đã train.

Chạy độc lập trong subprocess để tránh OOM với JAX kernel cũ.
Usage:
    python scripts/render_demo.py --checkpoint <path> --out /kaggle/working/g1_flat_demo.mp4
"""
import argparse
import gc
import inspect
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import flax.linen as nn
import imageio
import jax
import mujoco
import numpy as np
import yaml
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from mujoco import mjx
from mujoco_playground import locomotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="/kaggle/working/g1_flat_demo.mp4")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    print(f"[+] Checkpoint: {ckpt_path}")
    print(f"[+] JAX devices: {jax.devices()}")

    # Config
    if args.config:
        cfg_path = Path(args.config)
    else:
        # tìm resolved_config.yaml trong run dir
        run_dir = ckpt_path.parent.parent
        cfg_path = run_dir / "resolved_config.yaml"
        if not cfg_path.exists():
            cfg_path = Path(__file__).parent.parent / "configs" / "ppo_flat.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f"[+] Config: {cfg_path}")

    # Env
    task_name = cfg.get("env", {}).get("task_name", "G1JoystickFlatTerrain")
    pg_cfg = locomotion.get_default_config(task_name)
    if hasattr(pg_cfg, "impl"):
        pg_cfg.impl = "jax"
    env = locomotion.load(task_name, config=pg_cfg)
    print(f"[+] Env loaded: obs={env.observation_size} action={env.action_size}")

    # Network factory
    net_cfg = cfg.get("network", {})
    policy_hidden = tuple(net_cfg.get("policy_hidden_layer_sizes", [512, 256, 128]))
    value_hidden  = tuple(net_cfg.get("value_hidden_layer_sizes",  [512, 256, 128]))
    activation    = getattr(nn, net_cfg.get("activation", "swish"), nn.swish)
    valid_net_params = set(inspect.signature(ppo_networks.make_ppo_networks).parameters.keys())
    print(f"[+] Policy arch: hidden={policy_hidden}")

    def _make_factory(v_key):
        defaults = {
            "policy_hidden_layer_sizes": policy_hidden,
            "value_hidden_layer_sizes":  value_hidden,
            "activation": activation,
        }
        if isinstance(env.observation_size, dict):
            if "policy_obs_key" in valid_net_params:
                defaults["policy_obs_key"] = net_cfg.get("policy_obs_key", "state")
            if "value_obs_key" in valid_net_params and v_key in env.observation_size:
                defaults["value_obs_key"] = v_key
        if "distribution_type" in net_cfg and "distribution_type" in valid_net_params:
            defaults["distribution_type"] = net_cfg["distribution_type"]

        def make_networks(*a, **kw):
            m = dict(defaults)
            m.update({k: v for k, v in kw.items() if k in valid_net_params})
            return ppo_networks.make_ppo_networks(*a, **m)
        return make_networks

    policy_fn = None
    for v_key in (net_cfg.get("value_obs_key", "privileged_state"), "state"):
        try:
            policy_fn = ppo_checkpoint.load_policy(
                str(ckpt_path), network_factory=_make_factory(v_key), deterministic=True
            )
            print(f"[✓] Policy loaded (value_obs_key='{v_key}')")
            break
        except Exception as e:
            print(f"  load_policy({v_key}) failed: {e}")

    if policy_fn is None:
        sys.exit("ERROR: không load được policy")

    jit_policy = jax.jit(policy_fn)
    jit_reset  = jax.jit(env.reset)
    jit_step   = jax.jit(env.step)

    # Renderer
    mj_model = getattr(env, "mj_model", getattr(env, "_mj_model", None))
    renderer = None
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        os.environ["PYOPENGL_PLATFORM"] = backend
        try:
            renderer = mujoco.Renderer(mj_model, height=360, width=480)
            print(f"[✓] Renderer OK ({backend})")
            break
        except Exception as e:
            print(f"  [{backend}] failed: {e}")

    if renderer is None:
        sys.exit("ERROR: không khởi tạo được renderer")

    cam_id = -1
    for cname in ("track", "side", "back"):
        try:
            cid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cname)
            if cid >= 0:
                cam_id = cid
                break
        except Exception:
            pass
    if cam_id < 0 and mj_model.ncam > 0:
        cam_id = 0

    mj_data = mujoco.MjData(mj_model)

    def _render_frame(state_data):
        try:
            mjx.get_data_into(result=mj_data, model=mj_model, data=state_data)
        except Exception:
            mj_data.qpos[:] = np.asarray(state_data.qpos)
            mj_data.qvel[:] = np.asarray(state_data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
        if cam_id >= 0:
            renderer.update_scene(mj_data, camera=cam_id)
        else:
            renderer.update_scene(mj_data)
        return renderer.render().copy()

    # Rollout
    key = jax.random.PRNGKey(0)
    key, rk = jax.random.split(key)
    state = jit_reset(rk)
    frames = [_render_frame(state.data)]

    print(f"[+] Simulating {args.steps} steps...")
    for t in range(args.steps):
        key, ak = jax.random.split(key)
        action, _ = jit_policy(state.obs, ak)
        state = jit_step(state, action)
        frames.append(_render_frame(state.data))
        if bool(state.done):
            print(f"  Episode ended at step {t+1}")
            break

    renderer.close()
    gc.collect()
    print(f"[+] {len(frames)} frames collected")

    out_path = args.out
    imageio.mimsave(out_path, frames, fps=args.fps, macro_block_size=None)
    del frames
    gc.collect()
    print(f"[✓] Video saved: {out_path}")


if __name__ == "__main__":
    main()
