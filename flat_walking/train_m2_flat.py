"""Train M2 (larger MLP, input o_t) cho G1 trên mặt phẳng của Minh.

Dùng:
    python flat_walking/train_m2_flat.py --smoke      # kiểm tra nhanh env (vài giây)
    python flat_walking/train_m2_flat.py              # train đầy đủ

Chạy từ thư mục gốc repo. Không sửa file nào của Minh; xem g1_minh_env.py.
"""

import argparse
import functools
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))            # để import src.envs của Minh
sys.path.insert(0, str(Path(__file__).resolve().parent))  # g1_minh_env

import numpy as np
import yaml


def _patch_jax():
    """JAX mới đã bỏ device_put_replicated/sharded mà Brax còn gọi."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    def _has(name):
        try:
            getattr(jax, name)
            return True
        except AttributeError:
            return False

    if not _has("device_put_replicated"):
        def _replicated(x, devices):
            sharding = NamedSharding(Mesh(np.array(devices), ("x",)), P("x"))
            return jax.tree.map(
                lambda y: jax.device_put(jnp.stack([jnp.asarray(y)] * len(devices)), sharding), x)
        jax.device_put_replicated = _replicated
    if not _has("device_put_sharded"):
        def _sharded(shards, devices):
            sharding = NamedSharding(Mesh(np.array(devices), ("x",)), P("x"))
            return jax.tree.map(
                lambda *xs: jax.device_put(jnp.stack([jnp.asarray(x) for x in xs]), sharding), *shards)
        jax.device_put_sharded = _sharded
    if not _has("tree_map"):
        jax.tree_map = jax.tree.map


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return o if isinstance(o, (str, type(None))) else str(o)


def _mlp_params(in_dim, hidden, out_dim):
    n, d = 0, int(in_dim)
    for h in list(hidden) + [int(out_dim)]:
        n += d * int(h) + int(h)
        d = int(h)
    return n


def build_minh_env(cfg):
    """Tạo UnifiedG1TerrainEnv của Minh y như scripts/train_kaggle.py."""
    from src.envs.g1_terrain_env import G1EnvConfig, UnifiedG1TerrainEnv

    m = cfg["minh"]
    env_cfg = G1EnvConfig(
        model_id=cfg["experiment"]["model_id"],
        category=m["category"],
        difficulty=m["difficulty"],
        terrain_config_path=str(ROOT_DIR / m["terrain_config_path"]),
        num_envs=int(m["num_envs"]),
    )
    return UnifiedG1TerrainEnv(env_cfg)


def _network_factory(cfg, obs_size):
    import flax.linen as nn
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    net = cfg["network"]
    valid = set(inspect.signature(ppo_networks.make_ppo_networks).parameters)
    extra = {}
    if isinstance(obs_size, dict):
        if "policy_obs_key" in valid:
            extra["policy_obs_key"] = net["policy_obs_key"]
        if "value_obs_key" in valid:
            extra["value_obs_key"] = net["value_obs_key"]
    if "distribution_type" in net and "distribution_type" in valid:
        extra["distribution_type"] = net["distribution_type"]
    act = getattr(nn, net.get("activation", "swish"))

    def make(observation_size, action_size,
             preprocess_observations_fn=running_statistics.normalize):
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size, action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=tuple(net["policy_hidden_layer_sizes"]),
            value_hidden_layer_sizes=tuple(net["value_hidden_layer_sizes"]),
            activation=act, **extra)
    return make


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke(cfg):
    """Kiểm tra nhanh: env dựng được, reset/step chạy, spawn đúng, dữ liệu hợp lệ."""
    import jax
    from g1_minh_env import G1MinhTerrainEnv

    minh_env = build_minh_env(cfg)
    arena = minh_env.terrain_arena
    print(f"[smoke] Minh terrain: {arena.category} d={arena.difficulty} "
          f"{arena.size_x}x{arena.size_y} m, grid={arena.heights.shape}, "
          f"height=[{arena.heights.min():.3f}, {arena.heights.max():.3f}]")
    env = G1MinhTerrainEnv(minh_env, impl="jax")
    m = env.mj_model
    import mujoco
    floor_type = mujoco.mjtGeom(m.geom_type[env._floor_geom_id]).name
    print(f"[smoke] nu={env.action_size} nhfield={m.nhfield} "
          f"hfield_size={m.hfield_size[0].tolist()} floor_geom_type={floor_type}")
    assert floor_type == "mjGEOM_HFIELD", "geom floor chưa được thay bằng heightfield"
    print(f"[smoke] obs_size={env.observation_size}")
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    q = np.asarray(state.data.qpos)
    print(f"[smoke] spawn xyz=({q[0]:.2f}, {q[1]:.2f}, {q[2]:.3f})  "
          f"(arena {arena.size_x}x{arena.size_y})  command={np.asarray(state.info['command'])}")
    assert 0.25 * arena.size_x <= q[0] <= 0.75 * arena.size_x + 0.5
    for i in range(20):
        state = step(state, jax.numpy.zeros(env.action_size))
    print(f"[smoke] sau 20 bước: z={float(state.data.qpos[2]):.3f} "
          f"reward={float(state.reward):.4f} done={float(state.done)}")
    assert np.isfinite(np.asarray(state.data.qpos)).all()
    print("[smoke] OK")


def train(cfg, output_dir):
    import jax
    _patch_jax()
    from brax.training.agents.ppo import train as ppo_train
    from mujoco_playground import wrapper
    from g1_minh_env import G1MinhTerrainEnv

    exp, ppo = cfg["experiment"], cfg["ppo"]
    seed = int(exp["seed"])
    devices = jax.devices()
    print(f"[+] JAX devices ({len(devices)}): {[str(d) for d in devices]}")

    minh_env = build_minh_env(cfg)
    arena = minh_env.terrain_arena
    mcfg = minh_env.config
    num_envs = int(mcfg.num_envs)
    print(f"[+] Minh env: category={arena.category} d={arena.difficulty} "
          f"arena={arena.size_x}x{arena.size_y}m grid={arena.heights.shape} "
          f"num_envs={num_envs}")
    print(f"[+] Minh commands: vx={mcfg.command_vx_range} vy={mcfg.command_vy_range} "
          f"wz={mcfg.command_wz_range}  push={mcfg.push_enabled} "
          f"friction_switch={mcfg.friction_switch_enabled}")

    train_env = G1MinhTerrainEnv(minh_env, impl="jax")
    eval_env = G1MinhTerrainEnv(minh_env, impl="jax")
    obs_size, action_size = train_env.observation_size, int(train_env.action_size)
    pcfg = train_env._config
    print(f"[+] obs_size={obs_size} action_size={action_size} "
          f"episode_length={pcfg.episode_length}")

    net = cfg["network"]
    p_in = int(np.prod(obs_size[net["policy_obs_key"]]))
    v_in = int(np.prod(obs_size[net["value_obs_key"]]))
    params = {"policy_input_dim": p_in, "value_input_dim": v_in,
              "policy_params": _mlp_params(p_in, net["policy_hidden_layer_sizes"], 2 * action_size),
              "value_params": _mlp_params(v_in, net["value_hidden_layer_sizes"], 1)}
    print(f"[+] {exp['model_id']} params: {params}")

    run_name = f"{exp['name']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir).resolve() / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    terrain_yaml = ROOT_DIR / cfg["minh"]["terrain_config_path"]
    resolved = {
        **cfg,
        "resolved_from_minh": {
            "num_envs": num_envs, "category": arena.category, "difficulty": arena.difficulty,
            "command_vx_range": list(mcfg.command_vx_range),
            "command_vy_range": list(mcfg.command_vy_range),
            "command_wz_range": list(mcfg.command_wz_range),
            "control_dt": mcfg.control_dt, "physics_dt": mcfg.physics_dt,
            "push_enabled": mcfg.push_enabled,
            "friction_switch_enabled": mcfg.friction_switch_enabled,
            "nominal_friction": mcfg.nominal_friction,
            "terrain_yaml_sha256": _sha(terrain_yaml),
        },
        "git_commit": _git_commit(),
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(_json_safe(resolved), sort_keys=False))
    print(f"[+] Output: {run_dir}")

    t0 = time.perf_counter()
    metrics_file = run_dir / "metrics.jsonl"

    def progress_fn(num_steps, metrics):
        elapsed = time.perf_counter() - t0
        row = {"step": num_steps, "wall_time_s": round(elapsed, 2)}
        row.update({k: _json_safe(v) for k, v in metrics.items()})
        with open(metrics_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  step={num_steps:>11,}  elapsed={elapsed:6.0f}s  "
              f"sps={num_steps / max(elapsed, 1e-6):>9,.0f}  "
              f"eval_reward={row.get('eval/episode_reward', float('nan')):7.3f}  "
              f"eval_len={row.get('eval/avg_episode_length', float('nan')):6.1f}")

    kwargs = {
        "num_timesteps": int(ppo["num_timesteps"]), "num_evals": int(ppo["num_evals"]),
        "num_resets_per_eval": int(ppo["num_resets_per_eval"]),
        "reward_scaling": float(ppo["reward_scaling"]),
        "episode_length": int(pcfg.episode_length),
        "normalize_observations": bool(ppo["normalize_observations"]),
        "action_repeat": int(pcfg.action_repeat),
        "unroll_length": int(ppo["unroll_length"]),
        "num_minibatches": int(ppo["num_minibatches"]),
        "num_updates_per_batch": int(ppo["num_updates_per_batch"]),
        "discounting": float(ppo["discounting"]), "learning_rate": float(ppo["learning_rate"]),
        "entropy_cost": float(ppo["entropy_cost"]), "num_envs": num_envs,
        "batch_size": int(ppo["batch_size"]), "max_grad_norm": float(ppo["max_grad_norm"]),
        "clipping_epsilon": float(ppo["clipping_epsilon"]), "gae_lambda": float(ppo["gae_lambda"]),
        "normalize_advantage": bool(ppo["normalize_advantage"]),
        "deterministic_eval": bool(ppo["deterministic_eval"]),
        "num_eval_envs": int(ppo["num_eval_envs"]), "seed": seed,
        "network_factory": _network_factory(cfg, obs_size),
        "wrap_env_fn": wrapper.wrap_for_brax_training,
        "save_checkpoint_path": str(ckpt_dir),
    }
    if ppo.get("restore_checkpoint_path"):
        p = Path(ppo["restore_checkpoint_path"]).resolve()
        nums = sorted((d for d in p.iterdir() if d.is_dir() and d.name.isdigit()), key=lambda d: int(d.name))
        kwargs["restore_checkpoint_path"] = str(nums[-1] if nums else p)
        print(f"[+] Resume từ: {kwargs['restore_checkpoint_path']}")

    if (kwargs["batch_size"] * kwargs["num_minibatches"]) % num_envs:
        raise ValueError("batch_size*num_minibatches phải chia hết cho num_envs")
    if num_envs % len(devices):
        raise ValueError(f"num_envs phải chia hết cho số GPU ({len(devices)})")
    valid = set(inspect.signature(ppo_train.train).parameters)
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"[!] Brax PPO không hỗ trợ, bỏ qua: {dropped}")
    kwargs = {k: v for k, v in kwargs.items() if k in valid}

    print("[+] Starting PPO training...")
    _, _, final_metrics = functools.partial(ppo_train.train, **kwargs)(
        environment=train_env, eval_env=eval_env, progress_fn=progress_fn)

    total = time.perf_counter() - t0
    print(f"\n[✓] Done in {total:.0f}s. Checkpoint: {ckpt_dir}")
    meta = {"run_name": run_name, "seed": seed, "model_id": exp["model_id"],
            "param_counts": params, "checkpoint_dir": str(ckpt_dir),
            "total_wall_time_s": round(total, 2), "final_metrics": _json_safe(final_metrics)}
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "m2_flat.yaml"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--num-timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="chỉ kiểm tra nhanh env, không train")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.num_timesteps is not None:
        cfg["ppo"]["num_timesteps"] = a.num_timesteps
    if a.seed is not None:
        cfg["experiment"]["seed"] = a.seed
    if a.smoke:
        smoke(cfg)
    else:
        train(cfg, a.output_dir or cfg["experiment"]["output_dir"])


if __name__ == "__main__":
    main()
