"""Train Unitree G1 on flat terrain using MuJoCo Playground + Brax PPO.

Usage:
    python scripts/train_flat.py --config configs/ppo_flat.yaml
"""

import os
import sys
import json
import time
import argparse
import functools
import inspect
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import yaml
import numpy as np


# ---------------------------------------------------------------------------
# JAX / Brax compat shims (handles removed pmap helpers in JAX 0.9+/0.10+)
# ---------------------------------------------------------------------------

def _patch_jax():
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
            mesh = Mesh(np.array(devices), ("x",))
            sharding = NamedSharding(mesh, P("x"))
            return jax.tree.map(
                lambda y: jax.device_put(jnp.stack([jnp.asarray(y)] * len(devices)), sharding), x
            )
        setattr(jax, "device_put_replicated", _replicated)

    if not _has("device_put_sharded"):
        def _sharded(shards, devices):
            mesh = Mesh(np.array(devices), ("x",))
            sharding = NamedSharding(mesh, P("x"))
            return jax.tree.map(
                lambda *xs: jax.device_put(jnp.stack([jnp.asarray(x) for x in xs]), sharding), *shards
            )
        setattr(jax, "device_put_sharded", _sharded)

    if not _has("tree_map"):
        setattr(jax, "tree_map", jax.tree.map)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj) if not isinstance(obj, (str, type(None))) else obj


def _build_network_factory(cfg: dict, obs_size, action_size: int):
    """Builds a simple MLP policy/value factory compatible with Brax PPO."""
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    import flax.linen as nn

    net_cfg = cfg.get("network", {})
    policy_hidden = tuple(net_cfg.get("policy_hidden_layer_sizes", [512, 256, 128]))
    value_hidden = tuple(net_cfg.get("value_hidden_layer_sizes", [512, 256, 128]))
    activation_name = net_cfg.get("activation", "swish")
    activation = getattr(nn, activation_name, nn.swish)

    valid_net_params = set(inspect.signature(ppo_networks.make_ppo_networks).parameters.keys())
    extra_kwargs = {}
    if isinstance(obs_size, dict):
        p_key = net_cfg.get("policy_obs_key", "state")
        v_key = net_cfg.get("value_obs_key", "privileged_state")
        if p_key in obs_size and "policy_obs_key" in valid_net_params:
            extra_kwargs["policy_obs_key"] = p_key
        if v_key in obs_size and "value_obs_key" in valid_net_params:
            extra_kwargs["value_obs_key"] = v_key
        elif p_key in obs_size and "value_obs_key" in valid_net_params:
            extra_kwargs["value_obs_key"] = p_key
    if "distribution_type" in net_cfg and "distribution_type" in valid_net_params:
        extra_kwargs["distribution_type"] = net_cfg["distribution_type"]

    def make_networks(observation_size, action_size, preprocess_observations_fn=running_statistics.normalize):
        return ppo_networks.make_ppo_networks(
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            policy_hidden_layer_sizes=policy_hidden,
            value_hidden_layer_sizes=value_hidden,
            activation=activation,
            **extra_kwargs,
        )

    return make_networks


def resolve_latest_checkpoint(checkpoint_path: str | Path) -> Path:
    """Resolves the latest numbered step subfolder inside a Brax `checkpoints/` directory."""
    p = Path(checkpoint_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    digit_dirs = [d for d in subdirs if d.name.isdigit()]
    if digit_dirs:
        digit_dirs.sort(key=lambda d: int(d.name))
        return digit_dirs[-1]
    return p


def _get_domain_randomizer(locomotion, task_name: str):
    try:
        return locomotion.get_domain_randomizer(task_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: dict, output_dir: str):
    import jax
    _patch_jax()

    from brax.training.agents.ppo import train as ppo_train
    from mujoco_playground import locomotion
    from mujoco_playground import wrapper

    devices = jax.devices()
    print(f"[+] JAX devices ({len(devices)}): {[str(d) for d in devices]}")

    exp_cfg = cfg.get("experiment", {})
    ppo_cfg = cfg.get("ppo", {})
    env_cfg = cfg.get("env", {})

    seed = int(exp_cfg.get("seed", 42))
    run_name = str(exp_cfg.get("name", "g1_flat"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    out = Path(output_dir).resolve()
    run_dir = out / f"{run_name}_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Output: {run_dir}")
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    task_name = str(env_cfg.get("task_name", "G1JoystickFlatTerrain"))
    pg_cfg = locomotion.get_default_config(task_name)

    for key in (
        "impl", "episode_length", "action_repeat", "ctrl_dt", "sim_dt",
        "lin_vel_x", "lin_vel_y", "ang_vel_yaw",
    ):
        if key in env_cfg and hasattr(pg_cfg, key):
            setattr(pg_cfg, key, env_cfg[key])

    train_env = locomotion.load(task_name, config=pg_cfg)
    eval_env = locomotion.load(task_name, config=pg_cfg)

    obs_size = train_env.observation_size
    action_size = int(train_env.action_size)
    print(f"[+] obs_size={obs_size}  action_size={action_size}")

    network_factory = _build_network_factory(cfg, obs_size, action_size)
    randomization_fn = _get_domain_randomizer(locomotion, task_name) if env_cfg.get("domain_randomization", True) else None

    history: list = []
    t0 = time.perf_counter()
    metrics_file = run_dir / "metrics.jsonl"

    def progress_fn(num_steps: int, metrics: dict):
        elapsed = time.perf_counter() - t0
        row = {"step": num_steps, "wall_time_s": round(elapsed, 2)}
        for k, v in metrics.items():
            row[k] = _to_json_safe(v)
        history.append(row)
        with open(metrics_file, "a") as f:
            f.write(json.dumps(row) + "\n")
        rew = row.get("eval/episode_reward", float("nan"))
        ep_len = row.get("eval/avg_episode_length", float("nan"))
        sps = num_steps / max(elapsed, 1e-6)
        print(
            f"  step={num_steps:>10,}  elapsed={elapsed:6.0f}s  "
            f"sps={sps:>10,.0f}  eval_reward={rew:7.3f}  eval_len={ep_len:6.1f}"
        )

    train_kwargs = {
        "num_timesteps":        int(ppo_cfg.get("num_timesteps", 50_000_000)),
        "num_evals":            int(ppo_cfg.get("num_evals", 10)),
        "num_resets_per_eval":  int(ppo_cfg.get("num_resets_per_eval", 1)),
        "reward_scaling":       float(ppo_cfg.get("reward_scaling", 1.0)),
        "episode_length":       int(env_cfg.get("episode_length", 1000)),
        "normalize_observations": bool(ppo_cfg.get("normalize_observations", True)),
        "action_repeat":        int(env_cfg.get("action_repeat", 1)),
        "unroll_length":        int(ppo_cfg.get("unroll_length", 20)),
        "num_minibatches":      int(ppo_cfg.get("num_minibatches", 32)),
        "num_updates_per_batch": int(ppo_cfg.get("num_updates_per_batch", 4)),
        "discounting":          float(ppo_cfg.get("discounting", 0.97)),
        "learning_rate":        float(ppo_cfg.get("learning_rate", 3e-4)),
        "entropy_cost":         float(ppo_cfg.get("entropy_cost", 0.005)),
        "num_envs":             int(ppo_cfg.get("num_envs", 4096)),
        "batch_size":           int(ppo_cfg.get("batch_size", 256)),
        "max_grad_norm":        float(ppo_cfg.get("max_grad_norm", 1.0)),
        "clipping_epsilon":     float(ppo_cfg.get("clipping_epsilon", 0.2)),
        "gae_lambda":           float(ppo_cfg.get("gae_lambda", 0.95)),
        "normalize_advantage":  bool(ppo_cfg.get("normalize_advantage", True)),
        "deterministic_eval":   bool(ppo_cfg.get("deterministic_eval", True)),
        "num_eval_envs":        int(ppo_cfg.get("num_eval_envs", 128)),
        "seed":                 seed,
        "network_factory":      network_factory,
        "wrap_env_fn":          wrapper.wrap_for_brax_training,
        "save_checkpoint_path": str(ckpt_dir),
    }

    if randomization_fn is not None:
        train_kwargs["randomization_fn"] = randomization_fn

    valid_params = set(inspect.signature(ppo_train.train).parameters.keys())
    train_kwargs = {k: v for k, v in train_kwargs.items() if k in valid_params}

    print("[+] Starting PPO training...")
    make_inference_fn, params, final_metrics = functools.partial(ppo_train.train, **train_kwargs)(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=progress_fn,
    )

    total_time = time.perf_counter() - t0
    print(f"\n[✓] Training done in {total_time:.0f}s. Checkpoint at: {ckpt_dir}")

    meta = {
        "run_name": run_name,
        "seed": seed,
        "task": task_name,
        "checkpoint_dir": str(ckpt_dir),
        "total_wall_time_s": round(total_time, 2),
        "final_metrics": _to_json_safe(final_metrics),
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_flat.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.num_timesteps is not None:
        cfg.setdefault("ppo", {})["num_timesteps"] = args.num_timesteps
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    output_dir = args.output_dir or cfg.get("experiment", {}).get("output_dir", "results/flat")
    train(cfg, output_dir)


if __name__ == "__main__":
    main()
