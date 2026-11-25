"""Brax PPO training and evaluation runner for Unitree G1 (M1 & M2)."""

import csv
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.models.mlp_policy import (
    ModelArchitectureSpec,
    build_model_spec_from_config,
    build_parameter_comparison_table,
    format_parameter_table,
    make_model_network_factory,
)
from src.training.config_loader import save_resolved_config, validate_ppo_sharding
from src.training.env_builder import (
    build_g1_environment,
    build_perturbed_friction_env,
    get_domain_randomizer_fn,
)


def _get_git_commit() -> Optional[str]:
    """Returns current short git commit hash if running inside a git repository."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _to_json_safe(obj: Any) -> Any:
    """Converts NumPy/JAX scalars and arrays into JSON-serializable Python primitives."""
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


def inspect_model_setup(config: Dict[str, Any]) -> Tuple[ModelArchitectureSpec, str]:
    """Inspects observation/action dims (from Playground if installed, else G1 defaults)
    and builds the resolved ModelArchitectureSpec + ASCII parameter comparison table.
    """
    try:
        env, _, _ = build_g1_environment(config)
        obs_size = env.observation_size
        act_size = int(env.action_size)
        source = "mujoco_playground"
    except ImportError:
        # CPU-only inspection fallback matching README.md G1 dimensions
        obs_size = {"state": 58, "privileged_state": 58}
        act_size = 23
        source = "fallback_cpu_defaults(58/23)"

    spec = build_model_spec_from_config(config, obs_size, act_size)
    net_cfg = config.get("network", {})
    cm_cfg = net_cfg.get("capacity_matching", {})
    table = build_parameter_comparison_table(
        spec=spec,
        baseline_policy_hidden=cm_cfg.get("reference_policy_hidden", (512, 256, 128)),
        baseline_value_hidden=cm_cfg.get("reference_value_hidden", (512, 256, 128)),
        history_length=int(cm_cfg.get("history_length", 5)),
        include_actions=bool(cm_cfg.get("include_actions", True)),
    )
    summary = (
        f"[Model Setup Source: {source}]\n"
        f"Active Model: {spec.model_id} | Policy Obs: {spec.obs_dim} | "
        f"Value Obs: {spec.value_obs_dim} | Action Dim: {spec.action_dim}\n"
        f"Policy Hidden: {list(spec.policy_hidden_layer_sizes)} | "
        f"Value Hidden: {list(spec.value_hidden_layer_sizes)}\n"
        f"{format_parameter_table(table)}"
    )
    return spec, summary


def _ensure_jax_brax_compat() -> None:
    """Provides drop-in shims for JAX 0.9+/0.10+ removed pmap helpers used by PyPI Brax."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    def _has_attr(name: str) -> bool:
        try:
            getattr(jax, name)
            return True
        except AttributeError:
            return False

    if not _has_attr("device_put_replicated"):
        def _device_put_replicated(x, devices):
            mesh = Mesh(np.array(devices), ("x",))
            sharding = NamedSharding(mesh, P("x"))
            return jax.tree.map(
                lambda y: jax.device_put(jnp.stack([jnp.asarray(y)] * len(devices)), sharding),
                x,
            )

        setattr(jax, "device_put_replicated", _device_put_replicated)

    if not _has_attr("device_put_sharded"):
        def _device_put_sharded(shards, devices):
            mesh = Mesh(np.array(devices), ("x",))
            sharding = NamedSharding(mesh, P("x"))
            return jax.tree.map(
                lambda *xs: jax.device_put(jnp.stack([jnp.asarray(x) for x in xs]), sharding),
                *shards,
            )

        setattr(jax, "device_put_sharded", _device_put_sharded)

    if not _has_attr("tree_map"):
        setattr(jax, "tree_map", jax.tree.map)


def run_ppo_training(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes PPO training on JAX / Dual-T4 using Brax + MuJoCo Playground."""
    import jax
    _ensure_jax_brax_compat()
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    exp_cfg = config.get("experiment", {})
    ppo_cfg = config.get("ppo", {})
    env_cfg = config.get("env", {})

    seed = int(exp_cfg.get("seed", 42))
    run_name = str(exp_cfg.get("name", f"g1_{config.get('network', {}).get('model_id', 'm2').lower()}"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_out = Path(output_dir or exp_cfg.get("output_dir", "results")).resolve()
    run_dir = base_out / f"{run_name}_seed{seed}_{timestamp}"
    ckpt_dir = (run_dir / "checkpoints").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    devices = jax.devices()
    device_count = len(devices)
    validate_ppo_sharding(ppo_cfg, device_count)

    print("==================================================================")
    print(f" Starting PPO Training: {run_name} (seed={seed})")
    print(f" JAX Devices ({device_count}): {[f'{d.device_kind}:{d.id}' for d in devices]}")
    print(f" Output Directory: {run_dir}")
    print("==================================================================")

    # 1. Build training & evaluation environments
    train_env, pg_cfg, arena = build_g1_environment(config)
    eval_env, _, _ = build_g1_environment(config)

    obs_size = train_env.observation_size
    action_size = int(train_env.action_size)
    has_dict_obs = isinstance(obs_size, dict)

    # 2. Resolve model architecture (and auto-match M2 parameters if enabled)
    spec = build_model_spec_from_config(config, obs_size, action_size)
    net_cfg = config.get("network", {})
    cm_cfg = net_cfg.get("capacity_matching", {})

    # Record resolved layer sizes back into a copy of config for archival
    resolved_cfg = json.loads(json.dumps(config))
    resolved_cfg.setdefault("network", {})["policy_hidden_layer_sizes"] = list(spec.policy_hidden_layer_sizes)
    resolved_cfg.setdefault("network", {})["value_hidden_layer_sizes"] = list(spec.value_hidden_layer_sizes)
    save_resolved_config(resolved_cfg, run_dir / "resolved_config.yaml")

    param_table = build_parameter_comparison_table(
        spec=spec,
        baseline_policy_hidden=cm_cfg.get("reference_policy_hidden", (512, 256, 128)),
        baseline_value_hidden=cm_cfg.get("reference_value_hidden", (512, 256, 128)),
        history_length=int(cm_cfg.get("history_length", 5)),
        include_actions=bool(cm_cfg.get("include_actions", True)),
    )
    print("\n" + format_parameter_table(param_table) + "\n")

    with open(run_dir / "param_table.json", "w") as f:
        json.dump([r.to_dict() for r in param_table], f, indent=2)

    network_factory = make_model_network_factory(spec, has_dict_obs=has_dict_obs)
    randomization_fn = get_domain_randomizer_fn(config)

    # 3. Metrics logger callbacks
    metrics_jsonl = run_dir / "metrics.jsonl"
    metrics_csv = run_dir / "metrics.csv"
    history_records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    def progress_fn(num_steps: int, metrics: Dict[str, Any]):
        elapsed = time.perf_counter() - t_start
        row: Dict[str, Any] = {
            "step": int(num_steps),
            "wall_time_s": round(elapsed, 2),
        }
        for k, v in metrics.items():
            row[k] = _to_json_safe(v)
        history_records.append(row)

        with open(metrics_jsonl, "a") as fj:
            fj.write(json.dumps(row) + "\n")

        eval_rew = row.get("eval/episode_reward", float("nan"))
        eval_len = row.get("eval/avg_episode_length", float("nan"))
        sps = num_steps / max(elapsed, 1e-6)
        print(
            f"[step={num_steps:>10,}] wall={elapsed:7.1f}s | "
            f"sps={sps:9,.0f} | eval_reward={eval_rew:8.3f} | eval_len={eval_len:6.1f}"
        )

    # 4. Assemble Brax PPO train call
    train_kwargs: Dict[str, Any] = {
        "num_timesteps": int(ppo_cfg.get("num_timesteps", 50_000_000)),
        "num_evals": int(ppo_cfg.get("num_evals", 20)),
        "num_resets_per_eval": int(ppo_cfg.get("num_resets_per_eval", 1)),
        "reward_scaling": float(ppo_cfg.get("reward_scaling", 1.0)),
        "episode_length": int(env_cfg.get("episode_length", 1000)),
        "normalize_observations": bool(ppo_cfg.get("normalize_observations", True)),
        "action_repeat": int(env_cfg.get("action_repeat", 1)),
        "unroll_length": int(ppo_cfg.get("unroll_length", 20)),
        "num_minibatches": int(ppo_cfg.get("num_minibatches", 32)),
        "num_updates_per_batch": int(ppo_cfg.get("num_updates_per_batch", 4)),
        "discounting": float(ppo_cfg.get("discounting", 0.97)),
        "learning_rate": float(ppo_cfg.get("learning_rate", 3e-4)),
        "entropy_cost": float(ppo_cfg.get("entropy_cost", 0.005)),
        "num_envs": int(ppo_cfg.get("num_envs", 4096)),
        "batch_size": int(ppo_cfg.get("batch_size", 256)),
        "max_grad_norm": float(ppo_cfg.get("max_grad_norm", 1.0)),
        "clipping_epsilon": float(ppo_cfg.get("clipping_epsilon", 0.2)),
        "gae_lambda": float(ppo_cfg.get("gae_lambda", 0.95)),
        "normalize_advantage": bool(ppo_cfg.get("normalize_advantage", True)),
        "deterministic_eval": bool(ppo_cfg.get("deterministic_eval", True)),
        "num_eval_envs": int(ppo_cfg.get("num_eval_envs", 128)),
        "log_training_metrics": bool(ppo_cfg.get("log_training_metrics", False)),
        "seed": seed,
        "network_factory": network_factory,
        "wrap_env_fn": wrapper.wrap_for_brax_training,
        "save_checkpoint_path": str(ckpt_dir),
    }

    if randomization_fn is not None:
        train_kwargs["randomization_fn"] = randomization_fn

    if restore_checkpoint_path is not None:
        train_kwargs["restore_checkpoint_path"] = str(Path(restore_checkpoint_path).resolve())

    # Optional Brax PPO call compatibility filter for older/newer Brax releases
    import inspect
    valid_params = set(inspect.signature(ppo.train).parameters.keys())
    filtered_kwargs = {k: v for k, v in train_kwargs.items() if k in valid_params}

    train_fn = functools.partial(ppo.train, **filtered_kwargs)
    make_inference_fn, params, training_metrics = train_fn(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=progress_fn,
    )

    total_time = time.perf_counter() - t_start

    # Write flat CSV summary of eval checkpoints
    if history_records:
        all_keys: List[str] = []
        seen = set()
        for r in history_records:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with open(metrics_csv, "w", newline="") as fc:
            writer = csv.DictWriter(fc, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(history_records)

    # Save run metadata (PLAN-1.md Phase 9 reproducibility contract)
    metadata = {
        "run_name": run_name,
        "model_id": spec.model_id,
        "seed": seed,
        "git_commit": _get_git_commit(),
        "timestamp_utc": timestamp,
        "devices": [f"{d.device_kind} ({d.platform}:{d.id})" for d in devices],
        "env_task": str(env_cfg.get("task_name", "")),
        "terrain_mode": str(env_cfg.get("terrain_mode", "playground")),
        "terrain_category": arena.category if arena else str(env_cfg.get("terrain_category", "flat")),
        "terrain_difficulty": arena.difficulty if arena else 0.0,
        "obs_dim": spec.obs_dim,
        "value_obs_dim": spec.value_obs_dim,
        "action_dim": spec.action_dim,
        "policy_hidden_layer_sizes": list(spec.policy_hidden_layer_sizes),
        "value_hidden_layer_sizes": list(spec.value_hidden_layer_sizes),
        "actual_policy_params": spec.actual_policy_params,
        "actual_value_params": spec.actual_value_params,
        "actual_total_params": spec.actual_total_params,
        "capacity_matched": spec.capacity_matched,
        "match_reference": spec.match_reference,
        "total_wall_time_s": round(total_time, 2),
        "checkpoint_dir": str(ckpt_dir),
        "final_metrics": _to_json_safe(training_metrics),
    }
    with open(run_dir / "run_metadata.json", "w") as fm:
        json.dump(metadata, fm, indent=2)

    print(f"\n[✓] Training complete in {total_time:.1f}s. Artifacts saved to: {run_dir}")
    return metadata


def _resolve_latest_checkpoint(checkpoint_path: str | Path) -> Path:
    """Resolves either a specific step directory or the latest numbered subfolder inside `checkpoints/`."""
    p = Path(checkpoint_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {p}")

    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if not subdirs:
        return p

    # Check if `p` contains numbered checkpoint subdirectories (e.g. 000050000000 or 52920320)
    digit_dirs = [d for d in subdirs if d.name.isdigit()]
    if digit_dirs:
        digit_dirs.sort(key=lambda d: int(d.name))
        return digit_dirs[-1]

    import re
    with_nums = []
    for d in subdirs:
        m = re.findall(r"\d+", d.name)
        if m:
            with_nums.append((int(m[-1]), d))
    if with_nums:
        with_nums.sort(key=lambda x: x[0])
        return with_nums[-1][1]

    subdirs.sort(key=lambda d: d.stat().st_mtime)
    return subdirs[-1]


def _load_policy_inference_fn(
    checkpoint_path: Path,
    network_factory: Any,
    obs_size: Any,
    action_size: int,
    deterministic: bool = True,
) -> Any:
    """Robustly restores policy from checkpoint across Brax and Orbax variants."""
    ckpt_str = str(checkpoint_path)
    ppo_network = network_factory(obs_size, action_size)

    # 1. Try restoring params and instantiating policy via network_factory
    params = None
    try:
        from brax.training.agents.ppo import checkpoint as ppo_checkpoint
        if hasattr(ppo_checkpoint, "load"):
            params = ppo_checkpoint.load(ckpt_str)
    except Exception:
        pass

    if params is None:
        try:
            from brax.training import checkpoint as brax_checkpoint
            if hasattr(brax_checkpoint, "load"):
                params = brax_checkpoint.load(ckpt_str)
        except Exception:
            pass

    if params is None:
        try:
            import orbax.checkpoint as ocp
            try:
                params = ocp.StandardCheckpointer().restore(ckpt_str)
            except Exception:
                params = ocp.PyTreeCheckpointer().restore(ckpt_str)
        except Exception:
            pass

    if params is not None:
        return ppo_network.make_policy(params, deterministic=deterministic)

    # 2. Fallback to ppo_checkpoint.load_policy
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    try:
        return ppo_checkpoint.load_policy(ckpt_str, network_factory=network_factory, deterministic=deterministic)
    except TypeError:
        return ppo_checkpoint.load_policy(ckpt_str, deterministic=deterministic)


def evaluate_policy_rollouts(
    config: Dict[str, Any],
    checkpoint_path: str | Path,
    scenario: str = "nominal",
    num_episodes: int = 20,
    seed: int = 123,
    deterministic: bool = True,
    terrain_category: Optional[str] = None,
    terrain_difficulty: Optional[float] = None,
    switched_friction: float = 0.2,
    switch_step: int = 250,
    push_step: int = 300,
    push_vel_xy: Tuple[float, float] = (0.6, 0.4),
    recovery_thresh: float = 0.25,
    recovery_sustain_steps: int = 25,
) -> Dict[str, Any]:
    """Evaluates a saved Brax checkpoint on nominal, friction-switch, push, or terrain scenarios.

    Computes Plan-compliant metrics:
    - return, episode_length, fall_rate, success_rate
    - velocity tracking error (overall, pre-event, immediate post-event, integrated post-event)
    - recovery_time_steps (for recovered episodes) and censored_no_recovery_count
    """
    import jax
    import jax.numpy as jnp
    _ensure_jax_brax_compat()
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint

    ckpt = _resolve_latest_checkpoint(checkpoint_path)

    # Build primary evaluation environment
    env, _, _ = build_g1_environment(
        config,
        override_category=terrain_category,
        override_difficulty=terrain_difficulty,
        override_seed=seed,
    )
    obs_size = env.observation_size
    action_size = int(env.action_size)
    has_dict_obs = isinstance(obs_size, dict)

    spec = build_model_spec_from_config(config, obs_size, action_size)
    network_factory = make_model_network_factory(spec, has_dict_obs=has_dict_obs)
    inference_fn = _load_policy_inference_fn(
        checkpoint_path=ckpt,
        network_factory=network_factory,
        obs_size=obs_size,
        action_size=action_size,
        deterministic=deterministic,
    )

    jit_policy = jax.jit(inference_fn)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    jit_step_switched = None
    if scenario == "friction_switch":
        switched_env = build_perturbed_friction_env(
            config,
            friction=switched_friction,
            override_category=terrain_category,
            override_difficulty=terrain_difficulty,
        )
        jit_step_switched = jax.jit(switched_env.step)
    elif scenario == "held_out_friction":
        env = build_perturbed_friction_env(
            config,
            friction=switched_friction,
            override_category=terrain_category,
            override_difficulty=terrain_difficulty,
        )
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

    episode_length = int(config.get("env", {}).get("episode_length", 1000))
    event_step = switch_step if scenario == "friction_switch" else (push_step if scenario == "push" else -1)

    master_key = jax.random.PRNGKey(seed)
    ep_results: List[Dict[str, Any]] = []

    for ep_idx in range(num_episodes):
        master_key, reset_key, act_key = jax.random.split(master_key, 3)
        state = jit_reset(reset_key)

        ep_return = 0.0
        ep_len = 0
        fell = False
        vel_errors: List[float] = []

        for t in range(episode_length):
            act_key, step_key = jax.random.split(act_key)
            action, _ = jit_policy(state.obs, step_key)

            # Optional external push impulse applied to base horizontal velocity
            if scenario == "push" and t == push_step:
                qvel = state.data.qvel
                if qvel.shape[0] >= 2:
                    pushed_qvel = qvel.at[0].add(jnp.float32(push_vel_xy[0]))
                    pushed_qvel = pushed_qvel.at[1].add(jnp.float32(push_vel_xy[1]))
                    state = state.replace(data=state.data.replace(qvel=pushed_qvel))

            step_fn = jit_step_switched if (scenario == "friction_switch" and t >= switch_step and jit_step_switched is not None) else jit_step
            state = step_fn(state, action)

            ep_return += float(state.reward)
            ep_len += 1

            # Extract command vs actual local-frame planar velocity
            cmd = state.info.get("command") if isinstance(state.info, dict) else None
            if cmd is not None:
                local_vel = None
                try:
                    from mujoco_playground._src import mjx_env
                    mj_m = getattr(env, "mj_model", getattr(env, "_mj_model", None))
                    local_vel = mjx_env.get_sensor_data(mj_m, state.data, "local_linvel_pelvis")
                except Exception:
                    try:
                        local_vel = env.get_local_linvel(state.data, "pelvis")
                    except Exception:
                        local_vel = None

                if local_vel is not None and len(local_vel) >= 2:
                    vx_err = float(local_vel[0] - cmd[0])
                    vy_err = float(local_vel[1] - cmd[1])
                    vel_errors.append(float(np.hypot(vx_err, vy_err)))
                elif hasattr(state.data, "qvel") and state.data.qvel.shape[0] >= 2:
                    vx_err = float(state.data.qvel[0] - cmd[0])
                    vy_err = float(state.data.qvel[1] - cmd[1])
                    vel_errors.append(float(np.hypot(vx_err, vy_err)))

            if bool(state.done):
                # In G1Joystick, early termination before episode_length indicates a fall/collision
                if ep_len < episode_length:
                    fell = True
                break

        mean_vel_err = float(np.mean(vel_errors)) if vel_errors else float("nan")
        post_errors = vel_errors[event_step:] if (event_step >= 0 and len(vel_errors) > event_step) else []
        immediate_post_err = float(np.mean(post_errors[:25])) if post_errors else float("nan")
        integrated_post_err = float(np.sum(post_errors)) if post_errors else float("nan")

        recovery_time: Optional[int] = None
        if event_step >= 0 and not fell and len(post_errors) >= recovery_sustain_steps:
            for offset in range(len(post_errors) - recovery_sustain_steps + 1):
                window = post_errors[offset : offset + recovery_sustain_steps]
                if all(e <= recovery_thresh for e in window):
                    recovery_time = offset
                    break

        ep_results.append({
            "episode": ep_idx,
            "return": ep_return,
            "length": ep_len,
            "fell": fell,
            "success": not fell,
            "mean_vel_error": mean_vel_err,
            "immediate_post_event_error": immediate_post_err,
            "integrated_post_event_error": integrated_post_err,
            "recovery_time_steps": recovery_time,
            "censored_no_recovery": (event_step >= 0 and recovery_time is None),
        })

    recovered_times = [r["recovery_time_steps"] for r in ep_results if r["recovery_time_steps"] is not None]
    summary = {
        "scenario": scenario,
        "checkpoint": str(ckpt),
        "model_id": spec.model_id,
        "num_episodes": num_episodes,
        "mean_return": float(np.mean([r["return"] for r in ep_results])),
        "std_return": float(np.std([r["return"] for r in ep_results])),
        "mean_length": float(np.mean([r["length"] for r in ep_results])),
        "fall_rate": float(np.mean([1.0 if r["fell"] else 0.0 for r in ep_results])),
        "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in ep_results])),
        "mean_vel_error": float(np.nanmean([r["mean_vel_error"] for r in ep_results])),
        "mean_immediate_post_error": float(np.nanmean([r["immediate_post_event_error"] for r in ep_results])) if event_step >= 0 else None,
        "mean_integrated_post_error": float(np.nanmean([r["integrated_post_event_error"] for r in ep_results])) if event_step >= 0 else None,
        "mean_recovery_time_steps": float(np.mean(recovered_times)) if recovered_times else None,
        "recovery_rate": float(len(recovered_times) / num_episodes) if event_step >= 0 else None,
        "censored_no_recovery_count": sum(1 for r in ep_results if r["censored_no_recovery"]),
        "episodes": ep_results,
    }
    return summary
