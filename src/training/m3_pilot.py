"""Bounded M1 and M3 PPO training with matched initial/final evaluations."""

from contextlib import nullcontext
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
import math
import sys
import time
from importlib.metadata import version
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
import yaml
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks, train
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

from src.envs.curriculum import TerrainCurriculum
from src.envs.terrain import TerrainGenerator
from mujoco_playground._src.locomotion.g1 import g1_constants

from src.training.m3_env import HistoryJoystick
from src.training.m3_scene import write_scene

TASK = "G1JoystickFlatTerrain"
EVAL_SEEDS = tuple(range(10_000, 10_010))
ACTION_LIMIT_THRESHOLD = 0.95
FIXED_FORWARD_COMMANDS = (0.0, 0.5)


def enable_brax_jax_compatibility():
    """Support Brax versions that still call JAX's removed replication API."""
    if hasattr(jax, "device_put_replicated"):
        return

    def replicate(tree, devices):
        mesh = jax.sharding.Mesh(np.asarray(devices), ("replica",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("replica"))
        return jax.tree.map(
            lambda value: jax.device_put(jp.broadcast_to(value, (len(devices), *np.shape(value))), sharding),
            tree,
        )

    jax.device_put_replicated = replicate


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def with_action_metrics(compute_loss):
    def loss(params, normalizer_params, data, rng, **kwargs):
        value, metrics = compute_loss(params, normalizer_params, data, rng, **kwargs)
        near_limit = jp.abs(data.action) >= ACTION_LIMIT_THRESHOLD
        return value, dict(metrics, action_near_limit=jp.mean(near_limit),
                           any_action_near_limit=jp.mean(jp.any(near_limit, axis=-1)))

    return loss


def evaluate(env, policy, horizon, *, forward_command=None):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    act = jax.jit(policy)
    if forward_command is not None:
        command = jp.array([forward_command, 0., 0.], dtype=jp.float32)
        contact_sensor = env.mj_model.sensor_adr[env._right_foot_left_foot_found_sensor]

        with_command = jax.jit(partial(env.with_command, command=command))

        @jax.jit
        def forward_velocity(data):
            return env.get_local_linvel(data, "pelvis")[0]

    results = []
    for seed in EVAL_SEEDS:
        key = jax.random.PRNGKey(seed)
        state = reset(key)
        reward = distance = error = 0.
        if forward_command is not None:
            state = with_command(state)
        forward_speed = 0.
        for length in range(1, horizon + 1):
            key, subkey = jax.random.split(key)
            action, _ = act(state.obs, subkey)
            state = step(state, action)
            values = np.asarray([state.reward, state.metrics["m3_distance"],
                                 state.metrics["m3_tracking"], state.done])
            if not np.isfinite(values).all():
                raise ValueError("Non-finite evaluation transition")
            reward += float(values[0])
            distance += float(values[1])
            error += float(values[2])
            if forward_command is not None:
                forward_speed += float(forward_velocity(state.data))
            if values[3]:
                break
            if forward_command is not None:
                state = with_command(state)
        episode = dict(seed=seed, episode_return=reward, distance_m=distance,
                       fall=bool(values[3]), steps=length, tracking_error_mps=error / length)
        if forward_command is not None:
            episode.update(mean_forward_mps=forward_speed / length,
                           foot_to_foot_contact=bool(state.data.sensordata[contact_sensor] > 0))
        results.append(episode)
    return dict(episodes=results, means={k: float(np.mean([r[k] for r in results]))
                                        for k in ("episode_return", "distance_m", "fall", "tracking_error_mps")})


class EpisodeRecorder(wrapper.Wrapper):
    """Collect completed training episodes after vectorization, including time limits."""

    def __init__(self, env, record):
        super().__init__(env)
        self.record = record

    def reset(self, rng):
        state = self.env.reset(rng)
        state.info["m3_totals"] = jp.zeros((*state.reward.shape, 3))
        return state

    def step(self, state, action):
        totals = jp.where(state.done[..., None], 0, state.info["m3_totals"])
        result = self.env.step(state, action)
        totals += jp.stack((result.reward, result.metrics["m3_distance"],
                            jp.ones_like(result.reward)), axis=-1)
        result.info["m3_totals"] = totals

        def report():
            jax.debug.callback(self.record, result.done, totals, result.metrics["m3_fall"])

        jax.lax.cond(jp.any(result.done), report, lambda: None)
        return result


def verify_physics(env):
    """Exercise the same compiled step, contact sensors and model used by PPO."""
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(123))
    zero = jp.zeros(env.action_size)
    neutral = step(state, zero)
    driven = step(state, zero.at[0].set(0.5))
    if not float(neutral.data.time) > float(state.data.time):
        raise ValueError("The G1 physics clock did not advance")
    if np.array_equal(neutral.data.qvel, driven.data.qvel):
        raise ValueError("Different motor commands produced identical joint velocities")
    for _ in range(100):
        if bool(jp.any(neutral.info["last_contact"])):
            return
        if bool(neutral.done):
            break
        neutral = step(neutral, zero)
    raise ValueError("G1 foot sensors detected no contact with the generated floor")


def check_probe_action(actual, expected):
    """Compare bounded float32 actions across training and inference processes."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        raise ValueError("Checkpoint probe actions must have matching shapes and finite values")
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-6)
    return float(np.max(np.abs(actual - expected)))


def verify_checkpoint(directory):
    path = Path(directory).resolve()
    checkpoints = [p for p in (path / "checkpoints").iterdir() if p.name.isdigit()]
    latest = max(checkpoints, key=lambda p: int(p.name))
    settings = json.loads((path / "settings.json").read_text())
    with np.load(path / "reload_probe.npz") as probe:
        obs = {k: probe[k] for k in probe.files if k != "expected_action"}
        preprocess = running_statistics.normalize if settings["ppo"]["normalize_observations"] else lambda x, _: x
        network = networks.make_ppo_networks(
            {key: value.shape for key, value in obs.items()}, settings["actions"],
            preprocess_observations_fn=preprocess, **settings["network"],
        )
        policy = networks.make_inference_fn(network)(checkpoint.load(latest), deterministic=True)
        actual, _ = policy(obs, jax.random.PRNGKey(0))
        max_error = check_probe_action(actual, probe["expected_action"])
    save_json(path / "reload.json", {"checkpoint": latest.name, "action_verified": True, "max_abs_error": max_error})


def run(args):
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Use an empty run directory to preserve existing outputs: {out}")
    terrain_path = Path(args.terrain_config)
    contents = terrain_path.read_bytes()
    config = yaml.safe_load(contents)
    curriculum = TerrainCurriculum(config)
    category, difficulty = curriculum.sample_terrain(np.random.default_rng(args.seed))
    arena = TerrainGenerator(config).generate(category, difficulty)
    devices = jax.local_device_count()
    nenv = args.num_envs or devices
    if nenv % devices:
        raise ValueError("--num-envs must be divisible by the available device count")
    env_cfg = registry.get_default_config(TASK)
    # The upstream global buffer is sized for 8192 environments. Reserve
    # 256 contacts per local world for this pilot, retaining the upstream cap.
    env_cfg.naconmax = min(env_cfg.naconmax, 256 * (nenv // devices))
    template = g1_constants.task_to_xml("flat_terrain")
    scene = write_scene(str(template), arena, out / "terrain")
    env = HistoryJoystick(scene, env_cfg, args.history_len, Path("results/m3_robot_assets"),
                          actor_velocity=args.actor_velocity, actor_upstream=args.actor_upstream)
    observation_shapes = env.observation_size
    (actor_width,) = observation_shapes["state"]
    width, remainder = divmod(actor_width - args.history_len * env.action_size, args.history_len + 1)
    if width <= 0 or remainder:
        raise ValueError("Simulator observation shape disagrees with the configured actor width")

    ppo = locomotion_params.brax_ppo_config(TASK).to_dict()
    ppo["learning_rate"] = 0.0001
    network = ppo.pop("network_factory")
    unit = nenv * ppo["unroll_length"] * ppo["action_repeat"]
    default_updates = 2 if args.smoke else (50 if args.diagnose_ppo else 10)
    budget = args.num_timesteps or default_updates * unit
    if budget % unit:
        raise ValueError(f"--num-timesteps must be a multiple of {unit} for an exact pilot budget")
    ppo.update(num_envs=nenv, batch_size=nenv, num_minibatches=1, num_updates_per_batch=1,
               num_timesteps=budget, num_evals=1, num_eval_envs=nenv, num_resets_per_eval=0,
               run_evals=False)
    if args.diagnose_ppo:
        # Brax calls progress_fn with a device-averaged result at each epoch boundary.
        ppo["num_evals"] = budget // unit + 1
    settings = dict(model=args.model, architecture=("PPO MLP with current observation" if args.model == "M1"
                                                 else "PPO MLP with observation–action history stacking"),
        task=TASK, seed=args.seed, actor_fields=[
        "pelvis gyro", "projected gravity", "velocity command", "joint offsets", "joint velocities"],
        observation_width=width, actions=env.action_size, actor_width=actor_width,
        history_length=args.history_len, terrain_sha256=sha256(contents).hexdigest(),
        terrain=category, difficulty=difficulty, ppo=ppo, network=network,
        warp_contact_capacity=env_cfg.naconmax,
        packages={name: version(name) for name in ("jax", "brax", "mujoco")},
        control_dt=env.dt, physics_dt=env.sim_dt, simulator_backend=env_cfg.impl,
        observation_shapes=observation_shapes, optimizer_updates=budget // unit,
        terrain_config=str(terrain_path.resolve()), output_dir=str(out),
        devices=[f"{d}: {d.device_kind}" for d in jax.local_devices()])
    if args.diagnose_ppo:
        settings["ppo_diagnostic"] = dict(
            action_limit_threshold=ACTION_LIMIT_THRESHOLD,
            kl="Rollout distribution vs. pre-SGD distribution with updated normalization",
        )
    if args.actor_velocity:
        settings["actor_velocity"] = True
        settings["actor_fields"].append("pelvis forward and lateral velocity")
    if args.actor_upstream:
        settings["actor_upstream"] = True
        settings["actor_fields"] = [
            "noisy pelvis linear velocity", "noisy pelvis gyro", "noisy projected gravity",
            "velocity command", "noisy joint offsets", "noisy joint velocities",
            "previous action", "gait phase cosine and sine",
        ]
    print(f"Resolved {args.model} run configuration", flush=True)
    print(json.dumps(settings, indent=2, allow_nan=False), flush=True)
    save_json(out / "settings.json", settings)

    print("Checking G1 physics and terrain contact before PPO training.", flush=True)
    verify_physics(env)
    sample = jax.jit(env.reset)(jax.random.PRNGKey(0))
    if not all(np.isfinite(x).all() for x in sample.obs.values()):
        raise ValueError("Non-finite initial observation")

    initial_actor = None
    def snapshot(steps, make_policy, params):
        nonlocal initial_actor
        leaves = jax.tree_util.tree_leaves(params[1])
        policy = make_policy(params, deterministic=True)
        if int(steps) == 0:
            initial_actor = [np.array(x) for x in leaves]
            label = "initial"
        else:
            if int(steps) != budget:
                raise ValueError(f"PPO reported {steps} steps for a budget of {budget}")
            if not all(np.isfinite(x).all() for x in leaves):
                raise ValueError("Non-finite actor weights")
            if not any(np.any(a != b) for a, b in zip(initial_actor, leaves)):
                raise ValueError("PPO left every actor parameter unchanged")
            action, _ = policy(sample.obs, jax.random.PRNGKey(0))
            np.savez(out / "reload_probe.npz", expected_action=action, **sample.obs)
            label = "trained"
        save_json(out / f"{label}_evaluation.json", evaluate(env, policy, ppo["episode_length"]))
        if label == "trained" and args.fixed_command_eval:
            scenarios = [
                dict(forward_command_mps=speed, **evaluate(env, policy, ppo["episode_length"], forward_command=speed))
                for speed in FIXED_FORWARD_COMMANDS
            ]
            save_json(out / "fixed_command_evaluation.json", {"scenarios": scenarios})

    diagnostic_step = 0

    def progress(steps, metrics):
        nonlocal diagnostic_step
        if int(steps) != diagnostic_step + unit:
            raise ValueError(f"Expected PPO diagnostics at step {diagnostic_step + unit}, got {steps}")
        diagnostic_step = int(steps)
        # One minibatch and one SGD pass. The loss compares policies with the
        # same weights, before and after Brax updates the observation statistics.
        row = dict(environment_steps=diagnostic_step, optimizer_update=diagnostic_step // unit,
                   pre_sgd_kl=float(metrics["training/kl_mean"]),
                   action_near_limit_fraction=float(metrics["training/action_near_limit"]),
                   any_action_near_limit_fraction=float(metrics["training/any_action_near_limit"]))
        with (out / "ppo_diagnostic.jsonl").open("a") as log:
            log.write(json.dumps(row, allow_nan=False) + "\n")
        print(f'Update {row["optimizer_update"]}: pre-SGD KL {row["pre_sgd_kl"]:.4g}, '
              f'actions near limits {row["action_near_limit_fraction"]:.1%}', flush=True)

    def record(done, totals, falls):
        for (reward, distance, length), fall in zip(np.asarray(totals)[np.asarray(done, bool)],
                                                   np.asarray(falls)[np.asarray(done, bool)]):
            curriculum.record_episode(category, bool(not fall and distance >= curriculum.min_avg_distance),
                                      float(reward), int(length), float(distance),
                                      float(distance / (length * env.dt)), bool(fall))

    def wrap(environment, **kwargs):
        return EpisodeRecorder(wrapper.wrap_for_brax_training(environment, **kwargs), record)

    enable_brax_jax_compatibility()
    start = time.monotonic()
    network_factory = partial(networks.make_ppo_networks, **network)
    train_options = dict(save_checkpoint_path=str(out / "checkpoints"), policy_params_fn=snapshot)
    loss_hook = nullcontext()
    if args.diagnose_ppo:
        # Add auxiliary metrics to the existing loss, scoped to this training call.
        loss_hook = patch.object(train.ppo_losses, "compute_ppo_loss",
                                 with_action_metrics(train.ppo_losses.compute_ppo_loss))
        train_options = dict(progress_fn=progress,
                             policy_params_fn=lambda *params: snapshot(*params) if params[0] == 0 else None)
    with loss_hook:
        make_policy, params, metrics = train.train(
            environment=env, seed=args.seed, **ppo, network_factory=network_factory,
            wrap_env_fn=wrap, **train_options)
    jax.effects_barrier()
    if args.diagnose_ppo:
        if diagnostic_step != budget:
            raise ValueError(f"PPO diagnostic stopped at {diagnostic_step} of {budget} steps")
        snapshot(diagnostic_step, make_policy, params)
        metadata = checkpoint.network_config(observation_size=observation_shapes,
            action_size=env.action_size, normalize_observations=ppo["normalize_observations"],
            network_factory=network_factory)
        checkpoint.save(str(out / "checkpoints"), budget, params, metadata)
    metrics = {k: float(v) for k, v in metrics.items()}
    if not any("loss" in key for key in metrics) or not all(map(math.isfinite, metrics.values())):
        raise ValueError("Missing or non-finite PPO training diagnostics")
    save_json(out / "training.json", dict(environment_steps=budget, optimizer_updates=budget // unit,
              elapsed_seconds=time.monotonic() - start, metrics=metrics, curriculum=curriculum.update()))


if __name__ == "__main__":
    verify_checkpoint(sys.argv[1])
