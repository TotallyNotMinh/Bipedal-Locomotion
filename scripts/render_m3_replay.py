"""Render matched initial and checkpoint rollouts without running PPO."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import mujoco
from mujoco_playground import registry
import numpy as np
from PIL import Image, ImageDraw

from src.training.m3_env import HistoryJoystick
from src.training.m3_pilot import check_probe_action


def rollout(reset, step, policy, seed, horizon):
    key = jax.random.PRNGKey(seed)
    state = reset(key)
    states = [np.asarray(state.data.qpos)]
    act = jax.jit(policy)
    for _ in range(horizon):
        key, subkey = jax.random.split(key)
        action, _ = act(state.obs, subkey)
        state = step(state, action)
        states.append(np.asarray(state.data.qpos))
        if bool(state.done):
            break
    return states, bool(state.done)


def render(model, trajectories, output, dt, seed):
    width, height = 512, 320
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.elevation = -13
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    model.vis.quality.shadowsize = 1024
    frame_count = max(len(poses) for poses, _ in trajectories) + round(1 / dt)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{2 * width}x{2 * height}", "-r", str(1 / dt), "-i", "pipe:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for index in range(frame_count):
                frame = Image.new("RGB", (2 * width, 2 * height))
                draw = ImageDraw.Draw(frame)
                for column, ((poses, fell), label) in enumerate(
                    zip(trajectories, ("Initial policy", "Trained policy"))
                ):
                    current = min(index, len(poses) - 1)
                    data.qpos[:] = poses[current]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = [poses[0][0], poses[0][1], 0.63]
                    for row, (azimuth, view) in enumerate(((90, "Side"), (180, "Front"))):
                        camera.azimuth = azimuth
                        renderer.update_scene(data, camera=camera)
                        x, y = column * width, row * height
                        frame.paste(Image.fromarray(renderer.render()), (x, y))
                        draw.rectangle((x, y, x + width, y + 28), fill="#132236")
                        status = " | Fall" if fell and index >= len(poses) - 1 else ""
                        draw.text(
                            (x + 10, y + 7),
                            f"{label} | {view} | Seed {seed} | {current * dt:.2f}s{status}",
                            fill="white",
                        )
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed to encode the replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    run = args.run.resolve()
    settings = json.loads((run / "settings.json").read_text())
    verified = json.loads((run / "reload.json").read_text())
    network = networks.make_ppo_networks(
        {key: tuple(shape) for key, shape in settings["observation_shapes"].items()},
        settings["actions"],
        preprocess_observations_fn=(
            running_statistics.normalize
            if settings["ppo"]["normalize_observations"] else lambda x, _: x
        ),
        **settings["network"],
    )
    make_policy = networks.make_inference_fn(network)
    trained = checkpoint.load(run / "checkpoints" / verified["checkpoint"])
    global_key, _ = jax.random.split(jax.random.PRNGKey(settings["seed"]))
    actor_key, _ = jax.random.split(global_key)
    initial = (
        running_statistics.init_state(trained[0].mean),
        network.policy_network.init(actor_key),
        None,
    )
    trained_policy = make_policy(trained, deterministic=True)
    with np.load(run / "reload_probe.npz") as probe:
        obs = {key: probe[key] for key in settings["observation_shapes"]}
        action, _ = trained_policy(obs, jax.random.PRNGKey(0))
        check_probe_action(action, probe["expected_action"])

    config = registry.get_default_config(settings["task"])
    config.naconmax = settings["warp_contact_capacity"]
    terrain = run / "terrain"
    scene = ET.parse(terrain / "g1.xml")
    scene.find(".//hfield[@name='training_surface']").set("file", str(terrain / "terrain.bin"))
    with tempfile.TemporaryDirectory(dir=run) as temp:
        local_scene = Path(temp) / "g1.xml"
        scene.write(local_scene)
        env = HistoryJoystick(
            local_scene, config, settings["history_length"], Path("results/m3_robot_assets")
        )
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        trajectories = [
            rollout(reset, step, policy, args.seed, settings["ppo"]["episode_length"])
            for policy in (make_policy(initial, deterministic=True), trained_policy)
        ]
        print(f"Replay steps (initial, trained): {[len(poses) - 1 for poses, _ in trajectories]}", flush=True)
        output = run / f"replay_seed_{args.seed}.mp4"
        render(env.mj_model, trajectories, output, env.dt, args.seed)
        print(output)


if __name__ == "__main__":
    main()
