import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "kaggle/m1-baseline.ipynb"
CELLS = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}


def write(folder, name, value):
    (folder / name).write_text(json.dumps(value))


def test_existing_m1_trainer_skips_notebook_patch(monkeypatch, capsys):
    run = subprocess.run

    def check_command(command, **kwargs):
        assert command[0] != "git"
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", check_command)
    exec(CELLS["m1-4"], dict(repo=NOTEBOOK.parents[1], subprocess=subprocess, sys=sys))
    assert "M1 trainer ready" in capsys.readouterr().out


@pytest.fixture
def completed_run(tmp_path):
    folder = tmp_path / "results" / "m1"
    folder.mkdir(parents=True)
    settings = dict(model="M1", history_length=0, seed=42, actor_width=67, observation_width=67,
                    control_dt=0.02, network={"policy_obs_key": "state"},
                    ppo=dict(num_envs=8, num_timesteps=1_600_000, learning_rate=0.0001))
    write(folder, "settings.json", settings)
    write(folder, "training.json", {"environment_steps": 1_600_000})
    write(folder, "reload.json", {"action_verified": True, "checkpoint": "000001600000"})
    episodes = [dict(seed=seed, steps=50, tracking_error_mps=1.0, episode_return=0.0, fall=True)
                for seed in range(10_000, 10_010)]
    for stage in ("initial", "trained"):
        write(folder, f"{stage}_evaluation.json", {"episodes": episodes})
    return dict(run=folder, velocity_run=None, needs_training=False,
                repo=tmp_path, inputs=tmp_path / "input", seed=42,
                num_envs=8, budget=1_600_000, env={}, statistics=statistics,
                subprocess=subprocess, sys=sys, training_python=sys.executable,
                read=lambda path, name: json.loads((path / name).read_text()))


@pytest.mark.parametrize("has_m3_env", [True, False])
def test_training_python_when_kernel_lacks_brax(tmp_path, monkeypatch, has_m3_env):
    repo = tmp_path / "checkout"
    m3_python = repo / ".venv/bin/python"
    m3_python.parent.mkdir(parents=True)
    m3_python.touch(mode=0o755)
    installed_python = tmp_path / "bipedal-m1-training/bin/python"
    installs = []

    def run(command, **kwargs):
        if command[1:3] == ["-m", "venv"]:
            installed_python.parent.mkdir(parents=True, exist_ok=True)
            installed_python.touch(mode=0o755)
        if command[1:3] == ["-m", "pip"]:
            installs.append(command)
        ready = command[0] == str(m3_python if has_m3_env else installed_python)
        return subprocess.CompletedProcess(command, 0 if ready else 1, "",
                                           "" if ready else "No module named 'brax'")

    monkeypatch.setattr(subprocess, "run", run)
    scope = dict(repo=repo, needs_training=True, velocity_run=None, env={}, os=os, Path=Path, sys=sys,
                 gettempdir=lambda: str(tmp_path),
                 shutil=shutil, subprocess=subprocess)
    exec(CELLS["training-python"], scope)
    if has_m3_env:
        assert scope["training_python"] == str(m3_python)
        assert installs == []
    else:
        assert scope["training_python"] == str(installed_python)
        assert installs[0][3:5] == ["--python", str(installed_python)]
        assert "brax==0.14.2" in installs[0]
        exec(CELLS["training-python"], scope)
        assert len(installs) == 1


def test_m1_reports_without_meshes_or_reference(completed_run, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append(command))
    for cell in ("mesh-input", "m1-6", "m1-8", "compare-reference"):
        exec(CELLS[cell], completed_run)
    assert calls == [[sys.executable, "-u", "-m", "src.training.m3_pilot", str(completed_run["run"])]]
    output = capsys.readouterr().out
    assert "M1 / initial" in output and "M1 / trained" in output
    assert "M3 comparison skipped" in output


def test_recovery_rejects_m3_before_verification(completed_run, monkeypatch):
    folder = completed_run["run"]
    settings = completed_run["read"](folder, "settings.json")
    write(folder, "settings.json", dict(settings, model="M3", history_length=5))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Unexpected recovery"))
    with pytest.raises(ValueError, match="requested completed M1 run"):
        exec(CELLS["m1-6"], completed_run)


def test_fresh_run_prefers_trung1311_mesh_directory(completed_run):
    exec(CELLS["mesh-input"], completed_run)
    dataset = completed_run["inputs"] / "datasets/trung1311/robot-assets/results/m3_robot_assets"
    dataset.mkdir(parents=True)
    for name in completed_run["expected_meshes"]:
        (dataset / name).write_bytes(name.encode())
    (completed_run["inputs"] / "m3-robot-assets.zip").write_bytes(b"unused archive")
    completed_run.update(run=completed_run["repo"] / "results/fresh", needs_training=True, shutil=shutil)
    exec(CELLS["mesh-input"], completed_run)
    for mesh in dataset.iterdir():
        assert (completed_run["meshes"] / mesh.name).read_bytes() == mesh.read_bytes()


@pytest.mark.parametrize("kinds,expected", [
    (["Tesla T4", "Tesla T4"], True),
    (["Tesla T4"], False),
    (["NVIDIA L4", "NVIDIA L4"], False),
])
def test_training_python_checks_two_t4s(tmp_path, monkeypatch, kinds, expected):
    devices = [SimpleNamespace(platform="gpu", device_kind=kind) for kind in kinds]
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(local_devices=lambda: devices))
    monkeypatch.setitem(sys.modules, "src.training.m3_pilot", SimpleNamespace(run=None))
    monkeypatch.setattr("importlib.metadata.version", lambda name: "test")
    monkeypatch.setattr(sys, "argv", ["-c", "train"])
    monkeypatch.setattr(subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""))
    scope = dict(repo=tmp_path, needs_training=True, velocity_run=None, env={}, os=os, Path=Path,
                 gettempdir=lambda: str(tmp_path),
                 sys=sys, shutil=shutil, subprocess=subprocess)
    exec(CELLS["training-python"], scope)
    if expected:
        exec(scope["probe_code"], {})
    else:
        with pytest.raises(SystemExit, match="GPU T4 x2"):
            exec(scope["probe_code"], {})


def test_fresh_run_launches_with_selected_python(completed_run, monkeypatch):
    folder = completed_run["run"]
    completed_run.update(run=folder.with_name("fresh"), training_python="/training/bin/python")
    commands = []

    def train(command, **kwargs):
        commands.append(command)
        folder.rename(completed_run["run"])

    monkeypatch.setattr(subprocess, "run", train)
    exec(CELLS["m1-6"], completed_run)
    assert commands[0][0] == completed_run["training_python"]
    assert commands[0][commands[0].index("--num-envs") + 1] == "8"


def test_velocity_experiment_trains_and_reports_matched_runs(completed_run, monkeypatch, capsys):
    read = completed_run["read"]
    source = completed_run["run"]
    settings = read(source, "settings.json")
    settings.update(packages={}, devices=["T4", "T4"], terrain_sha256="same", terrain="flat",
                    difficulty=0.0, physics_dt=0.002, warp_contact_capacity=1024,
                    simulator_backend="warp", task="G1JoystickFlatTerrain", actions=29)
    control = source.with_name("control")
    variant = source.with_name("velocity")
    completed_run.update(run=control, velocity_run=variant, budget=160_000, needs_training=True)

    commands = []

    def train(command, **kwargs):
        commands.append(command)
        assert "--diagnose-ppo" in command and "--fixed-command-eval" in command
        folder = Path(command[command.index("--output-dir") + 1])
        width = 69 if "--actor-velocity" in command else 67
        assert width == (69 if folder == variant else 67)
        folder.mkdir()
        write(folder, "settings.json",
              dict(settings, actor_width=width, observation_width=width,
                   ppo=dict(settings["ppo"], num_timesteps=160_000)))
        write(folder, "training.json", {"environment_steps": 160_000})
        write(folder, "reload.json", {"action_verified": True, "checkpoint": "000000160000"})
        for stage in ("initial", "trained"):
            shutil.copyfile(source / f"{stage}_evaluation.json", folder / f"{stage}_evaluation.json")
        episodes = [dict(ep, mean_forward_mps=0.3, foot_to_foot_contact=True)
                    for ep in read(folder, "trained_evaluation.json")["episodes"]]
        write(folder, "fixed_command_evaluation.json", {"scenarios": [
            dict(forward_command_mps=speed, episodes=episodes) for speed in (0.0, 0.5)
        ]})

    monkeypatch.setattr(subprocess, "run", train)
    exec(CELLS["m1-6"], completed_run)
    exec(CELLS["m1-8"], completed_run)
    assert len(commands) == 2
    output = capsys.readouterr().out
    assert "Verified 67-input M1" in output and "Verified 69-input M1" in output
    assert "Foot collisions" in output and "0.50 m/s" in output
