import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / "kaggle/m1-baseline.ipynb"
CELLS = {cell["id"]: "".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text())["cells"]}


def write(folder, name, value):
    (folder / name).write_text(json.dumps(value))


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
    return dict(run=folder, repo=tmp_path, inputs=tmp_path / "input", seed=42,
                num_envs=8, budget=1_600_000, env={}, statistics=statistics,
                subprocess=subprocess, sys=sys, training_python=sys.executable,
                read=lambda path, name: json.loads((path / name).read_text()))


@pytest.mark.parametrize("has_m3_env", [True, False])
def test_training_python_when_kernel_lacks_brax(tmp_path, monkeypatch, has_m3_env):
    m3_python = tmp_path / ".venv/bin/python"
    m3_python.parent.mkdir(parents=True)
    m3_python.touch(mode=0o755)

    def probe(command, **kwargs):
        ready = has_m3_env and command[0] == str(m3_python)
        return subprocess.CompletedProcess(command, 0 if ready else 1, "",
                                           "" if ready else "No module named 'brax'")

    monkeypatch.setattr(subprocess, "run", probe)
    scope = dict(repo=tmp_path, env={}, os=os, Path=Path, sys=sys, shutil=shutil, subprocess=subprocess)
    if has_m3_env:
        exec(CELLS["training-python"], scope)
        assert scope["training_python"] == str(m3_python)
    else:
        with pytest.raises(RuntimeError, match="No module named 'brax'"):
            exec(CELLS["training-python"], scope)


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
    completed_run.update(run=completed_run["repo"] / "results/fresh", shutil=shutil)
    exec(CELLS["mesh-input"], completed_run)
    for mesh in dataset.iterdir():
        assert (completed_run["meshes"] / mesh.name).read_bytes() == mesh.read_bytes()


@pytest.mark.parametrize("devices,should_train", [
    ([["gpu", "Tesla T4"], ["gpu", "Tesla T4"]], True),
    ([["gpu", "Tesla T4"]], False),
    ([["gpu", "NVIDIA L4"], ["gpu", "NVIDIA L4"]], False),
])
def test_fresh_run_checks_two_t4s(completed_run, monkeypatch, devices, should_train):
    folder = completed_run["run"]
    completed_run.update(run=folder.with_name("fresh"), json=json)
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **kw: json.dumps(devices))
    commands = []

    def train(command, **kwargs):
        commands.append(command)
        folder.rename(completed_run["run"])

    monkeypatch.setattr(subprocess, "run", train)
    if should_train:
        exec(CELLS["m1-6"], completed_run)
        assert commands[0][commands[0].index("--num-envs") + 1] == "8"
    else:
        with pytest.raises(RuntimeError, match="GPU T4 x2"):
            exec(CELLS["m1-6"], completed_run)
        assert commands == []
