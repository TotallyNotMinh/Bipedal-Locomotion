# Report: M2 — Larger MLP Policy for Unitree G1 Walking

## 1. Goal

The goal of M2 is to train a **Unitree G1 humanoid robot to walk on flat ground** and follow a velocity command. The policy is a **large MLP** (896-448-128) that takes the current observation `o_t` as input.

## 2. Setup

**Robot and simulator**
- Unitree G1 humanoid with 29 actuated joints.
- MuJoCo Playground (MJX on JAX), task `G1JoystickFlatTerrain`.
- Physics runs at 500 Hz and the policy runs at 50 Hz. One episode lasts 1000 steps (20 s).

**Environment.** I kept the **default Playground config without any changes**:
- The robot must follow a random velocity command: vx in [-1, 1], vy in [-0.5, 0.5], yaw rate in [-1, 1].
- There is sensor noise and random pushes every 5–10 s.
- Domain randomization: floor friction U(0.4, 1.0), link masses ×U(0.9, 1.1), torso mass ±1 kg.
- The only setting I chose is `impl: jax`. This picks the simulation backend and does not change physics or rewards.

**Observations**
- Actor (`state`, 103 values): base velocity, angular velocity, gravity direction, command, joint positions, joint velocities, previous action, and gait phase.
- Critic (`privileged_state`, 216 values): the actor inputs plus extra true values, such as contact forces and base height. This is called an asymmetric actor-critic.

**Training (PPO, Brax).** I used the official Playground PPO settings for the G1:

| Parameter | Value |
|---|---|
| Total steps | 100M |
| Parallel envs | 8192 |
| Unroll length | 20 |
| Batch size × minibatches | 256 × 32 |
| Updates per batch | 4 |
| Learning rate | 3e-4 |
| Discount γ / GAE λ | 0.97 / 0.95 |
| Clip ε | 0.2 |
| Entropy cost | 0.005 |
| Hardware | 2× NVIDIA T4 (Kaggle) |

## 3. M2 Architecture

| Part | Input | Hidden layers | Output | Params |
|---|---|---|---|---|
| Actor | `state` (103) | 896-448-128 | 58 (mean + std for 29 joints) | 559,994 |
| Critic | `privileged_state` (216) | 512-256-128 | 1 (value) | 275,457 |

- Activation is swish, and the action distribution is tanh-normal.
- Observations are normalized with running statistics. The normalizer is not counted in the parameter numbers.

## 4. What Happened So Far

**First runs (pilot, old config).** Before the final setup, I ran two 50M-step trainings with a smaller 512-256-128 actor:

| Run | Change | Result |
|---|---|---|
| 1 | Default commands | In the video, the robot seemed to walk backward. |
| 2 | Only forward commands (vx in [0.3, 1.0]) | Final eval episode length about 340/1000 (318/1000 in a repeat run). In the video, the robot fell after 39 steps. |

**Problems I found**
1. **Not enough training.** Episode length was still going up at the end, so the policy had not converged. The official G1 budget is much larger than 50M steps.
2. **Bug in the demo script.** The video script used the default command range, not the training settings, and it did not show the command. So the "walking backward" in run 1 was most likely the robot correctly following a random backward command. In run 2, the robot got commands it had never seen during training.
3. **Changing the command range changed the environment.** I reverted this so the task stays the standard one.

**Fixes**
- The environment is back to the Playground defaults.
- The PPO settings are now the official ones, and the M2 network is in place.
- The demo uses the same environment as training and a fixed command (0.5 m/s forward), which is printed in the output.
- The training script now logs parameter counts, checks batch settings, and can resume from a checkpoint.

## 5. Current Status

This section describes the old pilot. It is superseded by sections 7 and 8, where M2 is
trained on Minh's flat terrain for 100M steps and evaluated. Final results are in section 8.

## 6. Notes and Limits

- The actor input already includes the **previous action**, so the policy has a very small "memory" of one step.
- One seed is not enough. At least 3 seeds are planned.
- On Kaggle, library versions can change between sessions, so I record the versions for each run.

---

## 7. Update: rebuilt on Minh's baseline (flat terrain only)

The runs described in sections 2–5 used the default Playground environment
(`G1JoystickFlatTerrain`). That is **not** the baseline the team agreed on. The
baseline is Minh's code on `main`, and it must not be changed. So the old training
code was deleted and M2 was rebuilt on top of Minh's files. Only **flat** terrain is
trained for now; other terrains come after flat works.

The old code is kept outside the repo for reference only
(`/home/oanh/Bipedal-old-M2-reference`). Its results (sections 4–5) are a pilot and
must not be compared with M1/M3/M4.

### 7.1 Minh's files used as they are (not edited)

| File | Used for |
|---|---|
| `configs/terrain.yaml` | Terrain stages, arena 80×80 m, 8 cm cells, seed 42 |
| `src/envs/terrain.py` (`TerrainGenerator`) | Builds the flat heightfield |
| `src/envs/terrain_utils.py` | `export_heightfield_to_mujoco_asset`, `patch_mjcf_with_terrain` put the heightfield into the robot scene |
| `src/envs/g1_terrain_env.py` (`G1EnvConfig`, `UnifiedG1TerrainEnv`) | Command ranges (vx 0.2–1.0, vy ±0.3, wz ±0.5), 50 Hz control / 500 Hz physics, spawn rule, `push_enabled=False`, `friction_switch_enabled=False`, `nominal_friction=1.0` |
| `scripts/train_kaggle.py` | Defaults: `--num-envs 4096`, `--seed 42`, `--difficulty 0.0` |

`git diff main` shows no change in any of these files. `scripts/benchmark_gpu.py`,
which I had edited earlier, was restored to `main`.

### 7.2 What I added (all new files, in `flat_walking/`)

Minh's `UnifiedG1TerrainEnv` has no physics, reward or PPO. It only creates terrain,
spawn points and commands. To make training possible I added:

| File | What it adds |
|---|---|
| `g1_minh_env.py` | An adapter. It builds Minh's flat terrain, replaces the `floor` of the Playground G1 scene with Minh's heightfield, and forces Minh's command ranges, spawn rule and push setting on the Playground G1 task |
| `train_m2_flat.py` | Training script (Brax PPO) and a `--smoke` check |
| `m2_flat.yaml` | Only what Minh does not define: PPO settings and the M2 network. No environment values here, they are read from Minh's code |
| `m2_flat_kaggle.ipynb` | Kaggle notebook: install, smoke check, train, zip results, evaluation, demo video |
| `eval_m2_flat.py` | Checks and backs up the checkpoint, analyses training curves, rolls out the saved policy and measures velocity tracking |
| `make_demo_video.py` | Renders the demo video: robot on the left, commanded vs actual velocity plots on the right |
| `readme_report.md` | This report |

### 7.3 Things taken from Playground because Minh's code does not define them

- **Robot:** Unitree G1 with **29 joints** (Playground) — Minh's README says 23 joints
  and 58 observations, but his code has no robot model.
- **Observation:** actor `state` 103 values, critic `privileged_state` 216 values.
- **Reward, termination, action scale (0.5), sensor noise, episode length (1000 steps).**
- **PPO settings:** the official Playground G1 settings (entropy 0.005, γ 0.97, GAE λ 0.95,
  batch 256 × 32 minibatches, 4 updates, lr 3e-4, unroll 20), 100M steps.
- **M2 network:** actor 896-448-128 (559,994 params), critic 512-256-128.

### 7.4 Differences from the old pilot

| | Old pilot | Now |
|---|---|---|
| Ground | Plane | Minh's flat heightfield (80×80 m) |
| Commands | vx −1..1, vy ±0.5, yaw ±1 | vx 0.2..1.0, vy ±0.3, yaw ±0.5 (Minh) |
| Zero command | 10% of the time | Never (Minh's rule) |
| Spawn | Near origin | Random in central region of the arena (Minh's rule) |
| Random pushes | On | Off (`push_enabled=False`) |
| Parallel envs | 8192 | 4096 (Minh's default) |
| Floor friction randomization | Playground randomizer, U(0.4, 1.0) | Not used |
| Domain randomization (masses) | On | Not used |

I did not use the Playground domain randomizer. Minh's setup has no randomization, and the
randomizer changes floor friction, joint friction loss, armature, link masses and the
default joint pose. Friction changes are also what Phase 6 of the plan wants to test later,
so they should not be mixed into training without a team decision.

### 7.5 Problems to report to Minh (not fixed by me)

1. `scripts/train_kaggle.py` stops at "Ready for PPO rollouts" and has no PPO loop.
2. `UnifiedG1TerrainEnv` does not connect to MuJoCo.
3. `HistoryConfig.obs_dim` defaults to 47, but `UnifiedG1TerrainEnv` uses 58, and the real
   Playground G1 observation is 103.
4. README says 23 actuated joints, but the Playground G1 has 29.
5. `UnifiedG1TerrainEnv.__init__` hard-codes `obs_dim = 58`, `action_dim = 23`.

### 7.6 Status

- Training finished (100M steps) and the policy was evaluated. See section 8.
- Not done yet: see sections 8.8 and 8.9.

## 8. Results (M2, flat terrain, seed 42)

Run: `g1_flat_m2_20261126_083748`, trained on Kaggle (2× Tesla T4).
Commit used for training: `4fc7f52`. Evaluation code: `eval_m2_flat.py` (`900db09`).

### 8.1 Training

| Item | Value |
|---|---|
| Environment steps | 102,727,680 (budget 100M, rounded up by Brax to whole batches) |
| Wall time | 14,029 s (about 3.9 h) |
| Throughput (average) | about 7,300 steps/s |
| Actor / critic parameters | 559,994 / 275,457 |
| Final eval reward | 42.28 (best, reached at the last eval) |
| Mean of last 3 evals | 41.39 ± 0.67 |
| Final eval episode length | 1000 / 1000 |

| Step | eval reward | eval length |
|---|---|---|
| 21M | 3.3 | 349 |
| 27M | 20.1 | 854 |
| 32M | 30.4 | 986 |
| 48M | 37.4 | 973 |
| 70M | 41.0 | 993 |
| 81M | 41.6 | 1000 |
| 103M | 42.3 | 1000 |

- Episode length reaches 95% of its maximum at 32M steps. After that, only the reward grows.
- Reward grows from 37.4 (48M) to 42.3 (103M), about +7% over the second half of training.
  It is flattening, but it was not fully flat at the end. The automatic script printed
  "converged" with a loose 10% rule, and also "still improving" with a 5% rule. I would
  say "close to converged"; more steps could give a small gain.
- Largest reward terms at the end (episode sums): `tracking_lin_vel` 929, `feet_phase` 866,
  `tracking_ang_vel` 669. Largest penalties: `feet_slip` −169, `joint_deviation_hip` −68, `pose` −48.

### 8.2 Checkpoint

- 19 checkpoints were saved (every 5.4M steps, from 5.4M to 102.7M), each 13–15 files and about 3.1 MB.
- `resolved_config.yaml`, `metrics.jsonl`, `run_metadata.json` are present.
- Backed up as `/kaggle/working/m2_flat_backup/g1_flat_m2_20261126_083748.zip` (59.2 MB)
  and `/kaggle/working/m2_flat_results.zip`. **Download these before the Kaggle session ends.**
- The final checkpoint (`000102727680`) was loaded again and used for the rollouts below, so
  loading works. A formal test that a reload gives identical actions is not written yet.

### 8.3 Rollout (deterministic policy, 64 envs × 1000 steps, first 1 s ignored)

Commands are inside Minh's range: vx 0.2–1.0, vy ±0.3, wz ±0.5.

| Scenario | Survived | Mean length | RMSE vx (m/s) | RMSE vy (m/s) | RMSE wz (rad/s) | Mean vx |
|---|---|---|---|---|---|---|
| Random commands | 100.0% | 1000.0 | 0.071 | 0.118 | 0.188 | +0.58 |
| Forward vx = 0.5 | 98.4% | 985.2 | 0.054 | 0.108 | 0.162 | +0.47 |
| Forward vx = 1.0 | 100.0% | 1000.0 | 0.080 | 0.105 | 0.179 | +0.94 |

- "Survived" is the share of envs not terminated within 1000 steps (20 s). With 64 envs,
  98.4% means one env fell in the vx = 0.5 test. The cause was not examined.
- The first version of the script skipped sideways, turning and standing-still scenarios.
  They used vx = 0, which is outside Minh's vx range (0.2–1.0), so the policy never saw such
  commands in training. These scenarios are now redefined with vx ≥ 0.2 (see 8.7).
- Velocity is measured from the pelvis (body frame, from `qpos`/`qvel`).
- Robot walks slightly slower than commanded (mean 0.47 for 0.5, 0.94 for 1.0).

### 8.4 Demo video

`demo_m2.mp4`: 18 s, 6 phases of 3 s (slow forward, fast forward, turn left, turn right,
diagonal left, diagonal right). The robot did not fall. Left: robot with tracking camera.
Right: commanded vs actual vx, vy, wz, with the current command and speed in the title.
RMSE over the video: vx 0.116, vy 0.156, wz 0.202. This is higher than the random-command
rollout because the video changes the command abruptly every 3 s, and the transient is counted.

### 8.5 Against the plan

| Plan item | Status |
|---|---|
| Baseline environment of Minh not modified | Done (`git diff main` shows no change in `src/`, `configs/`, `scripts/`) |
| M2 uses `o_t`, larger MLP (capacity control) | Done (actor 559,994 parameters) |
| Logs, config, git commit, seed, checkpoint (Phase 3) | Done |
| Deterministic evaluation on fixed scenarios | Done (fixed seed 123) |
| Velocity error and fall rate | Done for flat ground |
| Trajectory video with target and actual speed | Done |
| Training time, parameter count | Done |

### 8.6 Definitions used for the metrics

Code: `m2_metrics.py` (numpy only). Tests: `tests/test_m2_metrics.py` (12 tests on synthetic
trajectories: perfect tracking, offset error, warm-up, dead steps, falls, thresholds).

- **Fall:** the environment returns `done` within the 1000 steps (20 s) of an episode. The
  Playground termination rule is used. Minh's code defines none.
- **Success rate = 1 − fall rate.** Falls count as failures, and the step of each fall is recorded.
- **Velocity error:** RMSE of (actual − command) for vx, vy (body frame) and wz, over steps
  where the robot is still alive, ignoring the first 50 steps (1 s).
- For each fall, the script also records the pelvis height and uprightness at the step of `done`.

### 8.7 Added for the flat terrain (after review against the plan)

| Item | File | Status |
|---|---|---|
| Fixed scenario file with seed, shared by all models | `eval_scenarios.yaml` (10 scenarios, vx ≥ 0.2, inside Minh's range) | Written; a test checks the range |
| Metric unit tests | `tests/test_m2_metrics.py` | 12 tests pass on CPU |
| Checkpoint reload test | `tests/test_m2_checkpoint.py`, `eval_m2_flat.py --reload` | Written, **not run yet** (Kaggle, Cell 13) |
| Library versions, VRAM, inference latency | `eval_m2_flat.py --resources` → `resources.json` | Written, **not run yet** (Cell 13) |
| Diagnosis of the env that fell at vx = 0.5 | `eval_rollout.json` → `fall_diagnostics` | Needs the new rollout (Cell 10) |

The table in 8.3 comes from the first rollout (random, 0.5, 1.0 only). Re-run Cell 10 to get all
10 scenarios, then update 8.3.

### 8.8 Open questions for the team (not decided yet)

M2 is my own version, built on Minh's code. The team still has to decide which setup is final.

1. **Success thresholds.** `eval_scenarios.yaml` uses placeholders (success ≥ 95%, RMSE vx and
   vy < 0.25 m/s, wz < 0.30 rad/s), marked `provisional: true`.
2. **Randomization.** M2 trains with no pushes and no friction switch (Minh's defaults). Is this
   the frozen environment v1?
3. **Observation, reward, termination.** M2 uses Playground's G1 (29 joints, 103-dim actor input,
   Playground reward). Will M1, M3 and M4 use the same?
4. **Seeds and budget.** One seed (42), 100M steps, about 3.9 h on 2× T4. Is 3 seeds affordable?
5. The five code problems listed in 7.5.

### 8.9 Still not done / limits

1. One seed only.
2. Flat nominal only: no pushes, no friction switch, no held-out friction.
3. Reward still rose about 7% in the second half of training. It is close to converged, not fully flat.
4. Reload test, resource numbers and the 10-scenario rollout have not been run on Kaggle yet.
5. No comparison with M1, M3, M4 (not in this branch).
