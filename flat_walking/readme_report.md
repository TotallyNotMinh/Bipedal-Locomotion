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

M2 is **being retrained from scratch** with the final setup (100M steps). Results will be added here after training.

**How I will judge success**
- Eval episode length should get close to 1000 (no fall in 20 s).
- Low velocity tracking error.
- The robot walks forward smoothly in the demo video with the 0.5 m/s command.

**Planned tests after training**
- Velocity tracking error and fall rate.
- Recovery after pushes.
- Walking with different floor friction, including values outside the training range.
- Training time and inference cost.

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
| `m2_flat_kaggle.ipynb` | Kaggle notebook: install, smoke check, train, zip results |
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

- Check on CPU: environment builds, `floor` becomes a heightfield, reset and 20 steps run.
- **No real training has been done yet.** Run `m2_flat_kaggle.ipynb` on Kaggle.
