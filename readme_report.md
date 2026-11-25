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
