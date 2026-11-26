# M1 and M3 Training Report

The 1.6 million-step Kaggle runs produced working checkpoints for both models. The updated M1 run has much lower mean KL than the first run, but all ten evaluation episodes still end early. Verify which trainer produced the updated M1 checkpoint, then use that version to test what limits flat-ground walking. Once M1 walks reliably, train M3 with the same setup and compare the benefit of history.

## Work completed

- Set up the M1 Kaggle notebook to load the G1 meshes, reuse or install a compatible training runtime, verify both T4s, and train and evaluate M1 independently. The M3 comparison is optional. [M1 notebook](kaggle/m1-baseline.ipynb)
- Checked the imported M1 and M3 results. Both trained for 1.6 million environment steps on two T4s, with eight environments, training seed 42, learning rate `1e-4`, the same flat terrain, and the same ten evaluation seeds. Both checkpoints reproduced their saved probe actions on local reload. [M1 settings](results/m1_1p6m_env8_lr1e4/settings.json) · [M3 settings](results/m3_1p6m_env8_lr1e4/settings.json)
- Added `--diagnose-ppo` for M1 and ran it locally for 50 updates, or 8,000 environment steps. It logs KL before SGD and the fraction of sampled action components with magnitude at least 0.95 on every update. The diagnostic checkpoint passed reload, and 26 tests passed when the change was verified. [Diagnostic implementation](src/training/m3_pilot.py) · [Diagnostic data](results/m1_ppo_diagnostic_local/ppo_diagnostic.jsonl)

## What the runs show

The saved evaluations use deterministic actions. Mean duration is the mean episode length multiplied by the 0.02 s control period. Velocity error measures the difference between commanded and actual planar velocity. Each episode contributes its own mean tracking error, so a shorter episode can have lower error while still failing earlier.

| Policy | Mean duration | Mean velocity error | Mean return | Early terminations |
| --- | ---: | ---: | ---: | ---: |
| M1 initial, updated run | 1.064 s | 1.153 m/s | −3.76 | 10/10 |
| M1 trained, updated run | 0.964 s | 1.099 m/s | −2.75 | 10/10 |
| M3 initial | 1.068 s | 1.217 m/s | −3.66 | 10/10 |
| M3 trained | 1.028 s | 1.332 m/s | −9.81 | 10/10 |

[M1 initial evaluation](results/m1_1p6m_env8_lr1e4/initial_evaluation.json) · [M1 trained evaluation](results/m1_1p6m_env8_lr1e4/trained_evaluation.json) · [M3 initial evaluation](results/m3_1p6m_env8_lr1e4/initial_evaluation.json) · [M3 trained evaluation](results/m3_1p6m_env8_lr1e4/trained_evaluation.json)

The first M1 run averaged 0.812 s after training. The updated checkpoint reaches 0.964 s, short of its own initial policy's 1.064 s. In local replays, **all ten episodes from the updated M1 ended on foot-to-foot contact**. Only 0.2% of the deterministic action components had magnitude at least 0.95. The earlier M1 ended five episodes on foot-to-foot contact and five on torso inversion. The saved M3 ended all ten on torso inversion, including one with foot-to-foot contact. Some local episodes ended at different steps than in Kaggle.

A separate local check held lateral and turning commands at zero. With a stand-still command, M1 averaged 0.66 m/s forward over 1.02 s. With a 0.50 m/s forward command, it averaged 0.62 m/s over 1.00 s. Foot-to-foot contact ended all ten trials in each case. The positive mean under the stand-still command and the spread across seeds point to weak command tracking in both scenarios. [Fixed-command summary](results/m1_fixed_command_check/summary.json) · [Foot and velocity traces](results/m1_fixed_command_check/trajectories.csv)

## Why walking is still unreliable

**Observation normalization can change the policy before SGD.** The default fixed-rate Brax path collects a rollout using one set of observation statistics and refreshes them before computing the loss. The local pilot uses one minibatch and one SGD pass, so its pre-SGD KL compares the rollout and loss distributions at the same actor weights. KL started at **26.26**, fell to **0.028** by update 3, and reached **0.00031** at update 50. Near-limit action components fell from **12.5%** to **1.2%**. This local run used one RTX 2050 and JAX 0.12.1. [Brax 0.14.2 PPO source](https://github.com/google/brax/blob/v0.14.2/brax/training/agents/ppo/train.py) · [Diagnostic data](results/m1_ppo_diagnostic_local/ppo_diagnostic.jsonl)

**The updated M1 run has low mean KL and still falls.** The reported mean KL was **57.25** in the first M1 run, **0.00379** in the updated M1 run, and **74.01** in the saved M3 run. The updated M1 results folder contains a training summary. It has no per-update PPO trace or record of the normalization order. The recorded settings and package versions match the earlier runs. Preserve the Kaggle notebook and the trainer revision or patch so the lower-KL result can be reproduced. [M1 training metrics](results/m1_1p6m_env8_lr1e4/training.json) · [M3 training metrics](results/m3_1p6m_env8_lr1e4/training.json)

**The observation and rollout budget are two other candidates.** M1 sees 67 values and leaves out direct pelvis linear velocity and foot contacts. M3 adds five previous observation–action pairs for 547 inputs. At 50 Hz this covers 0.10 s and may help the actor infer how the robot responded to a recent action. It also adds 245,760 weights to the first layer. The pilot uses 160 transitions per PPO update for both models. Testing the upstream G1 actor inputs or collecting larger rollouts can help distinguish an observation limitation from a training limitation. [Actor observation code](src/training/m3_env.py) · [Playground G1 PPO configuration](https://github.com/google-deepmind/mujoco_playground/blob/0ce4c2b9589101d239cc833c92dd5f7c8da67b9e/rss2025/mujoco_playground/config/locomotion_params.py)

## What to run next on Kaggle

**1. Confirm the trainer used for the updated M1 run.** Record the Kaggle notebook version, checkout commit, and any edits to Brax. If the updated run used the after-SGD change, keep it as a completed long-run result and move on to the flat-ground walking test below. If the normalization order is still unclear or needs a controlled comparison, run the 160,000-step control. Open the M1 notebook in a GPU T4 x2 session. Set `run = repo / "results/m1_160k_normalizer_before_sgd"` in the first code cell and run through **Set up the training environment**. Confirm it reports two T4s and use the checkout that includes `--diagnose-ppo` and the default before-SGD normalization order.

```python
subprocess.run([
    training_python, "-u", "scripts/train_kaggle.py",
    "--model", "M1", "--num-envs", "8", "--seed", "42",
    "--num-timesteps", "160000", "--diagnose-ppo",
    "--output-dir", "results/m1_160k_normalizer_before_sgd",
], cwd=repo, env=env, check=True)
```

Keep `ppo_diagnostic.jsonl`, `trained_evaluation.json`, `training.json`, and the checkpoint. Inspect the full KL trace, especially after the first few updates.

**2. Compare normalization order if it is still unresolved.** The repository's diagnostic flag measures behavior using the selected trainer. The after-SGD variant requires the trainer implementation used in Kaggle or an equivalent change. Use the current actor and critic statistics to collect the rollout and compute the PPO loss. Apply the optimizer step, then update the observation statistics from the rollout for use on the next update. Keep the learning rate at `1e-4` so the comparison tests update order alone. Repeat the same 160,000-step run in `results/m1_160k_normalizer_after_sgd`. Use the same logging cadence and compare evaluation duration, early terminations, return, and velocity tracking.

| What you see | What to do |
| --- | --- |
| M1 stays upright longer and tracks the commands better | Run matched longer M1 trials and check whether the improvement holds across training seeds. |
| Pre-SGD KL drops but walking stays near one second | Test the upstream G1 actor inputs as a control on the same flat terrain. If that also fails, test a larger rollout batch in a separate M1 run. |
| The short run looks healthy but the 1.6M run fails | Keep logging throughout a longer matched run. Check when KL and near-limit actions rise relative to episode length. |

## How to get M3 walking

Run the M1 notebook with `experiment = "velocity"` before changing history length or terrain. It trains a 67-input actor and a 69-input actor that also receives the pelvis-frame forward and lateral velocities. Both start from scratch and train for 160,000 steps with eight environments, seed 42, and the same terrain, commands, and PPO settings. Use the Python environment and trainer revision from the low-KL run. The notebook evaluates both checkpoints at stand-still and 0.50 m/s forward and reports foot contact, survival, and actual velocity. [M1 notebook](kaggle/m1-baseline.ipynb)

A practical proposed gate is **at least 8 of 10 validation episodes surviving the 20 s horizon**, with measured velocity close to the target in both fixed-command scenarios. Reserve separate scenarios for final reporting.

Once M1 passes that gate, train **M3 with `--history-len 5`** using the same PPO settings, normalization order, terrain, and step budget. Train the first matched M1 and M3 policies from fresh initialization. If the successful M1 used the 69-input actor, add `--actor-velocity` to the M3 command. Run it from the checkout with the successful M1 trainer settings and use the same environment count and budget. The values below match the original 1.6M experiment.

```python
subprocess.run([
    training_python, "-u", "scripts/train_kaggle.py",
    "--model", "M3", "--history-len", "5", "--num-envs", "8",
    "--seed", "42", "--num-timesteps", "1600000",
    "--output-dir", "results/m3_k5_matched_walk",
], cwd=repo, env=env, check=True)
```

First verify that M3 can complete the same flat-ground walking test. Then compare M1 and M3 on the same pushes or traction changes. Measure recovery success, recovery time, and tracking error after each event. Check more than one training seed before drawing a conclusion about history. [Project evaluation plan](PLAN-1.md#phase-7--evaluation-and-statistical-analysis)
