# PLAN: Temporal Context for Adaptive Humanoid Locomotion

> **Status:** Proposed — ready for technical review  
> **Project:** Evaluating Temporal Context for Rapid Adaptation in Humanoid Reinforcement Learning  
> **Robot:** Unitree G1 · **Simulation:** MuJoCo Playground · **RL:** PPO  
> **Compute:** 1 × NVIDIA A10G (24 GB) on AWS EC2  
> **Document purpose:** Ordered, verifiable technical implementation plan; no calendar dates or staffing assignments.

## 1. Objective

Build a reproducible reinforcement-learning experiment that trains a Unitree G1 humanoid to walk, then measures whether observation–action history improves robustness, recovery, and generalization when physical dynamics change.

**Primary learning goal:** understand and explain the RL problem formulation, PPO/GAE, reward design, partial observability, neural policy design, training diagnostics, and controlled evaluation. **This is a course project, not a claim of novel RL algorithms.**

### Research questions

- **RQ1:** How does PPO learn stable velocity-tracking humanoid locomotion?
- **RQ2:** Does temporal context improve performance under changes in dynamics?
- **RQ3:** How does history-window length affect performance and compute cost?
- **RQ4:** When, if ever, does a learned history encoder outperform direct history stacking?

### Definition of done

- [ ] Reproduce a stable, evaluable PPO walking baseline for G1.
- [ ] Train/evaluate a capacity-control MLP, history-stacking policy, and history-encoder policy, or explicitly document resource-related omissions.
- [ ] Evaluate identical checkpoints/scenarios on nominal walking, friction switching, external pushes, and held-out friction settings.
- [ ] Build progressive terrain scenarios: mild rough ground and fixed slopes as advanced benchmarks; stationary stairs as an optional stretch goal.
- [ ] Report velocity error, falls, recovery behavior, learning curves, parameters, runtime, memory, and training sample budget.
- [ ] Run interpretable ablations and record at least one failure case.
- [ ] Publish reproducible code, pinned configuration, seed records, checkpoints or download instructions, analysis, and synchronized demo videos, including a staged multi-terrain presentation demo.
- [ ] Explain PPO, GAE, reward components, POMDP motivation, and limitations in the report.

## 2. Scope and non-goals

**In scope:** planar velocity-command locomotion; simulator physics/contact; PPO; reward analysis; short observation–action history; history encoder; friction changes *during episodes*; pushes; held-out friction; quantitative comparisons; advanced rough-ground and fixed-slope benchmarks (after core locomotion is reliable).

**Optional / stretch:** climbing **stationary physical stairs** (not an escalator or moving staircase); mass/payload randomization; additional history lengths or encoders. Only start stair-specific training after flat, friction, push, and basic terrain results are preserved.

**Out of scope:** physical robot deployment; escalators/moving stairs; running, jumping, stand-up-after-fall; inventing a new RL algorithm; distributed multi-GPU training. Stair climbing is explicitly a non-blocking stretch goal, not a core acceptance criterion.

## 3. Technical assumptions and decisions

| Area | Working decision | Validation / caveat |
|---|---|---|
| Simulation | MuJoCo Playground G1 locomotion task | Confirm task name/API at pinned revision; don't assume a particular environment identifier. |
| Physics engine | Benchmark available MJX-JAX and MJX-Warp paths | Choose **one validated backend** for all final comparisons; backend support may vary by task/version. |
| RL trainer | Reuse Playground/Brax PPO pipeline | Read PPO rollout, GAE, losses and optimizer; do not rewrite trainer without a reason. |
| Model framework | JAX ecosystem; Flax/Optax if compatible with pinned code | Do not add redundant frameworks to the training path. |
| Hardware | Single A10G 24 GB | Benchmark compilation, simulation steps/sec, wall time, VRAM, and checkpoint restore first. |
| Training execution | Sequential runs, persisted checkpoints | No assumption that 4 models × 3 seeds fits until pilot costs are measured. |
| Policy inputs | Proprioception + command (as exposed by task) | Actor must not receive hidden friction parameter; distinguish actor and critic input clearly. |
| Fair comparison | Same robot, reward, randomization, environment-step budget and evaluation scenarios | Record model-specific observation dimension, parameter counts, and architecture changes. |
| Research claims | Report evidence, including negative results | Robustness alone does not prove online adaptation. |

## 4. Target repository structure

These are **proposed files**; reconcile names with the actual upstream repository before implementation.

```text
.
├── PLAN.md
├── README.md
├── pyproject.toml / requirements-lock.txt
├── configs/
│   ├── environment.yaml
│   ├── ppo_baseline.yaml
│   ├── larger_mlp.yaml
│   ├── history_stack.yaml
│   ├── history_encoder.yaml
│   ├── evaluation.yaml
│   └── terrain.yaml
├── src/
│   ├── envs/                  # G1 setup, physics changes, wrappers
│   ├── models/                # base policy, stacking and encoder
│   ├── training/              # trainer integration, run/checkpoint logic
│   ├── evaluation/            # scenarios and metrics
│   └── utils/                 # configs, seeding, logging
├── scripts/
│   ├── benchmark_gpu.py
│   ├── train.py
│   ├── evaluate.py
│   ├── run_ablations.py
│   ├── evaluate_terrain.py
│   └── render_comparison.py
├── tests/
│   ├── test_observations.py
│   ├── test_history_buffer.py
│   ├── test_scenario_changes.py
│   ├── test_terrain_geometry.py
│   ├── test_metrics.py
│   └── test_checkpoint.py
├── results/                   # CSV/JSON summaries; no huge raw blobs in git
└── reports/                   # methods, plots, demos, final report
```

## 5. Ordered implementation plan

> For each phase, finish the **Acceptance criteria** before treating it as complete. Save outputs and commands in README or experiment logs. Tasks can be researched in parallel, but final experiments depend on the frozen environment and metrics.

### Phase 1 — RL theory and problem formulation

**Implement / study**
- [ ] Define agent, transition, episode, reward, return, policy, value function and discount factor for G1.
- [ ] Explain MDP versus partially observable settings: friction/mass are latent if not explicitly observed.
- [ ] Derive or reproduce Bellman relations, policy gradient, actor–critic, PPO clipped objective and GAE.
- [ ] Trace actual upstream PPO code: action sampling → rollout → advantage/value targets → minibatch losses → update.
- [ ] Define velocity-command tracking objective and what counts as success/failure.

**Deliverables:** concise RL notes, PPO flow diagram, G1 observation/action/reward specification.

**Acceptance criteria:** able to map every major PPO tensor to its mathematical meaning and identify where reward/advantage enter optimization.

### Phase 2 — AWS and simulator bring-up

**Implement**
- [ ] Check instance/GPU via `nvidia-smi`; verify driver/CUDA/JAX accelerator compatibility.
- [ ] Pin Python, MuJoCo Playground, MuJoCo, JAX, trainer and CUDA-dependent package versions.
- [ ] Load the G1 task; reset, simulate and render a short rollout.
- [ ] Inspect joints/actuators, command representation, observation channels, timestep, policy control rate, termination and safety limits.
- [ ] Benchmark candidate backend(s) and increasing parallel environment counts; record warm-up JIT time, steps/sec, peak VRAM and error modes.
- [ ] Establish persistent storage, logs, checkpoints and reliable noninteractive training sessions.

**Deliverables:** reproducible setup instructions, environment inspection report, A10G benchmark table.

**Acceptance criteria:** headless simulation runs on AWS, reproducibly saves outputs, and chosen backend/configuration fits memory with margin.

### Phase 3 — PPO walking baseline

**Implement**
- [ ] Reproduce upstream PPO on an unmodified G1 environment first.
- [ ] Document actor/critic inputs and architecture; inspect reward terms (tracking, orientation/posture, energy/torque, smoothness, termination).
- [ ] Log episodic return, individual reward components, tracking error, falls, policy loss, value loss, entropy, approximate KL and clip fraction when available.
- [ ] Save/restore checkpoints including model state, optimizer state when supported, seeds and configuration.
- [ ] Evaluate deterministically or using a documented action-selection protocol on fixed nominal scenarios.

**Deliverables:** baseline checkpoint, training logs, trajectory video, nominal metrics.

**Acceptance criteria:** stable repeatable nominal locomotion with clearly specified success threshold; no undocumented modifications to the upstream task.

### Phase 4 — Reward understanding and environment freeze

**Implement**
- [ ] Inspect reward scale and each component's behavioral effect.
- [ ] Run controlled reward ablations (one factor at a time) on the baseline; distinguish learning failure from controller failure.
- [ ] Define training dynamics distribution and prevent evaluation leakage.
- [ ] Freeze **environment v1**: action space, reward, control frequency, termination, randomization, observation contract and evaluation metric formulas.
- [ ] Version the frozen configuration; variants must share it for primary comparisons.

**Deliverables:** reward-ablation plots and immutable environment/benchmark config.

**Acceptance criteria:** reward changes are explained with actual metrics and final comparisons use the same frozen task.

### Phase 5 — Temporal inputs and comparison models

**Model definitions**

| ID | Policy input | Architecture | Scientific role |
|---|---|---|---|
| M1 | Current observation `o_t` | Standard PPO MLP | Baseline |
| M2 | `o_t` | Larger PPO MLP | Parameter-capacity control |
| M3 | `o_t` + last `k` observation–action pairs | PPO with history stacking | Value of explicit temporal input |
| M4 | `o_t` + encoded history `z_t = f_φ(h_t)` | PPO with history encoder | Value of learned history representation |

**Implement**
- [ ] Establish a causal history contract `h_t = {(o_{t-k}, a_{t-k}), …, (o_{t-1}, a_{t-1})}` and consistent action timing.
- [ ] Implement history buffering, padding, masking and reset on episode boundaries; no future observations.
- [ ] Verify observation/action-history normalization and exact shape consistency in batched JAX execution.
- [ ] Train M3; test several history durations based on actual policy Hz (e.g. around 0.1/0.3/0.5 s, subject to compute).
- [ ] Train M2 with parameter count reasonably matched to history variants; document unavoidable differences.
- [ ] Build M4 with a simple MLP history encoder and end-to-end PPO training; verify encoder gradients and checkpoint restoration.
- [ ] Use matched environment steps and compare both return and held-out metrics; do not tune against final test scenarios.

**Deliverables:** four model configurations/checkpoints, parameter-count table, history-buffer tests, pilot curves.

**Acceptance criteria:** input interfaces and model roles are documented; history does not cross episode boundaries; models train without numerical failure or silent leakage.

### Phase 6 — Controlled dynamics benchmark

**Implement**
- [ ] **Nominal walking:** fixed start states/velocity command sets and episode length.
- [ ] **Friction switching:** start with nominal friction, switch at a known in-episode event; record exact event step, old/new coefficient and contact behavior.
- [ ] **Push disturbance:** apply matched impulses or bounded forces from defined directions; record magnitude, duration and application point.
- [ ] **Held-out friction:** define disjoint test settings, including out-of-training-range values where physically meaningful.
- [ ] Maintain identical scenario files and evaluation seeds for each checkpoint; validate simulator parameter changes genuinely affect contact forces.
- [ ] Keep mass/payload adaptation optional, not a prerequisite for completion.

**Deliverables:** scripted deterministic benchmark scenarios and saved scenario metadata.

**Acceptance criteria:** parameter switching occurs at the correct timestep, evaluates identically across models, and does not expose true friction as an actor input.

### Phase 6A — Progressive terrain challenges (advanced scope)

**Goal:** extend controlled locomotion from flat ground to more complex **static** contact geometry without replacing the original temporal-context research question. Keep these experiments separate from the frozen core friction benchmark.

**Implement**
- [ ] **Rough terrain:** generate repeatable low-amplitude heightfields or uneven blocks, with configurable roughness amplitude, spatial frequency and deterministic terrain seeds.
- [ ] **Slopes:** add static uphill/downhill ramps with predefined mild inclines; validate joint clearance and correct friction/contact behavior.
- [ ] **Stationary stairs (stretch):** generate rigid, fixed-height steps; start with low risers and broad treads. **Do not** model escalators or moving steps.
- [ ] Define a **progressive curriculum**: flat ground → mild roughness → slopes → optional low stairs → more demanding stair settings only if training succeeds. Use training curricula only when necessary; do not label a terrain trained on as unseen evaluation.
- [ ] First test existing flat-trained checkpoints **zero-shot** on advanced terrain; if traversal fails, train terrain-aware variants and label results separately from the original flat-trained policy comparison.
- [ ] Maintain fair inputs across M1–M4. If adding terrain height scans, perception, or foot-contact features, make them available equally across policies and disclose the observation change.
- [ ] Add terrain metrics: distance/progress, traversal completion rate, fall rate, tracking error and contact/slip statistics where reliable. For stairs, count successfully negotiated steps and support-phase failures.
- [ ] Control for environmental confounds: same terrain geometry/seed, command, starting pose, episode time horizon and fixed camera across model comparisons.
- [ ] Profile simulation throughput and memory on the single A10G; cap terrain variants if contact complexity causes excessive compute cost.

**Deliverables:** `configs/terrain.yaml`, rough-ground/ramp generators, optional fixed-stair generator, terrain evaluation scripts, annotated videos and per-terrain result tables.

**Acceptance criteria:** rough ground and slopes are geometrically reproducible and runnable; evaluation distinguishes zero-shot transfer from terrain-trained results. Stair success is **optional** and must not block core project completion.

### Phase 7 — Evaluation and statistical analysis

**Implement**
- [ ] Define **success rate** and **fall rate** with explicit termination criteria.
- [ ] Compute velocity tracking error over identical windows and commands.
- [ ] Define **recovery time** from event onset to tracking error under a prespecified threshold sustained for a prespecified interval.
- [ ] Record never-recovered/fallen episodes as failures or censored observations; **never** encode as zero recovery time.
- [ ] Record immediate post-event error, integrated post-event tracking error and recovery success to separate robustness from recovery.
- [ ] Evaluate comparable scenario seeds for all M1–M4; report uncertainty across independent training runs when feasible.
- [ ] Report rough-ground and slope metrics in separate advanced benchmark tables; if fixed stairs are attempted, report step-completion and failure modes without conflating trained vs zero-shot tests.
- [ ] Compare environment steps, wall-clock training time, simulation throughput, memory, inference cost and parameter count.
- [ ] Separate training-seed variability from within-checkpoint episode variability; don't treat thousands of episodes as independent training runs.

**Deliverables:** evaluation CLI, structured results, comparison tables/plots and reproducible metric definitions.

**Acceptance criteria:** each metric is unit-tested on synthetic trajectories; conclusions include uncertainty and negative cases.

### Phase 8 — Ablations and interpretation

**Core studies**
- [ ] M1 vs M3: temporal input benefit.
- [ ] M1 vs M2: extra-capacity benefit.
- [ ] M3 vs M4: learned encoder vs raw stacking.
- [ ] Seen vs held-out friction: generalization.
- [ ] Pre- vs post-switch trajectories: transient loss and recovery.

**Supporting studies if feasible**
- [ ] History length vs recovery/compute.
- [ ] Valid vs masked/stale history, clearly flagged as potentially out-of-distribution intervention.
- [ ] Reward component ablation; optional encoder latent visualizations.

**Deliverables:** ablation matrix, plots, failure-case narratives, answers to RQ1–RQ4.

**Acceptance criteria:** each claim is linked to a named experiment and a result; correlation or history masking alone is not described as proof of online adaptation.

### Phase 9 — Reproducibility and delivery

**Implement**
- [ ] Pin dependency versions and record upstream repository revisions.
- [ ] Export per-run config, git commit, seed, backend/GPU, checkpoint path and summary metrics.
- [ ] Test fresh-environment installation, one smoke training run, checkpoint reload and deterministic evaluation.
- [ ] Produce synchronized comparison videos with common camera, timing, commands and event markers.
- [ ] Prepare the conference-style demo sequence specified in **Section 11**, including offline videos as the reliable presentation fallback.
- [ ] Write final report: theory → formulation → methods → experimental protocol → results → ablations → failures → limitations.
- [ ] Provide command examples that match the *actual* implemented CLI, not placeholder syntax.

**Deliverables:** repository, docs, results/figures, checkpoints or reproducible links, technical report and demo.

**Acceptance criteria:** someone unfamiliar with the work can install, run the smoke test, evaluate at least one checkpoint and locate evidence for reported conclusions.

## 6. Experimental controls and integrity rules

1. **No training on held-out tests.** Validation for choosing hyperparameters must be distinct from the final evaluation conditions.
2. **No silent differences:** record differences in policy input, architecture, parameter count, privileged critic inputs and normalization.
3. **Same primary budget:** compare models at the same number of environment interactions; report actual wall-clock cost separately.
4. **Same evaluation:** identical physics initializations, commands, perturbation schedules and termination rules.
5. **Multi-seed target:** aim for ≥3 independent training seeds per model **only if measured A10G budget allows**. Document reduced seed counts transparently.
6. **Claims grounded in measures:** strong post-change walking might reflect robustness rather than a memory-dependent adaptation mechanism.
7. **Preserve negatives:** log diverged runs, policy collapse, compilation errors, unexpected robot motion and implementation fixes.

## 7. Validation checklist

### Automated checks
- [ ] Observation/action dimensions, bounds and finite values validated.
- [ ] History buffer ordering, episode reset, padding and causal alignment tested.
- [ ] MLP/encoder forward pass, JAX JIT and batched rollout pass smoke tests.
- [ ] Checkpoint save/load returns consistent actions under fixed PRNG inputs.
- [ ] Friction-switch and push-event timing validated.
- [ ] Metric calculations tested on synthetic success, fall, no-recovery, and never-disturbed cases.
- [ ] Same scenario seed produces comparable initial conditions between models.

### Manual / experiment checks
- [ ] G1 walking video matches expected command direction and physical behavior.
- [ ] Reward-component plots correspond to actual behavior.
- [ ] GPU memory remains below safe capacity; compilation overhead is measured separately.
- [ ] No benchmark-specific tuning after final test results are inspected.
- [ ] Figure labels include units, number of seeds and meaning of uncertainty bands.

## 8. Risks and fallbacks

| Risk | Detection | Fallback |
|---|---|---|
| A10G throughput/VRAM insufficient | Measured steps/sec, OOM, compile duration | Reduce parallel environments or network size; keep same core benchmark and fewer pilot variants. |
| Backend/task incompatibility | Environment build or rollout fails | Use backend supported by pinned task; do not switch backend across model comparisons. |
| PPO locomotion not converging | Reward, tracking and fall diagnostics | Reproduce reference config/checkpoint, inspect action/reward semantics before changing algorithms. |
| Encoder instability | NaNs, gradient norms, collapse | Simplify encoder, reduce history length, preserve M3 as strong temporal baseline. |
| Weak or null history benefit | Controlled comparison shows overlap | Report negative findings, history cost and conditions where it is unnecessary. |
| Seed budget too expensive | Pilot wall time × planned runs | Preserve M1/M3/M4 evaluation; reduce optional ablations and state statistical limits. |
| Evaluation leakage | Test-specific tuning, input contamination | Restore frozen scenario definitions and rebuild independent held-out set. |
| Lost progress | Interrupted EC2 job or ephemeral storage | Periodic checkpoints on persistent EBS/S3 plus restore smoke tests. |

## 9. Priority and sequencing

**P0 — Core path:** theory → AWS/simulator → PPO baseline → frozen environment → history stacking → friction benchmark → evaluation → report/demo.

**P1 — Main comparisons:** larger MLP, history encoder, held-out friction, independent seeds, direct encoder-vs-stacking comparison.

**P2 — Additional depth:** rough terrain and fixed slopes (advanced demo/benchmark), stationary stairs (stretch only), multiple history windows, history interventions, expanded reward ablations, mass/payload and more advanced representation analysis.

**Stop rule:** when compute or implementation risk threatens core evaluation, stop P2 work; never sacrifice a credible baseline, common benchmark or reproducibility for terrain challenges. Stair climbing cannot block final delivery.

## 10. Open technical decisions to resolve by measurement

- [ ] Exact G1 Playground task/environment identifier and version.
- [ ] Working MJX backend(s) on this AWS image and throughput per configuration.
- [ ] Policy/control frequency and history durations expressed in actual policy steps.
- [ ] Operational thresholds for walking success, fall, velocity error and recovery.
- [ ] Training budget per run and feasible number of independent seeds.
- [ ] Whether default critic observations include privileged state; ensure documented parity across variants.

## 11. Presentation demo plan — same robot, same challenge, different memory

**Audience takeaway:** temporal memory may change a humanoid policy’s tracking, stability and recovery when conditions change. The demo must show **matched inputs and measurable outcomes**, not just selected successful clips.

### Demo A — synchronized controlled comparison (required)

1. **Opening (brief):** show the MuJoCo Unitree G1, the target velocity command and the four policy variants: M1 Standard PPO, M2 capacity-matched Larger MLP, M3 History Stacking, M4 History Encoder.
2. **Flat walking baseline:** show M1/M3/M4 side-by-side under matched initial states and command trace. Overlay target vs actual speed and a fall indicator.
3. **Friction switch (main result):** begin on normal friction; at a clearly marked frame switch to a lower coefficient **mid-episode**. Use exactly the same event step and scenario seed for every model. Add a vertical event line in the error plot.
4. **External push:** replay a matched impulse and show post-perturbation balance, recovery success and any fall. Do not imply a successful push recovery proves friction adaptation.
5. **Quantitative reveal:** present per-model mean/uncertainty where available for immediate tracking error, integrated post-event error, recovery time (or no recovery), recovery rate and fall rate.
6. **Negative example:** include at least one scenario where a history model fails or offers no gain, then briefly explain likely causes and limits.

### Demo B — progressive terrain showcase (advanced; attempt after Demo A works)

**Route:** `Flat Ground → Low-Friction Patch → Mild Rough Terrain → Fixed Slope → Stationary Stairs (stretch only)`.

- Build the route as separate controlled segments or connected sections with unambiguous boundary/event markers. Segmenting is preferred when a single connected course introduces uncontrolled carryover or reset bias.
- On **rough terrain**, show stable walking and a traverse-completion indicator; report terrain seed and roughness settings.
- On **fixed slopes**, show ascent and descent separately, with comparable commands and tracking error.
- On **stationary stairs**, if feasible, show a low-rise fixed staircase and report successful steps vs failed attempts. This is **not** an escalator and need not be attempted live.
- Use M1/M3/M4 synchronized playback for comparison if **all three** have been evaluated under matched conditions. Otherwise explicitly label the segment a *single-policy capability demo*, not evidence of temporal-memory superiority.
- Clearly state whether the policy was **zero-shot tested** from flat training or **retrained/fine-tuned** on terrain. Never mix these claims on one unlabeled comparison plot.

### On-screen overlays and deliverable assets

- [ ] Identical camera angle, playback speed, fixed scenario seed and control-command trace per comparison.
- [ ] Clearly visible policy name, terrain label, current target velocity, observed speed, friction/event marker, fall/recovery status.
- [ ] Small synchronized time-series plot of target/actual velocity and tracking error; readable units and event-time marker.
- [ ] Final comparison table with scenario count, training seeds, success/fall rate, recovery statistics and any non-recovered trials.
- [ ] Short annotated **offline MP4s** for each segment, plus an optional live replay, to avoid AWS/GPU/network failures during defense.
- [ ] One explanatory slide: *Why history could help* (POMDP/hidden dynamics); one slide: *What results actually show* (robustness vs adaptation vs generalization).

**Demo acceptance:** the core flat/friction/push demonstration works as prerecorded synchronized playback and matches reported metrics. Advanced rough/slope clips increase visual impact but are not required for scientific claims; stairs remain an explicitly optional challenge.

---

## 12. Primary references

- [Anthropic: Plan mode as the default starting point](https://academy.claude.com/courses/ai-native-sdlc-playbook/plan-mode) — implementation-plan style emphasizing scope, order, risks and proof through tests.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground)
- [MuJoCo documentation](https://mujoco.readthedocs.io/en/stable/)
- [JAX documentation](https://docs.jax.dev/)
- [PPO paper](https://arxiv.org/abs/1707.06347)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
- [Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034)

---

**Plan maintenance:** Update this document whenever implementation assumptions change; distinguish completed checkboxes from planned work and preserve the rationale for changes. `PLAN.md` is a repository convention for this project, not a special Claude Code runtime file.
