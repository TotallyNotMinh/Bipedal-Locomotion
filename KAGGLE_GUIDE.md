# Running on Kaggle with Dual NVIDIA Tesla T4 GPUs

This guide explains how to train and evaluate the Unitree G1 adaptive locomotion policy on **Kaggle Notebooks with Dual T4 GPUs (2 × 16GB VRAM)**.

---

## 1. Kaggle Notebook Settings

In your Kaggle notebook, configure the following settings in the right-hand panel:

1. **Accelerator:** Select **`GPU T4 x2`**.
2. **Internet:** Turn **`Internet ON`** (required for downloading dependencies and MuJoCo Playground).
3. **Environment:** Leave as Default / Latest.

---

## 2. One-Click Environment Setup

In the first code cell of your Kaggle Notebook, run:

```bash
!bash setup_kaggle.sh
```

### What this script does:
- Detects both Tesla T4 GPUs via `nvidia-smi`.
- Configures memory management variables to prevent OOM errors:
  ```bash
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  export TF_FORCE_GPU_ALLOW_GROWTH=true
  export CUDA_VISIBLE_DEVICES=0,1
  ```
- Installs JAX with CUDA 12 support (`jax[cuda12]`).
- Installs `mujoco`, `mujoco-playground`, `brax`, `flax`, `optax`, `orbax-checkpoint`, and dependencies.
- Runs a self-check verifying that JAX sees both GPU devices (`cuda(id=0)` and `cuda(id=1)`).

---

## 3. Verify Dual-GPU Acceleration & Throughput

In the next cell, run the benchmark script:

```bash
!python scripts/benchmark_gpu.py --batch-size 4096 --steps 500
```

This will measure:
- Multi-device discovery (`2 devices: Tesla T4`).
- Environment sharding across GPUs (e.g. 2,048 environments per GPU).
- JIT compilation time.
- Total stepping throughput (steps/second).

---

## 4. Run Training with Diverse Procedural Terrain

You can train any of the 4 experimental models defined in `PLAN-1.md`:
- **`M1`**: Standard PPO baseline (current observation only)
- **`M2`**: Parameter-capacity control MLP
- **`M3`**: Causal History Stacking ($[o_t, h_t]$)
- **`M4`**: Learned History Encoder ($z_t = f_\phi(h_t)$)

### Example: Train M3 (History Stacking) on Dual T4:
```bash
!python scripts/train_kaggle.py \
    --model M3 \
    --terrain-config configs/terrain.yaml \
    --num-envs 4096 \
    --history-len 5 \
    --seed 42 \
    --output-dir results/m3_run
```

---

## 5. Terrain Configuration & Customization

The procedural terrain arena is configured in `configs/terrain.yaml`:
- **Flat Ground:** Baseline reset & nominal walking (row 0 is always flat).
- **Uniform Rough Ground:** Stepped height variations.
- **Harmonic Perlin Ground:** Smooth multi-frequency undulations.
- **Slopes & Ramps:** Inclines, declines, and pyramid slopes.
- **Stationary Stairs:** Low-rise fixed steps for curriculum/stretch testing.

All terrains are completely reproducible: the same seed generates the identical heightfield matrix and MuJoCo `<hfield>` asset.

### Bounded M3 pilot in an existing runtime

From the repository root, run the smoke check and pilot in sequence. Both use the installed Playground G1 joystick task, the configured terrain, seed 42, and five previous observation–action pairs.

```bash
python scripts/train_kaggle.py --model M3 --smoke &&
python scripts/train_kaggle.py --model M3
```

The smoke run performs two optimizer updates in `results/m3_smoke/`. The pilot performs ten in `results/m3_pilot/`. Each uses one environment per visible JAX device by default. Use a fresh `--output-dir` for a repeat. `--num-timesteps` accepts an exact multiple of the rollout batch size, which the CLI reports for an incompatible budget.

Each directory contains the compiled scene inputs, resolved dimensions and PPO settings, initial and trained evaluations on ten matched seeds, training metrics, and the Brax inference checkpoint. A successful fresh-process action check writes `reload.json`. The pilot samples the initial curriculum stage for the run and feeds completed training episodes into a curriculum update at its end.

If the installed Playground package does not contain the Menagerie meshes, put
the G1 STL files from the Menagerie revision declared by Playground in
`results/m3_robot_assets/`. The pilot reads this directory as a fallback for
meshes missing from the installed package. On a memory-constrained GPU, run
the pilot with `XLA_PYTHON_CLIENT_PREALLOCATE=false` for that command.

### Train M1 and compare it with M3

Run [the M1 baseline notebook](kaggle/m1-baseline.ipynb) in the configured
Kaggle session. Enable Internet for the initial clone and attach the G1 mesh dataset
for training. The first cell clones the `m3-kaggle` branch into
`/kaggle/working/Bipedal-Locomotion` or reuses the existing branch checkout.
The mesh cell copies missing STL files from
`/kaggle/input/datasets/trung1311/robot-assets/results/m3_robot_assets`.
It also accepts `m3-robot-assets.zip` or another extracted folder. The
notebook checks that the training Python can import the PPO trainer. If the
kernel reports a missing `brax`, run **Select the training Python**. For a
custom M3 environment, set `m3_python` in that cell to its Python executable.
The notebook verifies the final M1 checkpoint and prints its initial and trained
evaluations. To compare later, attach the saved 1.6M-step M3 run and rerun
the optional comparison cell. Set `reference` there if the run directory has
a different name. The two models share the PPO trainer, critic inputs, terrain,
and evaluation seeds. M1 sees the current 67 values. M3 adds five previous
observation–action pairs for a total of 547 values.

From the checkout, the equivalent training command is:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -u scripts/train_kaggle.py \
  --model M1 --num-envs 8 --seed 42 --num-timesteps 1600000 \
  --output-dir results/m1_1p6m_env8_lr1e4
```
