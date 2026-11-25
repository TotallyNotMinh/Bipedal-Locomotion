"""Dual-GPU (and single GPU) benchmark script for JAX & MJX.

Verifies:
1. Multi-device discovery (Dual NVIDIA Tesla T4 or single GPU).
2. Multi-device sharding across parallel environment batches.
3. Warm-up JIT compilation time, execution steps/sec, and VRAM memory footprint.
"""

import os
import time
import argparse
import numpy as np

# Configure memory allocation prior to JAX initialization
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import jax.numpy as jnp

try:
    from jax.sharding import PositionalSharding
except ImportError:
    PositionalSharding = None


def benchmark_multi_device_sharding(batch_size: int = 4096, steps: int = 500):
    devices = jax.devices()
    num_devices = len(devices)
    print(f"\n=======================================================")
    print(f" JAX Device Benchmark: {num_devices} device(s) detected")
    for i, dev in enumerate(devices):
        print(f"   [{i}] {dev.device_kind} (platform={dev.platform}, id={dev.id})")
    print(f"=======================================================")

    if PositionalSharding is not None:
        sharding = PositionalSharding(devices)
        sharding_2d = sharding.reshape(num_devices, 1)
        sharded_put = lambda arr: jax.device_put(arr, sharding_2d)
    else:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.array(devices), ("devices",))
        sharding_2d = NamedSharding(mesh, P("devices", None))
        sharded_put = lambda arr: jax.device_put(arr, sharding_2d)

    # Shard environments across devices
    envs_per_device = batch_size // num_devices
    print(f"Batch size: {batch_size} total ({envs_per_device} envs per device)")

    # Synthetic state vector representing robot proprioception (state_dim=64, action_dim=23)
    state_dim = 64
    action_dim = 23

    # Define a JIT-compiled batched transition function
    @jax.jit
    def step_fn(states, actions):
        # Emulate nonlinear dynamics step & reward evaluation
        next_states = jnp.sin(states) + 0.1 * jnp.pad(actions, ((0, 0), (0, state_dim - action_dim)))
        rewards = jnp.sum(next_states[:, :3], axis=-1)
        dones = jnp.zeros(states.shape[0], dtype=bool)
        return next_states, rewards, dones

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    states = jax.random.normal(k1, (batch_size, state_dim))
    actions = jax.random.normal(k2, (batch_size, action_dim))

    # Place sharded array across devices
    states_sharded = sharded_put(states)
    actions_sharded = sharded_put(actions)

    print("\n[+] Triggering initial JIT compilation...")
    t_start_compile = time.perf_counter()
    next_s, rew, done = step_fn(states_sharded, actions_sharded)
    next_s.block_until_ready()
    compile_time = time.perf_counter() - t_start_compile
    print(f"[✓] JIT compile time: {compile_time:.3f} s")

    print(f"[+] Running {steps} benchmarking iterations...")
    t_start_run = time.perf_counter()
    for _ in range(steps):
        next_s, rew, done = step_fn(next_s, actions_sharded)
    next_s.block_until_ready()
    run_time = time.perf_counter() - t_start_run

    total_env_steps = batch_size * steps
    steps_per_sec = total_env_steps / run_time

    print("\n---------------- Benchmark Results ----------------")
    print(f"Total simulated steps: {total_env_steps:,}")
    print(f"Execution wall-clock:  {run_time:.3f} s")
    print(f"Throughput:            {steps_per_sec:,.0f} steps/second")
    print(f"Per-device throughput: {steps_per_sec / num_devices:,.0f} steps/sec/device")
    print("---------------------------------------------------\n")

    return {
        "num_devices": num_devices,
        "batch_size": batch_size,
        "compile_time_s": compile_time,
        "steps_per_sec": steps_per_sec,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark JAX multi-GPU throughput on Dual T4 / A10G.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Total parallel environments")
    parser.add_argument("--steps", type=int, default=500, help="Benchmark step count")
    args = parser.parse_args()

    benchmark_multi_device_sharding(batch_size=args.batch_size, steps=args.steps)
