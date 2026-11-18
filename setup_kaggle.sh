#!/usr/bin/env bash
# ==============================================================================
# Setup script for Kaggle Notebooks with Dual NVIDIA Tesla T4 GPUs
# Project: Temporal Context for Adaptive Humanoid Locomotion (Unitree G1)
# ==============================================================================

set -euo pipefail

echo "=================================================================="
echo " Starting Kaggle Dual-T4 Setup for Unitree G1 Locomotion & MJX"
echo " Date: $(date)"
echo " Host: $(uname -a)"
echo "=================================================================="

# 1. Verify NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "[-] ERROR: nvidia-smi not found. Ensure GPU accelerator is enabled in Kaggle Settings!"
    exit 1
fi

echo "[+] Detecting GPUs via nvidia-smi:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "[+] Detected $NUM_GPUS GPU(s)"
if [ "$NUM_GPUS" -lt 2 ]; then
    echo "[!] WARNING: Found $NUM_GPUS GPU(s). Dual T4 is recommended for full multi-device throughput."
fi

# 2. Configure Memory & JAX Environment Variables for Dual T4 (16GB each)
echo "[+] Configuring JAX / XLA memory flags..."
cat << 'EOF' > /tmp/env_jax.sh
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_PLATFORMS=cuda,cpu
EOF

# Source in current shell and append to bashrc for interactive sessions
# shellcheck disable=SC1091
source /tmp/env_jax.sh
if ! grep -q "XLA_PYTHON_CLIENT_PREALLOCATE" ~/.bashrc 2>/dev/null; then
    cat /tmp/env_jax.sh >> ~/.bashrc
fi

# 3. Detect CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' || echo "12")
echo "[+] Detected CUDA base version: $CUDA_VERSION"

# 4. Install / Upgrade JAX with CUDA 12 support
echo "[+] Installing JAX and JAXlib with CUDA 12 acceleration..."
pip install --upgrade "pip" "setuptools" "wheel"
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 5. Install Core RL & Simulation Dependencies
echo "[+] Installing MuJoCo, Brax, Flax, Optax, and simulation dependencies..."
pip install --upgrade \
    "mujoco>=3.1.0" \
    "flax" \
    "optax" \
    "brax" \
    "orbax-checkpoint" \
    "gymnasium" \
    "pyyaml" \
    "scipy" \
    "matplotlib" \
    "imageio" \
    "imageio-ffmpeg" \
    "pytest"

# 6. Install MuJoCo Playground (from source if not already installed)
echo "[+] Checking MuJoCo Playground installation..."
python3 -c "import mujoco_playground" 2>/dev/null || {
    echo "[+] Installing mujoco_playground from DeepMind GitHub..."
    pip install git+https://github.com/google-deepmind/mujoco_playground.git
}

# 7. Verify Dual-GPU JAX Acceleration and Devices
echo "[+] Verifying JAX dual-GPU setup..."
python3 - << 'EOF'
import os
import jax

devices = jax.devices()
print("\n" + "="*50)
print(f"[✓] JAX Version: {jax.__version__}")
print(f"[✓] Default Backend: {jax.default_backend()}")
print(f"[✓] Available JAX Devices ({len(devices)}):")
for i, d in enumerate(devices):
    print(f"    Device {i}: {d.device_kind} (platform: {d.platform}, id: {d.id})")
print("="*50 + "\n")

if len(devices) >= 2 and devices[0].platform == 'gpu':
    print("[✓] Dual GPU JAX acceleration is READY!")
elif len(devices) == 1 and devices[0].platform == 'gpu':
    print("[!] Single GPU detected. Running in single-device mode.")
else:
    print("[-] GPU not detected by JAX. Please inspect CUDA configuration.")
EOF

echo "[+] Kaggle environment setup completed successfully!"
