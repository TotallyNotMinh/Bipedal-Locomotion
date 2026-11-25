"""YAML configuration loader with inheritance (`base_config`) and PPO validation."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict
import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merges `override` into a copy of `base`."""
    merged = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = deepcopy(v)
    return merged


def load_training_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a training YAML config, resolving `base_config` inheritance if present."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    base_ref = cfg.pop("base_config", None)
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.exists():
            # Try resolving relative to repository root or parent dir
            alt_path = path.parent.parent / base_ref
            if alt_path.exists():
                base_path = alt_path
            else:
                alt_path2 = path.parent / base_ref
                if alt_path2.exists():
                    base_path = alt_path2
        base_cfg = load_training_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def save_resolved_config(config: Dict[str, Any], output_path: str | Path) -> Path:
    """Writes the fully resolved training configuration to disk for reproducibility."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    return out


def validate_ppo_sharding(ppo_cfg: Dict[str, Any], device_count: int) -> None:
    """Validates Brax PPO batch/minibatch/device divisibility constraints."""
    num_envs = int(ppo_cfg.get("num_envs", 4096))
    batch_size = int(ppo_cfg.get("batch_size", 256))
    num_minibatches = int(ppo_cfg.get("num_minibatches", 32))

    if device_count < 1:
        raise ValueError(f"Invalid device_count={device_count}")

    if num_envs % device_count != 0:
        raise ValueError(
            f"ppo.num_envs ({num_envs}) must be divisible by JAX device_count ({device_count}). "
            f"Try num_envs={((num_envs // device_count) + 1) * device_count}."
        )

    total_batch = batch_size * num_minibatches
    if total_batch % num_envs != 0:
        raise ValueError(
            f"Brax requires (batch_size * num_minibatches) % num_envs == 0, "
            f"got ({batch_size} * {num_minibatches}) = {total_batch} vs num_envs={num_envs}."
        )
