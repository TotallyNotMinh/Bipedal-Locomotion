"""Observation-action temporal history wrapper.

Implements the causal history contract:
    h_t = {(o_{t-k}, a_{t-k}), ..., (o_{t-1}, a_{t-1})}
Supports:
- M1 (Baseline): Current observation o_t only
- M2 (Capacity control): Current observation o_t only
- M3 (History stacking): Concatenation of [o_t, h_t]
- M4 (History encoder): Separate (o_t, h_t) for learned representation z_t = f_phi(h_t)

Guarantees:
- Zero cross-episode leakage: resets history buffer to zeros on done/reset.
- Strict causal ordering.
- Compatible with batched vector environments (JAX / NumPy).
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class HistoryConfig:
    history_length: int = 5  # Number of past timesteps k
    obs_dim: int = 47        # Standard G1 proprioceptive observation dimension
    action_dim: int = 23     # G1 actuation joints dimension
    include_actions: bool = True


class HistoryBuffer:
    """Manages observation-action history for multiple parallel environments.

    Stores:
    - past observations: shape (num_envs, history_length, obs_dim)
    - past actions: shape (num_envs, history_length, action_dim)
    """

    def __init__(self, num_envs: int, config: HistoryConfig):
        self.num_envs = num_envs
        self.k = config.history_length
        self.obs_dim = config.obs_dim
        self.action_dim = config.action_dim
        self.include_actions = config.include_actions

        self.pair_dim = self.obs_dim + (self.action_dim if self.include_actions else 0)
        self.history_dim = self.k * self.pair_dim

        self.obs_history = np.zeros((num_envs, self.k, self.obs_dim), dtype=np.float32)
        self.act_history = np.zeros((num_envs, self.k, self.action_dim), dtype=np.float32)

    def reset(self, env_indices: Optional[np.ndarray] = None):
        """Clears history to zero for specified environments (or all if None)."""
        if env_indices is None:
            self.obs_history.fill(0.0)
            self.act_history.fill(0.0)
        else:
            self.obs_history[env_indices] = 0.0
            self.act_history[env_indices] = 0.0

    def step(self, current_obs: np.ndarray, last_action: np.ndarray, dones: Optional[np.ndarray] = None):
        """Rolls history buffer forward and inserts (current_obs, last_action).

        If dones is provided, environments where done is True are reset to zeros.
        """
        # Shift past history left by 1
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :]
        self.act_history[:, :-1, :] = self.act_history[:, 1:, :]

        # Insert newest at index -1
        self.obs_history[:, -1, :] = current_obs
        self.act_history[:, -1, :] = last_action

        # Handle episode termination: mask out boundary crossing
        if dones is not None:
            done_idx = np.where(dones)[0]
            if len(done_idx) > 0:
                self.reset(done_idx)

    def get_stacked_history(self) -> np.ndarray:
        """Flattens history into a single 1D vector per environment: shape (num_envs, history_dim)."""
        if self.include_actions:
            combined = np.concatenate([self.obs_history, self.act_history], axis=-1)
        else:
            combined = self.obs_history
        return combined.reshape(self.num_envs, -1)

    def get_structured_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns structured history arrays for encoder architectures:
        (obs_history, act_history)
        """
        return self.obs_history.copy(), self.act_history.copy()


def format_policy_input(
    model_id: str,
    current_obs: np.ndarray,
    history_buffer: HistoryBuffer,
) -> Any:
    """Formats policy observation vector according to experimental model specification.

    M1: current_obs
    M2: current_obs (larger network)
    M3: concat([current_obs, stacked_history])
    M4: (current_obs, structured_history)
    """
    if model_id in ("M1", "M2"):
        return current_obs
    elif model_id == "M3":
        history = history_buffer.get_stacked_history()
        return np.concatenate([current_obs, history], axis=-1)
    elif model_id == "M4":
        return current_obs, history_buffer.get_structured_history()
    else:
        raise ValueError(f"Unknown model_id: {model_id}. Expected M1, M2, M3, or M4.")
