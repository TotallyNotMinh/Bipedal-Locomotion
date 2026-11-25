"""Unit tests for Model 2 (Capacity-Matched Larger MLP) & Config Inheritance."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.mlp_policy import (
    build_model_spec_from_config,
    build_parameter_comparison_table,
    count_actor_critic_params,
    count_mlp_params,
    estimate_m3_actor_critic_params,
    find_capacity_matched_hidden_sizes,
    format_parameter_table,
)
from src.training.config_loader import load_training_config, validate_ppo_sharding


def test_config_inheritance_parity():
    """Verify larger_mlp.yaml inherits exact PPO & env settings from ppo_baseline.yaml."""
    m1_cfg = load_training_config("configs/ppo_baseline.yaml")
    m2_cfg = load_training_config("configs/larger_mlp.yaml")

    assert m1_cfg["network"]["model_id"] == "M1"
    assert m2_cfg["network"]["model_id"] == "M2"
    assert m1_cfg["env"] == m2_cfg["env"], "M2 must share identical env config with M1"
    assert m1_cfg["ppo"] == m2_cfg["ppo"], "M2 must share identical PPO hyperparameters with M1"
    validate_ppo_sharding(m2_cfg["ppo"], device_count=2)


def test_exact_mlp_param_counting():
    """Verify analytical Dense parameter counting against manual calculation."""
    # Input=58, hidden=(512, 256, 128), output=46 (2*23)
    expected = (
        (58 * 512 + 512)
        + (512 * 256 + 256)
        + (256 * 128 + 128)
        + (128 * 46 + 46)
    )
    actual = count_mlp_params(58, (512, 256, 128), 46)
    assert actual == expected

    p, v, tot = count_actor_critic_params(
        policy_input_dim=58,
        value_input_dim=58,
        action_dim=23,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        distribution_type="tanh_normal",
    )
    assert p == expected
    assert tot == p + v


def test_m2_auto_capacity_matching():
    """Verify M2 solver matches M3 policy parameter count within < 5% across obs/action specs."""
    m2_cfg = load_training_config("configs/larger_mlp.yaml")

    for obs_size, act_size in [
        ({"state": 58, "privileged_state": 58}, 23),
        ({"state": 103, "privileged_state": 211}, 29),
    ]:
        spec = build_model_spec_from_config(m2_cfg, obs_size, act_size)
        obs_dim = obs_size["state"]
        m3_ref = estimate_m3_actor_critic_params(
            obs_dim=obs_dim,
            value_obs_dim=obs_size["privileged_state"],
            action_dim=act_size,
            history_length=5,
            include_actions=True,
        )

        rel_diff = abs(spec.actual_policy_params - m3_ref.policy_params) / m3_ref.policy_params
        assert spec.capacity_matched is True
        assert rel_diff < 0.05, f"Expected <5% param gap vs M3, got {rel_diff:.4f}"

        table = build_parameter_comparison_table(spec)
        ascii_table = format_parameter_table(table)
        assert "M1 (Baseline)" in ascii_table
        assert "M2 (Active)" in ascii_table
        assert "M3" in ascii_table


def test_find_capacity_matched_hidden_sizes_monotonic():
    """Verify hidden widths remain non-increasing and multiples of width_step."""
    hidden = find_capacity_matched_hidden_sizes(
        input_dim=103,
        output_dim=29,
        target_params=562_874,
        base_ratios=(512, 256, 128),
        width_step=16,
    )
    assert len(hidden) == 3
    assert all(h % 16 == 0 for h in hidden)
    assert hidden[0] >= hidden[1] >= hidden[2]


if __name__ == "__main__":
    test_config_inheritance_parity()
    test_exact_mlp_param_counting()
    test_m2_auto_capacity_matching()
    test_find_capacity_matched_hidden_sizes_monotonic()
    print("All Model 2 MLP tests passed successfully!")
