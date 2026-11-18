"""Unit tests for causal observation-action history buffer."""

import numpy as np
from src.envs.history_wrapper import HistoryBuffer, HistoryConfig, format_policy_input


def test_history_buffer_causality_and_shapes():
    num_envs = 4
    cfg = HistoryConfig(history_length=3, obs_dim=5, action_dim=2, include_actions=True)
    buf = HistoryBuffer(num_envs=num_envs, config=cfg)

    assert buf.obs_history.shape == (4, 3, 5)
    assert buf.act_history.shape == (4, 3, 2)
    assert buf.history_dim == 3 * (5 + 2)

    # Initial buffer should be all zeros
    assert np.all(buf.obs_history == 0.0)

    # Step 1
    obs_1 = np.ones((num_envs, 5), dtype=np.float32) * 1.0
    act_1 = np.ones((num_envs, 2), dtype=np.float32) * 0.1
    buf.step(obs_1, act_1)

    assert np.all(buf.obs_history[:, -1, :] == 1.0)
    assert np.all(buf.obs_history[:, 0, :] == 0.0)

    # Step 2
    obs_2 = np.ones((num_envs, 5), dtype=np.float32) * 2.0
    act_2 = np.ones((num_envs, 2), dtype=np.float32) * 0.2
    buf.step(obs_2, act_2)

    assert np.all(buf.obs_history[:, -2, :] == 1.0)
    assert np.all(buf.obs_history[:, -1, :] == 2.0)
    assert np.all(buf.act_history[:, -1, :] == 0.2)


def test_episode_boundary_reset():
    num_envs = 3
    cfg = HistoryConfig(history_length=3, obs_dim=4, action_dim=2)
    buf = HistoryBuffer(num_envs=num_envs, config=cfg)

    obs = np.ones((num_envs, 4), dtype=np.float32)
    act = np.ones((num_envs, 2), dtype=np.float32)
    buf.step(obs, act)

    # Environment index 1 terminates
    dones = np.array([False, True, False])
    buf.step(obs * 2, act * 2, dones=dones)

    # Env 1 should be completely reset to zero
    assert np.all(buf.obs_history[1] == 0.0)
    assert np.all(buf.act_history[1] == 0.0)

    # Env 0 and 2 should have valid history
    assert np.all(buf.obs_history[0, -1] == 2.0)
    assert np.all(buf.obs_history[0, -2] == 1.0)


def test_format_policy_input():
    num_envs = 2
    cfg = HistoryConfig(history_length=2, obs_dim=3, action_dim=1)
    buf = HistoryBuffer(num_envs=num_envs, config=cfg)

    current_obs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    # M1 and M2
    inp_m1 = format_policy_input("M1", current_obs, buf)
    assert np.array_equal(inp_m1, current_obs)

    inp_m2 = format_policy_input("M2", current_obs, buf)
    assert np.array_equal(inp_m2, current_obs)

    # M3: stacked
    inp_m3 = format_policy_input("M3", current_obs, buf)
    # obs_dim (3) + history_dim (2 * (3 + 1) = 8) -> 11
    assert inp_m3.shape == (2, 11)

    # M4: structured
    m4_curr, (m4_obs_hist, m4_act_hist) = format_policy_input("M4", current_obs, buf)
    assert np.array_equal(m4_curr, current_obs)
    assert m4_obs_hist.shape == (2, 2, 3)
    assert m4_act_hist.shape == (2, 2, 1)


if __name__ == "__main__":
    test_history_buffer_causality_and_shapes()
    test_episode_boundary_reset()
    test_format_policy_input()
    print("All history buffer tests passed successfully!")
