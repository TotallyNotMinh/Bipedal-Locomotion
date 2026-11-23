import numpy as np

from src.training.m3_history import initial_input, next_input


def test_causal_pairs_and_independent_resets():
    obs = np.array([[1.], [11.]])
    p0 = initial_input(obs, 1, 2, np)
    p1 = next_input(p0, obs + 2, obs + 1, np.array([False, False]), np)
    np.testing.assert_array_equal(p1, [[2, 0, 0, 1, 3], [12, 0, 0, 11, 13]])
    p2 = next_input(p1, obs + 3, obs + 2, np.array([False, True]), np)
    np.testing.assert_array_equal(p2, [[3, 1, 3, 2, 4], [13, 0, 0, 0, 0]])


def test_current_only_actor_on_reset_step_and_termination():
    obs = np.array([[1., 2.], [3., 4.]])
    previous = initial_input(obs, 1, 0, np)
    np.testing.assert_array_equal(previous, obs)
    current = obs + 10
    actual = next_input(previous, np.ones((2, 1)), current, np.array([False, True]), np)
    np.testing.assert_array_equal(actual, current)
