"""Unit tests for Singular Terrain Environment and dynamic scenario events."""

import numpy as np
from src.envs.g1_terrain_env import UnifiedG1TerrainEnv, G1EnvConfig


def test_unified_env_init_and_spawns():
    cfg = G1EnvConfig(
        model_id="M3",
        category="rolling_hills",
        difficulty=0.5,
        num_envs=32,
        history_length=4,
        friction_switch_enabled=True,
        friction_switch_step=10,
        switched_friction=0.2,
    )
    env = UnifiedG1TerrainEnv(cfg)

    # Check singular terrain arena was generated
    assert env.terrain_arena is not None
    assert env.terrain_arena.heights.shape[0] > 0
    assert env.terrain_arena.category == "rolling_hills"

    # Check spawn sampling
    spawns, headings = env.sample_spawn_positions(20)
    assert spawns.shape == (20, 3)
    assert headings.shape == (20,)


def test_friction_switch_and_history_integration():
    cfg = G1EnvConfig(
        model_id="M3",
        category="flat",
        difficulty=0.0,
        num_envs=4,
        history_length=3,
        friction_switch_enabled=True,
        friction_switch_step=2,
        nominal_friction=1.0,
        switched_friction=0.2,
    )
    env = UnifiedG1TerrainEnv(cfg)

    obs_0 = env.reset()
    assert obs_0.shape == (4, env.obs_dim + 3 * (env.obs_dim + env.action_dim))
    assert np.all(env.current_friction == 1.0)

    raw_obs_1 = np.ones((4, env.obs_dim), dtype=np.float32)
    actions = np.zeros((4, env.action_dim), dtype=np.float32)
    dones = np.zeros(4, dtype=bool)

    obs_1, info_1 = env.step(actions, raw_obs_1, dones)
    assert np.all(info_1["friction"] == 1.0)

    # Step 2: friction switch event trigger
    raw_obs_2 = np.ones((4, env.obs_dim), dtype=np.float32) * 2.0
    obs_2, info_2 = env.step(actions, raw_obs_2, dones)

    assert "friction_switched" in info_2["events"]
    assert len(info_2["events"]["friction_switched"]) == 4
    assert np.all(info_2["friction"] == 0.2)


if __name__ == "__main__":
    test_unified_env_init_and_spawns()
    test_friction_switch_and_history_integration()
    print("All singular terrain scenario tests passed successfully!")
