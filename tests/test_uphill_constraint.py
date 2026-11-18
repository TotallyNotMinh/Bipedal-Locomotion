"""Unit tests verifying uphill command direction constraint for steep slopes and obstacles."""

import numpy as np
from src.envs.g1_terrain_env import UnifiedG1TerrainEnv, G1EnvConfig


def test_uphill_command_direction():
    cfg = G1EnvConfig(category="steep_slopes_obstacles", difficulty=0.8)
    env = UnifiedG1TerrainEnv(cfg)

    assert env.terrain_arena.uphill_direction == (1.0, 0.0)
    ux, uy = env.terrain_arena.uphill_direction

    # Sample 200 commands and verify all have positive uphill dot product
    has_lateral = False
    for _ in range(200):
        vx, vy, wz = env.sample_velocity_command(max_heading_deviation_deg=45.0)
        dot = vx * ux + vy * uy
        assert dot > 0.0, f"Command ({vx}, {vy}) must point uphill (dot={dot})"
        if abs(vy) > 0.05:
            has_lateral = True

    assert has_lateral, "Must allow natural diagonal wandering lines while progressing uphill"


def test_steep_slope_spawn_near_base():
    cfg = G1EnvConfig(category="steep_slopes_obstacles", difficulty=0.8)
    env = UnifiedG1TerrainEnv(cfg)

    spawns, headings = env.sample_spawn_positions(50)

    # All spawns must be in rear region x in [5m, 12m] to provide full forward run-out
    assert np.all(spawns[:, 0] >= 5.0)
    assert np.all(spawns[:, 0] <= 12.0)

    # Initial headings must face uphill (+x) within +/- 20 degrees
    for h in headings:
        assert abs(h) <= np.radians(20.0), f"Initial heading {np.degrees(h)} deg must face uphill"


if __name__ == "__main__":
    test_uphill_command_direction()
    test_steep_slope_spawn_near_base()
    print("All uphill constraint tests passed successfully!")
