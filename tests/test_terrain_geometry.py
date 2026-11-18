"""Unit tests for Singular Terrain Arena generation and reproducibility."""

import numpy as np
from src.envs.terrain import TerrainGenerator, SingularTerrainArena


def test_singular_terrain_reproducibility():
    """Verify identical seed generates bit-for-bit identical terrain arrays."""
    config = {
        "terrain": {
            "seed": 1234,
            "active_category": "rolling_hills",
            "arena_size": {"x": 40.0, "y": 40.0},
            "horizontal_scale": 0.08,
        }
    }

    gen = TerrainGenerator(config)
    arena1 = gen.generate(category="rolling_hills", difficulty=0.6, seed=1234)
    arena2 = gen.generate(category="rolling_hills", difficulty=0.6, seed=1234)

    assert np.array_equal(arena1.heights, arena2.heights), "Heights must be bit-identical for identical seed"
    assert arena1.size_x == arena2.size_x == 40.0
    assert arena1.size_y == arena2.size_y == 40.0


def test_singular_terrain_diversity():
    """Verify different seeds generate different heightfield instances."""
    config = {
        "terrain": {
            "active_category": "rolling_hills",
            "arena_size": {"x": 40.0, "y": 40.0},
            "horizontal_scale": 0.08,
        }
    }

    gen = TerrainGenerator(config)
    arena_a = gen.generate(category="rolling_hills", difficulty=0.5, seed=101)
    arena_b = gen.generate(category="rolling_hills", difficulty=0.5, seed=202)

    assert not np.array_equal(arena_a.heights, arena_b.heights), "Different seeds must produce distinct terrain variations"


def test_level4_has_discrete_obstacles():
    """Verify Level 4 has distinct physical obstacle features compared to smooth Level 1 slope."""
    config = {
        "terrain": {
            "arena_size": {"x": 40.0, "y": 40.0},
            "horizontal_scale": 0.08,
        }
    }

    gen = TerrainGenerator(config)
    slope = gen.generate(category="gentle_slopes", difficulty=1.0)
    steep_obs = gen.generate(category="steep_slopes_obstacles", difficulty=1.0)

    # Gentle slopes has 0 lateral variance (smooth in y)
    assert np.var(np.diff(slope.heights, axis=1)) < 1e-6

    # Steep slopes & obstacles must have lateral variance from discrete obstacle platforms
    assert np.var(np.diff(steep_obs.heights, axis=1)) > 1e-4, "Level 4 must contain discrete obstacle platforms"
    assert steep_obs.uphill_direction == (1.0, 0.0)


if __name__ == "__main__":
    test_singular_terrain_reproducibility()
    test_singular_terrain_diversity()
    test_level4_has_discrete_obstacles()
    print("All singular terrain geometry tests passed successfully!")
