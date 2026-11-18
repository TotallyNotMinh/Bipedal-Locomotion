"""Singular Procedural Terrain Generator for Bipedal Locomotion.

An environment uses exactly ONE singular terrain category for training.
Supports:
- Level 0: flat (constant elevation)
- Level 1: gentle_slopes (linear elevation gradient)
- Level 2: rolling_hills (continuous multi-frequency harmonic hills)
- Level 3: rough_terrain (stepped random noise)
- Level 4: steep_slopes_obstacles (steep slopes and discrete obstacle platforms, strictly uphill)

All parameters scale continuously via difficulty coefficient d in [0, 1].
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
import numpy as np


@dataclass
class SingularTerrainArena:
    """Represents an environment arena composed of a single terrain category."""
    category: str
    difficulty: float
    heights: np.ndarray  # shape: (nx, ny) in meters
    horizontal_scale: float
    vertical_scale: float
    size_x: float  # length in meters (along forward / uphill direction)
    size_y: float  # width in meters
    min_height: float
    max_height: float
    uphill_direction: Optional[Tuple[float, float]] = None  # (ux, uy) unit vector


# ---------------------------------------------------------------------------
# Procedural Generator Functions with Continuous Difficulty Scaling d in [0,1]
# ---------------------------------------------------------------------------

def generate_flat(nx: int, ny: int) -> np.ndarray:
    """Level 0: Constant elevation floor."""
    return np.zeros((nx, ny), dtype=np.float32)


def generate_gentle_slopes(
    nx: int,
    ny: int,
    horizontal_scale: float,
    d: float,
    slope_min: float = 0.05,
    slope_max: float = 0.22,
    direction: str = "incline",
) -> np.ndarray:
    """Level 1: Linear elevation gradient scaled by d."""
    slope = slope_min + d * (slope_max - slope_min)
    x = np.arange(nx) * horizontal_scale

    if direction == "incline":
        profile = x * slope
    elif direction == "decline":
        profile = (x[-1] - x) * slope
    else:
        profile = np.abs(x - x[nx // 2]) * slope

    return np.outer(profile, np.ones(ny, dtype=np.float32)).astype(np.float32)


def generate_rolling_hills(
    nx: int,
    ny: int,
    horizontal_scale: float,
    d: float,
    rng: np.random.Generator,
    amp_min: float = 0.08,
    amp_max: float = 0.40,
    wave_min: float = 3.0,
    wave_max: float = 7.0,
    octaves: int = 3,
) -> np.ndarray:
    """Level 2: Continuous Gaussian / harmonic wave rolling hills."""
    amplitude = amp_min + d * (amp_max - amp_min)
    base_wavelength = wave_max - d * (wave_max - wave_min)
    base_freq = 1.0 / max(0.5, base_wavelength)

    x = np.linspace(0, nx * horizontal_scale, nx)
    y = np.linspace(0, ny * horizontal_scale, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    heights = np.zeros((nx, ny), dtype=np.float32)
    cur_amp = amplitude
    cur_freq = base_freq

    for _ in range(octaves):
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(0, np.pi)
        kx = cur_freq * np.cos(angle)
        ky = cur_freq * np.sin(angle)

        wave = np.sin(2 * np.pi * (kx * xx + ky * yy) + phase_x) * np.cos(
            2 * np.pi * (-ky * xx + kx * yy) + phase_y
        )
        heights += (wave * (cur_amp * 0.5)).astype(np.float32)
        cur_amp *= 0.5
        cur_freq *= 2.0

    return heights


def generate_rough_terrain(
    nx: int,
    ny: int,
    d: float,
    rng: np.random.Generator,
    noise_min: float = 0.02,
    noise_max: float = 0.10,
    step_size: float = 0.005,
) -> np.ndarray:
    """Level 3: Higher-frequency random stepped / rough noise scaled by d."""
    max_amp = noise_min + d * (noise_max - noise_min)
    levels = int(np.round(2 * max_amp / step_size)) + 1
    bins = np.linspace(-max_amp, max_amp, levels)
    idx = rng.integers(0, levels, size=(nx, ny))
    return bins[idx].astype(np.float32)


def generate_steep_slopes_obstacles(
    nx: int,
    ny: int,
    horizontal_scale: float,
    d: float,
    rng: np.random.Generator,
    steep_min: float = 0.12,
    steep_max: float = 0.30,
    obs_height_min: float = 0.06,
    obs_height_max: float = 0.25,
) -> np.ndarray:
    """Level 4: Steep uphill slope COMBINED with discrete obstacle platforms, stairs, and raised hurdles.

    Features:
    - Base steep incline ascending along +x (strictly uphill)
    - Periodic transverse stair tiers / ridges across the slope
    - Procedurally placed 3D rectangular obstacle boxes / stepping platforms protruding above the slope
    """
    x = np.arange(nx) * horizontal_scale
    y = np.arange(ny) * horizontal_scale

    # 1. Base steep incline ascending along +x
    slope = steep_min + d * (steep_max - steep_min)
    base_heights = np.outer(x * slope, np.ones(ny, dtype=np.float32)).astype(np.float32)

    # 2. Add periodic transverse stair steps / ridges rising above the slope
    step_interval = 3.0 # meters between major stair steps
    step_indices = (x / step_interval).astype(int)
    step_rise = 0.05 + d * 0.10 # 5cm to 15cm additional step rise
    stair_ridges = np.outer(step_indices * step_rise, np.ones(ny, dtype=np.float32)).astype(np.float32)

    heights = base_heights + stair_ridges

    # 3. Procedurally generate discrete 3D obstacle boxes and stepping platforms
    platform_h = obs_height_min + d * (obs_height_max - obs_height_min)
    box_w_pts = max(3, int(1.2 / horizontal_scale))  # ~1.2m wide boxes
    box_l_pts = max(3, int(1.2 / horizontal_scale))  # ~1.2m long boxes
    stride_x = max(box_l_pts + 2, int(3.5 / horizontal_scale))
    stride_y = max(box_w_pts + 2, int(3.0 / horizontal_scale))

    # Staggered placement of obstacle platforms along the uphill course
    # Start placing after the spawn zone (x > 15m)
    x_start_idx = int(14.0 / horizontal_scale)

    for row_idx, i in enumerate(range(x_start_idx, nx - box_l_pts, stride_x)):
        # Stagger lateral offset every other row
        y_offset = (row_idx % 2) * (stride_y // 2)
        for j in range(int(8.0 / horizontal_scale) + y_offset, ny - int(8.0 / horizontal_scale) - box_w_pts, stride_y):
            # Random jitter to position
            ji = i + rng.integers(-1, 2)
            jj = j + rng.integers(-1, 2)
            ji = np.clip(ji, 0, nx - box_l_pts)
            jj = np.clip(jj, 0, ny - box_w_pts)

            # Protrude platform above the local slope height
            h_boost = rng.uniform(platform_h * 0.7, platform_h * 1.3)
            heights[ji : ji + box_l_pts, jj : jj + box_w_pts] += h_boost

    return heights.astype(np.float32)


# ---------------------------------------------------------------------------
# Singular Terrain Generator Class
# ---------------------------------------------------------------------------

class TerrainGenerator:
    """Generates an environment arena composed entirely of ONE singular terrain category."""

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        t_cfg = cfg.get("terrain", {})

        self.seed = int(t_cfg.get("seed", 42))
        self.active_category = str(t_cfg.get("active_category", "rolling_hills"))

        arena = t_cfg.get("arena_size", {"x": 80.0, "y": 80.0})
        self.size_x = float(arena.get("x", 80.0))
        self.size_y = float(arena.get("y", 80.0))

        self.horizontal_scale = float(t_cfg.get("horizontal_scale", 0.08))
        self.vertical_scale = float(t_cfg.get("vertical_scale", 0.002))
        self.border_size = float(t_cfg.get("border_size", 4.0))

        self.categories_cfg = t_cfg.get("categories", {})

    def generate(
        self,
        category: Optional[str] = None,
        difficulty: float = 0.5,
        seed: Optional[int] = None,
    ) -> SingularTerrainArena:
        """Generates an entire arena composed of ONE singular terrain category at difficulty d in [0, 1]."""
        cat = category or self.active_category
        d = float(np.clip(difficulty, 0.0, 1.0))
        effective_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(effective_seed)

        nx = int(np.round(self.size_x / self.horizontal_scale))
        ny = int(np.round(self.size_y / self.horizontal_scale))

        cat_cfg = self.categories_cfg.get(cat, {})
        uphill_dir = None

        if cat == "flat":
            raw_heights = generate_flat(nx, ny)
        elif cat == "gentle_slopes":
            s_min = float(cat_cfg.get("slope_min", 0.05))
            s_max = float(cat_cfg.get("slope_max", 0.22))
            direction = str(cat_cfg.get("direction", "incline"))
            raw_heights = generate_gentle_slopes(nx, ny, self.horizontal_scale, d, s_min, s_max, direction)
            uphill_dir = (1.0, 0.0) if direction == "incline" else (-1.0, 0.0)
        elif cat == "rolling_hills":
            a_min = float(cat_cfg.get("amplitude_min", 0.08))
            a_max = float(cat_cfg.get("amplitude_max", 0.40))
            w_min = float(cat_cfg.get("wavelength_min", 3.0))
            w_max = float(cat_cfg.get("wavelength_max", 7.0))
            octaves = int(cat_cfg.get("octaves", 3))
            raw_heights = generate_rolling_hills(nx, ny, self.horizontal_scale, d, rng, a_min, a_max, w_min, w_max, octaves)
        elif cat == "rough_terrain":
            n_min = float(cat_cfg.get("noise_amplitude_min", 0.02))
            n_max = float(cat_cfg.get("noise_amplitude_max", 0.10))
            step = float(cat_cfg.get("step_size", 0.005))
            raw_heights = generate_rough_terrain(nx, ny, d, rng, n_min, n_max, step)
        elif cat == "steep_slopes_obstacles":
            s_min = float(cat_cfg.get("steep_slope_min", 0.15))
            s_max = float(cat_cfg.get("steep_slope_max", 0.45))
            h_min = float(cat_cfg.get("obstacle_height_min", 0.06))
            h_max = float(cat_cfg.get("obstacle_height_max", 0.25))
            raw_heights = generate_steep_slopes_obstacles(nx, ny, self.horizontal_scale, d, rng, s_min, s_max, h_min, h_max)
            uphill_dir = (1.0, 0.0)
        else:
            raw_heights = generate_flat(nx, ny)

        min_h = float(np.min(raw_heights))
        max_h = float(np.max(raw_heights))

        return SingularTerrainArena(
            category=cat,
            difficulty=d,
            heights=raw_heights,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            size_x=self.size_x,
            size_y=self.size_y,
            min_height=min_h,
            max_height=max_h,
            uphill_direction=uphill_dir,
        )
