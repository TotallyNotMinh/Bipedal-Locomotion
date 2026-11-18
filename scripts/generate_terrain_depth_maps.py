"""Generate top-down depth maps of all singular terrain environments at maximum difficulty (d=1.0).

Outputs:
- assets/terrain_all_categories_depth.png: 5-panel comparison of all singular terrain environments
"""

import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.envs.terrain import TerrainGenerator


def generate_depth_maps():
    assets_dir = ROOT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = ROOT_DIR / "configs" / "terrain.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    gen = TerrainGenerator(cfg)

    categories = [
        ("flat", "Level 0: Flat Ground", 0.0),
        ("gentle_slopes", "Level 1: Gentle Slopes", 1.0),
        ("rolling_hills", "Level 2: Rolling Hills", 1.0),
        ("rough_terrain", "Level 3: Rough Terrain", 1.0),
        ("steep_slopes_obstacles", "Level 4: Steep Slopes & Obstacles", 1.0),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(24, 4.8), constrained_layout=True)

    for ax, (cat, title, diff) in zip(axes, categories):
        arena = gen.generate(category=cat, difficulty=diff, seed=42)
        h = arena.heights
        extent = [0, arena.size_x, 0, arena.size_y]

        im = ax.imshow(
            h.T,
            origin="lower",
            extent=extent,
            cmap="viridis",
            interpolation="nearest",
        )
        ax.set_title(f"{title}\n(Difficulty d={diff:.1f}, {arena.size_x:.0f}m×{arena.size_y:.0f}m)", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("X [m] (Forward / Uphill →)", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Y [m] (Lateral)", fontsize=10)

        # Draw safe spawn band indicator (x in [5m, 12m])
        ax.axvline(5.0, color="cyan", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axvline(12.0, color="cyan", linestyle="--", linewidth=1.0, alpha=0.8)

        # Annotate height range
        min_val = float(np.min(h))
        max_val = float(np.max(h))
        ax.text(
            0.05, 0.05, f"Range: [{min_val:.2f}m, {max_val:.2f}m]",
            transform=ax.transAxes, fontsize=9,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Height [m]", fontsize=9)

    out_categories = assets_dir / "terrain_all_categories_depth.png"
    plt.savefig(out_categories, dpi=200)
    plt.close()
    print(f"[✓] Saved all singular terrain depth maps to: {out_categories}")


if __name__ == "__main__":
    generate_depth_maps()
