"""Interactive 3D MuJoCo Viewer for Singular Terrain Environment.

Launches the native MuJoCo GUI on your display.
Inspects the environment's singular terrain at difficulty d in [0, 1].

Usage:
    python scripts/view_terrain_mujoco.py
    python scripts/view_terrain_mujoco.py --category rolling_hills --difficulty 1.0
    python scripts/view_terrain_mujoco.py --category steep_slopes_obstacles --difficulty 0.8
"""

import sys
import argparse
from pathlib import Path
import yaml
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import mujoco
import mujoco.viewer

from src.envs.terrain import TerrainGenerator


def build_terrain_mjcf_xml(nx: int, ny: int, half_x: float, half_y: float, elevation_z: float, base_z: float) -> str:
    xml = f"""<mujoco model="terrain_viewer">
  <compiler angle="degree" coordinate="local"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1920" offheight="1080"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.25 0.35 0.45" rgb2="0.15 0.2 0.25"/>
    <material name="grid_mat" texture="grid" texrepeat="25 25" reflectance="0.2"/>
    <material name="hfield_mat" specular="0.2" shininess="0.4" rgba="0.65 0.75 0.65 1"/>
    <hfield name="terrain" nrow="{nx}" ncol="{ny}" size="{half_x:.3f} {half_y:.3f} {elevation_z:.3f} {base_z:.3f}"/>
  </asset>

  <worldbody>
    <light directional="true" pos="0 0 25" dir="0.2 0.3 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>
    <light directional="true" pos="0 0 25" dir="-0.3 -0.2 -1" diffuse="0.4 0.4 0.5"/>

    <!-- Heightfield Ground Geom -->
    <geom name="terrain_geom" type="hfield" hfield="terrain" material="hfield_mat"
          pos="{half_x:.3f} {half_y:.3f} 0" condim="3" friction="1.0 0.005 0.0001"/>

    <!-- Test Agent Marker -->
    <body name="agent_marker" pos="3.0 {half_y:.3f} 1.2">
      <freejoint/>
      <geom type="capsule" size="0.12 0.4" rgba="0.9 0.2 0.2 1" mass="35"/>
      <geom type="sphere" size="0.16" pos="0 0 0.5" rgba="0.2 0.7 0.9 1" mass="5"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml


def main():
    parser = argparse.ArgumentParser(description="View singular terrain in native MuJoCo 3D viewer")
    parser.add_argument("--config", type=str, default="configs/terrain.yaml", help="Terrain config path")
    parser.add_argument("--category", type=str, default=None,
                        choices=["flat", "gentle_slopes", "rolling_hills", "rough_terrain", "steep_slopes_obstacles"],
                        help="Specific terrain category to view")
    parser.add_argument("--difficulty", type=float, default=0.5, help="Difficulty coefficient d in [0, 1]")
    parser.add_argument("--seed", type=int, default=42, help="Seed for terrain generation")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

    gen = TerrainGenerator(cfg)
    cat = args.category or gen.active_category

    print(f"\n=======================================================")
    print(f" Singular Terrain Environment Viewer")
    print(f" Category:   {cat}")
    print(f" Difficulty: d = {args.difficulty:.2f}")
    print(f" Seed:       {args.seed}")
    print(f"=======================================================")

    arena = gen.generate(category=cat, difficulty=args.difficulty, seed=args.seed)

    nx, ny = arena.heights.shape
    min_h = arena.min_height
    max_h = arena.max_height

    elevation_z = max(0.02, max_h)
    base_z = max(0.02, -min_h if min_h < 0 else 0.05)
    total_range = elevation_z + base_z

    normalized_data = (arena.heights + base_z) / total_range
    normalized_data = np.clip(normalized_data, 0.0, 1.0).astype(np.float32)

    half_x = arena.size_x / 2.0
    half_y = arena.size_y / 2.0

    print(f"[+] Arena dimensions: {arena.size_x:.1f}m x {arena.size_y:.1f}m")
    print(f"[+] Grid resolution:  {nx} x {ny} points ({arena.horizontal_scale*100:.1f} cm/cell)")
    print(f"[+] Height range:     [{min_h:.3f}m, {max_h:.3f}m]")
    if arena.uphill_direction:
        print(f"[+] Uphill Direction: {arena.uphill_direction} (Ascending along +x)")

    mjcf_xml = build_terrain_mjcf_xml(nx, ny, half_x, half_y, elevation_z, base_z)
    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)

    model.hfield_data[:] = normalized_data.ravel()

    print("\n" + "=" * 60)
    print(" LAUNCHING MUJOCO 3D VIEWER")
    print(" Controls:")
    print("   - Left click + drag:  Orbit camera")
    print("   - Right click + drag: Pan camera")
    print("   - Scroll wheel:       Zoom in / out")
    print("   - Spacebar:           Pause / unpause simulation")
    print("   - Backspace:          Reset simulation")
    print("=" * 60 + "\n")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
