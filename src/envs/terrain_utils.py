"""MuJoCo XML and MJX integration utilities for procedural heightfield terrains."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np

from src.envs.terrain import SingularTerrainArena


def export_heightfield_to_mujoco_asset(
    arena: Any,
    output_dir: Path,
    asset_name: str = "terrain_hfield",
) -> Tuple[Path, dict]:
    """Saves the heightfield array to disk and computes MuJoCo <hfield> attributes.

    MuJoCo hfield specification:
    - size: (size_x, size_y, elevation_z, base_z)
      size_x, size_y: half-widths of the terrain in meters
      elevation_z: max elevation above zero (meters)
      base_z: depth below zero (meters)
    - nrow, ncol: number of grid points along x and y
    - Data: normalized float array or uint16 grayscale image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heights = arena.heights
    nx, ny = heights.shape

    min_h = float(np.min(heights))
    max_h = float(np.max(heights))

    elevation_z = max(0.01, max_h)
    base_z = max(0.01, -min_h if min_h < 0 else 0.05)
    total_range = elevation_z + base_z

    normalized_data = (heights + base_z) / total_range
    normalized_data = np.clip(normalized_data, 0.0, 1.0).astype(np.float32)

    # Save as raw float binary
    binary_path = output_dir / f"{asset_name}.bin"
    normalized_data.tofile(binary_path)

    size_x = getattr(arena, "size_x", getattr(arena, "total_size_x", 32.0))
    size_y = getattr(arena, "size_y", getattr(arena, "total_size_y", 32.0))
    half_x = size_x / 2.0
    half_y = size_y / 2.0

    attributes = {
        "name": asset_name,
        "nrow": str(nx),
        "ncol": str(ny),
        "size": f"{half_x:.3f} {half_y:.3f} {elevation_z:.3f} {base_z:.3f}",
        "file": str(binary_path.name),
        "half_x": half_x,
        "half_y": half_y,
        "elevation_z": elevation_z,
        "base_z": base_z,
        "nx": nx,
        "ny": ny,
    }

    return binary_path, attributes


def patch_mjcf_with_terrain(
    base_mjcf_path: Path,
    hfield_attrs: dict,
    output_mjcf_path: Path,
    ground_geom_name: str = "floor",
) -> Path:
    """Replaces or updates the flat floor geom in an MJCF model with the procedural heightfield geom."""
    tree = ET.parse(base_mjcf_path)
    root = tree.getroot()

    # Find or create <asset>
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Add or update <hfield> asset
    hfield_elem = None
    for h in asset.findall("hfield"):
        if h.get("name") == hfield_attrs["name"]:
            hfield_elem = h
            break
    if hfield_elem is None:
        hfield_elem = ET.SubElement(asset, "hfield")

    hfield_elem.set("name", hfield_attrs["name"])
    hfield_elem.set("nrow", hfield_attrs["nrow"])
    hfield_elem.set("ncol", hfield_attrs["ncol"])
    hfield_elem.set("size", hfield_attrs["size"])

    # Update worldbody floor geom
    worldbody = root.find("worldbody")
    if worldbody is not None:
        floor_geom = None
        for g in worldbody.findall("geom"):
            if g.get("name") == ground_geom_name:
                floor_geom = g
                break

        if floor_geom is not None:
            # Modify to hfield
            floor_geom.set("type", "hfield")
            floor_geom.set("hfield", hfield_attrs["name"])
            floor_geom.set("pos", f"{hfield_attrs['half_x']:.3f} {hfield_attrs['half_y']:.3f} 0")
        else:
            # Add heightfield geom
            ET.SubElement(
                worldbody,
                "geom",
                {
                    "name": ground_geom_name,
                    "type": "hfield",
                    "hfield": hfield_attrs["name"],
                    "pos": f"{hfield_attrs['half_x']:.3f} {hfield_attrs['half_y']:.3f} 0",
                    "condim": "3",
                    "friction": "1.0 0.005 0.0001",
                },
            )

    output_mjcf_path = Path(output_mjcf_path)
    output_mjcf_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_mjcf_path)
    return output_mjcf_path
