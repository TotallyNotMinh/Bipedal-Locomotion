"""Build the pilot collision scene from the installed G1 template."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET


def write_scene(template, arena, directory):
    """Replace only the floor geometry, retaining its name and contact sensors."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    scene = ET.parse(template)
    heights = arena.heights.T  # MuJoCo rows run along y, columns along x.
    span = max(arena.max_height - arena.min_height, 0.01)
    surface = directory / "terrain.bin"
    with surface.open("wb") as output:
        output.write(struct.pack("<ii", *heights.shape))
        output.write(((heights - arena.min_height) / span).astype("<f4").tobytes())
    asset = scene.find("asset")
    if asset is None:
        asset = ET.SubElement(scene.getroot(), "asset")
    ET.SubElement(asset, "hfield", name="training_surface", file=str(surface),
                  size=f"{arena.size_x / 2} {arena.size_y / 2} {span} 0.05")
    floor = scene.find(".//worldbody/geom[@name='floor']")
    if floor is None:
        raise ValueError("Installed G1 scene has no named floor geometry")
    floor.attrib.update(type="hfield", hfield="training_surface", pos=f"0 0 {arena.min_height}")
    floor.attrib.pop("size", None)
    path = directory / "g1.xml"
    scene.write(path)
    return path
