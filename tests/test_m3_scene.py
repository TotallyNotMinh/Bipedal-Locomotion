import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from src.training.m3_scene import write_scene


def test_collision_asset_keeps_physical_heights_and_axes():
    heights = np.array([[-2., 0.], [-1., 1.], [0., 2.]])
    arena = SimpleNamespace(heights=heights, min_height=-2., max_height=2., size_x=3., size_y=2.)
    with TemporaryDirectory() as directory:
        template = Path(directory) / "upstream.xml"
        template.write_text('<mujoco><worldbody><geom name="floor" type="plane" size="0 0 1" '
                            'friction="1 0.1 0.1"/></worldbody></mujoco>')
        scene = ET.parse(write_scene(template, arena, Path(directory) / "output"))
        field = scene.find("asset/hfield")
        floor = scene.find("worldbody/geom")
        assert floor.get("hfield") == field.get("name")
        assert floor.get("friction") == "1 0.1 0.1"
        payload = Path(field.get("file")).read_bytes()
        rows, cols = struct.unpack("<ii", payload[:8])
        values = np.frombuffer(payload[8:], dtype="<f4").reshape(rows, cols)
        sx, sy, scale, _ = map(float, field.get("size").split())
        assert (2 * sx, 2 * sy) == (arena.size_x, arena.size_y)
        z = float(floor.get("pos").split()[2])
        np.testing.assert_allclose(z + values.T * scale, heights)
