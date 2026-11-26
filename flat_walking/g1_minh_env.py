"""Adapter: chạy G1 (MuJoCo Playground, MJX) trên địa hình + cấu hình của Minh.

File này CHỈ THÊM MỚI, không sửa bất kỳ file nào của Minh. Nó nối:
  - TerrainGenerator / UnifiedG1TerrainEnv / configs/terrain.yaml   (Minh)
  - export_heightfield_to_mujoco_asset / patch_mjcf_with_terrain     (Minh)
với robot + reward + termination của Playground (phần Minh chưa có).

Những gì lấy từ Minh (không tự đặt):
  - Heightfield mặt đất (80x80 m, 8 cm/cell) -> thay geom "floor" của scene G1.
  - Tần số: physics_dt=0.002 (500 Hz), control_dt=0.02 (50 Hz).
  - Dải lệnh vận tốc: command_vx/vy/wz_range của G1EnvConfig.
  - Vùng spawn của sample_spawn_positions() (nhánh flat/omnidirectional).
  - push_enabled / friction_switch_enabled (mặc định False => không đẩy,
    không đổi ma sát khi train). Ma sát nominal = nominal_friction (1.0),
    đúng bằng friction của geom hfield mà patch_mjcf_with_terrain tạo ra.

Những gì lấy từ Playground (Minh chưa định nghĩa): robot G1 29 khớp, obs, reward,
termination, độ dài episode (1000 bước), action_scale, nhiễu cảm biến.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground import locomotion
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

from src.envs.terrain_utils import (
    export_heightfield_to_mujoco_asset,
    patch_mjcf_with_terrain,
)

PLAYGROUND_TASK = "G1JoystickFlatTerrain"  # chỉ để lấy default config của Playground
SCENE_TASK = "flat_terrain"                # scene XML có geom tên "floor"


class G1MinhTerrainEnv(g1_joystick.Joystick):
    """G1 Joystick env chạy trên heightfield của Minh (hiện hỗ trợ: flat)."""

    def __init__(self, minh_env, impl: str = "jax", config_overrides=None):
        arena = minh_env.terrain_arena
        mcfg = minh_env.config

        if arena.category != "flat":
            raise NotImplementedError(
                f"Mới hỗ trợ category 'flat' (nhận '{arena.category}'). "
                "Các địa hình khác chỉ làm sau khi flat đã đi ổn."
            )

        # ---- Config Playground, ép các giá trị lấy từ Minh -----------------
        config = locomotion.get_default_config(PLAYGROUND_TASK)
        config.impl = impl
        if not np.isclose(config.ctrl_dt, mcfg.control_dt) or not np.isclose(config.sim_dt, mcfg.physics_dt):
            raise ValueError(
                f"Playground dt (ctrl={config.ctrl_dt}, sim={config.sim_dt}) khác Minh "
                f"(control={mcfg.control_dt}, physics={mcfg.physics_dt})"
            )
        config.lin_vel_x = list(mcfg.command_vx_range)
        config.lin_vel_y = list(mcfg.command_vy_range)
        config.ang_vel_yaw = list(mcfg.command_wz_range)
        config.push_config.enable = bool(mcfg.push_enabled)

        self._cmd_vx = tuple(float(v) for v in mcfg.command_vx_range)
        self._cmd_vy = tuple(float(v) for v in mcfg.command_vy_range)
        self._cmd_wz = tuple(float(v) for v in mcfg.command_wz_range)

        # ---- Gắn heightfield của Minh vào scene G1 (thay geom "floor") -----
        self._tmp = tempfile.TemporaryDirectory(prefix="g1_minh_terrain_")
        tmp = Path(self._tmp.name)
        hfield_bin, attrs = export_heightfield_to_mujoco_asset(
            arena=arena, output_dir=tmp, asset_name=f"g1_{arena.category}_hfield"
        )
        patched_xml = patch_mjcf_with_terrain(
            base_mjcf_path=Path(str(g1_constants.task_to_xml(SCENE_TASK))),
            hfield_attrs=attrs,
            output_mjcf_path=tmp / "scene_g1_minh_terrain.xml",
            ground_geom_name="floor",
        )

        # locomotion.load() thường tự tải mujoco_menagerie (mesh robot); ta bỏ qua
        # load() nên phải gọi tay.
        mjx_env.ensure_menagerie_exists()

        # Không gọi Joystick.__init__ (nó hard-code XML của Playground).
        g1_base.G1Env.__init__(
            self, xml_path=patched_xml.as_posix(), config=config,
            config_overrides=config_overrides,
        )

        # patch_mjcf_with_terrain không ghi dữ liệu độ cao => nạp sau khi compile,
        # rồi đưa lại model lên MJX.
        data = np.fromfile(hfield_bin, dtype=np.float32)
        if self._mj_model.hfield_data.size != data.size:
            raise ValueError(
                f"hfield_data {self._mj_model.hfield_data.size} != file {data.size}"
            )
        self._mj_model.hfield_data[:] = data
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Dữ liệu arena để dựng spawn theo quy tắc của Minh.
        self._heights = jp.asarray(arena.heights, dtype=jp.float32)
        self._hscale = float(arena.horizontal_scale)
        self._size_x = float(arena.size_x)
        self._size_y = float(arena.size_y)

        self._post_init()

    # ---- Lệnh vận tốc: đúng như UnifiedG1TerrainEnv.sample_velocity_command
    #      (nhánh không-uphill): vx, vy, wz ~ U(range), không có lệnh 0 ngẫu nhiên.
    def sample_command(self, rng: jax.Array) -> jax.Array:
        r1, r2, r3 = jax.random.split(rng, 3)
        vx = jax.random.uniform(r1, minval=self._cmd_vx[0], maxval=self._cmd_vx[1])
        vy = jax.random.uniform(r2, minval=self._cmd_vy[0], maxval=self._cmd_vy[1])
        wz = jax.random.uniform(r3, minval=self._cmd_wz[0], maxval=self._cmd_wz[1])
        return jp.hstack([vx, vy, wz])

    # ---- Spawn: đúng như sample_spawn_positions (flat/omnidirectional):
    #      x,y ~ U(0.25..0.75 * size), z = độ cao mặt đất tại ô đó, yaw ~ U(-pi, pi)
    #      (yaw ngẫu nhiên [-3.14, 3.14] đã có sẵn trong Joystick.reset).
    def reset(self, rng: jax.Array):
        rng, spawn_rng = jax.random.split(rng)
        state = super().reset(rng)

        kx, ky = jax.random.split(spawn_rng)
        x = jax.random.uniform(kx, minval=0.25 * self._size_x, maxval=0.75 * self._size_x)
        y = jax.random.uniform(ky, minval=0.25 * self._size_y, maxval=0.75 * self._size_y)
        nx, ny = self._heights.shape
        gi = jp.clip((x / self._hscale).astype(jp.int32), 0, nx - 1)
        gj = jp.clip((y / self._hscale).astype(jp.int32), 0, ny - 1)
        ground_z = self._heights[gi, gj]

        qpos = state.data.qpos
        qpos = qpos.at[0].add(x).at[1].add(y).at[2].add(ground_z)
        data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))

        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sid]] > 0
            for sid in self._feet_floor_found_sensor
        ])
        obs = self._get_obs(data, state.info, contact)
        return state.replace(data=data, obs=obs)
