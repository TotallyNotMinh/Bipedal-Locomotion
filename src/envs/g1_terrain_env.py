"""Unified Unitree G1 Locomotion Environment with Singular Terrain Category.

Each environment instance runs exactly ONE singular terrain category.
Difficulty d in [0, 1] scales continuously via the curriculum manager.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import numpy as np

from src.envs.terrain import TerrainGenerator, SingularTerrainArena
from src.envs.curriculum import TerrainCurriculum
from src.envs.history_wrapper import HistoryBuffer, HistoryConfig, format_policy_input


@dataclass
class G1EnvConfig:
    """Configuration for G1 singular terrain locomotion."""
    model_id: str = "M1"  # "M1", "M2", "M3", "M4"
    terrain_config_path: str = "configs/terrain.yaml"
    category: Optional[str] = None  # None = use config active_category
    difficulty: float = 0.5
    num_envs: int = 2048
    history_length: int = 5
    control_dt: float = 0.02  # 50 Hz policy control rate
    physics_dt: float = 0.002 # 500 Hz simulation rate
    command_vx_range: Tuple[float, float] = (0.2, 1.0)
    command_vy_range: Tuple[float, float] = (-0.3, 0.3)
    command_wz_range: Tuple[float, float] = (-0.5, 0.5)

    # In-episode dynamics switching (Phase 6 benchmark)
    friction_switch_enabled: bool = False
    friction_switch_step: int = 250
    nominal_friction: float = 1.0
    switched_friction: float = 0.2

    # Disturbance / Push
    push_enabled: bool = False
    push_step: int = 300
    push_force: float = 80.0


class UnifiedG1TerrainEnv:
    """Unified environment running ONE singular terrain category."""

    def __init__(self, config: Optional[G1EnvConfig] = None):
        self.config = config or G1EnvConfig()

        # 1. Load terrain config & generate singular terrain arena
        t_cfg_path = Path(self.config.terrain_config_path)
        raw_cfg = {}
        if t_cfg_path.exists():
            with open(t_cfg_path, "r") as f:
                raw_cfg = yaml.safe_load(f)

        self.terrain_gen = TerrainGenerator(raw_cfg)
        self.curriculum = TerrainCurriculum(raw_cfg)

        chosen_cat = self.config.category or self.curriculum.frontier_category
        initial_diff = self.config.difficulty if self.config.difficulty != 0.5 else self.curriculum.difficulties.get(chosen_cat, 0.0)

        self.terrain_arena: SingularTerrainArena = self.terrain_gen.generate(
            category=chosen_cat,
            difficulty=initial_diff,
        )

        # 2. Setup History Buffer
        self.obs_dim = 58
        self.action_dim = 23

        self.history_config = HistoryConfig(
            history_length=self.config.history_length,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            include_actions=True,
        )
        self.history_buffer = HistoryBuffer(
            num_envs=self.config.num_envs,
            config=self.history_config,
        )

        # 3. Dynamic step tracking
        self.episode_steps = np.zeros(self.config.num_envs, dtype=np.int32)
        self.current_friction = np.full(self.config.num_envs, self.config.nominal_friction, dtype=np.float32)

    def set_terrain(self, category: str, difficulty: float):
        """Regenerates the singular terrain arena for a new category or difficulty."""
        self.config.category = category
        self.config.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.terrain_arena = self.terrain_gen.generate(
            category=category,
            difficulty=self.config.difficulty,
        )

    def set_difficulty(self, new_difficulty: float):
        """Regenerates the singular terrain arena at the updated curriculum difficulty d."""
        self.set_terrain(self.terrain_arena.category, new_difficulty)

    def update_curriculum(self) -> Dict[str, Any]:
        """Steps the curriculum manager, checking prerequisite health and stage promotion."""
        state = self.curriculum.update()
        current_cat = self.terrain_arena.category
        new_cat = state["frontier_category"] if not state["in_intervention"] else (state["failing_stage"] or current_cat)
        new_diff = state["difficulties"].get(new_cat, self.config.difficulty)

        if state["stage_promoted"] or (new_cat != current_cat) or not np.isclose(self.config.difficulty, new_diff, atol=1e-3):
            self.set_terrain(new_cat, new_diff)
        return state

    def reset(self, env_ids: Optional[np.ndarray] = None) -> Any:
        """Resets specified environments and returns formatted policy input."""
        if env_ids is None:
            self.episode_steps.fill(0)
            self.current_friction.fill(self.config.nominal_friction)
            self.history_buffer.reset()
            initial_obs = np.zeros((self.config.num_envs, self.obs_dim), dtype=np.float32)
        else:
            self.episode_steps[env_ids] = 0
            self.current_friction[env_ids] = self.config.nominal_friction
            self.history_buffer.reset(env_ids)
            initial_obs = np.zeros((len(env_ids), self.obs_dim), dtype=np.float32)

        return format_policy_input(self.config.model_id, initial_obs, self.history_buffer)

    def sample_spawn_positions(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Samples spawn positions (x, y, z) and initial yaw headings (radians).

        Uses safe spawn zones giving >65 meters of unobstructed forward run-out
        so agents can complete long training episodes without boundary collisions.
        """
        sx = self.terrain_arena.size_x
        sy = self.terrain_arena.size_y
        positions = []
        headings = []

        for _ in range(num_samples):
            if self.terrain_arena.uphill_direction is not None or self.terrain_arena.category in ("steep_slopes_obstacles", "gentle_slopes"):
                # Spawn near rear of the 80m arena (x in [5m, 12m]), centered laterally (y in [25m, 55m])
                spawn_x = np.random.uniform(5.0, 12.0)
                spawn_y = np.random.uniform(sy * 0.35, sy * 0.65)
                # Aligned uphill (+x) within +/- 15 deg
                yaw = np.random.uniform(-np.radians(15.0), np.radians(15.0))
            else:
                # Flat or omnidirectional: spawn in central region with generous border margins
                spawn_x = np.random.uniform(sx * 0.25, sx * 0.75)
                spawn_y = np.random.uniform(sy * 0.25, sy * 0.75)
                yaw = np.random.uniform(-np.pi, np.pi)

            grid_i = int(np.clip(spawn_x / self.terrain_arena.horizontal_scale, 0, self.terrain_arena.heights.shape[0] - 1))
            grid_j = int(np.clip(spawn_y / self.terrain_arena.horizontal_scale, 0, self.terrain_arena.heights.shape[1] - 1))
            ground_z = float(self.terrain_arena.heights[grid_i, grid_j])

            positions.append([spawn_x, spawn_y, ground_z])
            headings.append(yaw)

        return np.array(positions, dtype=np.float32), np.array(headings, dtype=np.float32)

    def sample_velocity_command(self, max_heading_deviation_deg: float = 45.0) -> Tuple[float, float, float]:
        """Samples target velocity command (vx, vy, wz).

        For steep_slopes_obstacles (or any uphill-directed terrain):
        Strictly guarantees uphill progression: dot(v_cmd, uphill_dir) > 0,
        allowing natural lateral wandering up to max_heading_deviation_deg.
        """
        speed = float(np.random.uniform(self.config.command_vx_range[0], self.config.command_vx_range[1]))
        wz = float(np.random.uniform(self.config.command_wz_range[0], self.config.command_wz_range[1]))

        if self.terrain_arena.uphill_direction is not None:
            ux, uy = self.terrain_arena.uphill_direction
            uphill_angle = np.arctan2(uy, ux)
            max_rad = np.radians(max_heading_deviation_deg)
            dev = np.random.uniform(-max_rad, max_rad)
            cmd_angle = uphill_angle + dev

            vx = speed * np.cos(cmd_angle)
            vy = speed * np.sin(cmd_angle)
        else:
            vx = speed
            vy = float(np.random.uniform(self.config.command_vy_range[0], self.config.command_vy_range[1]))

        return (float(vx), float(vy), float(wz))

    def check_dynamic_events(self) -> Dict[str, Any]:
        """Evaluates friction switches and push events at the current timestep."""
        events = {}
        if self.config.friction_switch_enabled:
            switch_mask = self.episode_steps == self.config.friction_switch_step
            if np.any(switch_mask):
                self.current_friction[switch_mask] = self.config.switched_friction
                events["friction_switched"] = np.where(switch_mask)[0]

        if self.config.push_enabled:
            push_mask = self.episode_steps == self.config.push_step
            if np.any(push_mask):
                events["push_applied"] = np.where(push_mask)[0]

        return events

    def step(self, actions: np.ndarray, current_raw_obs: np.ndarray, dones: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Advances environment state, updates history buffer, and tracks dynamic events."""
        self.episode_steps += 1
        self.history_buffer.step(current_raw_obs, actions, dones=dones)

        if np.any(dones):
            self.episode_steps[dones] = 0
            self.current_friction[dones] = self.config.nominal_friction

        events = self.check_dynamic_events()
        policy_obs = format_policy_input(self.config.model_id, current_raw_obs, self.history_buffer)

        info = {
            "events": events,
            "friction": self.current_friction.copy(),
            "episode_steps": self.episode_steps.copy(),
        }

        return policy_obs, info
