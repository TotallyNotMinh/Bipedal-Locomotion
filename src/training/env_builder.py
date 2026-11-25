"""MuJoCo Playground Unitree G1 environment builder & singular terrain integration.

Does NOT modify `configs/terrain.yaml` or `src/envs/terrain.py`; consumes them read-only
when `terrain_mode == "singular"`.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import xml.etree.ElementTree as ET
import numpy as np
import yaml

from src.envs.terrain import SingularTerrainArena, TerrainGenerator


def _import_playground_locomotion():
    """Imports mujoco_playground.locomotion with clear guidance on v0.2/rss2025 vs v1.0."""
    try:
        from mujoco_playground import locomotion
        return locomotion
    except ImportError as exc:
        raise ImportError(
            "Could not import `mujoco_playground.locomotion`.\n"
            "Note: `google-deepmind/mujoco_playground` main branch moved JAX/Brax locomotion "
            "(G1JoystickFlatTerrain) into `rss2025/` in v1.0+.\n"
            "On Kaggle, install the JAX/MJX locomotion package with one of:\n"
            '  pip install --upgrade "git+https://github.com/google-deepmind/mujoco_playground.git#subdirectory=rss2025"\n'
            'or pin the pre-v1.0 release:\n'
            '  pip install --upgrade "git+https://github.com/google-deepmind/mujoco_playground.git@v0.2.0"'
        ) from exc


def _apply_env_config_overrides(mjx_cfg: Any, env_cfg: Dict[str, Any]) -> Any:
    """Applies supported config overrides onto a MuJoCo Playground ConfigDict."""
    for key in ("impl", "episode_length", "action_repeat", "ctrl_dt", "sim_dt"):
        if key in env_cfg and hasattr(mjx_cfg, key):
            setattr(mjx_cfg, key, env_cfg[key])
    return mjx_cfg


def _inject_singular_terrain_arena(
    env: Any,
    arena: SingularTerrainArena,
    spawn_x: float = 8.5,
    spawn_y: Optional[float] = None,
    downsample: int = 1,
    impl: Optional[str] = None,
) -> Any:
    """Rebuilds `env._mj_model` and `env._mjx_model` with a SingularTerrainArena heightfield.

    Maps `arena.heights` (shape `(nx, ny)` over `[0, size_x] x [0, size_y]`) into MuJoCo's
    `<hfield>` layout `(nrow=ny, ncol=nx)` and shifts the floor geom so world `(0, 0)`
    matches `(spawn_x, spawn_y)` at surface height `z = 0.0`.
    """
    import mujoco
    from mujoco import mjx
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.g1 import base as g1_base, g1_constants

    ds = max(1, int(downsample))
    heights = arena.heights[::ds, ::ds]
    nx, ny = heights.shape

    min_h = float(np.min(heights))
    max_h = float(np.max(heights))
    elevation_z = max(0.01, max_h - min_h)
    base_z = 0.50
    normalized = np.clip((heights - min_h) / elevation_z, 0.0, 1.0).astype(np.float32)

    half_x = arena.size_x / 2.0
    half_y = arena.size_y / 2.0

    # Locate spawn point inside arena and align its height to world z = 0.0
    sy = arena.size_y / 2.0 if spawn_y is None else float(spawn_y)
    eff_scale = arena.horizontal_scale * ds
    si = int(np.clip(round(spawn_x / eff_scale), 0, nx - 1))
    sj = int(np.clip(round(sy / eff_scale), 0, ny - 1))
    spawn_h = float(heights[si, sj])

    pos_x = half_x - float(spawn_x)
    pos_y = half_y - sy
    pos_z = min_h - spawn_h

    xml_path = Path(g1_constants.task_to_xml("rough_terrain"))
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    hfield_elem = None
    for h in asset.findall("hfield"):
        if h.get("name") == "hfield":
            hfield_elem = h
            break
    if hfield_elem is None:
        hfield_elem = ET.SubElement(asset, "hfield")

    # Remove `file` attribute so MuJoCo allocates data buffer directly from nrow/ncol
    hfield_elem.attrib.pop("file", None)
    hfield_elem.set("name", "hfield")
    hfield_elem.set("nrow", str(ny))
    hfield_elem.set("ncol", str(nx))
    hfield_elem.set("size", f"{half_x:.4f} {half_y:.4f} {elevation_z:.4f} {base_z:.4f}")

    worldbody = root.find("worldbody")
    if worldbody is not None:
        floor_geom = None
        for g in worldbody.findall("geom"):
            if g.get("name") == "floor":
                floor_geom = g
                break
        if floor_geom is None:
            floor_geom = ET.SubElement(worldbody, "geom", {"name": "floor"})
        floor_geom.set("type", "hfield")
        floor_geom.set("hfield", "hfield")
        floor_geom.set("pos", f"{pos_x:.4f} {pos_y:.4f} {pos_z:.4f}")

    xml_str = ET.tostring(root, encoding="unicode")
    mj_model = mujoco.MjModel.from_xml_string(xml_str, assets=g1_base.get_assets())
    mj_model.opt.timestep = env.sim_dt
    mj_model.vis.global_.offwidth = 3840
    mj_model.vis.global_.offheight = 2160

    # MuJoCo hfield_data has shape (nrow=ny, ncol=nx) where axis 0 is Y and axis 1 is X
    mj_model.hfield_data[:] = normalized.T.ravel()

    # Re-apply joint range restrictions if helper exists
    if hasattr(g1_constants, "RESTRICTED_JOINT_RANGE") and len(mj_model.jnt_range) > 1:
        try:
            mj_model.jnt_range[1:] = g1_constants.RESTRICTED_JOINT_RANGE
        except Exception:
            pass

    target_impl = impl or getattr(env._config, "impl", "jax")
    env._mj_model = mj_model
    try:
        env._mjx_model = mjx.put_model(mj_model, impl=target_impl)
    except TypeError:
        env._mjx_model = mjx.put_model(mj_model)
    return env


def build_g1_environment(
    config: Dict[str, Any],
    override_category: Optional[str] = None,
    override_difficulty: Optional[float] = None,
    override_seed: Optional[int] = None,
) -> Tuple[Any, Any, Optional[SingularTerrainArena]]:
    """Builds the MuJoCo Playground G1 environment and its underlying config.

    Returns:
        `(env, playground_env_cfg, singular_arena_or_none)`
    """
    locomotion = _import_playground_locomotion()
    env_cfg = config.get("env", {})

    terrain_mode = str(env_cfg.get("terrain_mode", "playground")).lower()
    task_name = str(env_cfg.get("task_name", "G1JoystickFlatTerrain"))
    category = override_category or str(env_cfg.get("terrain_category", "flat"))
    difficulty = float(
        env_cfg.get("terrain_difficulty", 0.0)
        if override_difficulty is None
        else override_difficulty
    )
    impl = str(env_cfg.get("impl", "jax"))

    # In singular mode with non-flat category, start from rough_terrain scene so hfield sensors match
    load_task = task_name
    if terrain_mode == "singular" and category != "flat":
        load_task = "G1JoystickRoughTerrain"

    pg_cfg = locomotion.get_default_config(load_task)
    _apply_env_config_overrides(pg_cfg, env_cfg)

    env = locomotion.load(load_task, config=pg_cfg)
    arena: Optional[SingularTerrainArena] = None

    if terrain_mode == "singular":
        t_cfg_path = Path(env_cfg.get("terrain_config_path", "configs/terrain.yaml"))
        raw_t_cfg = {}
        if t_cfg_path.exists():
            with open(t_cfg_path, "r") as f:
                raw_t_cfg = yaml.safe_load(f) or {}

        t_gen = TerrainGenerator(raw_t_cfg)
        seed = override_seed if override_seed is not None else int(config.get("experiment", {}).get("seed", t_gen.seed))
        arena = t_gen.generate(category=category, difficulty=difficulty, seed=seed)

        if category != "flat":
            downsample = int(env_cfg.get("terrain_downsample", 1))
            env = _inject_singular_terrain_arena(
                env=env,
                arena=arena,
                downsample=downsample,
                impl=impl,
            )

    return env, pg_cfg, arena


def build_perturbed_friction_env(
    config: Dict[str, Any],
    friction: float,
    override_category: Optional[str] = None,
    override_difficulty: Optional[float] = None,
) -> Any:
    """Builds a G1 environment with a fixed floor/foot contact friction coefficient."""
    import jax.numpy as jnp
    from mujoco import mjx

    env, _, _ = build_g1_environment(
        config,
        override_category=override_category,
        override_difficulty=override_difficulty,
    )
    impl = str(config.get("env", {}).get("impl", "jax"))

    # Update both MuJoCo pair_friction and geom_friction on CPU model, then upload to MJX
    mj_model = getattr(env, "mj_model", getattr(env, "_mj_model", None))
    if mj_model is not None:
        if mj_model.npair > 0:
            mj_model.pair_friction[: min(2, mj_model.npair), :2] = float(friction)
        if mj_model.ngeom > 0:
            mj_model.geom_friction[0, 0] = float(friction)
        try:
            env._mjx_model = mjx.put_model(mj_model, impl=impl)
        except TypeError:
            env._mjx_model = mjx.put_model(mj_model)

    # Also patch pair_friction directly in case put_model preserves defaults
    if (
        hasattr(env, "_mjx_model")
        and hasattr(env._mjx_model, "tree_replace")
        and hasattr(env._mjx_model, "pair_friction")
        and env._mjx_model.pair_friction.shape[0] > 0
    ):
        n_pairs = min(2, env._mjx_model.pair_friction.shape[0])
        updated_pf = env._mjx_model.pair_friction.at[:n_pairs, :2].set(jnp.float32(friction))
        env._mjx_model = env._mjx_model.tree_replace({"pair_friction": updated_pf})

    return env


def get_domain_randomizer_fn(
    config: Dict[str, Any],
) -> Optional[Callable[..., Any]]:
    """Fetches MuJoCo Playground's G1 domain randomization function if enabled."""
    env_cfg = config.get("env", {})
    if not bool(env_cfg.get("domain_randomization", True)):
        return None

    locomotion = _import_playground_locomotion()
    task_name = str(env_cfg.get("task_name", "G1JoystickFlatTerrain"))
    try:
        return locomotion.get_domain_randomizer(task_name)
    except Exception:
        return None
