"""Playground G1 joystick task on the repository's generated collision surface."""

import jax.numpy as jp
import mujoco
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base, joystick

from src.training.m3_history import initial_input, next_input


class HistoryJoystick(joystick.Joystick):
    def __init__(self, scene, config, length, mesh_directory, *, actor_velocity=False):
        mjx_env.MjxEnv.__init__(self, config)
        self._xml_path = str(scene)
        self._model_assets = base.get_assets()
        mjx_env.update_assets(self._model_assets, mesh_directory, "*.STL")
        self._mj_model = mujoco.MjModel.from_xml_string(scene.read_text(), assets=self._model_assets)
        self._mj_model.opt.timestep = self.sim_dt
        if config.restricted_joint_range:
            from mujoco_playground._src.locomotion.g1.g1_constants import RESTRICTED_JOINT_RANGE
            self._mj_model.jnt_range[1:] = RESTRICTED_JOINT_RANGE
            self._mj_model.actuator_ctrlrange[:] = RESTRICTED_JOINT_RANGE
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=config.impl)
        self._post_init()
        self.history_length = length
        self.actor_velocity = actor_velocity

    def _actor_obs(self, state):
        data = state.data
        fields = (
            self.get_gyro(data, "pelvis"),
            -data.site_xmat[self._pelvis_imu_site_id][2],
            state.info["command"],
            data.qpos[7:] - self._default_pose,
            data.qvel[6:],
        )
        if self.actor_velocity:
            fields += (self.get_local_linvel(data, "pelvis")[:2],)
        return jp.concatenate(fields)

    def reset(self, rng):
        state = super().reset(rng)
        vector = initial_input(self._actor_obs(state), self.action_size, self.history_length, jp)
        obs = dict(state.obs, state=vector)
        metrics = dict(state.metrics, m3_distance=jp.zeros(()), m3_tracking=jp.zeros(()),
                       m3_fall=jp.zeros(()))
        return state.replace(obs=obs, metrics=metrics)

    def step(self, state, action):
        previous = state.obs["state"]
        xy = state.data.qpos[:2]
        command = state.info["command"][:2]
        result = super().step(state, action)
        obs = next_input(previous, action, self._actor_obs(result), result.done.astype(bool), jp)
        velocity = self.get_local_linvel(result.data, "pelvis")[:2]
        metrics = dict(result.metrics, m3_distance=jp.linalg.norm(result.data.qpos[:2] - xy),
                       m3_tracking=jp.linalg.norm(velocity - command), m3_fall=result.done)
        return result.replace(obs=dict(result.obs, state=obs), metrics=metrics)
