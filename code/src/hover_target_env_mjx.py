from mujoco_playground._src.mjx_env import MjxEnv, State, init, step
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict

import mediapy as media

from constants import SCENE_PATH
TARGET_POS = jnp.array([0.0, 0.0, 0.5])

# Crazyflie env based on the MuJoCo playground mjx env that uses GPU acceleration 
# https://github.com/google-deepmind/mujoco_playground
class CrazyflieEnv(MjxEnv):
    """Simple MJX environment using the Crazyflie drone model."""

    def __init__(self):
        # Define environment configuration (control and sim time steps)
        cfg = config_dict.ConfigDict()
        cfg.ctrl_dt = 0.02    # control step (50Hz)
        cfg.sim_dt = 0.002    # simulation step (500Hz)
        super().__init__(cfg)

        # Load MuJoCo model and convert to MJX
        self._mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self._mjx_model = mjx.put_model(self._mj_model)

        # Define control/action size as number of actuators
        self._action_size = self._mj_model.nu

    @property
    def xml_path(self):
        return SCENE_PATH

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model

    @property
    def action_size(self):
        return self._action_size



    def reset(self, rng: jax.Array) -> State:
        """Reset the Crazyflie to default state."""
        qpos = jnp.array(self._mj_model.key_qpos[0])
        qvel = jnp.zeros(self._mj_model.nv)
        data = init(self._mjx_model, qpos=qpos, qvel=qvel)
        obs = self._get_obs(data)
        reward = jnp.array(0.0)
        done = jnp.array(False)
        
        return State(data=data, obs=obs, reward=reward, done=done, metrics={}, info={})


    def step(self, state: State, action: jax.Array) -> State:
        """Advance Crazyflie dynamics by one control step."""
        data = step(self._mjx_model, state.data, action, n_substeps=self.n_substeps)
        obs = self._get_obs(data)

        # Simple hover reward: penalize deviation from origin
        pos = data.qpos[:3]

        # Reward: - Euclidean distance to target
        reward = -jnp.linalg.norm(pos - TARGET_POS)

        # Done: if drone goes too far from target or crashes into the ground
        done = jnp.any(jnp.abs(pos - TARGET_POS) > 2.0) | (pos[2] < 0.05)

        return State(data=data, obs=obs, reward=reward, done=done, metrics={}, info={})


    def _get_obs(self, data: mjx.Data):
        return jnp.concatenate([data.qpos, data.qvel])


# Simple rollout example
# Based on Rollout section from https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb
if __name__ == "__main__":
    env = CrazyflieEnv()
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)
    state = jit_env_reset(key)
    rollout = [state]

    episode_length = 1000

    for i in range(episode_length):
        # Hover action with random offset
        key, subkey = jax.random.split(key)
        hover_offset = jax.random.uniform(subkey, (), minval=-0.002, maxval=0.002)
        action = jax.numpy.array([0.26487 + hover_offset, 0.0, 0.0, 0.0])

        state = jit_env_step(state, action)
        rollout.append(state)

        print(f"Step {i+1:04d}: reward={state.reward:.3f}, done={state.done}")
        if bool(state.done):
            break


    # Utility to render target as geom in scene
    target_geom_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, b"target")
    def move_target_geom(scene: mujoco.MjvScene, target_pos: np.ndarray):
        scene.geoms[target_geom_id].pos[:] = target_pos

    modify_target_fn = lambda scene: move_target_geom(scene, TARGET_POS)

    # Collect frames, use track camera from xml scene and update geom pos to target pos
    frames = env.render(rollout, camera="track", modify_scene_fns=[lambda scene: move_target_geom(scene, TARGET_POS)]*len(rollout))

    # Need to install ffmpeg for visualization - using the mjx_env renderer
    # Cannot use MuJoCo viewer which operates on CPU since this env state runs on Jax which operates on GPU
    # https://www.ffmpeg.org/ (e.g. on Windows, download a release, add ffmpeg/.../bin to PATH env variable)
    media.write_video("crazyflie_rollout_mjx.mp4", frames, fps=1.0 / env.dt)
    print("Saved rollout to crazyflie_rollout_mjx.mp4")

