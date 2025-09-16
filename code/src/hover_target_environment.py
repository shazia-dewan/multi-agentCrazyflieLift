import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional


class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self, 
        xml_path: str, 
        num_drones: int = 1, 
        target_pos: np.ndarray = np.array([0.0, 0.0, 0.5], dtype=np.float32),
        max_steps: int = 100,
        debug: bool = False,
    ):
        """
        Initialize the training environment

        Parameters
        ----------
        xml_path : string 
            Path to xml MuJoCo scene
        num_drones : int
            Number of drones in the MuJoCo scene
        target_pos : float
            Target position the drone will fly to and hover at
        max_steps : int
            Termination condition for episodes (when we have reached max_steps)
        debug : bool
            Enables some logging (may remove later)
        """
        super().__init__()

        # Store config
        self.num_drones = num_drones
        assert target_pos[2] > 0.4, "Target position too close to the ground"
        self.target_pos = target_pos
        self.debug = debug

        # MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Drone obervation space
        # [pos(3) + quaternion(4) + velocity(3) + angular_velocity(3)] = 13 per drone
        obs_high = np.inf * np.ones(13 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Drone action space (see aicraft axes: https://en.wikipedia.org/wiki/Aircraft_principal_axes)
        # thrust + roll + pitch + yaw = 4 per drone
        # Controls use PD (Proportional-Derivative) control + RL residual
        act_high = np.tile([0.35, 0.2, 0.2, 0.2], self.num_drones).astype(np.float32)
        act_low = np.tile([0.0, 0.2, 0.2, 0.2], self.num_drones).astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Viewer
        self.viewer = None

        # 7 qpos, 6 qvel, and 4 action controls per drone
        self.qpos_per_drone = 7
        self.qvel_per_drone = 6
        self.ctrl_per_drone = 4

        self.timestep = 0
        self.max_steps = max_steps


    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the training environment

        Parameters
        ----------
        seed : int | None
            Optional seed for consistent environments.

        Returns
        -------
        observation : np.ndarray
            The initial observation of the environment
        info : dict
            Additional reset information
        """
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Set drone Z position to random during training (learn rise / drop about target)
        # NOTE: add flag/option to initialize drone/target position manually (not randomly) during eval
        self.target_pos = np.array([0.0, 0.0, np.random.uniform(0.1, 1.0)], dtype=np.float32)
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            self.data.qpos[base_qpos + 2] = np.random.uniform(0.1, 1.0)

        # Start drone(s) with 0 velocity
        for i in range(self.num_drones):
            base_qvel = i * self.qvel_per_drone
            self.data.qvel[base_qvel: base_qvel + self.qvel_per_drone] = 0.0

        # Track previous positions
        self.prev_pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            self.prev_pos[i] = self.data.qpos[base_qpos: base_qpos + 3]

        self.timestep = 0

        return self._get_obs(), {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take a step (action) and track the observation

        Parameters
        ----------
        action : np.ndarray
            Control action for the drones

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        # Render the target geom at the target position 
        # Keeping here for now rather than reset() in case we switch to moving targets
        target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.model.geom_pos[target_geom_id] = self.target_pos

        # Clip action within action space
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Termination values
        reward = 0.0
        done = False

        # Fill control array
        ctrl = np.zeros(self.num_drones * self.ctrl_per_drone, dtype=np.float32)

        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone
            base_ctrl = i * self.ctrl_per_drone

            # Current state
            pos = self.data.qpos[base_qpos: base_qpos + 3]
            quat = self.data.qpos[base_qpos + 3: base_qpos + 7]
            vel = self.data.qvel[base_qvel: base_qvel + 3]
            ang_vel = self.data.qvel[base_qvel + 3: base_qvel + 6]

            #############################################
            # RL Controls
            #############################################

            thrust = action[base_ctrl + 0]
            roll  = action[base_ctrl + 1]
            pitch = action[base_ctrl + 2]
            yaw   = action[base_ctrl + 3]

            # Clip controls to valid ranges (see crazyflie XML actuator)
            ctrl[0] = np.clip(thrust, 0, 0.35)
            # Not using roll, yaw, or pitch for now (vertical hover test)
            ctrl[1] = np.clip(0, -1, 1)
            ctrl[2] = np.clip(0, -1, 1)
            ctrl[3] = np.clip(0, -1, 1)


            ##############################################
            # Rewards
            ##############################################

            # Reward for moving closer to target
            # Compare new & old distance to generate a per step reward (-dist_new alone may not signal improvement)
            dist_old = np.linalg.norm(self.prev_pos[i] - self.target_pos)
            dist_new = np.linalg.norm(pos - self.target_pos)
            reward += dist_old - dist_new
            
            # Update previous position for next timestep
            self.prev_pos[i] = pos.copy()

            # Termination
            self.timestep += 1
            if self.timestep >= self.max_steps:
                done = True
            
        # Apply control and step
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        return obs, reward, done, False, {}


    def _get_obs(self) -> np.ndarray:
        """
        Retrieve an observation of the current drone state

        Returns
        -------
        obs : np.ndarray
            Concatenated observation vector for all drones
        """
        obs = []
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone

            # Observation
            # qpos stores (position[x, y, z], orientation[qz, qy, qz, q2])
            # qvel stores (velocity[vx, vy, vz], angular velocity[wx, wy, wz])

            pos = self.data.qpos[base_qpos: base_qpos + 3]
            quat = self.data.qpos[base_qpos + 3: base_qpos + 7]
            vel = self.data.qvel[base_qvel: base_qvel + 3]
            ang_vel = self.data.qvel[base_qvel + 3: base_qvel + 6]

            obs.extend(np.concatenate([pos, quat, vel, ang_vel]))

        return np.array(obs, dtype=np.float32)


    def render(self) -> None:
        """
        Render the model in MuJoCo
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()


    def close(self) -> None:
        """
        Close the MuJoCo viewer
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
