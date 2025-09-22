import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional


BASE_HOVER_THRUST = 0.26487
TRAINING_POS_RANGE = 1.0
TRAINING_QUAT_RANGE = np.pi / 18

class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self, 
        xml_path: str, 
        num_drones: int = 1, 
        target_pos: np.ndarray = np.array([0.0, 0.0, 0.5], dtype=np.float32),
        max_steps: int = 400,
        random_initialization: bool = True,
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
        random_initialization : bool
            Whether to randomly initialize target pos and drone pos/rotation
        debug : bool
            Enables some logging (may remove later)
        """
        super().__init__()

        # MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # 7 qpos, 6 qvel, and 4 action controls per drone
        self.qpos_per_drone = 7
        self.qvel_per_drone = 6
        self.ctrl_per_drone = 4

        # Store user config
        self.num_drones = num_drones
        assert target_pos[2] > 0.0, "Target position is under the ground"
        self.target_pos_param = target_pos
        self.max_steps = max_steps
        self.random_initialization = random_initialization
        self.debug = debug

        # Drone obervation space: 33 features (base features and some engineered ones)
        obs_high = np.inf * np.ones(33 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.obs_index_to_name = {
            0: "pos_x", 1: "pos_y", 2: "pos_z",
            3: "quat_w", 4: "quat_x", 5: "quat_y", 6: "quat_z",
            7: "vel_x", 8: "vel_y", 9: "vel_z",
            10: "ang_x", 11: "ang_y", 12: "ang_z",
            13: "norm_x", 14: "norm_y", 15: "norm_z",
            16: "dist_to_target",
            17: "err_x", 18: "err_y", 19: "err_z",
            20: "derivative_err_x", 21: "derivative_err_y", 22: "derivative_err_z",
            23: "sec_derivative_err_x", 24: "sec_derivative_err_y", 25: "sec_derivative_err_z",
            26: "integral_err_x", 27: "integral_err_y", 28: "integral_err_z",
            29: "prev_ctrl_0", 30: "prev_ctrl_1", 31: "prev_ctrl_2", 32: "prev_ctrl_3",
        }

        # Drone action space (see aicraft axes: https://en.wikipedia.org/wiki/Aircraft_principal_axes)
        # thrust + roll + pitch + yaw = 4 per drone
        act_high = np.tile([0.35, 1, 1, 1], self.num_drones).astype(np.float32)
        act_low = np.tile([0.0, -1, -1, -1], self.num_drones).astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Viewer
        self.viewer = None


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
        
        drone_spacing_x = 0.4
        drone_offsets_x = np.linspace(-(self.num_drones - 1) / 2, (self.num_drones - 1) / 2, self.num_drones) * drone_spacing_x
        
        # Start target and drone at some random pos [X, Y, Z] (and quat [w, x, y, z] for drone)
        # Currently hover (Z) only
        if self.random_initialization:

            # Start target at some random [X, Y, Z]
            self.target_pos = np.array(
                [
                    self.np_random.uniform(-TRAINING_POS_RANGE / 2, TRAINING_POS_RANGE / 2),
                    self.np_random.uniform(-TRAINING_POS_RANGE / 2, TRAINING_POS_RANGE / 2),
                    self.np_random.uniform(0.1, TRAINING_POS_RANGE)
                ],
                dtype=np.float32
            )

            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                # Random position
                start_x = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)
                start_y = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)
                start_z =self.np_random.uniform(0.1, TRAINING_POS_RANGE)

                self.data.qpos[base_qpos + 0] = start_x + drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = start_y
                self.data.qpos[base_qpos + 2] = start_z

                # Random rotation axis with small rotation angle
                axis = self.np_random.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = self.np_random.uniform(-TRAINING_QUAT_RANGE, TRAINING_QUAT_RANGE)

                w = np.cos(angle / 2.0)
                x, y, z = axis * np.sin(angle / 2.0)

                quat = np.array([w, x, y, z], dtype=np.float64)
                self.data.qpos[base_qpos + 3: base_qpos + 7] = quat
        else:
            self.target_pos = self.target_pos_param
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone

                self.data.qpos[base_qpos + 0] = drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = 0
                self.data.qpos[base_qpos + 2] = 0.05 + np.random.uniform(0.05, 0.1)

        for i in range(self.num_drones):
            # Start drone(s) with 0 velocity
            base_qvel = i * self.qvel_per_drone
            self.data.qvel[base_qvel: base_qvel + self.qvel_per_drone] = 0.0
        
        self.timestep = 0

        # Set up tracking for previous drone(s) e.g. for certain features like derivative
        self.prev_action = np.zeros((self.num_drones, self.ctrl_per_drone), dtype=np.float32)
        self.prev_pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_derivative_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)

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
        terminated = False
        truncated = False

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
            base_ctrl = i * self.ctrl_per_drone

            thrust = action[base_ctrl + 0]
            roll   = action[base_ctrl + 1]
            pitch  = action[base_ctrl + 2]
            yaw    = action[base_ctrl + 3]

            # Not using roll/pitch/yaw for now
            ctrl[base_ctrl + 0] = np.clip(thrust, 0.0, 0.35)
            ctrl[base_ctrl + 1] = 0.0
            ctrl[base_ctrl + 2] = 0.0
            ctrl[base_ctrl + 3] = 0.0


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

            # Update prev action for next timestep (to store in observations)
            self.prev_action[i] = ctrl.copy()

            # Termination (just truncation in this case, no terminating condition)
            terminated = False
            truncated = self.timestep >= self.max_steps

            if self.debug:
                print(f"Ctrl: {[f'{c:.4f}' for c in ctrl]}, Height: {pos[2]:.2f}, Reward: {reward:.4f}")
            
        # Apply control and step
        self.data.ctrl[:] = ctrl
        self.timestep += 1
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()

        return obs, reward, terminated, truncated, {}


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
            base_ctrl = i * self.ctrl_per_drone

            ###########################################
            # Base Features in Observation Space:
            ###########################################

            # position[x, y, z]
            pos = self.data.qpos[base_qpos: base_qpos + 3]

            # quaternion[qw, qx, qy, qz,]
            quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]

            # velocity[vx, vy, vz]
            vel = self.data.qvel[base_qvel: base_qvel + 3]

            # angular velocity[wx, wy, wz]
            ang_vel = self.data.qvel[base_qvel + 3 : base_qvel + 6]

            #################################################################
            # Engineered Features in Observation Space
            # Using PPO w/ tanh so trying to normalize features to [-1, 1]
            # Including derivative and integral info (used in PID controller)
            #################################################################

            # pos_error is a vector for the XYZ position error between the drone and target
            pos_error = pos - self.target_pos

            # normalize_pos stores [x, y, z] normalized by the l2 (euclidean) norm of target
            normalized_pos = 2 * (pos / (np.linalg.norm(self.target_pos) + 1e-9)) - 1

            # distance_to_target is a scalar storing absolute distance to target
            distance_to_target = np.linalg.norm(pos_error)

            # 1st and 2nd derivative position error over one step ~ velocity and acceleration
            derivative_error = pos_error - self.prev_pos_error[i]
            second_derivative_error = derivative_error - self.prev_derivative_error[i]

            # Integral position error: exponential moving average to avoid unbounded growth
            self.integral_pos_error[i] = 0.95 * self.integral_pos_error[i] + 0.05 * pos_error

            # Update prev pos and derivative error
            self.prev_pos_error[i] = pos_error.copy()
            self.prev_derivative_error[i] = derivative_error.copy()
                        
            # normalized_prev_ctrl stores normalized previous action controls
            normalized_prev_ctrl = [
                2 * (self.prev_action[i][0] / 0.35) - 1,
                self.prev_action[i][1],
                self.prev_action[i][2],
                self.prev_action[i][3]
            ]

            obs.extend(np.concatenate(
                [
                    pos, quat, vel, ang_vel, 
                    normalized_pos,
                    np.array([distance_to_target], dtype=np.float32),
                    pos_error,
                    derivative_error,
                    second_derivative_error,
                    self.integral_pos_error[i],
                    normalized_prev_ctrl
                ]))

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
