import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional

from util import quat_to_rotation_matrix


BASE_HOVER_THRUST = 0.26487
DRONE_TILT_EPISLON = np.deg2rad(10)
TRAINING_POS_RANGE = 2.0
TRAINING_QUAT_RANGE = np.pi / 18

class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str, 
        num_drones: int = 1, 
        target_pos: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
        max_steps: int = 1200,
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
        self.target_pos = target_pos
        self.max_steps = max_steps
        self.random_initialization = random_initialization
        self.debug = debug

        # Index of observations for logging afterwards (e.g. which observations were important)
        self.obs_index_to_name = {
            # Base features
            0: "pos_x", 1: "pos_y", 2: "pos_z",
            3: "quat_w", 4: "quat_x", 5: "quat_y", 6: "quat_z",
            7: "vel_x", 8: "vel_y", 9: "vel_z",
            10: "ang_x", 11: "ang_y", 12: "ang_z",
            # Positional features
            13: "pos_norm_x", 14: "pos_norm_y", 15: "pos_norm_z",
            16: "pos_err_x", 17: "pos_err_y", 18: "pos_err_z",
            19: "dist_to_target", 20: "squared_dist_to_target",
            21: "derivative_pos_err_x", 22: "derivative_pos_err_y", 23: "derivative_pos_err_z",
            24: "sec_derivative_pos_err_x", 25: "sec_derivative_pos_err_y", 26: "sec_derivative_pos_err_z",
            27: "integral_pos_err_x", 28: "integral_pos_err_y", 29: "integral_pos_err_z",
            30: "prev_ctrl_thrust", 31: "prev_ctrl_roll", 32: "prev_ctrl_pitch", 33: "prev_ctrl_yaw",
            34: "vel_dist_x", 35: "vel_dist_y", 36: "vel_dist_z", 
            37: "vel_along_err",
            38: "vel_norm_x", 39: "vel_norm_y", 40: "vel_norm_z",
            # Rotational features
            41: "drone_up_x", 42: "drone_up_y", 43: "drone_up_z",
            44: "quat_err_x", 45: "quat_err_y", 46: "quat_err_z",
            47: "derivative_quat_err_x", 48: "derivative_quat_err_y", 49: "derivative_quat_err_z",
            50: "integral_quat_err_x", 51: "integral_quat_err_y", 52: "integral_quat_err_z",
            53: "vel_rotation_x", 54: "vel_rotation_y",
            55: "pos_error_rotation_x", 56: "pos_error_rotation_y"
        }
        obs_high = np.inf * np.ones(57 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

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

        # Set up tracking for previous drone(s) e.g. for certain features like derivative
        self.prev_action = np.zeros((self.num_drones, self.ctrl_per_drone), dtype=np.float32)
        
        self.prev_pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_derivative_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)

        self.prev_quat_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_quat_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        
        # If using multiple drones, offset them along X axis
        drone_spacing_x = 0.4
        drone_offsets_x = np.linspace(-(self.num_drones - 1) / 2, (self.num_drones - 1) / 2, self.num_drones) * drone_spacing_x
        
        # Start drone at some random pos [X, Y, Z] and quat [w, x, y, z]
        if self.random_initialization:
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                # Random position
                start_x = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)
                start_y = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)
                start_z =self.np_random.uniform(0.1, TRAINING_POS_RANGE)

                self.data.qpos[base_qpos + 0] = 0.0 + drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = 0.0
                self.data.qpos[base_qpos + 2] = 1.0

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]

                # Random rotation axis with small rotation angle
                axis = self.np_random.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = self.np_random.uniform(-TRAINING_QUAT_RANGE, TRAINING_QUAT_RANGE)

                w = np.cos(angle / 2.0)
                x, y, z = axis * np.sin(angle / 2.0)

                quat = np.array([w, x, y, z], dtype=np.float64)
                self.data.qpos[base_qpos + 3: base_qpos + 7] = quat
        else:
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                self.data.qpos[base_qpos + 0] = drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = 0
                self.data.qpos[base_qpos + 2] = 1.0

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]

        for i in range(self.num_drones):
            # Start drone(s) with 0 velocity
            base_qvel = i * self.qvel_per_drone
            self.data.qvel[base_qvel: base_qvel + self.qvel_per_drone] = 0.0
        
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

            thrust = np.clip(action[base_ctrl + 0], 0.0, 0.35)
            roll   = np.clip(action[base_ctrl + 1], -1, 1)
            pitch  = np.clip(action[base_ctrl + 2], -1, 1)
            yaw    = np.clip(action[base_ctrl + 3], -1, 1)

            ctrl[base_ctrl + 0] = BASE_HOVER_THRUST
            ctrl[base_ctrl + 1] = roll
            ctrl[base_ctrl + 2] = pitch
            ctrl[base_ctrl + 3] = 0.0


            ##############################################
            # Rewards
            ##############################################

            # Reward for moving closer to target
            # Compare new & old distance to generate a per step reward (-dist_new alone may not signal improvement)
            dist_old = np.linalg.norm(self.prev_pos[i] - self.target_pos)
            dist_new = np.linalg.norm(pos - self.target_pos)
            reward += dist_old - dist_new

            # Reward for staying upright
            w, x, y, z = quat[:]
            drone_z_up = np.array([
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y)
            ], dtype=np.float32)
            # tilt angle = arccos(dot product of drone_z_up and world_z_up)
            # In this case, just equal to drone_z_up[2] since world_z_up = [0, 0, 1]
            tilt_angle = np.arccos(np.clip(drone_z_up[2], -1.0, 1.0))
            if tilt_angle < DRONE_TILT_EPISLON:
                reward += 0.001


            # Update previous position for next timestep
            self.prev_pos[i] = pos.copy()

            # Update prev action for next timestep (to store in observations)
            self.prev_action[i] = ctrl.copy()

            # Termination (just truncation in this case, no terminating condition)
            terminated = False
            truncated = self.timestep >= self.max_steps

            if self.debug:
                print(f"Ctrl: {[f'{c:.4f}' for c in ctrl]}, Position: {[f'{p:.2f}' for p in pos]}, Reward: {reward:.4f}")
            
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
            #################################################################

            # ---------------------------------------------------------------------
            # Features that are mainly built to help thrust control
            # E.g. derivative and integral pos error to help model PID controller
            # ---------------------------------------------------------------------

            # normalize_pos stores [x, y, z] normalized by the l2 (euclidean) norm of target + scaled to [-1, 1]
            normalized_pos = 2 * pos / (np.linalg.norm(self.target_pos) + 1e-9) - 1

            # pos_error is a vector for the XYZ position error between the drone and target
            pos_error = pos - self.target_pos

            # distance_to_target and squared_distance_to_target are scalars storing abs/squared distance to target
            distance_to_target = np.linalg.norm(pos_error)
            squared_distance_to_target = distance_to_target ** 2

            # 1st and 2nd derivative position error over one step ~ velocity and acceleration of error
            derivative_error = pos_error - self.prev_pos_error[i]
            second_derivative_error = derivative_error - self.prev_derivative_error[i]

            # Integral position error: exponential moving average to avoid unbounded growth
            self.integral_pos_error[i] = 0.95 * self.integral_pos_error[i] + 0.05 * pos_error

            # Update prev pos and derivative error
            self.prev_pos_error[i] = pos_error.copy()
            self.prev_derivative_error[i] = derivative_error.copy()
            
            # normalized_prev_ctrl stores normalized [-1, 1] previous action controls
            normalized_prev_ctrl = [
                2 * (self.prev_action[i][0] / 0.35) - 1,
                self.prev_action[i][1],
                self.prev_action[i][2],
                self.prev_action[i][3]
            ]

            # Velocity over distance
            vel_over_dist = vel / (distance_to_target + 1e-9)

            # Projection of velocity along the error vector
            # scalar value: moving toward (+) or away (-) from the target, and how fast relative to the gap
            vel_along_error = np.dot(vel, pos_error) / (distance_to_target + 1e-9)

            # Normalized velocity
            vel_norm = vel / (np.linalg.norm(vel) + 1e-9)

            # ---------------------------------------------------------------------
            # Features that are mainly built to help orientation control
            # ---------------------------------------------------------------------
            w, x, y, z = quat[:]

            # drone_up_vector is where the drones local body Z axis is pointing
            # It is the Z column in the Quaternion-derived rotation matrix: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
            drone_up_vector = np.array([
                2 * (x * z + w * y),
                2 * (y * z - w * x),
                1 - 2 * (x * x + y * y)
            ])

            # quat_error: the difference between the drones Z-up axis and the world Z-up axis (how far from level)
            quat_error = drone_up_vector - np.array([0, 0, 1])
            quat_derivative_error = quat_error - self.prev_quat_error[i]
            self.integral_quat_error[i] = 0.95 * self.integral_quat_error[i] + 0.05 * quat_error
            self.prev_quat_error[i] = quat_error.copy()

            # From quaternion -> rotation matrix
            R = quat_to_rotation_matrix(quat)
             # Drone's body x and y axes in world frame
            rotation_x_vector = R[:, 0]
            rotation_y_vector   = R[:, 1]

            # Project velocity onto drone axes, how much is drone moving on roll/pitch axes
            vel_rotation_x = np.dot(vel, rotation_x_vector)
            vel_rotation_y   = np.dot(vel, rotation_y_vector)

            # Project pos_error onto drone axes, how is pos error moving on roll/pitch axes
            pos_error_rotation_x = np.dot(pos_error, rotation_x_vector)
            pos_error_rotation_y   = np.dot(pos_error, rotation_y_vector)


            obs.extend(np.concatenate(
                [
                    # Base features
                    pos, quat, vel, ang_vel, 
                    # Positional features
                    normalized_pos,
                    pos_error,
                    np.array([distance_to_target, squared_distance_to_target], dtype=np.float32),
                    derivative_error,
                    second_derivative_error,
                    self.integral_pos_error[i],
                    normalized_prev_ctrl,
                    vel_over_dist, np.array([vel_along_error], dtype=np.float32), vel_norm,
                    # Rotational features
                    drone_up_vector,
                    quat_error,
                    quat_derivative_error,
                    self.integral_quat_error[i],
                    np.array([vel_rotation_x, vel_rotation_y], dtype=np.float32),
                    np.array([pos_error_rotation_x, pos_error_rotation_y], dtype=np.float32)
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
