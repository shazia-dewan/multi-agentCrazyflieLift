import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional
from scipy.spatial.transform import Rotation as R


BASE_HOVER_THRUST = 0.26487
DRONE_TILT_EPISLON = np.deg2rad(10)
TRAINING_POS_RANGE = 2.0
TRAINING_QUAT_RANGE = np.pi / 36

class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str, 
        num_drones: int = 1, 
        target_pos: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
        max_steps: int = 1500,
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

        # MuJoCo scene
        self.mujoco_scene = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.mujoco_scene)

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
        self.observation_names = []

        # Define the observation space with n features based on how many we assign in _get_obs()
        obs_high = np.inf * np.ones(115 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Drone action space (see aicraft axes: https://en.wikipedia.org/wiki/Aircraft_principal_axes)
        # thrust + roll + pitch + yaw = 4 per drone
        act_high = np.tile([0.35, 1, 1, 1], self.num_drones).astype(np.float32)
        act_low = np.tile([0.0, -1, -1, -1], self.num_drones).astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Viewer
        self.viewer = None

    def add_feature_names(self, name_list: list[str], values: np.ndarray) -> np.ndarray:
        """
        Add a feature from observations to the tracked list of feature names

        Parameters
        ----------
        name_list : list[str]
            The list of feature names
        values : np.ndarray
            The list of feature values
        """
        # Only build the name index once
        if self.timestep == 1:
            self.observation_names.extend(name_list)
        return values


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

        mujoco.mj_resetData(self.mujoco_scene, self.data)

        # Position tracking for previous drone(s) feature
        self.prev_pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.derivative_prev_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)

        # Velocity tracking for previous drone(s) feature
        self.prev_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.derivative_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        # Angular velocities
        self.prev_ang_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.derivative_ang_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_ang_vel = np.zeros((self.num_drones, 3), dtype=np.float32)

        # Rotation tracking for previous drone(s) feature
        self.prev_quat_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_quat_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_quat = np.zeros((self.num_drones, 4), dtype=np.float32)
        self.integral_quat = np.zeros((self.num_drones, 4), dtype=np.float32)
        self.prev_direction_to_target_in_body = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_direction_to_target_in_body = np.zeros((self.num_drones, 3), dtype=np.float32)

        # Control tracking for previous drone(s) feature
        self.prev_control = np.zeros((self.num_drones, self.ctrl_per_drone), dtype=np.float32)
        self.derivative_control = np.zeros((self.num_drones, self.ctrl_per_drone), dtype=np.float32)
        self.integral_control = np.zeros((self.num_drones, self.ctrl_per_drone), dtype=np.float32)
        self.prev_world_thrust = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_world_thrust = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_thrust_alignment = np.zeros(self.num_drones, dtype=np.float32)
        self.integral_thrust_alignment = np.zeros(self.num_drones, dtype=np.float32)
        
        # If using multiple drones, offset them along X axis
        drone_spacing_x = 0.4
        drone_offsets_x = np.linspace(-(self.num_drones - 1) / 2, (self.num_drones - 1) / 2, self.num_drones) * drone_spacing_x
        
        # Start drone at some random pos [X, Y, Z] and quat [w, x, y, z]
        if self.random_initialization:
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                # Random position
                start_x = self.np_random.uniform(-TRAINING_POS_RANGE / 5, TRAINING_POS_RANGE / 5)
                start_y = self.np_random.uniform(-TRAINING_POS_RANGE / 5, TRAINING_POS_RANGE / 5)
                start_z = self.np_random.uniform(0.1, TRAINING_POS_RANGE)

                self.data.qpos[base_qpos + 0] = 0.0 + drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = start_y
                self.data.qpos[base_qpos + 2] = start_z

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]

                # Random rotation axis with small rotation angle
                # axis = self.np_random.normal(size=3)
                axis = np.array([1.0, 0.0, 0.0])
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

                # # Test starting rotation in render / eval run
                # axis = np.array([1.0, 0.0, 0.0])
                # axis /= np.linalg.norm(axis)

                # angle = self.np_random.uniform(-TRAINING_QUAT_RANGE, TRAINING_QUAT_RANGE)

                # w = np.cos(angle / 2.0)
                # x, y, z = axis * np.sin(angle / 2.0)

                # quat = np.array([w, x, y, z], dtype=np.float64)
                # self.data.qpos[base_qpos + 3: base_qpos + 7] = quat

        for i in range(self.num_drones):
            # Start drone(s) with 0 velocity
            base_qvel = i * self.qvel_per_drone
            self.data.qvel[base_qvel: base_qvel + self.qvel_per_drone] = 0.0
        
        self.timestep = 0
        # For each drone, track how many timesteps we have been at the same pos (detect hover or stall)
        self.same_pos_steps = np.zeros((self.num_drones, 1), dtype=np.float32)

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
        target_geom_id = mujoco.mj_name2id(self.mujoco_scene, mujoco.mjtObj.mjOBJ_GEOM, "target")
        self.mujoco_scene.geom_pos[target_geom_id] = self.target_pos

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

            ctrl[base_ctrl + 0] = thrust
            ctrl[base_ctrl + 1] = roll
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

            ##############################################
            # Termination and Truncation
            ##############################################

            # If we have been at the same pos (within eps) for n timesteps, terminate (hover or stall)
            if np.linalg.norm(pos - self.prev_pos[i]) < 1e-6:
                self.same_pos_steps[i] += 1
                if self.same_pos_steps[i] > 400:
                    terminated = True
            else:
                self.same_pos_steps[i] = 0

            # Detect if we have exceeded max steps (truncate)
            truncated = self.timestep >= self.max_steps

            ##############################################
            # Next-state update
            ##############################################

            # Track how the action changes between steps (1-step derivative)
            self.derivative_control[i] = np.array([
                (2 * (ctrl[base_ctrl + 0] / 0.35) - 1) - (2 * (self.prev_control[i][0] / 0.35) - 1),
                ctrl[base_ctrl + 1] - self.prev_control[i][1],
                ctrl[base_ctrl + 2] - self.prev_control[i][2],
                ctrl[base_ctrl + 3] - self.prev_control[i][3]
            ], dtype=np.float32)

            # Update prev pos and action for next timestep (to use in observations)
            self.prev_pos[i] = pos.copy()
            self.prev_control[i] = ctrl[base_ctrl : base_ctrl + self.ctrl_per_drone].copy()

            if self.debug:
                print(f"Step {self.timestep} - Ctrl: {[f'{c:.4f}' for c in ctrl]}, Position: {[f'{p:.2f}' for p in pos]}, Reward: {reward:.4f}")
            
        # Apply control and step
        self.data.ctrl[:] = ctrl
        self.timestep += 1
        mujoco.mj_step(self.mujoco_scene, self.data)
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

            # ------------------------------------------------
            # Base Features in Observation Space:
            # ------------------------------------------------

            # position[x, y, z]
            pos = self.data.qpos[base_qpos: base_qpos + 3]

            # quaternion[qw, qx, qy, qz,]
            quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]

            # velocity[vx, vy, vz]
            vel = self.data.qvel[base_qvel: base_qvel + 3]

            # angular velocity[wx, wy, wz]
            ang_vel = self.data.qvel[base_qvel + 3 : base_qvel + 6]

            obs.extend(self.add_feature_names(["pos_x", "pos_y", "pos_z"], pos))
            obs.extend(self.add_feature_names(["quat_w", "quat_x", "quat_y", "quat_z"], quat))
            obs.extend(self.add_feature_names(["vel_x", "vel_y", "vel_z"], vel))
            obs.extend(self.add_feature_names(["ang_x", "ang_y", "ang_z"], ang_vel))


            # ------------------------------------------------------------------------
            # Extract some other useful features that can be used in multiple areas
            # ------------------------------------------------------------------------

            # Experimentally, doing pos - target instead of target - pos works better here as pos_error
            pos_error = pos - self.target_pos
            distance_to_target = np.linalg.norm(pos_error)

            # Direction to target in world coordinates (unit vector of pos_error)
            direction_to_target = (self.target_pos - pos) / (np.linalg.norm(distance_to_target) + 1e-9)

            # From quaternion -> rotation matrix (get orientation of drone in world coordinates)
            quat_xyzw = np.roll(quat, -1) # SciPy expects [x, y, z, w], so reorder
            rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

            # Direction to target in drone's local body coordinates
            direction_to_target_in_body = rotation_matrix.T @ direction_to_target

            # ---------------------------------------------------------------------
            # Positional engineered features
            # E.g. derivative and integral pos error to help model PID controller
            # ---------------------------------------------------------------------

            # normalize_pos stores [x, y, z] normalized by the l2 (euclidean) norm of target
            normalized_pos = 2 * pos / (np.linalg.norm(self.target_pos) + 1e-9) - 1

            squared_distance_to_target = distance_to_target ** 2

            # 1st and 2nd derivative position error over one step ~ velocity and acceleration of error
            derivative_error = pos_error - self.prev_pos_error[i]
            second_derivative_error = derivative_error - self.derivative_prev_pos_error[i]

            # Integral position error: exponential moving average to avoid unbounded growth
            self.integral_pos_error[i] = 0.95 * self.integral_pos_error[i] + 0.05 * pos_error

            # Update prev pos and derivative error
            self.prev_pos_error[i] = pos_error.copy()
            self.derivative_prev_pos_error[i] = derivative_error.copy()

            obs.extend(self.add_feature_names(["pos_norm_x", "pos_norm_y", "pos_norm_z"], normalized_pos))
            obs.extend(self.add_feature_names(["pos_err_x", "pos_err_y", "pos_err_z"], pos_error))
            obs.extend(self.add_feature_names(["dist_to_target", "squared_dist_to_target"], np.array([distance_to_target, squared_distance_to_target], dtype=np.float32)))
            obs.extend(self.add_feature_names(["derivative_pos_err_x", "derivative_pos_err_y", "derivative_pos_err_z"], derivative_error))
            obs.extend(self.add_feature_names(["sec_derivative_pos_err_x", "sec_derivative_pos_err_y", "sec_derivative_pos_err_z"], second_derivative_error))
            obs.extend(self.add_feature_names(["integral_pos_err_x", "integral_pos_err_y", "integral_pos_err_z"], self.integral_pos_error[i]))


            # ---------------------------------------------------------------------
            # Velocity (base and angular) engineered features
            # ---------------------------------------------------------------------

            # Projection of velocity along the pos error vector
            # scalar value: moving toward (+) or away (-) from the target, and how fast relative to the gap
            vel_along_error = np.dot(vel, pos_error) / (distance_to_target + 1e-9)

            # Velocity over distance
            vel_over_dist = vel / (distance_to_target + 1e-9)

            vel_norm = vel / (np.linalg.norm(vel) + 1e-9)
            vel_derivative = vel - self.prev_vel[i]
            self.integral_vel[i] = 0.95 * self.integral_vel[i] + 0.05 * vel
            self.prev_vel[i] = vel.copy()

            ang_vel_norm = ang_vel / (np.linalg.norm(ang_vel) + 1e-9)
            ang_vel_derivative = ang_vel - self.prev_ang_vel[i]
            self.integral_ang_vel[i] = 0.95 * self.integral_ang_vel[i] + 0.05 * ang_vel
            self.prev_ang_vel[i] = ang_vel.copy()

            # Skid component (lateral velocity): Velocity - Velocity along target direction
            skid_velocity = vel - np.dot(vel, direction_to_target) * direction_to_target

            # Approximate the orbital angular momentum: indicator of circling (orbitting) about the target
            angular_momentum_proxy = np.cross(pos_error, vel)

            obs.extend(self.add_feature_names(["vel_along_err"], np.array([vel_along_error], dtype=np.float32)))
            obs.extend(self.add_feature_names(["vel_over_dist_x", "vel_over_dist_y", "vel_over_dist_z"], vel_over_dist))
            obs.extend(self.add_feature_names(["vel_norm_x", "vel_norm_y", "vel_norm_z"], vel_norm))
            obs.extend(self.add_feature_names(["derivative_vel_x", "derivative_vel_y", "derivative_vel_z"], vel_derivative))
            obs.extend(self.add_feature_names(["integral_vel_x", "integral_vel_y", "integral_vel_z"], self.integral_vel[i]))
            obs.extend(self.add_feature_names(["ang_vel_norm_x", "ang_vel_norm_y", "ang_vel_norm_z"], ang_vel_norm))
            obs.extend(self.add_feature_names(["derivative_ang_vel_x", "derivative_ang_vel_y", "derivative_ang_vel_z"], ang_vel_derivative))
            obs.extend(self.add_feature_names(["integral_ang_vel_x", "integral_ang_vel_y", "integral_ang_vel_z"], self.integral_ang_vel[i]))
            obs.extend(self.add_feature_names(["skid_vel_x", "skid_vel_y", "skid_vel_z"], skid_velocity))
            obs.extend(self.add_feature_names(["angular_momentum_x", "angular_momentum_y", "angular_momentum_z"], angular_momentum_proxy))


            # ---------------------------------------------------------------------
            # Rotational engineered features
            # ---------------------------------------------------------------------

            # drone_up_vector is where the drones local body Z axis is pointing in world coordinates
            drone_up_vector = np.array([
                rotation_matrix[0][2],
                rotation_matrix[1][2],
                rotation_matrix[2][2],
            ])

            # quat_error: the difference between the drones Z-up axis and the world Z-up axis (how far from level)
            quat_error = drone_up_vector - np.array([0, 0, 1])
            quat_derivative_error = quat_error - self.prev_quat_error[i]
            self.integral_quat_error[i] = 0.95 * self.integral_quat_error[i] + 0.05 * quat_error
            self.prev_quat_error[i] = quat_error.copy()
            
            # Drone's body x and y axes in world frame
            rotation_x_vector = rotation_matrix[:, 0]
            rotation_y_vector = rotation_matrix[:, 1]

            # Project velocity onto drone axes, how much is drone moving on roll/pitch axes
            vel_rotation_x = np.dot(vel, rotation_x_vector)
            vel_rotation_y   = np.dot(vel, rotation_y_vector)

            # Project pos_error onto drone axes, how is pos error moving on roll/pitch axes
            pos_error_rotation_x = np.dot(pos_error, rotation_x_vector)
            pos_error_rotation_y   = np.dot(pos_error, rotation_y_vector)

            # Quat derivative and integral (using quat directly, not quat error)
            quat_derivative = quat - self.prev_quat[i]
            self.integral_quat[i] = 0.95 * self.integral_quat[i] + 0.05 * quat
            self.prev_quat[i] = quat.copy()

            derivative_direction_to_target_in_body = direction_to_target_in_body - self.prev_direction_to_target_in_body[i]
            self.integral_direction_to_target_in_body[i] = 0.95 * self.integral_direction_to_target_in_body[i] + 0.05 * direction_to_target_in_body
            self.prev_direction_to_target_in_body[i] = direction_to_target_in_body

            obs.extend(self.add_feature_names(["drone_up_x", "drone_up_y", "drone_up_z"], drone_up_vector))
            obs.extend(self.add_feature_names(["quat_err_x", "quat_err_y", "quat_err_z"], quat_error))
            obs.extend(self.add_feature_names(["derivative_quat_err_x", "derivative_quat_err_y", "derivative_quat_err_z"], quat_derivative_error))
            obs.extend(self.add_feature_names(["integral_quat_err_x", "integral_quat_err_y", "integral_quat_err_z"], self.integral_quat_error[i]))
            obs.extend(self.add_feature_names(["vel_rotation_x", "vel_rotation_y"], np.array([vel_rotation_x, vel_rotation_y], dtype=np.float32)))
            obs.extend(self.add_feature_names(["pos_error_rotation_x", "pos_error_rotation_y"], np.array([pos_error_rotation_x, pos_error_rotation_y], dtype=np.float32)))
            obs.extend(self.add_feature_names(["derivative_quat_w", "derivative_quat_x", "derivative_quat_y", "derivative_quat_z"], quat_derivative))
            obs.extend(self.add_feature_names(["integral_quat_w", "integral_quat_x", "integral_quat_y", "integral_quat_z"], self.integral_quat[i]))
            obs.extend(self.add_feature_names(["dir_to_target_body_x", "dir_to_target_body_y", "dir_to_target_body_z"], direction_to_target_in_body))
            obs.extend(self.add_feature_names(["derivative_dir_to_target_body_x", "derivative_dir_to_target_body_y", "derivative_dir_to_target_body_z"], direction_to_target_in_body))
            obs.extend(self.add_feature_names(["integral_dir_to_target_body_x", "integral_dir_to_target_body_y", "integral_dir_to_target_body_z"], direction_to_target_in_body))


            # ---------------------------------------------------------------------
            # Control (action) engineered features
            # Derivative control updated in step()
            # ---------------------------------------------------------------------

            # normalized_prev_ctrl stores normalized [-1, 1] previous action controls
            # Roll/Pitch/Yaw each operate by adjusting the relative motor speed for 2/4 of the drone motors
            # The strength relative to thrustt: More thrust --> more roll/pitch/yaw, so scale roll/pitch/yaw by thrust
            normalized_prev_ctrl = np.array([
                2 * (self.prev_control[i][0] / 0.35) - 1,
                self.prev_control[i][1] * (self.prev_control[i][0] / 0.35),
                self.prev_control[i][2] * (self.prev_control[i][0] / 0.35),
                self.prev_control[i][3] * (self.prev_control[i][0] / 0.35)
            ], dtype=np.float32)

            # Update action integral
            self.integral_control[i] = 0.95 * self.integral_control[i] + 0.05 * normalized_prev_ctrl

            # Thrust in world frame (body thrust is just [0, 0, thrust], transform to world thrust based on drone rotation)
            world_thrust = rotation_matrix @ np.array([0, 0, self.prev_control[i][0]])
            world_thrust = world_thrust / (np.linalg.norm(world_thrust) + 1e-9)
            derivative_world_thrust = world_thrust - self.prev_world_thrust[i]
            self.integral_world_thrust[i] = 0.95 * self.integral_world_thrust[i] + 0.05 * world_thrust
            self.prev_world_thrust[i] = world_thrust
            
            # Alignment in [-1, 1] of thrust towards the target
            thrust_alignment = np.dot(world_thrust, direction_to_target)
            derivative_thrust_alignment = thrust_alignment - self.prev_thrust_alignment[i]
            self.integral_thrust_alignment[i] = 0.95 * self.integral_thrust_alignment[i] + thrust_alignment
            self.prev_thrust_alignment[i] = thrust_alignment

            obs.extend(self.add_feature_names(["prev_ctrl_thrust", "prev_ctrl_roll", "prev_ctrl_pitch", "prev_ctrl_yaw"], normalized_prev_ctrl))
            obs.extend(self.add_feature_names(["derivative_ctrl_thrust", "derivative_ctrl_roll", "derivative_ctrl_pitch", "derivative_ctrl_yaw"], self.derivative_control[i]))
            obs.extend(self.add_feature_names(["integral_ctrl_thrust", "integral_ctrl_roll", "integral_ctrl_pitch", "integral_ctrl_yaw"], self.integral_control[i]))
            obs.extend(self.add_feature_names(["thrust_x", "thrust_y", "thrust_z"], world_thrust))
            obs.extend(self.add_feature_names(["derivative_thrust_x", "derivative_thrust_y", "derivative_thrust_z"], derivative_world_thrust))
            obs.extend(self.add_feature_names(["integral_thrust_x", "integral_thrust_y", "integral_thrust_z"], self.integral_world_thrust[i]))
            obs.extend(self.add_feature_names(
                ["thrust_alignment", "derivative_thrust_alignment", "integral_thrust_alignment"], 
                np.array([thrust_alignment, derivative_thrust_alignment, self.integral_thrust_alignment[i]], dtype=np.float32)
            ))


        return np.array(obs, dtype=np.float32)


    def render(self) -> None:
        """
        Render the model in MuJoCo
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.mujoco_scene, self.data)
        self.viewer.sync()


    def close(self) -> None:
        """
        Close the MuJoCo viewer
        """
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
