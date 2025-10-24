import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional
from scipy.spatial.transform import Rotation as R

from reward_system import RewardTracker

BASE_HOVER_THRUST = 0.26487

# Initial state ranges for random initialization during training (used for standard deviations)
TRAINING_POS_RANGE = 2.0
TRAINING_QUAT_RANGE = np.pi / 8
TRAINING_VEL_RANGE = 0.2
TRAINING_ANG_VEL_RANGE = 0.1

OUT_OF_BOUNDS_RANGE = 2.0

TERMINATION_PENALTY = -1.0

DRONE_PROXIMITY_THRESHOLD = 0.25
DRONE_COLLISION_THRESHOLD = 0.1

class CrazyflieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str, 
        num_drones: int = 1, 
        target_pos: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
        max_steps: int = 1500,
        random_initialization: bool = True,
        manual_override = False,
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
        manual_override : bool
            If true, uses sampled action directly in step()
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
        self.target_pos = target_pos
        self.max_steps = max_steps
        self.random_initialization = random_initialization
        self.manual_override = manual_override
        self.debug = debug

        # Index of observations for logging afterwards (e.g. which observations were important)
        self.observation_names = []

        # Define the observation space with n features based on how many we assign in _get_obs()
        obs_high = np.inf * np.ones(146 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Drone action space (see aicraft axes: https://en.wikipedia.org/wiki/Aircraft_principal_axes)
        # thrust + roll + pitch + yaw = 4 per drone
        act_high = np.tile([0.35, 1, 1, 1], self.num_drones).astype(np.float32)
        act_low = np.tile([0.0, -1, -1, -1], self.num_drones).astype(np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Number of MuJoCo physics steps per call to env step()
        self.frame_skip = 10

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
        self.state = np.zeros((self.num_drones, self.qpos_per_drone + self.qvel_per_drone), dtype=np.float32)

        self.starting_pos = np.zeros((self.num_drones, 3), dtype=np.float32)

        # Position tracking for previous drone(s) feature
        self.prev_pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.derivative_prev_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_pos_error = np.zeros((self.num_drones, 3), dtype=np.float32)

        # Velocity tracking for previous drone(s) feature
        self.prev_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        # Angular velocities
        self.prev_ang_vel = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_ang_vel = np.zeros((self.num_drones, 3), dtype=np.float32)

        # Rotation tracking for previous drone(s) feature
        self.prev_direction_to_target_in_body = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.integral_direction_to_target_in_body = np.zeros((self.num_drones, 3), dtype=np.float32)

        self.prev_rotation_matrix = np.tile(np.eye(3, dtype=np.float32), (self.num_drones, 1, 1))
        self.integral_rotation_matrix = np.tile(np.eye(3, dtype=np.float32), (self.num_drones, 1, 1))

        # Control tracking for previous drone(s) feature
        self.prev_control = np.zeros((self.num_drones, self.ctrl_per_drone), dtype=np.float32)

        # Store last n control actions (action history) to compensate for thrust motor lag
        self.action_history_len = 16
        self.action_history = np.zeros((self.num_drones, self.action_history_len, self.ctrl_per_drone), dtype=np.float32)
        
        # If using multiple drones, offset them along X axis to avoid starting collisions
        drone_spacing_x = 0.4
        drone_offsets_x = np.linspace(-(self.num_drones - 1) / 2, (self.num_drones - 1) / 2, self.num_drones) * drone_spacing_x

        if self.random_initialization:
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                base_qvel = i * self.qvel_per_drone
                
                # Random starting position around the target based on normal distribution
                std_pos = TRAINING_POS_RANGE / 2

                start_x = self.np_random.normal(loc=self.target_pos[0], scale=std_pos)
                start_y = self.np_random.normal(loc=self.target_pos[1], scale=std_pos)
                start_z = max(self.np_random.normal(loc=self.target_pos[2], scale=std_pos), 0.1)

                self.data.qpos[base_qpos + 0] = start_x + drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = start_y
                self.data.qpos[base_qpos + 2] = start_z

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]


                # Random rotation axis with small rotation angle based on normal distribution
                axis = self.np_random.normal(size=3)
                axis /= np.linalg.norm(axis)

                std_angle = TRAINING_QUAT_RANGE / 2.0
                angle = self.np_random.normal(loc=0.0, scale=std_angle)

                w = np.cos(angle)
                x, y, z = axis * np.sin(angle)

                quat = np.array([w, x, y, z], dtype=np.float64)
                self.data.qpos[base_qpos + 3: base_qpos + 7] = quat

                # Random starting linear velocity based on normal distribution
                std_vel = TRAINING_VEL_RANGE / 2
                vel_y = self.np_random.normal(loc=0.0, scale=std_vel)
                vel_z = self.np_random.normal(loc=0.0, scale=std_vel)
                vel_x = self.np_random.normal(loc=0.0, scale=std_vel)
                self.data.qvel[base_qvel + 0: base_qvel + 3] = np.array([vel_x, vel_y, vel_z], dtype=np.float32)

                # Random starting angular velocity based on normal distribution
                std_ang_vel = TRAINING_ANG_VEL_RANGE / 2
                ang_vel_x = self.np_random.normal(loc=0.0, scale=std_ang_vel)
                ang_vel_y = self.np_random.normal(loc=0.0, scale=std_ang_vel)
                ang_vel_z = self.np_random.normal(loc=0.0, scale=std_ang_vel)
                self.data.qvel[base_qvel + 3: base_qvel + 6] = np.array([ang_vel_x, ang_vel_y, ang_vel_z], dtype=np.float32)
        else:
            # In non-training (eval) run, start drone(s) on the floor
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                self.data.qpos[base_qpos + 0] = drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = 0
                self.data.qpos[base_qpos + 2] = 0.03

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]

        # Store initial state and starting pos
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone

            self.state[i] = np.concatenate([
                self.data.qpos[base_qpos : base_qpos + 3],
                self.data.qpos[base_qpos + 3 : base_qpos + 7],
                self.data.qvel[base_qvel : base_qvel + 3],
                self.data.qvel[base_qvel + 3 : base_qvel + 6]
            ])

            self.starting_pos[i] = self.data.qpos[base_qpos : base_qpos + 3].copy()
            self.initial_distance_to_target = np.linalg.norm(self.target_pos - self.starting_pos[i])

            # Start at neutral control
            self.prev_control[i] = np.array([BASE_HOVER_THRUST, 0.0, 0.0, 0.0], dtype=np.float32)

        self.timestep = 0

        self.reward_tracker = RewardTracker()

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

        # Termination values
        reward = 0.0
        terminated = False
        truncated = False

        # Fill control array
        ctrl = np.zeros(self.num_drones * self.ctrl_per_drone, dtype=np.float32)

        # Extract all drone positions for later proximity calculations
        drone_positions = np.zeros((self.num_drones, 3), dtype=np.float32)
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            pos = self.data.qpos[base_qpos: base_qpos + 3]
            drone_positions[i] = pos

        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone
            base_ctrl = i * self.ctrl_per_drone

            # Current state
            pos = self.data.qpos[base_qpos: base_qpos + 3]
            quat = self.data.qpos[base_qpos + 3: base_qpos + 7]
            vel = self.data.qvel[base_qvel: base_qvel + 3]
            ang_vel = self.data.qvel[base_qvel + 3: base_qvel + 6]

            self.state[i] = np.concatenate([pos, quat, vel, ang_vel])

            # Set ctrl to clipped action (within action space)
            ctrl_action = np.clip(
                action[base_ctrl : base_ctrl + 4],
                self.action_space.low[base_ctrl : base_ctrl + 4],
                self.action_space.high[base_ctrl : base_ctrl + 4],
            )
            ctrl[base_ctrl : base_ctrl + 4] = ctrl_action
            
            # Update action history: shift left and append current roll and pitch
            self.action_history[i, :-1] = self.action_history[i, 1:]
            self.action_history[i, -1] = ctrl_action

            ##############################################
            # Rewards, Termination, and Truncation
            ##############################################

            # Surival bonus
            self.reward_tracker.update("survival_bonus", -1 * TERMINATION_PENALTY / self.max_steps)

            pos_error = self.target_pos - pos
            distance_to_target = np.linalg.norm(pos_error)

            # Reward/penalty for moving towards/away from target
            distance_to_target_old = np.linalg.norm(self.target_pos - self.prev_pos[i])
            distance_improvement = distance_to_target_old - distance_to_target
            self.reward_tracker.update("distance_improvement", distance_improvement)

            # Reward/peanlty for distance to target (normalized by initial distance to target)
            proximity_bonus = 0.01 * (1 - np.tanh(distance_to_target))
            self.reward_tracker.update("proximity_bonus", proximity_bonus)
            
            reward = self.reward_tracker.step_total()

            # Crash, terminate with penalty
            if pos[2] < 0.05 and self.timestep > 100:
                self.reward_tracker.update("crash", TERMINATION_PENALTY)
                reward = self.reward_tracker.step_total()
                terminated = True

            # Out of bounds, terminate with penalty (if we go much further from the target compared to starting pos)
            if distance_to_target > self.initial_distance_to_target + OUT_OF_BOUNDS_RANGE:
                self.reward_tracker.update("out_of_bounds", TERMINATION_PENALTY)
                reward = self.reward_tracker.step_total()
                terminated = True
            
            # MARL checks: penalize proximity to other drone(s), terminate if we collide
            min_distance_to_other = np.inf
            for j in range(self.num_drones):
                if j == i:
                    continue
                distance_to_drone = np.linalg.norm(drone_positions[i] - drone_positions[j])
                min_distance_to_other = min(min_distance_to_other, distance_to_drone)

                # Too close
                if distance_to_drone < DRONE_PROXIMITY_THRESHOLD:
                    penalty = -0.01 * (1.0 - distance_to_drone / DRONE_PROXIMITY_THRESHOLD)
                    self.reward_tracker.update(f"proximity_penalty_{j}", penalty)

                # Collision
                if distance_to_drone < DRONE_COLLISION_THRESHOLD:
                    self.reward_tracker.update(f"collision_{j}", TERMINATION_PENALTY)
                    terminated = True

            # Detect if we have exceeded max steps (truncate)
            truncated = self.timestep >= self.max_steps

            # Update prev pos and prev action
            self.prev_pos[i] = pos.copy()
            self.prev_control[i] = ctrl_action.copy()

            if self.debug:
                print(f"Step {self.timestep} - Ctrl: {[f'{c:.4f}' for c in ctrl]}, Position: {[f'{p:.2f}' for p in pos]}, Reward: {reward:.4f}")
        
        # Apply control
        self.data.ctrl[:] = ctrl
        self.timestep += 1

        # Step for several actual physics steps in MuJoCo
        # Using 2ms timestep and RK4 integrator, so 10 steps = 20ms (50Hz) (see cf2.xml = self.mujoco_scene.opt.timestep)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.mujoco_scene, self.data)
            
        obs = self._get_obs()

        info = {
            "terminated": terminated,
            "truncated": truncated
        }
        return obs, reward, terminated, truncated, info


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
            # Not used directly, we use relative pos
            pos = self.data.qpos[base_qpos: base_qpos + 3]

            # quaternion[qw, qx, qy, qz]
            # Not using quat (can be ambiguous to nn), using rotation_matrix instead (orientation in world coordinates)
            quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]
            quat_xyzw = np.roll(quat, -1) # SciPy expects [x, y, z, w], so reorder
            rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

            # velocity[vx, vy, vz] in body frame
            vel = rotation_matrix.T @ self.data.qvel[base_qvel: base_qvel + 3]
            obs.extend(self.add_feature_names(["vel_x", "vel_y", "vel_z"], vel))

            # angular velocity[wx, wy, wz]
            ang_vel = self.data.qvel[base_qvel + 3 : base_qvel + 6]
            obs.extend(self.add_feature_names(["ang_x", "ang_y", "ang_z"], ang_vel))

            # ---------------------------------------------------------------------
            # Positional engineered features
            # ---------------------------------------------------------------------

            # Relative pos to target in body coordinates, normalized by initial distance (initial pos error magnitude)
            pos_error = (self.target_pos - pos) / self.initial_distance_to_target
            pos_error = rotation_matrix.T @ pos_error
            obs.extend(self.add_feature_names(["pos_err_x", "pos_err_y", "pos_err_z"], pos_error))

            # 1st and 2nd derivative position error over one step ~ velocity and acceleration of error
            derivative_error = pos_error - self.prev_pos_error[i]
            obs.extend(self.add_feature_names(["derivative_pos_err_x", "derivative_pos_err_y", "derivative_pos_err_z"], derivative_error))
            second_derivative_error = derivative_error - self.derivative_prev_pos_error[i]
            obs.extend(self.add_feature_names(["sec_derivative_pos_err_x", "sec_derivative_pos_err_y", "sec_derivative_pos_err_z"], second_derivative_error))

            # Integral position error: exponential moving average to avoid unbounded growth
            self.integral_pos_error[i] = 0.95 * self.integral_pos_error[i] + 0.05 * pos_error
            obs.extend(self.add_feature_names(["integral_pos_err_x", "integral_pos_err_y", "integral_pos_err_z"], self.integral_pos_error[i]))

            # Update prev pos and derivative error
            self.prev_pos_error[i] = pos_error.copy()
            self.derivative_prev_pos_error[i] = derivative_error.copy()

            # Some positional attributes used below
            distance_to_target = np.linalg.norm(self.target_pos - pos)
            direction_to_target = pos_error / (np.linalg.norm(pos_error) + 1e-9)

            # ---------------------------------------------------------------------
            # Velocity (base and angular) engineered features
            # ---------------------------------------------------------------------

            # Projection of velocity along the pos error vector
            # scalar value: moving toward (+) or away (-) from the target, and how fast relative to the gap
            vel_along_error = np.dot(vel, pos_error) / (distance_to_target + 1e-9)
            obs.extend(self.add_feature_names(["vel_along_err"], np.array([vel_along_error], dtype=np.float32)))

            # Velocity over distance: relate velocity to target distance
            vel_over_dist = vel / (distance_to_target + 1e-9)
            obs.extend(self.add_feature_names(["vel_over_dist_x", "vel_over_dist_y", "vel_over_dist_z"], vel_over_dist))

            ang_vel_over_dist = ang_vel / (distance_to_target + 1e-9)
            obs.extend(self.add_feature_names(["ang_vel_over_dist_x", "ang_vel_over_dist_y", "ang_vel_over_dist_z"], ang_vel_over_dist))

            vel_norm = vel / (np.linalg.norm(vel) + 1e-9)
            obs.extend(self.add_feature_names(["vel_norm_x", "vel_norm_y", "vel_norm_z"], vel_norm))

            vel_derivative = vel - self.prev_vel[i]
            obs.extend(self.add_feature_names(["derivative_vel_x", "derivative_vel_y", "derivative_vel_z"], vel_derivative))
            self.prev_vel[i] = vel.copy()

            ang_vel_norm = ang_vel / (np.linalg.norm(ang_vel) + 1e-9)
            obs.extend(self.add_feature_names(["ang_vel_norm_x", "ang_vel_norm_y", "ang_vel_norm_z"], ang_vel_norm))

            ang_vel_derivative = ang_vel - self.prev_ang_vel[i]
            obs.extend(self.add_feature_names(["derivative_ang_vel_x", "derivative_ang_vel_y", "derivative_ang_vel_z"], ang_vel_derivative))
            self.prev_ang_vel[i] = ang_vel.copy()

            # Skid component (lateral velocity): Velocity - Velocity along target direction
            skid_velocity = vel - np.dot(vel, direction_to_target) * direction_to_target
            obs.extend(self.add_feature_names(["skid_vel_x", "skid_vel_y", "skid_vel_z"], skid_velocity))

            # Approximate the orbital angular momentum: indicator of circling (orbitting) about the target
            # Normally, orbital ang. momentum = pos x linear momentum, we are doing pos_error x velocity
            angular_momentum_proxy = np.cross(pos_error, vel)
            obs.extend(self.add_feature_names(["angular_momentum_x", "angular_momentum_y", "angular_momentum_z"], angular_momentum_proxy))


            # ---------------------------------------------------------------------
            # Rotational engineered features
            # ---------------------------------------------------------------------

            # Rotation matrix from quaternion
            obs.extend(self.add_feature_names(["rot_matrix_x0", "rot_matrix_x1", "rot_matrix_x2"], rotation_matrix[0]))
            obs.extend(self.add_feature_names(["rot_matrix_y0", "rot_matrix_y1", "rot_matrix_y2"], rotation_matrix[1]))
            obs.extend(self.add_feature_names(["rot_matrix_z0", "rot_matrix_z1", "rot_matrix_z2"], rotation_matrix[2]))

            derivative_rotation_matrix = rotation_matrix - self.prev_rotation_matrix[i]
            obs.extend(self.add_feature_names(["derivative_rot_matrix_x0", "derivative_rot_matrix_x1", "derivative_rot_matrix_x2"], derivative_rotation_matrix[0]))
            obs.extend(self.add_feature_names(["derivative_rot_matrix_y0", "derivative_rot_matrix_y1", "derivative_rot_matrix_y2"], derivative_rotation_matrix[1]))
            obs.extend(self.add_feature_names(["derivative_rot_matrix_z0", "derivative_rot_matrix_z1", "derivative_rot_matrix_z2"], derivative_rotation_matrix[2]))

            self.integral_rotation_matrix[i] = 0.95 * self.integral_rotation_matrix[i] + 0.05 * rotation_matrix
            obs.extend(self.add_feature_names(["integral_rot_matrix_x0", "integral_rot_matrix_x1", "integral_rot_matrix_x2"], self.integral_rotation_matrix[i][0]))
            obs.extend(self.add_feature_names(["integral_rot_matrix_y0", "integral_rot_matrix_y1", "integral_rot_matrix_y2"], self.integral_rotation_matrix[i][1]))
            obs.extend(self.add_feature_names(["integral_rot_matrix_z0", "integral_rot_matrix_z1", "integral_rot_matrix_z2"], self.integral_rotation_matrix[i][2]))
            self.prev_rotation_matrix[i] = rotation_matrix.copy()
            
            # Direction to target in drone's local body coordinates
            obs.extend(self.add_feature_names(["dir_to_target_body_x", "dir_to_target_body_y", "dir_to_target_body_z"], direction_to_target))
            
            derivative_direction_to_target_in_body = direction_to_target - self.prev_direction_to_target_in_body[i]
            obs.extend(self.add_feature_names(
                ["derivative_dir_to_target_body_x", "derivative_dir_to_target_body_y", "derivative_dir_to_target_body_z"],
                derivative_direction_to_target_in_body
            ))
            
            self.integral_direction_to_target_in_body[i] = 0.95 * self.integral_direction_to_target_in_body[i] + 0.05 * direction_to_target
            obs.extend(self.add_feature_names(
                ["integral_dir_to_target_body_x", "integral_dir_to_target_body_y", "integral_dir_to_target_body_z"], 
                self.integral_direction_to_target_in_body[i]
            ))
            self.prev_direction_to_target_in_body[i] = direction_to_target


            # ---------------------------------------------------------------------
            # Control (action) engineered features
            # ---------------------------------------------------------------------

            # Thrust in world frame (body thrust is just [0, 0, thrust], transform to world thrust based on drone rotation)
            world_thrust = rotation_matrix @ np.array([0, 0, self.prev_control[i][0]])
            world_thrust /= (np.linalg.norm(world_thrust) + 1e-9)
            obs.extend(self.add_feature_names(["world_thrust_x", "world_thrust_y", "world_thrust_z"], world_thrust))

            # Action History, last n actions
            flattened_action_history = self.action_history[i].flatten()
            obs.extend(self.add_feature_names(
                [f"ctrl_hist_{t}_{name}" for t in range(self.action_history_len) for name in ["thrust", "roll", "pitch", "yaw"]],
                flattened_action_history
            ))

        return np.array(obs, dtype=np.float32)
