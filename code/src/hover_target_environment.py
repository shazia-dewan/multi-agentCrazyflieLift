import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional
from scipy.spatial.transform import Rotation as R

from reward_system import RewardTracker

BASE_HOVER_THRUST = 0.26487
TRAINING_POS_RANGE = 2.0
TRAINING_QUAT_RANGE = np.pi / 36

STEPS_BEFORE_IDLE = 400

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
        obs_high = np.inf * np.ones(118 * self.num_drones, dtype=np.float32)
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
                start_x = self.np_random.uniform(-TRAINING_POS_RANGE / 2, TRAINING_POS_RANGE / 2)
                start_y = self.np_random.uniform(-TRAINING_POS_RANGE / 2, TRAINING_POS_RANGE / 2)
                start_z = self.np_random.uniform(0.1, TRAINING_POS_RANGE)

                self.data.qpos[base_qpos + 0] = 0.0 + drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = start_y
                self.data.qpos[base_qpos + 2] = start_z

                self.starting_pos[i] = np.array([self.data.qpos[base_qpos + 0], self.data.qpos[base_qpos + 1], self.data.qpos[base_qpos + 2]])
                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]


                # Random rotation axis with small rotation angle
                # axis = self.np_random.normal(size=3)
                # TESTING ON X AXIS ONLY FOR NOW
                axis = np.array([1.0, 0.0, 0.0])
                axis /= np.linalg.norm(axis)

                angle = self.np_random.uniform(-TRAINING_QUAT_RANGE, TRAINING_QUAT_RANGE)

                w = np.cos(angle / 2.0)
                x, y, z = axis * np.sin(angle / 2.0)

                quat = np.array([w, x, y, z], dtype=np.float64)
                self.data.qpos[base_qpos + 3: base_qpos + 7] = quat
        else:
            # Start drone(s) at consistent Z with X offset per drone
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                self.data.qpos[base_qpos + 0] = drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = 0
                self.data.qpos[base_qpos + 2] = 1.0

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]
        
        self.timestep = 0

        # Define reward system
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
            distance_to_target_old = np.linalg.norm(self.target_pos - self.prev_pos[i])
            pos_error = self.target_pos - pos
            distance_to_target = np.linalg.norm(pos_error)
            distance_improvement = distance_to_target_old - distance_to_target
            self.reward_tracker.update("distance_improvement", distance_improvement)

            above_the_ground = pos[2] > 0.05
            if above_the_ground:
                y_target_proximity_function = abs(np.tanh(pos_error[1]))

                # Get drone rotation matrix for drone-relative coordinates
                quat_xyzw = np.roll(quat, -1) # scipy uses [x, y, z, w]
                rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

                # Base reward for rolling, penalize large roll
                base_roll_reward = 0.001 / (abs(roll) + 1)

                # Reward/penalty for rolling towards/away from target on drone Y (roll-controlled) axis
                # Scaled by the proximity function because rolling towards target is more important when far away
                direction_to_target = pos_error / (distance_to_target + 1e-9)
                direction_to_target_body = rotation_matrix.T @ direction_to_target
                roll_towards_target = np.sign(roll) * direction_to_target_body[1] * base_roll_reward * y_target_proximity_function
                self.reward_tracker.update("roll_towards_target", roll_towards_target)

                # Reward/penalty for rolling away from/into velocity on drone Y (roll-controlled) axis to counteract velocity
                velocity_direction = vel / (np.linalg.norm(vel) + 1e-9)
                velocity_direction_body = rotation_matrix.T @ velocity_direction
                roll_away_from_velocity = -1 * np.sign(roll) * velocity_direction_body[1] * base_roll_reward
                self.reward_tracker.update("roll_away_from_velocity", roll_away_from_velocity)
            else:
                self.reward_tracker.update("roll_towards_target", 0.0)
                self.reward_tracker.update("roll_away_from_velocity", 0.0)

            reward = self.reward_tracker.step_total()

            ##############################################
            # Termination and Truncation
            ##############################################

            # Crash
            if not above_the_ground:
                self.reward_tracker.update("crash", -1)
                reward = self.reward_tracker.step_total()
                terminated = True

            # If we go much further from the target compared to starting pos, terminate with out of bounds penalty
            initial_distance_to_target = np.linalg.norm(self.target_pos - self.starting_pos[i])
            if distance_to_target > initial_distance_to_target + 1:
                self.reward_tracker.update("out_of_bounds", -1)
                reward = self.reward_tracker.step_total()
                terminated = True

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

            # Update prev pos and action for next timestep
            self.prev_pos[i] = pos.copy()
            self.prev_control[i] = ctrl[base_ctrl : base_ctrl + self.ctrl_per_drone].copy()

            if self.debug:
                print(f"Step {self.timestep} - Ctrl: {[f'{c:.4f}' for c in ctrl]}, Position: {[f'{p:.2f}' for p in pos]}, Reward: {reward:.4f}")
            
        # Apply control and step
        self.data.ctrl[:] = ctrl
        self.timestep += 1
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
            pos = self.data.qpos[base_qpos: base_qpos + 3]
            obs.extend(self.add_feature_names(["pos_x", "pos_y", "pos_z"], pos))

            # quaternion[qw, qx, qy, qz,]
            # Not using quat directly (can be ambiguous to nn), using rotation_matrix below
            quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]

            # velocity[vx, vy, vz]
            vel = self.data.qvel[base_qvel: base_qvel + 3]
            obs.extend(self.add_feature_names(["vel_x", "vel_y", "vel_z"], vel))

            # angular velocity[wx, wy, wz]
            ang_vel = self.data.qvel[base_qvel + 3 : base_qvel + 6]
            obs.extend(self.add_feature_names(["ang_x", "ang_y", "ang_z"], ang_vel))

            # ---------------------------------------------------------------------
            # Positional engineered features
            # E.g. derivative and integral pos error to help model PID controller
            # ---------------------------------------------------------------------

            # Relative pos
            pos_error = self.target_pos - pos
            obs.extend(self.add_feature_names(["pos_err_x", "pos_err_y", "pos_err_z"], pos_error))

            distance_to_target = np.linalg.norm(pos_error)
            squared_distance_to_target = distance_to_target ** 2
            obs.extend(self.add_feature_names(["dist_to_target", "squared_dist_to_target"], np.array([distance_to_target, squared_distance_to_target], dtype=np.float32)))

            direction_to_target = pos_error / (distance_to_target + 1e-9)

            # normalize_pos stores [x, y, z] normalized by the l2 (euclidean) norm of target
            normalized_pos = 2 * pos / (np.linalg.norm(self.target_pos) + 1e-9) - 1
            obs.extend(self.add_feature_names(["pos_norm_x", "pos_norm_y", "pos_norm_z"], normalized_pos))

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

            # From quaternion -> rotation matrix (get orientation of drone in world coordinates)
            quat_xyzw = np.roll(quat, -1) # SciPy expects [x, y, z, w], so reorder
            rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
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
            
            # Drone's body x and y axes in world frame
            rotation_x_vector = rotation_matrix[:, 0]
            rotation_x_vector /= (np.linalg.norm(rotation_x_vector) + 1e-9)
            rotation_y_vector = rotation_matrix[:, 1]
            rotation_y_vector /= (np.linalg.norm(rotation_y_vector) + 1e-9)

            # Project velocity onto drone axes, how much is drone moving on roll (X) and pitch (Y) axes
            vel_rotation_x = np.dot(vel, rotation_x_vector)
            vel_rotation_y = np.dot(vel, rotation_y_vector)
            obs.extend(self.add_feature_names(["vel_rotation_x", "vel_rotation_y"], np.array([vel_rotation_x, vel_rotation_y], dtype=np.float32)))

            # Project pos_error onto drone axes, how is pos error moving on roll/pitch axes
            pos_error_rotation_x = np.dot(pos_error, rotation_x_vector)
            pos_error_rotation_y   = np.dot(pos_error, rotation_y_vector)
            obs.extend(self.add_feature_names(["pos_error_rotation_x", "pos_error_rotation_y"], np.array([pos_error_rotation_x, pos_error_rotation_y], dtype=np.float32)))

            # Direction to target in drone's local body coordinates
            direction_to_target_in_body = rotation_matrix.T @ direction_to_target
            obs.extend(self.add_feature_names(["dir_to_target_body_x", "dir_to_target_body_y", "dir_to_target_body_z"], direction_to_target_in_body))
            
            derivative_direction_to_target_in_body = direction_to_target_in_body - self.prev_direction_to_target_in_body[i]
            obs.extend(self.add_feature_names(
                ["derivative_dir_to_target_body_x", "derivative_dir_to_target_body_y", "derivative_dir_to_target_body_z"],
                derivative_direction_to_target_in_body
            ))
            
            self.integral_direction_to_target_in_body[i] = 0.95 * self.integral_direction_to_target_in_body[i] + 0.05 * direction_to_target_in_body
            obs.extend(self.add_feature_names(
                ["integral_dir_to_target_body_x", "integral_dir_to_target_body_y", "integral_dir_to_target_body_z"], 
                self.integral_direction_to_target_in_body[i]
            ))
            self.prev_direction_to_target_in_body[i] = direction_to_target_in_body


            # Minimal rotation to align forward with target direction, drones forward axis is X (roll)
            # rotation_error_vector is similar to direction_to_target_in_body, but encoded in a different way
            if np.linalg.norm(direction_to_target) < 1e-8:
                rotation_error_matrix = R.identity()
                rotation_error_vector = np.zeros(3)
            else:
                rotation_error_matrix, _ = R.align_vectors([direction_to_target], [rotation_x_vector])
                rotation_error_vector = rotation_error_matrix.as_rotvec()

            obs.extend(self.add_feature_names(
                ["rot_err_x", "rot_err_y", "rot_err_z"], 
                rotation_error_vector
            ))




            # ---------------------------------------------------------------------
            # Control (action) engineered features
            # Derivative control updated in step()
            # ---------------------------------------------------------------------

            # normalized_prev_ctrl stores normalized [-1, 1] previous action controls
            # Roll/Pitch/Yaw each operate by adjusting the relative motor speed for 2/4 of the drone motors
            # The strength relative to thrust: More thrust --> more roll/pitch/yaw, so scale roll/pitch/yaw by thrust
            normalized_prev_ctrl = np.array([
                2 * (self.prev_control[i][0] / 0.35) - 1,
                self.prev_control[i][1] * (self.prev_control[i][0] / 0.35),
                self.prev_control[i][2] * (self.prev_control[i][0] / 0.35),
                self.prev_control[i][3] * (self.prev_control[i][0] / 0.35)
            ], dtype=np.float32)
            obs.extend(self.add_feature_names(["prev_ctrl_thrust", "prev_ctrl_roll", "prev_ctrl_pitch", "prev_ctrl_yaw"], normalized_prev_ctrl))

            # Derivative control computed in step()
            obs.extend(self.add_feature_names(["derivative_ctrl_thrust", "derivative_ctrl_roll", "derivative_ctrl_pitch", "derivative_ctrl_yaw"], self.derivative_control[i]))

            # Update action integral
            self.integral_control[i] = 0.95 * self.integral_control[i] + 0.05 * normalized_prev_ctrl
            obs.extend(self.add_feature_names(["integral_ctrl_thrust", "integral_ctrl_roll", "integral_ctrl_pitch", "integral_ctrl_yaw"], self.integral_control[i]))

            # Thrust in world frame (body thrust is just [0, 0, thrust], transform to world thrust based on drone rotation)
            world_thrust = rotation_matrix @ np.array([0, 0, self.prev_control[i][0]])
            world_thrust /= (np.linalg.norm(world_thrust) + 1e-9)
            obs.extend(self.add_feature_names(["world_thrust_x", "world_thrust_y", "world_thrust_z"], world_thrust))

            derivative_world_thrust = world_thrust - self.prev_world_thrust[i]
            obs.extend(self.add_feature_names(["derivative_world_thrust_x", "derivative_world_thrust_y", "derivative_world_thrust_z"], derivative_world_thrust))

            self.integral_world_thrust[i] = 0.95 * self.integral_world_thrust[i] + 0.05 * world_thrust
            obs.extend(self.add_feature_names(["integral_world_thrust_x", "integral_world_thrust_y", "integral_world_thrust_z"], self.integral_world_thrust[i]))
            self.prev_world_thrust[i] = world_thrust

            # Alignment in [-1, 1] of thrust towards the target
            thrust_alignment = np.dot(world_thrust, direction_to_target)
            derivative_thrust_alignment = thrust_alignment - self.prev_thrust_alignment[i]
            self.integral_thrust_alignment[i] = 0.95 * self.integral_thrust_alignment[i] + 0.05 * thrust_alignment
            self.prev_thrust_alignment[i] = thrust_alignment
            obs.extend(self.add_feature_names(
                ["thrust_alignment", "derivative_thrust_alignment", "integral_thrust_alignment"], 
                np.array([thrust_alignment, derivative_thrust_alignment, self.integral_thrust_alignment[i]], dtype=np.float32)
            ))

        return np.array(obs, dtype=np.float32)


    def render(self) -> None:
        """
        Render the scene in MuJoCo
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
