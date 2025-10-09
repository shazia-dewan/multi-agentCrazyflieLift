import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional
from scipy.spatial.transform import Rotation as R

from PID_controller import NeutralHoverController
from reward_system import RewardTracker

TRAINING_POS_RANGE = 0.1

OUT_OF_BOUNDS_RANGE = 0.2
AT_TARGET_RANGE = 0.01

TERMINATION_PENALTY = -1.0

RL_RESIDUAL_THRUST = 0.05
RL_RESIDUAL_ROTATE = 0.1

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
        self.action_per_drone = 3

        # Store user config
        self.num_drones = num_drones
        self.target_pos = target_pos
        self.max_steps = max_steps
        self.random_initialization = random_initialization
        self.debug = debug

        # Index of observations for logging afterwards (e.g. which observations were important)
        self.observation_names = []

        # Define the observation space with n features based on how many we assign in _get_obs()
        obs_high = np.inf * np.ones(3 * self.num_drones, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        # Action = Desired velocity XYZ in drone body coordinates
        act_high = np.tile([1.0, 1.0, 1.0], self.num_drones).astype(np.float32)
        act_low = np.tile([-1.0, -1.0, -1.0], self.num_drones).astype(np.float32)
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

        # Track state for PID control
        self.state = np.zeros((self.num_drones, self.qpos_per_drone + self.qvel_per_drone), dtype=np.float32)

        # Track previous and starting position for reward / termination tracking
        self.starting_pos = np.zeros((self.num_drones, 3), dtype=np.float32)
        self.prev_pos = np.zeros((self.num_drones, 3), dtype=np.float32)

        # If using multiple drones, offset them along X axis
        drone_spacing_x = 0.4
        drone_offsets_x = np.linspace(-(self.num_drones - 1) / 2, (self.num_drones - 1) / 2, self.num_drones) * drone_spacing_x

        # Store last n control actions to compensate for control lag
        self.action_history_len = 4
        self.action_history = np.zeros((self.num_drones, self.action_history_len, self.action_per_drone), dtype=np.float32)

        # Start drone at some random pos [X, Y, Z] around the target with random velocity
        if self.random_initialization:
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                base_qvel = i * self.qvel_per_drone
                
                # Random starting position
                start_x = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)
                start_y = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)
                start_z = self.np_random.uniform(-TRAINING_POS_RANGE, TRAINING_POS_RANGE)

                self.data.qpos[base_qpos + 0] = start_x
                self.data.qpos[base_qpos + 1] = start_y + self.target_pos[1]
                self.data.qpos[base_qpos + 2] = start_z + self.target_pos[2]

                self.prev_pos[i] = self.data.qpos[base_qpos : base_qpos + 3]
        else:
            # Start drone(s) at consistent Z with X offset per drone
            for i in range(self.num_drones):
                base_qpos = i * self.qpos_per_drone
                self.data.qpos[base_qpos + 0] = drone_offsets_x[i]
                self.data.qpos[base_qpos + 1] = 0
                self.data.qpos[base_qpos + 2] = 1.0

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

        # Stability PID controller
        self.controller = NeutralHoverController(target_yaw=0.0)

        self.timestep = 0

        self.reward_tracker = RewardTracker()

        return self._get_obs(), {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take a step (action) and track the observation

        Parameters
        ----------
        action : np.ndarray
            Desired velocity XYZ for the drones

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

        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone
            base_ctrl = i * self.ctrl_per_drone
            base_action = i * self.action_per_drone

            # Current state
            pos = self.data.qpos[base_qpos: base_qpos + 3]
            quat = self.data.qpos[base_qpos + 3: base_qpos + 7]
            vel = self.data.qvel[base_qvel: base_qvel + 3]
            ang_vel = self.data.qvel[base_qvel + 3: base_qvel + 6]

            self.state[i] = np.concatenate([pos, quat, vel, ang_vel])

            ###########################################################
            # Stability PID control + RL Control for navigation
            ###########################################################

            # Get euler for stability controller, scipy uses [x, y, z, w]
            quat_xyzw = np.roll(quat, -1)
            euler = R.from_quat(quat_xyzw).as_euler('xyz')
            pid_ctrl = self.controller.update(pos, vel, euler, ang_vel)

            # Map desired velocity on XYZ to thrust/roll/pitch residuals to be added to the PID controller
            thrust_rl = np.clip(action[base_action + 2] * RL_RESIDUAL_THRUST, -RL_RESIDUAL_THRUST, RL_RESIDUAL_THRUST)
            roll_rl   = np.clip(action[base_action + 1] * RL_RESIDUAL_ROTATE, -RL_RESIDUAL_ROTATE, RL_RESIDUAL_ROTATE)
            pitch_rl  = -1 * np.clip(action[base_action + 0] * RL_RESIDUAL_ROTATE, -RL_RESIDUAL_ROTATE, RL_RESIDUAL_ROTATE)

            # Add residuals to PID control, clip within drone control limits
            ctrl_action = np.array([
                np.clip(pid_ctrl["thrust"] + thrust_rl, 0.0, 0.35),
                np.clip(pid_ctrl["roll"] + roll_rl, -1.0, 1.0),
                np.clip(pid_ctrl["pitch"] + pitch_rl, -1.0, 1.0),
                np.clip(pid_ctrl["yaw"], -1.0, 1.0)
            ])

            ctrl[base_ctrl + 0] = ctrl_action[0]
            ctrl[base_ctrl + 1] = ctrl_action[1]
            ctrl[base_ctrl + 2] = ctrl_action[2]
            ctrl[base_ctrl + 3] = ctrl_action[3]

            # Update action history: shift left and append current roll and pitch
            self.action_history[i, :-1] = self.action_history[i, 1:]
            self.action_history[i, -1] = action[base_action : base_action + 3]

            ##############################################
            # Rewards, Termination, and Truncation
            ##############################################

            pos_error = self.target_pos - pos
            distance_to_target = np.linalg.norm(pos_error)

            # Reward for moving closer to target
            distance_to_target_old = np.linalg.norm(self.target_pos - self.prev_pos[i])
            distance_improvement = distance_to_target_old - distance_to_target
            self.reward_tracker.update("distance_improvement", distance_improvement)

            reward = self.reward_tracker.step_total()

            # # Crash, terminate with penalty
            # if pos[2] < 0.05:
            #     self.reward_tracker.update("crash", TERMINATION_PENALTY)
            #     reward = self.reward_tracker.step_total()
            #     terminated = True

            # # Out of bounds, terminate with penalty (if we go much further from the target compared to starting pos)
            # initial_distance_to_target = np.linalg.norm(self.target_pos - self.starting_pos[i])
            # if distance_to_target > initial_distance_to_target + OUT_OF_BOUNDS_RANGE:
            #     self.reward_tracker.update("out_of_bounds", TERMINATION_PENALTY)
            #     reward = self.reward_tracker.step_total()
            #     terminated = True

            # Detect if we have exceeded max steps (truncate)
            truncated = self.timestep >= self.max_steps

            # Update prev pos
            self.prev_pos[i] = pos.copy()

            if self.debug:
                print(f"Desired velocity direction: ({np.sign(action[0])}, {np.sign(action[1])}, {np.sign(action[2])})")
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

            # ------------------------------------------------
            # Base Features in Observation Space:
            # ------------------------------------------------

            # position[x, y, z]
            pos = self.data.qpos[base_qpos: base_qpos + 3]
            # obs.extend(self.add_feature_names(["pos_x", "pos_y", "pos_z"], pos))

            # # Not using quat directly (can be ambiguous to nn), using rotation_matrix below
            # quat = self.data.qpos[base_qpos + 3 : base_qpos + 7]
            # quat_xyzw = np.roll(quat, -1) # SciPy expects [x, y, z, w], so reorder

            # # From quaternion -> rotation matrix (get orientation of drone in world coordinates)
            # rotation_matrix = R.from_quat(quat_xyzw).as_matrix()
            # obs.extend(self.add_feature_names(["rot_matrix_x0", "rot_matrix_x1", "rot_matrix_x2"], rotation_matrix[0]))
            # obs.extend(self.add_feature_names(["rot_matrix_y0", "rot_matrix_y1", "rot_matrix_y2"], rotation_matrix[1]))
            # obs.extend(self.add_feature_names(["rot_matrix_z0", "rot_matrix_z1", "rot_matrix_z2"], rotation_matrix[2]))

            # # velocity[vx, vy, vz]
            # vel = self.data.qvel[base_qvel: base_qvel + 3]
            # obs.extend(self.add_feature_names(["vel_x", "vel_y", "vel_z"], vel))

            # # angular velocity[wx, wy, wz]
            # ang_vel = self.data.qvel[base_qvel + 3 : base_qvel + 6]
            # obs.extend(self.add_feature_names(["ang_x", "ang_y", "ang_z"], ang_vel))

            # Relative pos
            pos_error = self.target_pos - pos
            obs.extend(self.add_feature_names(["pos_err_x", "pos_err_y", "pos_err_z"], pos_error))

            # # Action history
            # flattened_action_history = self.action_history[i].flatten()
            # obs.extend(self.add_feature_names(
            #     [f"ctrl_hist_{t}_{name}" for t in range(self.action_history_len) for name in ["desired_vel_x", "desired_vel_y", "desired_vel_z"]],
            #     flattened_action_history
            # ))

        return np.array(obs, dtype=np.float32)

