import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Any, Optional
from scipy.spatial.transform import Rotation as R

from PID_controller import NeutralHoverController
from reward_system import RewardTracker

STARTING_QUAT_RANGE = np.pi / 180

class PIDTestEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        xml_path: str, 
        num_drones: int = 1, 
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
        self.debug = debug

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
        
        for i in range(self.num_drones):
            base_qpos = i * self.qpos_per_drone
            base_qvel = i * self.qvel_per_drone

            # Random starting rotation to test PID stability
            axis = self.np_random.normal(size=3)

            # Instead of random axis, uncomment to rotate about X or Y axis below to inspect roll/pitch/yaw
            # axis = np.array([1.0, 0.0, 0.0])
            # axis = np.array([0.0, 1.0, 0.0])
            # axis = np.array([0.0, 0.0, 1.0])

            axis /= np.linalg.norm(axis)


            angle = STARTING_QUAT_RANGE

            w = np.cos(angle / 2.0)
            x, y, z = axis * np.sin(angle / 2.0)
            quat = np.array([w, x, y, z], dtype=np.float64)

            self.data.qpos[base_qpos + 3: base_qpos + 7] = quat

            # Initial state
            self.state[i] = np.concatenate([
                self.data.qpos[base_qpos : base_qpos + 3],
                self.data.qpos[base_qpos + 3 : base_qpos + 7],
                self.data.qvel[base_qvel : base_qvel + 3],
                self.data.qvel[base_qvel + 3 : base_qvel + 6]
            ])

        self.controller = NeutralHoverController(target_yaw=0.0)

        
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

            self.state[i] = np.concatenate([pos, quat, vel, ang_vel])
            
            quat_xyzw = np.roll(quat, -1) # scipy uses [x, y, z, w]
            # Euler = current roll/pitch/yaw
            euler = R.from_quat(quat_xyzw).as_euler('xyz')

            # Take action based on the PID controller
            cmd = self.controller.update(pos, vel, euler, ang_vel)
            
            ctrl[base_ctrl + 0] = cmd["thrust"]
            ctrl[base_ctrl + 1] = cmd["roll"]
            ctrl[base_ctrl + 2] = cmd["pitch"]
            ctrl[base_ctrl + 3] = cmd["yaw"]

            # Update action history: shift left and append current action
            self.action_history[i, :-1] = self.action_history[i, 1:]
            self.action_history[i, -1] = ctrl[base_ctrl: base_ctrl + self.ctrl_per_drone]

            # Update prev pos for next timestep
            self.prev_pos[i] = pos.copy()
            self.prev_control[i] = ctrl[base_ctrl : base_ctrl + self.ctrl_per_drone].copy()

            if self.debug:
                print(f"Step {self.timestep} - Ctrl: {[f'{c:.4f}' for c in ctrl]}, Position: {[f'{p:.2f}' for p in pos]}")
            
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
        return