"""
Hardware Deployment Script for Running Trained Models on Real Crazyflie Drones

This script loads a GPU-trained model and runs it using real-time sensor data from
physical Crazyflie drones. It interfaces with the crazyflie-lib-python library to:
- Connect to drones via radio
- Collect sensor data (position, velocity, orientation, etc.)
- Run the trained policy network
- Send computed actions to the drones

Usage Examples:
    For one drone:
    python run_hardware.py --uri radio://0/80/2M/E7E7E7E701
    
    For multiple drones:
    python run_hardware.py --uri radio://0/80/2M/E7E7E7E701 radio://0/80/2M/E7E7E7E7E7
"""

# Base imports
import argparse
import time
import numpy as np
from typing import List
import sys
from scipy.spatial.transform import Rotation as R
import jax
from flax.training import checkpoints
import os
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks

# Crazyflie library imports
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Custom imports
from Actor import create_train_state, NUM_UPDATES, PER_AGENT_OBS_DIM, HOVER_THRUST
from motive_frame import motive_to_policy_position, motive_to_policy_quaternion
from mocap_stream import NatNetPoseStream

# send_action maps sim force [0, 0.35] then scales down for the real motors.
# Dummy hover must invert this or the drone only skims the floor (~55% PWM).
HARDWARE_THRUST_SCALE = 0.72
SIM_THRUST_MAX = 0.35


class CrazyflieHardwareInterface:
    """Interface for collecting sensor data from a single Crazyflie drone"""
        
    def __init__(self, uri: str, drone_id: int, pos_offset: np.ndarray):
        """
        Initialize hardware interface for a single drone
        
        Parameters
        ----------
        uri : str
            Crazyflie URI (e.g., 'radio://0/80/2M/E7E7E7E7E7')
        drone_id : int
            Unique identifier for this drone
        """
        self.uri = uri
        self.drone_id = drone_id
        self.position_offset = pos_offset
        self.use_mocap = False
        
        # State variables (updated by log callbacks, or OptiTrack when use_mocap)
        self.position = np.zeros(3, dtype=np.float32) # [x, y, z] in meters
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # [qx, qy, qz, qw]
        self.velocity = np.zeros(3, dtype=np.float32) # [vx, vy, vz] in m/s
        self.angular_velocity = np.zeros(3, dtype=np.float32) # [gx, gy, gz] in radians
        self._mocap_prev_pos = None
        self._mocap_prev_t = None
        
        self.connected = False
        self.scf = None
        
    def connect(self):
        """Establish connection to the Crazyflie"""
        print(f"Connecting to drone {self.drone_id} at {self.uri}...")
        try:
            self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache='./cache'))
            self.scf.open_link()
            self.connected = True
            print(f"✓ Drone {self.drone_id} connected")

            # Flow deck pose drifts if the drone was moved after power-on.
            # Absolute x/y then trips bound_range and the script sends sub-hover
            # instead of the policy (looks like sliding, never climbing).
            self.scf.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.1)
            self.scf.cf.param.set_value("kalman.resetEstimation", "0")
            print(f"  Reset Kalman estimator; keep the drone still for 2s")
            time.sleep(2.0)
            
            # Set up logging configuration
            self._setup_logging()
            
        except Exception as e:
            print(f"✗ Failed to connect to drone {self.drone_id}: {e}")
            self.connected = False
            
    def _setup_logging(self):
        """
        Configure log variables to stream from the Crazyflie.
        """

        # NOTE: Max log size is 26. bytes - can't print all info in one block, split into many
        # Log less critical data at lower frequency to converve bandwitdh
        
        # Block 1: Position
        log_pos = LogConfig(name='pos', period_in_ms=20)

        log_pos.add_variable('stateEstimate.x', 'float')
        log_pos.add_variable('stateEstimate.y', 'float')
        log_pos.add_variable('stateEstimate.z', 'float')

        self.scf.cf.log.add_config(log_pos)
        log_pos.data_received_cb.add_callback(self._log_callback)
        log_pos.start()

        # Block 2: Velocity
        log_vel = LogConfig(name='vel', period_in_ms=20)

        log_vel.add_variable('stateEstimate.vx', 'float')
        log_vel.add_variable('stateEstimate.vy', 'float')
        log_vel.add_variable('stateEstimate.vz', 'float')

        self.scf.cf.log.add_config(log_vel)
        log_vel.data_received_cb.add_callback(self._log_callback)
        log_vel.start()
        
        # Block 3: Quaternion
        log_quat = LogConfig(name='quat', period_in_ms=30)

        log_quat.add_variable('stateEstimate.qx', 'float')
        log_quat.add_variable('stateEstimate.qy', 'float')
        log_quat.add_variable('stateEstimate.qz', 'float')
        log_quat.add_variable('stateEstimate.qw', 'float')

        self.scf.cf.log.add_config(log_quat)
        log_quat.data_received_cb.add_callback(self._log_callback)
        log_quat.start()

        # Block 4: Angular velocity (gyro)
        log_gyro = LogConfig(name='gyro', period_in_ms=10)

        log_gyro.add_variable('gyro.x', 'float')
        log_gyro.add_variable('gyro.y', 'float')
        log_gyro.add_variable('gyro.z', 'float')

        self.scf.cf.log.add_config(log_gyro)
        log_gyro.data_received_cb.add_callback(self._log_callback)
        log_gyro.start()
        
    def _log_callback(self, timestamp, data, logconf):
        """
        Callback function for log data reception - each log does not contain all values,
        update values that are present.
        """
        for key, value in data.items():
            if self.use_mocap:
                if key == "gyro.x":
                    self.angular_velocity[0] = np.deg2rad(value)
                elif key == "gyro.y":
                    self.angular_velocity[1] = np.deg2rad(value)
                elif key == "gyro.z":
                    self.angular_velocity[2] = np.deg2rad(value)
                continue

            if key == 'stateEstimate.x': self.position[0] = value
            elif key == 'stateEstimate.y': self.position[1] = value
            elif key == 'stateEstimate.z': self.position[2] = value

            elif key == 'stateEstimate.vx': self.velocity[0] = value
            elif key == 'stateEstimate.vy': self.velocity[1] = value
            elif key == 'stateEstimate.vz': self.velocity[2] = value

            elif key == 'stateEstimate.qx': self.quaternion[0] = value
            elif key == 'stateEstimate.qy': self.quaternion[1] = value
            elif key == 'stateEstimate.qz': self.quaternion[2] = value
            elif key == 'stateEstimate.qw': self.quaternion[3] = value

            elif key == 'gyro.x': self.angular_velocity[0] = np.deg2rad(value)
            elif key == 'gyro.y': self.angular_velocity[1] = np.deg2rad(value)
            elif key == 'gyro.z': self.angular_velocity[2] = np.deg2rad(value)

    def apply_mocap_pose(
        self,
        xyz_motive: np.ndarray,
        quat_xyzw_motive: np.ndarray,
        timestamp: float,
        flip_x: bool,
        flip_y: bool,
    ) -> None:
        """Overwrite world pose from Motive (Y-up) after converting to policy Z-up."""
        pos = motive_to_policy_position(xyz_motive, flip_x=flip_x, flip_y=flip_y)
        quat = motive_to_policy_quaternion(quat_xyzw_motive, flip_x=flip_x, flip_y=flip_y)
        if self._mocap_prev_pos is not None and self._mocap_prev_t is not None:
            dt = timestamp - self._mocap_prev_t
            if dt > 1e-4:
                self.velocity = ((pos - self._mocap_prev_pos) / dt).astype(np.float32)
        self.position = pos
        self.quaternion = quat
        self._mocap_prev_pos = pos.copy()
        self._mocap_prev_t = timestamp

        
    def get_observation(
        self,
        target_pos: np.ndarray,
        all_drones: List['CrazyflieHardwareInterface'],
        initial_distance: float,
        action_history: np.ndarray,
    ) -> np.ndarray:
        """
        Hardware observation construction matching the training environment format from the colab notebook.

        Parameters
        ----------
        target_pos : np.ndarray
            Target position [x, y, z]
        all_drones : List[CrazyflieHardwareInterface]
            All drones in the system (including self)
        initial_distance : float
            Initial distance to target (for normalization)
        action_history : np.ndarray, optional
            Per-drone action history, shape (N, A) for N past actions

        Returns
        -------
        np.ndarray
            Flattened observation vector aligned with training environment expectation.
        """

        # NOTE: Need flow deck connected for XY observations

        # Add starting offset (for multiple drones, since each thinks it starts at [0, 0, 0])
        pos = self.position + self.position_offset

        # Catch invalid (0-magnitude) quats
        try:
            quat = self.quaternion
            rotation_matrix = R.from_quat(quat).as_matrix()
        except:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            rotation_matrix = R.from_quat(quat).as_matrix()

        vel = self.velocity

        ang_vel = self.angular_velocity

        # Relative target pos in body coordinates
        pos_error_world = target_pos - pos
        pos_error_norm = pos_error_world / (initial_distance + 1e-9)
        rel_pos_body = rotation_matrix.T @ pos_error_norm

        # Linear velocity in body frame
        linear_vel_body = rotation_matrix.T @ vel

        # Relative positions to all drones (body frame, currently includes self since notebook does)
        rel_drone_positions_body = []
        for d in all_drones:
            rel_drone = d.position + d.position_offset - pos
            rel_drone_body = rotation_matrix.T @ rel_drone
            rel_drone_positions_body.append(rel_drone_body)

        rel_drone_positions_body = np.array(rel_drone_positions_body).flatten()

        # Build core observation
        obs_flat = np.concatenate([
            pos,
            vel,
            ang_vel,
            rotation_matrix.reshape(-1),
            rel_pos_body,
            linear_vel_body,
            rel_drone_positions_body,
            action_history.flatten()
        ]).astype(np.float32)

        # Also return tuple for easy parsing / debugging
        obs_tuple = (
            pos,
            vel,
            ang_vel,
            rotation_matrix.reshape(-1),
            rel_pos_body,
            linear_vel_body,
            rel_drone_positions_body,
            action_history.flatten(),
        )

        return obs_flat, obs_tuple

        
    def send_action(self, action: np.ndarray):
        """
        Send action commands to the Crazyflie
        
        Parameters
        ----------
        action : np.ndarray
            Action vector [thrust, roll_torque, pitch_torque, yaw_torque] from policy
        """
        if not self.connected:
            print(f"Drone {self.drone_id} not connected, cannot send action")
            return
        
        # NOTE: The MuJoCo menagerie model we tested with has its control ranges as somewhat arbitrary.
        # As such, we have experimentally adjusted the controls here (thrust scaling and deg/s clamping)
            
        # Extract action components from simulation format
        thrust = np.clip(float(action[0]), 0.0, SIM_THRUST_MAX)
        thrust *= HARDWARE_THRUST_SCALE
        
        roll = np.clip(float(action[1]), -1.0, 1.0)  # Simulation: [-1, 1] normalized torque
        pitch = np.clip(float(action[2]), -1.0, 1.0) # Simulation: [-1, 1] normalized torque  
        yaw = np.clip(float(action[3]), -1.0, 1.0)   # Simulation: [-1, 1] normalized torque
        
        thrust_percent = np.clip((thrust / SIM_THRUST_MAX) * 100.0, 0.0, 100.0)
        
        # Map [-1, 1] limits to deg/s limits for safety
        rate_max = 1.0
        roll_rate_deg  = np.clip(roll  * rate_max, -rate_max, rate_max)
        pitch_rate_deg = np.clip(pitch * rate_max, -rate_max, rate_max)
        yaw_rate_deg   = np.clip(yaw   * rate_max, -rate_max, rate_max)
        
        # Send command via commander
        self.scf.cf.commander.send_setpoint_manual(
            roll_rate_deg,
            pitch_rate_deg,
            yaw_rate_deg,
            thrust_percent,
            rate=True
        )
        
    def disconnect(self):
        """Close connection to the Crazyflie"""
        if self.scf:
            # Send zero command before disconnecting
            self.scf.cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.1)
            self.scf.close_link()
            print(f"Drone {self.drone_id} disconnected")


class HardwareDeploymentController:
    """Main controller for running trained models on hardware"""
    
    def __init__(
            self, 
            model_checkpoint_path: str, 
            uris: List[str], 
            target_pos: np.ndarray, 
            control_rate: float,
            drone_offset: np.ndarray,
            mocap_stream: NatNetPoseStream | None = None,
            mocap_rb_ids: List[int] | None = None,
            mocap_flip_x: bool = False,
            mocap_flip_y: bool = False,
        ):
        """
        Initialize hardware deployment controller
        
        Parameters
        ----------
        model_checkpoint_path : str
            Path to the flax checkpoint folder with the trained model info
        uris : List[str]
            List of Crazyflie URIs to connect to
        target_pos : np.ndarray, optional
            Target position for hovering [x, y, z]
        control_rate : float
            Control loop frequency in Hz
        """
        self.num_drones = len(uris)
        self.target_pos = target_pos
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        
        # Initialize drones with URI, ID, and pos offset (for multiple drones since they all think they start at [0,0,0])
        drone_pos_offsets = [drone_offset * i for i in range(self.num_drones)]
        self.drones = [CrazyflieHardwareInterface(uri, i, drone_pos_offsets[i]) for i, uri in enumerate(uris)]
        self.mocap_stream = mocap_stream
        self.mocap_rb_ids = mocap_rb_ids or []
        self.mocap_flip_x = mocap_flip_x
        self.mocap_flip_y = mocap_flip_y
        if self.mocap_stream is not None:
            for drone in self.drones:
                drone.use_mocap = True
        
        # Load agent (single or double)
        if self.num_drones > 1:
            init_rng = jax.random.PRNGKey(0)
            init_state, _, _ = create_train_state(init_rng, num_updates=NUM_UPDATES)
            loaded_state = checkpoints.restore_checkpoint(
                ckpt_dir=model_checkpoint_path,
                target=init_state,
            )
            def inference_fn(obs_per_agent):
                apply_fn = loaded_state.policy_state.apply_fn
                params = loaded_state.policy_state.params
                mean, _ = apply_fn(params, obs_per_agent)
                return mean
            
            self.agent = inference_fn
        else:
            ppo_net = ppo_networks.make_ppo_networks(
                observation_size=PER_AGENT_OBS_DIM,
                action_size=4,
            )
            params = model.load_params(model_checkpoint_path)
            inference_fn = ppo_networks.make_inference_fn(ppo_net)(params=params, deterministic=True)
            self.agent = inference_fn


        
        
    def connect_all(self):
        """Connect to all drones"""
        print("\n" + "="*50)
        print("CONNECTING TO DRONES")
        print("="*50)
        
        # Initialize drivers
        cflib.crtp.init_drivers()
        
        # Connect each drone
        for drone in self.drones:
            drone.connect()
            
        # Check if all connected
        all_connected = all(drone.connected for drone in self.drones)
        if not all_connected:
            print("\n✗ Not all drones connected. Aborting.")
            self.disconnect_all()
            return False
            
        print("\n✓ All drones connected successfully")
        return True
        

    def disconnect_all(self):
        """Disconnect from all drones"""
        print("\nDisconnecting from all drones...")
        for drone in self.drones:
            if drone.connected:
                drone.disconnect()



    def run_control_loop(
            self, 
            duration: float, 
            dummy_policy: bool, 
            bound_range: float = 2.0,
            land_at_time_left: float = 2.0
        ):
        """Run a control loop with dummy (basic hover) or actual policy based on self.num_drones"""
        print("\nRunning control loop...")
        start_time = time.time()
        step_count = 0

        if self.mocap_stream is not None:
            if len(self.mocap_rb_ids) != self.num_drones:
                raise ValueError("Pass one --mocap-rb-id per URI (Motive Streaming ID).")
            for rb_id in self.mocap_rb_ids:
                self.mocap_stream.wait_for_pose(rb_id)
            self._apply_mocap_to_drones()
            print(
                f"Mocap pose locked at {self.drones[0].position}; "
                f"policy target {self.target_pos}"
            )

        initial_drone_dists = []
        for drone in self.drones:
            initial_drone_dists.append(np.linalg.norm(drone.position - self.target_pos))

        prev_action = np.zeros((self.num_drones, 4), dtype=np.float32)

        log_path = os.path.join(os.path.dirname(__file__), "drone_obs.log")

        safety_triggered = False
        
        try:
            with open(log_path, "w") as log_file:
                while (time.time() - start_time) < duration:
                    loop_start = time.time()
                    if self.mocap_stream is not None:
                        self._apply_mocap_to_drones()

                    observations_per_drone = []
                    obs_tuples = []
                    for i, drone in enumerate(self.drones):
                        obs_flat, obs_tuple = drone.get_observation(
                            self.target_pos, 
                            self.drones,
                            initial_drone_dists[i],
                            prev_action[i],
                        )
                        observations_per_drone.append(obs_flat)
                        obs_tuples.append(obs_tuple)

                    # Command override check
                    # When the run is almost over or we are out of bounds, send a sub-hover command to land
                    time_left = duration - (time.time() - start_time)
                    if time_left < land_at_time_left:
                        safety_triggered = True
                    
                    for drone in self.drones:
                        if np.any(np.abs(drone.position) > bound_range):
                            safety_triggered = True
                            break

                    # Action selection
                    if safety_triggered:
                        # Force safe control sub-hover thrust for all drones
                        safe_thrust = HOVER_THRUST * 0.8
                        safe_action = np.array([safe_thrust, 0.0, 0.0, 0.0], dtype=np.float32)
                        action = np.tile(safe_action, self.num_drones)
                    elif dummy_policy:
                        # Invert HARDWARE_THRUST_SCALE so dummy hover is sim hover on the motors.
                        dummy_thrust = min(SIM_THRUST_MAX, HOVER_THRUST / HARDWARE_THRUST_SCALE)
                        single_action = np.array([dummy_thrust, 0.0, 0.0, 0.0])
                        action = np.tile(single_action, self.num_drones)
                    else:
                        observations_per_drone = np.array(observations_per_drone, dtype=np.float32)
                        if self.num_drones > 1:
                            # Custom MAPPO model for two drones
                            action = self.agent(observations_per_drone).reshape((-1,))
                        else:
                            # Brax model for one drone
                            action, _ = self.agent(observations_per_drone, jax.random.PRNGKey(0))
                            action = np.array(action).reshape((-1,))
                        
                    # Send action(s) to drone(s)
                    for i, drone in enumerate(self.drones):
                        drone_action = action[4 * i : 4 * (i+1)]
                        drone.send_action(drone_action)
                    
                    # Update action history
                    prev_action = action.reshape(self.num_drones, 4)

                    # Logs
                    for i, drone in enumerate(self.drones):
                        pos, vel, ang_vel, rot_mat, rel_pos_body, lin_vel_body, rel_drones, _ = obs_tuples[i]
                        log_file.write(
                            f"Step {step_count}, Drone {i+1}, "
                            f"Pos={pos}, "
                            f"Vel={vel}, "
                            f"AngVel={ang_vel}, "
                            f"R={rot_mat}, "
                            f"RelPosBody={rel_pos_body}, "
                            f"LinVelBody={lin_vel_body}, "
                            f"RelDronePosBody={rel_drones}, "
                            f"ActionHistory={prev_action[0]}\n"
                        )
                        log_file.flush()
                        
                    step_count += 1
                    
                    # Sleep to maintain control rate
                    elapsed = time.time() - loop_start
                    if elapsed < self.dt:
                        time.sleep(self.dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n\nDummy control loop interrupted by user")
            
        finally:
            print("\nShutting down dummy control loop...")
            for drone in self.drones:
                if drone.connected:
                    drone.send_action(np.zeros(4))
            time.sleep(0.2)
            
            elapsed_total = time.time() - start_time
            print(f"\nDummy control loop completed:")
            print(f"  Duration: {elapsed_total:.1f}s")
            print(f"  Steps: {step_count}")
            print(f"  Average rate: {step_count/elapsed_total:.1f} Hz")

    def _apply_mocap_to_drones(self) -> None:
        for drone, rb_id in zip(self.drones, self.mocap_rb_ids):
            packed = self.mocap_stream.get_raw_pose(rb_id)
            if packed is None:
                continue
            xyz_m, quat_m, valid, ts = packed
            if not valid:
                continue
            drone.apply_mocap_pose(
                xyz_m,
                quat_m,
                ts,
                flip_x=self.mocap_flip_x,
                flip_y=self.mocap_flip_y,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run trained model on real Crazyflie hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            Single drone:
                python run_hardware_deployment.py --uri radio://0/100/2M/E7E7E7E7E7
            
            Single drone with OptiTrack:
                python run_hardware.py --uri radio://0/80/2M/E7E7E7E703 --mocap \\
                    --mocap-server 192.168.x.x --mocap-client 192.168.y.y \\
                    --mocap-rb-id 81 --mocap-sdk-dir /path/to/NatNetSDK/Samples \\
                    --target 0 0 0.3 --starting_pos_offset 0 0 0 --duration 5
        """
    )
    
    parser.add_argument(
        "--uri",
        type=str,
        nargs='+',
        required=True,
        help="Crazyflie URI(s) for connection (e.g., radio://0/80/2M/E7E7E7E7E7), determines number of drones"
    )
    
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[1.5, 0.0, 0.3],
        help="Target position [x y z] in meters (policy frame). With mocap, this is "
             "absolute Motive-converted position, not 'hover here'."
    )

    parser.add_argument(
        "--starting_pos_offset",
        type=float,
        nargs=3,
        default=[0.5, 0.0, 0.0],
        help="Starting drone position offsets (for multiple drones) in meters."
    )
    
    parser.add_argument(
        "--dummy_policy",
        type=bool,
        default=False,
        help="Run a dummy hover policy on control loop"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration to run in seconds"
    )

    parser.add_argument(
        "--bound_range",
        type=float,
        default=2.0,
        help="Send a sub-hover command to land if any drone exceeds this position range (meters) on any XYZ"
    )

    parser.add_argument(
        "--land_at_time_left",
        type=float,
        default=2.0,
        help="Send a sub-hover command to land when less than this much time (s) is left in the run"
    )

    parser.add_argument(
        "--control_rate",
        type=float,
        default=50.0,
        help="Control loop frequency in Hz (simulated with 50Hz in MuJoCo)"
    )
    parser.add_argument(
        "--mocap",
        action="store_true",
        help="Use OptiTrack NatNet pose instead of Flow-deck x/y/z and quaternion",
    )
    parser.add_argument(
        "--mocap-server",
        type=str,
        default=None,
        help="Motive PC IP (NatNet server), same as PythonSample.py",
    )
    parser.add_argument(
        "--mocap-client",
        type=str,
        default=None,
        help="This machine's IP (NatNet client), same as PythonSample.py",
    )
    parser.add_argument(
        "--mocap-multicast",
        action="store_true",
        help="Use multicast (default is unicast)",
    )
    parser.add_argument(
        "--mocap-sdk-dir",
        type=str,
        default=None,
        help="Folder containing NatNetClient.py (SDK Samples or PythonClient)",
    )
    parser.add_argument(
        "--mocap-rb-id",
        type=int,
        nargs="+",
        default=None,
        help="Motive rigid-body Streaming ID, one per --uri (see Properties)",
    )
    parser.add_argument(
        "--mocap-flip-x",
        action="store_true",
        help="Flip policy X after Y-up conversion if the cage axis is mirrored",
    )
    parser.add_argument(
        "--mocap-flip-y",
        action="store_true",
        help="Flip policy Y after Y-up conversion if the cage axis is mirrored",
    )
    
    args = parser.parse_args()
    
    # Determine number of drones
    num_drones = len(args.uri)

    # Configure model path based on num_drones
    if num_drones > 1:
        model_path = os.path.abspath(
            os.path.join(os.getcwd(), "..", "..", "rl_models", "gpu", "MAPPO_model_checkpoint")
        )
    else:
        model_path = os.path.abspath(
            os.path.join(os.getcwd(), "..", "..", "rl_models", "gpu", "PPO_brax_model_checkpoint.pkl")
        )
    
    target_pos = np.array(args.target, dtype=np.float32)

    mocap_stream = None
    mocap_rb_ids = None
    if args.mocap:
        if not args.mocap_server or not args.mocap_client:
            print("With --mocap you must set --mocap-server and --mocap-client (same IPs as PythonSample.py)")
            sys.exit(1)
        if not args.mocap_rb_id:
            print("With --mocap you must set --mocap-rb-id (Motive Streaming ID)")
            sys.exit(1)
        if len(args.mocap_rb_id) != num_drones:
            print("Pass as many --mocap-rb-id values as --uri values")
            sys.exit(1)
        mocap_rb_ids = list(args.mocap_rb_id)
        mocap_stream = NatNetPoseStream(
            server_ip=args.mocap_server,
            client_ip=args.mocap_client,
            multicast=args.mocap_multicast,
            sdk_dir=args.mocap_sdk_dir,
        )
        try:
            mocap_stream.start()
        except Exception as e:
            print(f"Failed to start NatNet: {e}")
            sys.exit(1)
    
    # Create controller
    controller = HardwareDeploymentController(
        model_checkpoint_path=model_path,
        uris=args.uri,
        target_pos=target_pos,
        control_rate=args.control_rate,
        drone_offset=np.array(args.starting_pos_offset),
        mocap_stream=mocap_stream,
        mocap_rb_ids=mocap_rb_ids,
        mocap_flip_x=args.mocap_flip_x,
        mocap_flip_y=args.mocap_flip_y,
    )
    
    # Connect to drones
    if not controller.connect_all():
        sys.exit(1)
        
    # Give drones time to stabilize
    print("\nWaiting 2 seconds for stabilization...")
    time.sleep(2.0)
    
    try:
        # Run control loop
        controller.run_control_loop(
            duration=args.duration,
            dummy_policy=args.dummy_policy,
            bound_range=args.bound_range,
            land_at_time_left=args.land_at_time_left
        )
    finally:
        # Ensure cleanup
        controller.disconnect_all()
        if mocap_stream is not None:
            mocap_stream.shutdown()
        
        for i in range(num_drones):
            from cflib.utils.power_switch import PowerSwitch
            PowerSwitch(args.uri[i]).stm_power_cycle()

        print("\n✓ Hardware deployment completed")


if __name__ == "__main__":
    main()
