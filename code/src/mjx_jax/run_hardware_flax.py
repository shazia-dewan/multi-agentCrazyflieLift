"""
Hardware Deployment Script for Running Trained Models on Real Crazyflie Drones

This script loads a trained .pt model and runs it using real-time sensor data from
physical Crazyflie drones. It interfaces with the crazyflie-lib-python library to:
- Connect to drones via radio
- Collect sensor data (position, velocity, orientation, etc.)
- Run the trained policy network
- Send computed actions to the drones

Usage:
    python run_hardware_deployment.py --model_path ../rl_models/ppo_model.pt --uri radio://0/80/2M/E7E7E7E7E7
    
    For multiple drones:
    python run_hardware_deployment.py --model_path ../rl_models/mappo_model.pt --num_drones 2 \
        --uri radio://0/80/2M/E7E7E7E701 radio://0/80/2M/E7E7E7E708
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
from Actor import create_train_state, NUM_UPDATES, PER_AGENT_OBS_DIM


class CrazyflieHardwareInterface:
    """Interface for collecting sensor data from a single Crazyflie drone"""
        
    def __init__(self, uri: str, drone_id: int = 0):
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
        
        # State variables (updated by log callbacks)
        self.position = np.zeros(3, dtype=np.float32) # [x, y, z] in meters
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32) # [qx, qy, qz, qw]
        self.velocity = np.zeros(3, dtype=np.float32) # [vx, vy, vz] in m/s
        self.angular_velocity = np.zeros(3, dtype=np.float32) # [gx, gy, gz] in radians
        
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

        # Construct base observations (rotation as matrix and ang_vel in radians)
        pos = self.position

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
            rel_drone = d.position - pos
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
            
        # Extract action components from simulation format
        thrust = float(action[0])       # Simulation: [0, 0.35] thrust force
        roll = float(action[1])  # Simulation: [-1, 1] normalized torque
        pitch = float(action[2]) # Simulation: [-1, 1] normalized torque  
        yaw = float(action[3])   # Simulation: [-1, 1] normalized torque
        
        thrust_percent = np.clip((thrust / 0.35) * 100.0, 0.0, 100.0)
        
        # Map [-1, 1] limits to deg/s limits for safety
        rate_max = 10.0
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
            target_pos: np.ndarray = None, 
            control_rate: float = 50.0
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
        self.target_pos = target_pos if target_pos is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        
        # Initialize drones
        self.drones = [CrazyflieHardwareInterface(uri, i) for i, uri in enumerate(uris)]
        
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



    def run_dummy_control_loop(self, duration: float = 10.0):
        """Run a dummy control loop with a basic hover action for testing"""
        print("\nRunning dummy hover control loop...")
        start_time = time.time()
        step_count = 0

        initial_drone_dists = []
        action_histories = []
        for drone in self.drones:
            initial_drone_dists.append(np.linalg.norm(drone.position - self.target_pos))
            action_histories.append(np.zeros((4, 4), dtype=np.float32))

        log_path = os.path.join(os.path.dirname(__file__), "drone_obs.log")
        
        try:
            with open(log_path, "w") as log_file:
                while (time.time() - start_time) < duration:
                    loop_start = time.time()

                    observations = []
                    obs_tuple = []
                    for i, drone in enumerate(self.drones):
                        obs_flat, obs_tuple = drone.get_observation(
                            self.target_pos, 
                            self.drones,
                            initial_drone_dists[i],
                            action_histories[i]
                        )
                        observations.append(obs_flat)
                    
                    # Send hover action to all drones, can test different values, should hover at ~0.26487
                    for i, drone in enumerate(self.drones):
                        hover_thrust = 0.26487
                        # Experimentally, the actual hover thrust is less than the mujoco hover thrust
                        test_hover_thrust = hover_thrust * 0.8
                        action = np.array([test_hover_thrust, 0.0, 0.0, 0.0])
                        drone.send_action(action)

                        # Update action history
                        action_histories[i][:-1] = action_histories[i][1:]
                        action_histories[i][-1] = action


                    # Logs
                    for i, drone in enumerate(self.drones):

                        # Log observations
                        pos, vel, ang_vel, rot_mat, rel_pos_body, lin_vel_body, rel_drones, _ = obs_tuple
                        log_file.write(
                            f"Step {step_count}, Drone {i+1}, "
                            f"Pos={pos}, "
                            f"Vel={vel}, "
                            f"AngVel={ang_vel}, "
                            f"R={rot_mat}, "
                            f"RelPosBody={rel_pos_body}, "
                            f"LinVelBody={lin_vel_body}, "
                            f"RelDronePosBody={rel_drones}, "
                            f"ActionHistory={action_histories[i]}\n"
                        )
                        log_file.flush()

                    # Log action that agent would have taken (self.agent can be single or multi inference fn)
                    observations = np.array(observations, dtype=np.float32)
                    # Messy temporary fix: adjust obs for multi-agent case
                    if self.num_drones > 1:
                        obs_per_drone = observations.reshape((self.num_drones, observations.shape))
                        agent_actions = self.agent(obs_per_drone).reshape((-1,))
                    else:
                        obs_per_drone = observations
                        action, _ = self.agent(obs_per_drone, jax.random.PRNGKey(0))
                        agent_actions = np.array(action).reshape((-1,))

                    log_file.write(f"Step {step_count}, Agent(s) would have taken action: {agent_actions}\n")
                    log_file.write("-------------\n")
                        
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

              
    # TODO: ADJUST FOR FLAX, CURRENTLY TORCH (+ observation changes)
    # def run_control_loop(self, duration: float = 30.0, verbose: bool = True):
    #     """
    #     Run the main control loop
        
    #     Parameters
    #     ----------
    #     duration : float
    #         How long to run in seconds
    #     verbose : bool
    #         Whether to print status updates
    #     """
    #     print("\n" + "="*50)
    #     print("STARTING CONTROL LOOP")
    #     print("="*50)
    #     print(f"Target position: {self.target_pos}")
    #     print(f"Duration: {duration}s")
    #     print(f"Control rate: {self.control_rate} Hz")
    #     print("\nPress Ctrl+C to stop")
    #     print("="*50 + "\n")
        
    #     start_time = time.time()
    #     step_count = 0
        
    #     try:
    #         while (time.time() - start_time) < duration:
    #             loop_start = time.time()
                
    #             # Collect observations from all drones
    #             observations = []
    #             for drone in self.drones:
    #                 obs = drone.get_observation(self.target_pos, self.drones)
    #                 observations.append(obs)
                    
    #             # Stack observations for multi-agent case
    #             if self.num_drones > 1:
    #                 obs_array = np.concatenate(observations)
    #             else:
    #                 obs_array = observations[0]
                    
    #             # Get action from policy (deterministic for deployment)
    #             with torch.no_grad():
    #                 action, _, _ = self.agent.sample_action(obs_array, deterministic=True)
                    
    #             # Send actions to drones
    #             if self.num_drones > 1:
    #                 # Split actions for each drone
    #                 for i, drone in enumerate(self.drones):
    #                     drone_action = action[i*4:(i+1)*4]
    #                     drone.send_action(drone_action)
    #             else:
    #                 self.drones[0].send_action(action)
                    
    #             # Verbose output
    #             if verbose and step_count % 50 == 0:  # Print every 0.5s at 100Hz
    #                 elapsed = time.time() - start_time
    #                 print(f"[{elapsed:.1f}s] Step {step_count}")
    #                 for i, drone in enumerate(self.drones):
    #                     pos = drone.position
    #                     dist = np.linalg.norm(pos - self.target_pos)
    #                     print(f"  Drone {i}: pos={pos}, dist to target={dist:.3f}m")
                        
    #             step_count += 1
                
    #             # Sleep to maintain control rate
    #             elapsed = time.time() - loop_start
    #             if elapsed < self.dt:
    #                 time.sleep(self.dt - elapsed)
                    
    #     except KeyboardInterrupt:
    #         print("\n\nControl loop interrupted by user")
            
    #     finally:
    #         # Send zero commands and disconnect
    #         print("\nShutting down...")
    #         for drone in self.drones:
    #             if drone.connected:
    #                 drone.send_action(np.zeros(4))
    #         time.sleep(0.2)
            
    #         elapsed_total = time.time() - start_time
    #         print(f"\nControl loop completed:")
    #         print(f"  Duration: {elapsed_total:.1f}s")
    #         print(f"  Steps: {step_count}")
    #         print(f"  Average rate: {step_count/elapsed_total:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Run trained model on real Crazyflie hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            Single drone:
                python run_hardware_deployment.py --num_drones 1 --uri radio://0/100/2M/E7E7E7E7E7
            
            Multiple drones:
                python run_hardware_deployment.py --num_drones 2 --uri radio://0/100/2M/E7E7E7E7E7 radio://0/100/2M/E7E7E7E7E8
        """
    )
    
    parser.add_argument(
        "--uri",
        type=str,
        nargs='+',
        required=True,
        help="Crazyflie URI(s) for connection (e.g., radio://0/80/2M/E7E7E7E7E7)"
    )
    
    parser.add_argument(
        "--num_drones",
        type=int,
        default=None,
        help="Number of drones (default: inferred from number of URIs)"
    )
    
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 1.0],
        help="Target position [x y z] in meters (default: 0 0 1)"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration to run in seconds (default: 3)"
    )
    
    parser.add_argument(
        "--control_rate",
        type=float,
        default=50.0,
        help="Control loop frequency in Hz (default: 50)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output during control loop"
    )
    
    args = parser.parse_args()
    
    # Determine number of drones
    num_drones = args.num_drones if args.num_drones else len(args.uri)
    if num_drones != len(args.uri):
        print(f"Warning: num_drones ({num_drones}) doesn't match number of URIs ({len(args.uri)})")
        print(f"Using {len(args.uri)} URIs provided")
        num_drones = len(args.uri)

    # Configure model path based on num_drones
    if num_drones > 1:
        model_path = os.path.abspath(
            os.path.join(os.getcwd(), "MAPPO_model_checkpoint")
        )
    else:
        model_path = os.path.abspath(
            os.path.join(os.getcwd(), "PPO_model_checkpoint.pkl")
        )
    
    target_pos = np.array(args.target, dtype=np.float32)
    
    # Create controller
    controller = HardwareDeploymentController(
        model_checkpoint_path=model_path,
        uris=args.uri,
        target_pos=target_pos,
        control_rate=args.control_rate
    )
    
    # Connect to drones
    if not controller.connect_all():
        sys.exit(1)
        
    # Give drones time to stabilize
    print("\nWaiting 2 seconds for stabilization...")
    time.sleep(2.0)
    
    try:
        # Run control loop
        # TODO; Replace with actual model
        controller.run_dummy_control_loop(
            duration=args.duration
        )
    finally:
        # Ensure cleanup
        controller.disconnect_all()
        
        for i in range(num_drones):
            from cflib.utils.power_switch import PowerSwitch
            PowerSwitch(args.uri[i]).stm_power_cycle()

        print("\n✓ Hardware deployment completed")


if __name__ == "__main__":
    main()
