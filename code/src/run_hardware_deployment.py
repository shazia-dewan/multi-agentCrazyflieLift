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
        --uri radio://0/80/2M/E7E7E7E7E7 radio://0/80/2M/E7E7E7E7E8
"""

import argparse
import time
import numpy as np
import torch
from typing import List, Optional
import sys

# Crazyflie library imports
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.swarm import Swarm
except ImportError:
    print("ERROR: crazyflie-lib-python not installed!")
    print("Install with: pip install cflib")
    sys.exit(1)

from PPO_agent import PPOAgentVec
from MAPPO_agent import MAPPOAgent


class CrazyflieHardwareInterface:
    """Interface for collecting sensor data from a single Crazyflie drone"""
    
    # Action scaling factors (tunable for your specific setup)
    # These convert simulation torques to hardware angles/rates
    THRUST_SCALE = 60000 / 0.35      # Sim [0, 0.35] → Hardware PWM [0, 60000]
    ANGLE_SCALE = 30.0               # Sim torque [-1, 1] → Angle ±30°
    YAW_RATE_SCALE = 100.0           # Sim torque [-1, 1] → Yaw rate ±100°/s
    MAX_ANGLE = 30.0                 # Maximum roll/pitch angle (safety limit)
    MAX_YAW_RATE = 200.0             # Maximum yaw rate (safety limit)
    
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
        self.position = np.zeros(3, dtype=np.float32)  # [x, y, z]
        self.velocity = np.zeros(3, dtype=np.float32)  # [vx, vy, vz]
        self.acceleration = np.zeros(3, dtype=np.float32)  # [ax, ay, az]
        self.orientation = np.zeros(3, dtype=np.float32)  # [roll, pitch, yaw] in degrees
        self.angular_velocity = np.zeros(3, dtype=np.float32)  # [gx, gy, gz]
        
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
        """Configure log variables to stream from the Crazyflie"""
        
        # Create log configuration for state estimation data
        log_config = LogConfig(name='StateEstimate', period_in_ms=10)  # 100 Hz
        
        # Position
        log_config.add_variable('stateEstimate.x', 'float')
        log_config.add_variable('stateEstimate.y', 'float')
        log_config.add_variable('stateEstimate.z', 'float')
        
        # Velocity
        log_config.add_variable('stateEstimate.vx', 'float')
        log_config.add_variable('stateEstimate.vy', 'float')
        log_config.add_variable('stateEstimate.vz', 'float')
        
        # Acceleration
        log_config.add_variable('stateEstimate.ax', 'float')
        log_config.add_variable('stateEstimate.ay', 'float')
        log_config.add_variable('stateEstimate.az', 'float')
        
        # Orientation (roll, pitch, yaw)
        log_config.add_variable('stateEstimate.roll', 'float')
        log_config.add_variable('stateEstimate.pitch', 'float')
        log_config.add_variable('stateEstimate.yaw', 'float')
        
        # Gyroscope (angular velocity)
        log_config.add_variable('gyro.x', 'float')
        log_config.add_variable('gyro.y', 'float')
        log_config.add_variable('gyro.z', 'float')
        
        # Add log configuration and start logging
        self.scf.cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(self._log_callback)
        log_config.start()
        
    def _log_callback(self, timestamp, data, logconf):
        """Callback function to update state variables from log data"""
        # Position
        self.position[0] = data.get('stateEstimate.x', 0.0)
        self.position[1] = data.get('stateEstimate.y', 0.0)
        self.position[2] = data.get('stateEstimate.z', 0.0)
        
        # Velocity
        self.velocity[0] = data.get('stateEstimate.vx', 0.0)
        self.velocity[1] = data.get('stateEstimate.vy', 0.0)
        self.velocity[2] = data.get('stateEstimate.vz', 0.0)
        
        # Acceleration (note: in Gs, might need conversion)
        self.acceleration[0] = data.get('stateEstimate.ax', 0.0)
        self.acceleration[1] = data.get('stateEstimate.ay', 0.0)
        self.acceleration[2] = data.get('stateEstimate.az', 0.0)
        
        # Orientation (in degrees)
        self.orientation[0] = data.get('stateEstimate.roll', 0.0)
        self.orientation[1] = data.get('stateEstimate.pitch', 0.0)
        self.orientation[2] = data.get('stateEstimate.yaw', 0.0)
        
        # Angular velocity (in deg/s)
        self.angular_velocity[0] = data.get('gyro.x', 0.0)
        self.angular_velocity[1] = data.get('gyro.y', 0.0)
        self.angular_velocity[2] = data.get('gyro.z', 0.0)
        
    def get_observation(self, target_pos: np.ndarray, other_drones: List['CrazyflieHardwareInterface'] = None) -> np.ndarray:
        """
        Construct observation vector matching the training environment format
        
        Parameters
        ----------
        target_pos : np.ndarray
            Target position [x, y, z]
        other_drones : List[CrazyflieHardwareInterface], optional
            Other drones for multi-agent observations
            
        Returns
        -------
        np.ndarray
            Observation vector matching training format
        """
        obs = []
        
        # Convert orientation from degrees to radians
        orientation_rad = np.deg2rad(self.orientation)
        
        # Target position relative to drone
        rel_target = target_pos - self.position
        obs.extend(rel_target)
        
        # Drone state
        obs.extend(self.position)
        obs.extend(self.velocity)
        obs.extend(orientation_rad)
        obs.extend(np.deg2rad(self.angular_velocity))  # Convert to rad/s
        obs.extend(self.acceleration)
        
        # Add other drone positions if multi-agent
        if other_drones:
            for other in other_drones:
                if other.drone_id != self.drone_id:
                    rel_pos = other.position - self.position
                    obs.extend(rel_pos)
                    obs.extend(other.velocity)
        
        # Pad observation to match expected dimension (146 per drone in training)
        # This is a simplified version - you may need to adjust based on your exact observation space
        obs_array = np.array(obs, dtype=np.float32)
        
        # Pad to 146 if necessary
        if len(obs_array) < 146:
            obs_array = np.pad(obs_array, (0, 146 - len(obs_array)), mode='constant')
        
        return obs_array[:146]  # Truncate if too long
        
    def send_action(self, action: np.ndarray):
        """
        Send action commands to the Crazyflie
        
        WARNING: This is an approximation of the simulation control!
        
        In simulation (MuJoCo):
          - Actions are direct torques/thrust: [thrust, roll_torque, pitch_torque, yaw_torque]
          - Ranges: thrust [0, 0.35], torques [-1, 1]
          - Low-level control applied directly to rigid body physics
        
        On hardware (Crazyflie):
          - send_setpoint expects: (roll_angle, pitch_angle, yaw_rate, thrust_PWM)
          - The drone has onboard PID controllers that convert angles to motor commands
          - This is a HIGH-LEVEL control interface, NOT direct torque control
        
        This function attempts to bridge the gap by:
          1. Interpreting torque commands as desired angles (imperfect mapping)
          2. Scaling to appropriate ranges
          3. Sending to onboard controller
        
        For better sim-to-real transfer, consider:
          - Retraining with attitude control in simulation
          - Domain randomization to account for control differences
          - Fine-tuning scaling factors based on hardware tests
        
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
        roll_torque = float(action[1])  # Simulation: [-1, 1] normalized torque
        pitch_torque = float(action[2]) # Simulation: [-1, 1] normalized torque  
        yaw_torque = float(action[3])   # Simulation: [-1, 1] normalized torque
        
        # ========================================================================
        # SCALING: Convert simulation torques to hardware angles/rates
        # Adjust class constants at top of CrazyflieHardwareInterface if needed
        # ========================================================================
        
        # Thrust: Scale from [0, 0.35] to PWM [0, 60000]
        # Note: Crazyflie uses PWM for motor control, not normalized thrust
        thrust_scaled = int(np.clip(thrust * self.THRUST_SCALE, 0, 60000))
        
        # Roll/Pitch: Interpret torques as desired angles in degrees
        # In sim: torques cause angular acceleration
        # On hardware: angles are setpoints for onboard PID controller
        roll_angle = np.clip(roll_torque * self.ANGLE_SCALE, -self.MAX_ANGLE, self.MAX_ANGLE)
        pitch_angle = np.clip(pitch_torque * self.ANGLE_SCALE, -self.MAX_ANGLE, self.MAX_ANGLE)
        
        # Yaw: Interpret torque as yaw RATE (deg/s) not angle
        # Crazyflie's send_setpoint uses yaw rate, not yaw angle
        yaw_rate = np.clip(yaw_torque * self.YAW_RATE_SCALE, -self.MAX_YAW_RATE, self.MAX_YAW_RATE)
        
        # Send command via commander (roll, pitch, yaw_rate, thrust)
        # This goes to the onboard attitude controller
        self.scf.cf.commander.send_setpoint(roll_angle, pitch_angle, yaw_rate, thrust_scaled)
        
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
    
    def __init__(self, model_path: str, uris: List[str], target_pos: np.ndarray = None, 
                 control_rate: float = 100.0):
        """
        Initialize hardware deployment controller
        
        Parameters
        ----------
        model_path : str
            Path to the trained .pt model file
        uris : List[str]
            List of Crazyflie URIs to connect to
        target_pos : np.ndarray, optional
            Target position for hovering [x, y, z]
        control_rate : float
            Control loop frequency in Hz
        """
        self.model_path = model_path
        self.uris = uris
        self.num_drones = len(uris)
        self.target_pos = target_pos if target_pos is not None else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        
        # Initialize drones
        self.drones = [CrazyflieHardwareInterface(uri, i) for i, uri in enumerate(uris)]
        
        # Load agent
        print(f"\nLoading model from {model_path}...")
        self._load_agent()
        
    def _load_agent(self):
        """Load the trained agent from file"""
        # Determine observation and action dimensions
        # This is based on your training environment
        obs_dim = 146 * self.num_drones
        action_dim = 4 * self.num_drones
        
        if self.num_drones > 1:
            self.agent = MAPPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_drones=self.num_drones
            )
            print(f"Using MAPPOAgent for {self.num_drones} drones")
        else:
            self.agent = PPOAgentVec(
                obs_dim=obs_dim,
                action_dim=action_dim
            )
            print("Using PPOAgentVec for single drone")
            
        # Load model weights
        self.agent.load(self.model_path)
        print("✓ Model loaded successfully")
        
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
                
    def run_control_loop(self, duration: float = 30.0, verbose: bool = True):
        """
        Run the main control loop
        
        Parameters
        ----------
        duration : float
            How long to run in seconds
        verbose : bool
            Whether to print status updates
        """
        print("\n" + "="*50)
        print("STARTING CONTROL LOOP")
        print("="*50)
        print(f"Target position: {self.target_pos}")
        print(f"Duration: {duration}s")
        print(f"Control rate: {self.control_rate} Hz")
        print("\nPress Ctrl+C to stop")
        print("="*50 + "\n")
        
        start_time = time.time()
        step_count = 0
        
        try:
            while (time.time() - start_time) < duration:
                loop_start = time.time()
                
                # Collect observations from all drones
                observations = []
                for drone in self.drones:
                    obs = drone.get_observation(self.target_pos, self.drones)
                    observations.append(obs)
                    
                # Stack observations for multi-agent case
                if self.num_drones > 1:
                    obs_array = np.concatenate(observations)
                else:
                    obs_array = observations[0]
                    
                # Get action from policy (deterministic for deployment)
                with torch.no_grad():
                    action, _, _ = self.agent.sample_action(obs_array, deterministic=True)
                    
                # Send actions to drones
                if self.num_drones > 1:
                    # Split actions for each drone
                    for i, drone in enumerate(self.drones):
                        drone_action = action[i*4:(i+1)*4]
                        drone.send_action(drone_action)
                else:
                    self.drones[0].send_action(action)
                    
                # Verbose output
                if verbose and step_count % 50 == 0:  # Print every 0.5s at 100Hz
                    elapsed = time.time() - start_time
                    print(f"[{elapsed:.1f}s] Step {step_count}")
                    for i, drone in enumerate(self.drones):
                        pos = drone.position
                        dist = np.linalg.norm(pos - self.target_pos)
                        print(f"  Drone {i}: pos={pos}, dist to target={dist:.3f}m")
                        
                step_count += 1
                
                # Sleep to maintain control rate
                elapsed = time.time() - loop_start
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
                    
        except KeyboardInterrupt:
            print("\n\nControl loop interrupted by user")
            
        finally:
            # Send zero commands and disconnect
            print("\nShutting down...")
            for drone in self.drones:
                if drone.connected:
                    drone.send_action(np.zeros(4))
            time.sleep(0.2)
            
            elapsed_total = time.time() - start_time
            print(f"\nControl loop completed:")
            print(f"  Duration: {elapsed_total:.1f}s")
            print(f"  Steps: {step_count}")
            print(f"  Average rate: {step_count/elapsed_total:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Run trained model on real Crazyflie hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single drone:
    python run_hardware_deployment.py --model_path ../rl_models/ppo_model.pt \\
        --uri radio://0/80/2M/E7E7E7E7E7
  
  Multiple drones:
    python run_hardware_deployment.py --model_path ../rl_models/mappo_model.pt \\
        --uri radio://0/80/2M/E7E7E7E7E7 radio://0/80/2M/E7E7E7E7E8 \\
        --num_drones 2
        
  Custom target and duration:
    python run_hardware_deployment.py --model_path ../rl_models/ppo_model.pt \\
        --uri radio://0/80/2M/E7E7E7E7E7 \\
        --target 0.5 0.5 1.5 --duration 60
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained .pt model file"
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
        default=30.0,
        help="Duration to run in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--control_rate",
        type=float,
        default=100.0,
        help="Control loop frequency in Hz (default: 100)"
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
    
    target_pos = np.array(args.target, dtype=np.float32)
    
    # Create controller
    controller = HardwareDeploymentController(
        model_path=args.model_path,
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
        controller.run_control_loop(
            duration=args.duration,
            verbose=not args.quiet
        )
    finally:
        # Ensure cleanup
        controller.disconnect_all()
        print("\n✓ Hardware deployment completed")


if __name__ == "__main__":
    main()
