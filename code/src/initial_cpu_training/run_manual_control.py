# Run a MuJoCo crazyflie sim with manual input controls for the drone
# Purpose
#   1. Better understand how an ideal drone behaves in relation to state and control (e.g angular velocity and roll).
#       Subsequently, better understand how to shape features and rewards for RL task.
#   2. Save manual data to potentially train the PPO agent via imitation

import argparse
import datetime
import os
import time
from tabulate import tabulate
import torch
import numpy as np
import mujoco.viewer
from pynput import keyboard
import pickle

from constants import SCENE_PATH, MULTI_SCENE_PATH
from hover_target_env import CrazyflieEnv
from PPO_agent import PPOAgentVec

# Store manual step info (obs, action, reward, done) so it can be used to train the agent by imitation
step_data = []

# Step increments for controls
THRUST_STEP = 0.001
ROLL_STEP   = 0.001
PITCH_STEP  = 0.001
YAW_STEP    = 0.001

TRAINING_POS_RANGE = 2.0

# Current pressed keys
pressed_keys = set()

# Control key mapping: Arrows + number keys 4-7
# This key scheme is bizarre but avoids clashing with MuJoCo key commands (e.g. WASD clashes)
key_map = {
    # Thrust (arrow up/down)
    keyboard.Key.up:    ("thrust", +THRUST_STEP),
    keyboard.Key.down:  ("thrust", -THRUST_STEP),

    # Roll (arrow left/right)
    keyboard.Key.left:  ("roll", -ROLL_STEP),
    keyboard.Key.right: ("roll", +ROLL_STEP),

    # Pitch (6/7)
    "6": ("pitch", +PITCH_STEP),
    "7": ("pitch", -PITCH_STEP),

    # Yaw (4/5)
    "4": ("yaw", +YAW_STEP),
    "5": ("yaw", -YAW_STEP),
}



def on_press(key):
    try:
        if key in key_map or (hasattr(key, "char") and key.char in key_map):
            pressed_keys.add(key)
    except AttributeError:
        pass


def on_release(key):
    try:
        if key in pressed_keys:
            pressed_keys.remove(key)
        elif hasattr(key, "char") and key.char in pressed_keys:
            pressed_keys.remove(key.char)
    except AttributeError:
        pass


def load_manual_steps(agent: PPOAgentVec, manual_steps_dir: str) -> None:
    """
    Loads all .pkl files of manually controlled drone steps rom the manual step directory into
    the agent for imitation training (pre-training, offline, bootstrapping)

    Parameters
    ----------
    agent : PPOAgentVec
        The PPO agent
    manual_steps_path : str
        The path to the .pkl file with the manual steps
    """
    print("\nPre-training from manual control episodes...")

    manual_step_files = [f for f in os.listdir(manual_steps_dir) if f.endswith(".pkl")]
    if not manual_step_files:
        print(f"No demo files found in {manual_step_files}")
        return
    
    total_steps = 0
    for filename in manual_step_files:

        fpath = os.path.join(manual_steps_dir, filename)
        with open(fpath, "rb") as f:
            manual_control_data = pickle.load(f)

        for (obs, action, reward, done) in manual_control_data:
            obs = np.array(obs, dtype=np.float32)
            action = np.array(action, dtype=np.float32)

            # Recompute log_prob & value from current policy (not stored in .pkl files)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            mean, std = agent.policy_network(obs_tensor)
            dist = torch.distributions.Normal(mean, std)

            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
            log_prob = dist.log_prob(action_tensor).sum(dim=-1)
            value = agent.value_network(obs_tensor).view(-1)

            terminated = np.array([done], dtype=np.float32)

            agent.buffer.store(
                observations=obs[np.newaxis, :],
                actions=action[np.newaxis, :],
                rewards=np.array([reward]),
                terminateds=terminated,
                log_probs=log_prob,
                values=value
            )
            total_steps += 1

    print(f"{total_steps} manual steps loaded from {len(manual_step_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a manual control simulation of drone(s) in MuJoCo.")
    parser.add_argument(
        "--num_drones",
        type=int,
        default=1,
        help="Number of drones in the environment. If > 1, apply manual control to all drones."
    )
    args = parser.parse_args()

    start_x = np.random.uniform(-TRAINING_POS_RANGE / 2, TRAINING_POS_RANGE / 2)
    start_y = np.random.uniform(-TRAINING_POS_RANGE / 2, TRAINING_POS_RANGE / 2)
    start_z = np.random.uniform(0.5, TRAINING_POS_RANGE)

    single_action = np.array([0.26487, 0.0, 0.0, 0.0], dtype=np.float32)
    if args.num_drones > 1:
        scene = MULTI_SCENE_PATH
        action = np.tile(single_action, args.num_drones)
    else:
        scene = SCENE_PATH
        action = single_action

    env = CrazyflieEnv(
        xml_path=scene,
        num_drones=args.num_drones,
        target_pos=np.array([0.0, start_y, start_z], dtype=np.float32),
        max_steps=3000,
        random_initialization=False,
        manual_override = True,
        debug=True
    )

    obs, _ = env.reset()

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    total_reward = 0.0

    with mujoco.viewer.launch_passive(env.mujoco_scene, env.data) as viewer:
        # Use the fixed camera for control so it's easier to navigate
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = mujoco.mj_name2id(env.mujoco_scene, mujoco.mjtObj.mjOBJ_CAMERA, "track")

        print("Letting MuJoCo start up... 3")
        time.sleep(1)
        print("Letting MuJoCo start up... 2")
        time.sleep(1)
        print("Letting MuJoCo start up... 1")
        time.sleep(1)

        # Adjust real-time render based on env frame skip and timestep
        sim_dt = env.mujoco_scene.opt.timestep * getattr(env, "frame_skip", 1)
        last_time = time.perf_counter()

        for step in range(env.max_steps):
            # Reset action deltas each step
            thrust_delta, roll_delta, pitch_delta, yaw_delta = 0, 0, 0, 0

            # Apply all pressed key changes
            for k in pressed_keys:
                ctrl, delta = key_map.get(k, key_map.get(getattr(k, "char", None), (None, 0)))
                if ctrl == "thrust":
                    thrust_delta += delta
                elif ctrl == "roll":
                    roll_delta += delta
                elif ctrl == "pitch":
                    pitch_delta += delta
                elif ctrl == "yaw":
                    yaw_delta += delta

            # Update action
            if args.num_drones > 1:
                for i in range(args.num_drones):
                    base_ctrl = i * 4
                    print(base_ctrl)
                    action[base_ctrl] = np.clip(action[0] + thrust_delta, 0.0, 0.35)
                    action[base_ctrl + 1] = np.clip(action[1] + roll_delta, -1, 1)
                    action[base_ctrl + 2] = np.clip(action[2] + pitch_delta, -1, 1)
                    action[base_ctrl + 3] = np.clip(action[3] + yaw_delta, -1, 1)
            else:
                action[0] = np.clip(action[0] + thrust_delta, 0.0, 0.35)
                action[1] = np.clip(action[1] + roll_delta, -1, 1)
                action[2] = np.clip(action[2] + pitch_delta, -1, 1)
                action[3] = np.clip(action[3] + yaw_delta, -1, 1)

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Track steps
            step_data.append((obs, action.copy(), reward, done))

            # Track total reward
            total_reward += reward

            # Render
            viewer.sync()
            
            # Sleep the sim for real-time rendering
            elapsed = time.perf_counter() - last_time
            sleep_time = sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.perf_counter()

            if done:
                break

    listener.stop()

    # Track which rewards/penalties were important during this run
    print(f"Reward summary:\n")
    summary = env.reward_tracker.summary()

    print(f"Total reward: {summary['total']:.6f}")

    abs_total = 0
    for name, stats in summary["reward_summary"].items():
        abs_total += stats["abs_sum"]
    
    rows = []
    for name, stats in summary["reward_summary"].items():
        rows.append([name, stats["sum"], stats["abs_sum"], f"{(100 * stats['abs_sum'] / abs_total):.2f}%"])
    
    print(tabulate(rows, headers=["Name", "Sum (+/-)", "Absolute Sum", "Impact Ratio"], floatfmt=".6f", tablefmt="fancy_grid"))


    # Save manual steps to pickle file
    out_dir = os.path.join(os.path.dirname(__file__), "model_manual_step_data")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"manual_ep_reward_{total_reward:.2f}_{timestamp}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(step_data, f)
    print(f"Saved manual episode to {out_path}")



