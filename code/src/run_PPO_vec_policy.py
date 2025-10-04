import os
import csv
import time
import argparse
import numpy as np
import mujoco.viewer
import torch
from tabulate import tabulate
import logging

from run_manual_control import load_manual_steps
from constants import SCENE_PATH, MODEL_SAVE_PATH
from PPO_vec_agent import PPOAgentVec
from hover_target_environment import CrazyflieEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Set up logging, can be a lot of logs so save to a log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("run_logs_PPO_training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

####################################
# Env factory for parallelism
####################################
def make_env(xml_path, rank, seed=0, num_drones=1):
    """
    Returns a function that creates a new environment instance.
    Used by SubprocVecEnv/DummyVecEnv from SB3.
    """
    def _init():
        env = CrazyflieEnv(
            xml_path=xml_path,
            num_drones=num_drones
        )
        env.reset(seed=seed + rank)
        return env
    return _init


####################################
# Training PPO
####################################
def train_PPO(
    agent: PPOAgentVec,
    envs,
    total_steps: int,
    update_steps: int,
    print_logs: bool = True,
    save_model: bool = True
):
    """
    Train PPO with parallel environments

    Parameters
    -----------
    envs : SubprocVecEnv or DummyVecEnv
        The SB3 vectorized wrapper for our envs
    """
    print("\nBeginning PPO training...")
    num_envs = envs.num_envs
    num_updates = total_steps // update_steps

    obs = envs.reset()
    episode_returns = np.zeros(num_envs)

    for update in range(num_updates):
        for _ in range(update_steps):
            actions, log_probs, values = agent.sample_action(obs)
            next_obs, rewards, dones, infos = envs.step(actions)

            # SB3 VecEnv treats step as {next_obs, rewards, dones, infos} so recover terminated & truncated here from infos
            # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#
            terminateds = np.array([info.get("terminated", False) for info in infos])
            truncateds  = np.array([info.get("truncated", False) or info.get("TimeLimit.truncated", False) for info in infos])

            # Store batched transition
            agent.store_transition(obs, actions, rewards, terminateds, log_probs, values)

            obs = next_obs
            episode_returns += rewards

            # Reset reward tracking for finished episodes (terminated or truncated)
            dones = np.logical_or(terminateds, truncateds)
            for i, done in enumerate(dones):
                if done:
                    if print_logs:
                        end_type = "terminated" if terminateds[i] else "TRUNCATED"
                        logger.info(f"Update {update + 1}/{num_updates}: env {i + 1:2d} {end_type} an episode with return: {episode_returns[i]:.2f}")
                    episode_returns[i] = 0.0

        # At the end of rollout, bootstrap last state values for any non-terminal episodes
        # Terminal episodes will not bootstrapped since the PPO agent compute_gae() method handles the terminal case
        _, _, last_values = agent.sample_action(obs, deterministic=True)
        agent.update_policy(last_values)

        # Anneal lr param for policy and value optimizers (decreases lr over time based on number of updates)
        lr_now = agent.lr * (1.0 - update / num_updates)
        for optimizer in [agent.policy_optimizer, agent.value_optimizer]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

    if save_model:
        agent.save(MODEL_SAVE_PATH)
        
    print("PPO Training Complete")


####################
# Rendering PPO
####################
def render_PPO(agent: PPOAgentVec, env: CrazyflieEnv, seed: int = 42, sleep_time: float = 1/120):
    """
    Render trained PPO in a single environment
    """
    obs, _ = env.reset(seed)

    total_reward = 0.0
    with mujoco.viewer.launch_passive(env.mujoco_scene, env.data) as viewer:
        print("\nRunning final visualization based on learned policy")
        for _ in range(env.max_steps):
            action, _, _ = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            viewer.sync()
            time.sleep(sleep_time)

            if done:
                break
    print("\nSimulation complete.\n")



################
# Main
################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or render PPO")
    parser.add_argument(
        "--load_model",
        nargs="?",
        const=True,
        default=False,
        help="Optionally provide a model path. If omitted, uses default PPO save path."
    )
    parser.add_argument(
        "--manual_steps",
        action="store_true",
        help="Before PPO training, bootstrap the model offline from manual control steps."
    )
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total_steps", type=int, default=150_000, help="Total timesteps for training")
    parser.add_argument("--update_steps", type=int, default=1500, help="Number of timesteps before policy updates")
    parser.add_argument("--device", type=str, default="cpu", help="Device for tensor computations")
    args = parser.parse_args()

    # Parallel training environments
    env_fns = [make_env(SCENE_PATH, rank=i, seed=42) for i in range(args.num_envs)]
    if args.num_envs == 1:
        envs = DummyVecEnv(env_fns)
    else:
        envs = SubprocVecEnv(env_fns)

    # Initialize agent (get obs/action dimensions using the first env)
    example_env = env_fns[0]()
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        if args.device != "cpu":
            print("Device not supported, falling back to cpu")
        device = "cpu"

    agent = PPOAgentVec(
        obs_dim=example_env.observation_space.shape[0],
        action_dim=example_env.action_space.shape[0],
        device=device
    )

    if not args.load_model:
        # Pre-train drone with steps from manual control 
        if args.manual_steps:
            load_manual_steps(agent, "../model_manual_step_data")

            # Bootstrap policy update using demo buffer
            _, _, last_values = agent.sample_action(agent.buffer.observations[-1], deterministic=True)
            agent.update_policy(last_values)
    
        # Train the agent with PPO
        train_PPO(agent, envs, total_steps=args.total_steps, update_steps=args.update_steps)
    else:
        # Use provided model if present (as string), otherwise attempt to use default stored model
        if args.load_model is True:
            model_path = MODEL_SAVE_PATH 
        else:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models", 
                f"{args.load_model}.pt"
            )
        print(f"Loading model from: {model_path}")
        agent.load(model_path)

    # Single environment for rendering
    render_env = CrazyflieEnv(
        xml_path=SCENE_PATH,
        num_drones=1,
        target_pos=np.array([0.0, 0.00, 1.0], dtype=np.float32),
        max_steps=5000,
        random_initialization=False,
        debug=True
    )
    agent.track_obs_gradient = True
    render_PPO(agent, render_env, seed=42, sleep_time=1/1000)

    # Feature importance for each action
    rows = []
    for idx, feature_name in enumerate(render_env.observation_names):
        scores = []
        for a in range(agent.action_dim):
            counts = agent.obs_counts[a][idx]
            if counts > 0:
                scores.append(agent.obs_importance[a][idx] / counts)
            else:
                scores.append(0.0)
       
        score_sum = sum(scores)
        rows.append([idx, feature_name] + scores + [score_sum])

    headers = ["Idx", "Feature", "Thrust", "Roll", "Pitch", "Yaw", "Sum"]
    sortby = "Sum"
    col_idx = headers.index(sortby)
    rows.sort(key=lambda x: x[col_idx], reverse=True)

    # Format value columns (decimal precision)
    rows_formatted = [[r[0], r[1]] + [f"{s:.4f}" for s in r[2:]] for r in rows]

    # Save feature importance results to a CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output_feature_importance")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feature_importance.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in rows:
            writer.writerow(r)

    print(f"\nFeature importance CSV saved to {out_path}\n")

    # Track which rewards/penalties were important during this run
    summary = render_env.reward_tracker.summary()
    print(f"\nTotal reward: {summary['total']:.6f}")

    abs_total = 0
    for name, stats in summary["reward_summary"].items():
        abs_total += stats["abs_sum"]
    
    rows = []
    for name, stats in summary["reward_summary"].items():
        rows.append([name, stats["sum"], stats["abs_sum"], f"{(100 * stats['abs_sum'] / abs_total):.2f}%"])
    
    print(tabulate(rows, headers=["Name", "Sum (+/-)", "Absolute Sum", "Impact Ratio"], floatfmt=".6f", tablefmt="fancy_grid"))