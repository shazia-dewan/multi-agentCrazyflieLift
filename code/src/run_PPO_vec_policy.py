import os
import time
import argparse
import numpy as np
import mujoco.viewer
import torch

from constants import SCENE_PATH, MODEL_SAVE_PATH
from PPO_vec_agent import PPOAgentVec
from hover_target_environment import CrazyflieEnv
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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
    total_timesteps: int,
    num_steps: int,
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
    num_envs = envs.num_envs
    num_updates = total_timesteps // num_steps

    obs = envs.reset()
    episode_returns = np.zeros(num_envs)

    for update in range(num_updates):
        for _ in range(num_steps):
            actions, log_probs, values = agent.sample_action(obs)
            next_obs, rewards, dones, infos = envs.step(actions)

            # SB3 VecEnv treats step as {next_obs, rewards, dones, infos} so recover terminated & truncated here
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
                        print(f"Update {update + 1}/{num_updates}: env {i + 1:2d} finished an episode with return: {episode_returns[i]:.2f}")
                    episode_returns[i] = 0.0

        # At the end of rollout, bootstrap last state values for any non-terminal episodes
        # Terminal episodes will not bootstrapped since the PPO agent compute_returns() method handles the terminal case
        _, _, last_values = agent.sample_action(obs, deterministic=True)
        agent.update_policy(last_values)

        # Anneal lr param for policy and value optimizers (decreases lr over time based on number of updates)
        lr_now = agent.lr * (1.0 - update / num_updates)
        for optimizer in [agent.policy_optimizer, agent.value_optimizer]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_now

    if save_model:
        agent.save(MODEL_SAVE_PATH)


####################
# Rendering PPO
####################
def render_PPO(agent: PPOAgentVec, env: CrazyflieEnv, seed: int = 42):
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
            time.sleep(1/120)

            if done:
                break
        print(f"Simulation ended with total reward {total_reward:.2f}")



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
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--total_timesteps", type=int, default=150_000, help="Total timesteps for training")
    parser.add_argument("--num_steps", type=int, default=1500, help="Number of timesteps before policy updates")
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
        train_PPO(agent, envs, total_timesteps=args.total_timesteps, num_steps=args.num_steps)
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
        target_pos=np.array([0.0, -0.5, 1.0], dtype=np.float32),
        max_steps=4000,
        random_initialization=False,
        debug=True
    )
    agent.track_obs_gradient = True
    render_PPO(agent, render_env)

    # Checking which observations were important to the agent over the render run
    print("\nFeature importance ranking:")
    for name, score in agent.get_obs_importance().items():
        # Extract obs index e.g. obs_15 --> 15, look it up in env obs map
        idx = int(name.split("_")[1])
        feature_name = render_env.obs_index_to_name.get(idx, f"obs_{idx}")
        print(f"{feature_name:15s} (idx {idx:2d}): {score:.4f}")
