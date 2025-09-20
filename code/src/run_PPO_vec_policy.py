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

MAX_CURRICULUM_STAGE = 3

####################################
# Env factory for parallelism
####################################
def make_env(xml_path, rank, seed=0, num_drones=1, max_steps=400):
    """
    Returns a function that creates a new environment instance.
    Used by SubprocVecEnv/DummyVecEnv from SB3.
    """
    def _init():
        env = CrazyflieEnv(
            xml_path=xml_path,
            num_drones=num_drones,
            curriculum=False,
            max_curriculum_stage=MAX_CURRICULUM_STAGE,
            max_steps=max_steps
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
    updates_before_curriculum = num_updates // (MAX_CURRICULUM_STAGE + 1)
    current_curriculum = 0

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
                        print(f"Update {update + 1}/{num_updates}: env {i + 1} finished an episode with return: {episode_returns[i]:.2f} (curriculum stage {current_curriculum})")
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

        # Advance the curriculum (env target and drone pos ranges) based on #updates
        # Motive: Start with small ranges (fine control) and move on from there
        if (update + 1) % updates_before_curriculum == 0 and current_curriculum < MAX_CURRICULUM_STAGE:
            envs.env_method("advance_curriculum")
            current_curriculum += 1     

    if save_model:
        agent.save(MODEL_SAVE_PATH)


####################
# Rendering PPO
####################
def render_PPO(agent: PPOAgentVec, env: CrazyflieEnv, seed: int = 42, num_steps: int = 3000):
    """
    Render trained PPO in a single environment
    """
    env.max_steps = num_steps
    env.debug = True
    env.curriculum = False
    obs, _ = env.reset(seed)

    total_reward = 0.0
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("\nRunning final visualization based on learned policy")
        for step in range(num_steps):
            action, _, _ = agent.sample_action(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            viewer.sync()
            time.sleep(1/60)

            if done:
                print(f"Simulation ended at step {step+1}/{num_steps} with total reward {total_reward:.2f}")
                break


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
    parser.add_argument("--total_timesteps", type=int, default=120_000, help="Total timesteps for training")
    parser.add_argument("--num_steps", type=int, default=1200, help="Number of timesteps before policy updates")
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
        target_pos=np.array([0.0, 0.0, 3.0], dtype=np.float32),
        curriculum=False
    )
    render_PPO(agent, render_env)
