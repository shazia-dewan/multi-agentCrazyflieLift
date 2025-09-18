import os
import time
import mujoco.viewer
from PPO_agent import PPOAgent
from constants import SCENE_PATH, MODEL_SAVE_PATH
from hover_target_environment import CrazyflieEnv
import numpy as np
import argparse

def train_PPO(
    agent: PPOAgent, 
    env: CrazyflieEnv,
    total_timesteps: int = 150_000, 
    num_steps: int = 1500, 
    print_logs: bool = True,
    save_model: bool = True,
    increment_seed: bool = False
):
    """
    Trains the PPO agent in the environment

    Parameters
    ----------
    agent : PPOAgent
        The PPOAgent
    env : CrazyflieEnv
        Training environment
    total_timesteps : int
        Total steps for the training
    num_steps : int
        Number of steps before a policy update (can span multiple episodes)
    print_logs : bool
        If true, prints some logs
    save_model : bool
        If true, saves the model to a local .pt file
    increment_seed : bool
        Whether to use an incrementing seed as opposed to random
    """
    # Run script args
    parser = argparse.ArgumentParser(description="Train or load a model")
    parser.add_argument(
        "--load_model",
        nargs="?",
        const=True,
        default=False,
        help="Optionally provide a model path. If omitted, uses default PPO save path."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model (if not specified, only loads/runs the model)"
    )
    args = parser.parse_args()

    # Load existing PPO model or train new one
    if args.load_model:
        # Use user-provided model if present (as string), otherwise use default stored model
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

    # Allow training existing pre-trained models
    if args.train or not args.load_model:
        # Number of policy updates
        num_updates = int(total_timesteps // num_steps)

        # Reset environment
        if increment_seed:
            seed = 1
            obs, _ = env.reset(seed)
        else:
            obs, _ = env.reset()

        episode_counter = 1

        # Iterate over a number of steps rather than episodes which may terminate early
        for update in range(num_updates):

            episode_return = 0

            for _ in range(num_steps):
                # Sample action from policy
                action, log_prob, value = agent.sample_action(obs)
                next_obs, reward, done, _, _ = env.step(action)

                # Store transition in agent's buffer
                agent.store_transition(obs, action, reward, done, log_prob, value)

                # Accumulate reward for logging
                episode_return += reward
                obs = next_obs

                # Reset env if episode ends
                if done:
                    if increment_seed:
                        seed += 1
                        obs, _ = env.reset(seed)
                    else:
                        obs, _ = env.reset()
                    if print_logs:
                        print(f"Episode {episode_counter} (update {update + 1}/{num_updates}) finished with return: {episode_return:.2f}")
                    episode_counter += 1
                    episode_return = 0

            # Bootstrap truncated episodes
            _, _, last_value = agent.sample_action(obs)
            agent.update_policy(last_value)

            # Anneal lr param for policy and value optimizers (decreases lr over time based on number of updates)
            lr_now = agent.lr * (1.0 - update / num_updates)
            for optimizer in [agent.policy_optimizer, agent.value_optimizer]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now
                
        if save_model:
            agent.save(MODEL_SAVE_PATH)


def run_PPO(agent: PPOAgent, env: CrazyflieEnv, env_seed: int = 42, num_steps: int = 2000):
    """
    Runs the trained PPO agent in the environment and returns total reward.

    Parameters
    ----------
    agent : PPOAgent
        The PPOAgent
    env : CrazyflieEnv
        Training environment
    env_seed : int
        Seed for the environment (applies on reset)
    num_steps : int
        Number of steps to run the episode for
    """
    obs, _ = env.reset(env_seed)
    env.max_steps = num_steps
    total_reward = 0.0
    done = False

    while not done:
        # Sample action from trained policy
        action, _, _ = agent.sample_action(obs, deterministic=True)

        # Step the environment
        obs, reward, done, _, _ = env.step(action)

        # Accumulate total reward
        total_reward += reward

    return total_reward


# Evaluation / Visualization - run learned policy in the sim
def render_PPO(agent: PPOAgent, env: CrazyflieEnv):
    """
    Runs the trained PPO agent in the environment and renders in MuJoCo

    Parameters
    ----------
    agent : PPOAgent
        The PPOAgent
    env : CrazyflieEnv
        Training environment
    """
    num_sim_steps = 2500
    total_reward = 0.0
    env.max_steps = num_sim_steps
    env.debug = True
    env.random_start_pos = False
    obs, _ = env.reset(42)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("\nRunning final visualization based on learned policy")
        for step in range(num_sim_steps):
            # Sample action from trained PPO policy
            # NOT RUNNING DETERMINISTICALLY FOR NOW - Drone seems to fly straight up
            action, _, _ = agent.sample_action(obs, deterministic=False)

            # Step the environment
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

            # Render with sleep for real-time visualization
            viewer.sync()
            time.sleep(1/60)

            if done:
                print(f"Simulation ended at step {step+1}/{num_sim_steps} with total reward {total_reward}")
                break

if __name__ == "__main__":

    # Training and evaluation env
    env = CrazyflieEnv(
        xml_path = SCENE_PATH,
        num_drones = 1,
        target_pos = np.array([0.0, 0.0, 0.5], dtype=np.float32),
        random_start_pos=False
    )
    agent = PPOAgent(
        obs_dim = env.observation_space.shape[0],
        action_dim = env.action_space.shape[0]
    )
    train_PPO(agent, env)
    render_PPO(agent, env)