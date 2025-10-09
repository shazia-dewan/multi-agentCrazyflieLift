import os
import csv
import argparse
from typing import Any, Callable, Union
import torch

from constants import SCENE_PATH
from PPO_vec_agent import PPOAgentVec
from code.src.hover_target_env_velocity import CrazyflieEnv
from run_PPO_vec_policy import train_PPO, make_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# Result directory for output files
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_results"
)

def save_results(results: list[dict[str, Any]], filename: str) -> None:
    """
    Save experiment results to a CSV file.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of results dictionaries
    filename : str
        Results file name
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
        f.write("\n")


def evaluate(agent: PPOAgentVec, env_fn: Callable[[], CrazyflieEnv], num_runs: int = 5):
    """
    Runs deterministic (mean-greedy) PPO policy

    Parameters
    -----------
    agent : PPOAgentVec
        The PPO agent
    env_fn : Callable[[], CrazyflieEnv]
        The environment factory function (creates an environment)
    num_runs : int
        The number of test runs under this policy

    Returns
    -------
    Average return (total return / number of runs)
    """
    total_reward = 0
    for i in range(num_runs):
        env = env_fn()
        obs, _ = env.reset(seed=i)
        done = False
        ep_reward = 0
        while not done:
            action, _, _ = agent.sample_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_reward += ep_reward
    return total_reward / num_runs


def check_hyperparameter(args, param_name: str, values: list[Union[int, float]]) -> None:
    """
    Run PPO against a specific hyperparameter e.g. num_minibatches

    Parameters
    ----------
    args : CLI args
        Used to specific the number of trial runs for each hyperparameter-trained model

    param_name : str
        The PPO agent hyperparameter name

    values : list[Union[int, float]]
        The list of values to test for the selected hyperparameter
    """
    print(f"\nTesting PPO with different {param_name} values")
    results = []

    for value in values:
        # Parallel envs for training
        env_fns = [make_env(SCENE_PATH, rank=i, seed=42) for i in range(args.num_envs)]
        envs = SubprocVecEnv(env_fns) if args.num_envs > 1 else DummyVecEnv(env_fns)

        # Initialize an agent for the current hyperparameter value
        example_env = env_fns[0]()
        agent_kwargs = dict(
            obs_dim=example_env.observation_space.shape[0],
            action_dim=example_env.action_space.shape[0],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        agent_kwargs[param_name] = value
        agent = PPOAgentVec(**agent_kwargs)

        print(f"Training with {param_name} = {value}...")
        train_PPO(
            agent, 
            envs, 
            total_steps=args.total_steps,
            update_steps=args.update_steps,
            print_logs=False,
            save_model=False
        )

        avg_reward = evaluate(agent, env_fns[0], num_runs=args.num_runs)
        results.append({param_name: value, "average_reward": avg_reward})

    save_results(results, f"vecPPO_{param_name}_results.csv")



def check_env_max_steps(args, max_step_values: list[int]) -> None:
    """
    Test PPO agent performance with different max_steps values in the environment.

    args : CLI args
        Used to specific the number of trial runs for each hyperparameter-trained model

    step_values : list[int]
        The list of values to test for max_steps
    """
    print("\nTesting PPO with different environment max_steps")
    results = []

    for steps in max_step_values:
        # Pass current max_steps value to env factory function
        env_fns = [make_env(SCENE_PATH, rank=i, seed=42, max_steps=steps) for i in range(args.num_envs)]
        envs = SubprocVecEnv(env_fns) if args.num_envs > 1 else DummyVecEnv(env_fns)

        example_env = env_fns[0]()
        agent = PPOAgentVec(
            obs_dim=example_env.observation_space.shape[0],
            action_dim=example_env.action_space.shape[0],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Training with max_steps = {steps}...")
        train_PPO(
            agent,
            envs,
            total_steps=args.total_steps,
            update_steps=args.update_steps,
            print_logs=False,
            save_model=False
        )

        avg_reward = evaluate(agent, env_fns[0], num_runs=args.num_runs)
        results.append({"max_steps": steps, "average_reward": avg_reward})

    save_results(results, "vecPPO_env_steps_results.csv")

def check_training_schedule(args, schedule_values: list[tuple[int, int]]) -> None:
    """
    Test PPO agent performance with different training schedules.

    args : CLI args
        Used to specific the number of trial runs for each hyperparameter-trained model

    schedule_values : list[tuple[int, int]]
        Pairs of (total_steps, update_steps)
    """
    print("\nTesting PPO with different training schedules")
    results = []

    for total_steps, update_steps in schedule_values:
        env_fns = [make_env(SCENE_PATH, rank=i, seed=42) for i in range(args.num_envs)]
        envs = SubprocVecEnv(env_fns) if args.num_envs > 1 else DummyVecEnv(env_fns)

        example_env = env_fns[0]()
        agent = PPOAgentVec(
            obs_dim=example_env.observation_space.shape[0],
            action_dim=example_env.action_space.shape[0],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Training with total_steps={total_steps}, update_steps={update_steps}...")
        train_PPO(
            agent,
            envs,
            total_steps=total_steps,
            update_steps=update_steps,
            print_logs=False,
            save_model=False
        )

        avg_reward = evaluate(agent, env_fns[0], num_runs=args.num_runs)
        results.append({
            "total_steps": total_steps,
            "update_steps": update_steps,
            "average_reward": avg_reward
        })

    save_results(results, "vecPPO_training_schedule_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vectorized PPO with varying parameters")
    parser.add_argument("--num_runs", type=int, default=20, help="Number of evaluation runs per trained model")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs during training")
    parser.add_argument("--total_steps", type=int, default=150_000, help="Default total training timesteps")
    parser.add_argument("--update_steps", type=int, default=1500, help="Default rollout steps per update")
    args = parser.parse_args()

    # Hyperparameter variations
    # check_hyperparameter(args, "lr", [3e-3, 1e-3, 7e-4, 5e-4])
    # check_hyperparameter(args, "gamma", [0.98, 0.99, 0.995])
    # check_hyperparameter(args, "clip_eps", [0.15, 0.2, 0.25])
    # check_hyperparameter(args, "update_epochs", [1, 2, 4, 8])
    # check_hyperparameter(args, "num_minibatches", [1, 2, 4, 8])
    # check_hyperparameter(args, "entropy_coefficient", [0.01, 0.02, 0.03, 0.04])
    # check_hyperparameter(args, "kl_threshold", [0.01, 0.02, 0.3, 0.04])

    # Environment variations
    # check_env_max_steps(args, [1500, 2000])

    # Training schedule
    # check_training_schedule(args, [(50_000, 200), (50_000, 400)])
