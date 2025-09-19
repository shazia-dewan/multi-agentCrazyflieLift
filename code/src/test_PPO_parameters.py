import csv
import pandas as pd
import os
from typing import Any, Union
import numpy as np
import argparse

from PPO_agent import PPOAgent
from run_PPO_policy import train_PPO, run_PPO
from hover_target_environment import CrazyflieEnv
from constants import SCENE_PATH

# Result files
HYPERPARAMETERS_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_results",
    "PPO_hyperparameter_results.csv"
)
TRAINING_STEPS_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_results",
    "PPO_training_steps_results.csv"
)
ENV_STEPS_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "test_results",
    "PPO_env_steps_results.csv"
)


# Training and evaluation env
env = CrazyflieEnv(
    xml_path = SCENE_PATH,
    num_drones = 1,
    target_pos = np.array([0.0, 0.0, 0.7], dtype=np.float32)
)


def check_hyperparameter(args, param_name: str, value_range: list[Union[int, float]]) -> None:
    """
    Run PPO against specific hyperparameter(s) e.g. num_minibatches

    Parameters
    ----------
    args : CLI args
        Used to specific the number of trial runs for each hyperparameter-trained model

    param_name : str
        The hyperparameter name

    value_range : list[Union[int, float]]
        The list of values to test for the selected hyperparameter
    """
    print(f"\n\nTesting PPO with different {param_name} values\n")
    results = []

    for value in value_range:
        agent_kwargs = {
            "obs_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
        }

        # Set the current hyperparameter
        agent_kwargs[param_name] = value

        # Keep other defaults (these could be customized if needed)
        agent = PPOAgent(**agent_kwargs)

        print(f"Training and testing with {param_name} = {value}...")
        train_PPO(
            agent, 
            env=env, 
            print_logs=False, 
            save_model=False, 
            increment_seed=True
        )

        total_reward = 0
        for i in range(args.num_runs):
            # Run tests against the same env (use i as seed for each hyperparameter)
            total_reward += run_PPO(agent, env, i)
        
        avg_reward = total_reward / args.num_runs
        results.append({param_name: value, "average_reward": avg_reward})

    print(f"Average rewards for {param_name} over {args.num_runs} runs")
    for entry in results:
        print(f"{param_name}={entry[param_name]}: {entry['average_reward']}")
    
    save_results(results, HYPERPARAMETERS_CSV, args.num_runs)



def check_env_max_steps(args, value_range: list[int]) -> None:
    """
    Test PPO agent performance with different max_steps values in the environment.

    args : CLI args
        Used to specific the number of trial runs for each hyperparameter-trained model

    value_range : list[int]
        The list of values to test for the selected max_steps
    """
    print(f"\nTesting PPO with different max_steps values\n")
    results = []

    for steps in value_range:
        # Create a new environment each time with a different max_steps
        test_env = CrazyflieEnv(
            xml_path=SCENE_PATH,
            num_drones=1,
            target_pos=np.array([0.0, 0.0, 0.7], dtype=np.float32),
            max_steps=steps
        )

        agent = PPOAgent(
            obs_dim=test_env.observation_space.shape[0],
            action_dim=test_env.action_space.shape[0]
        )

        print(f"Training and testing with max_steps={steps}...")
        train_PPO(
            agent, 
            env=env, 
            print_logs=False, 
            save_model=False, 
            increment_seed=True
        )

        total_reward = 0
        for i in range(args.num_runs):
            # Run tests against the same env (use i as seed for each max_steps value)
            total_reward += run_PPO(agent, env, i)

        avg_reward = total_reward / args.num_runs
        results.append({"max_steps": steps, "average_reward": avg_reward})

    print(f"Average rewards for max_steps over {args.num_runs} runs")
    for entry in results:
        print(f"max_steps={entry['max_steps']}: {entry['average_reward']}")
    
    save_results(results, ENV_STEPS_CSV, args.num_runs)


def check_training_schedule(args, schedule_values: list[tuple[int, int]]) -> None:
    """
    Test PPO agent performance with different training schedules.

    args : CLI args
        Used to specific the number of trial runs for each hyperparameter-trained model

    schedule_values : list[tuple[int, int]]
        Pairs of (total_timesteps, num_steps)
    """
    print(f"\nTesting PPO with different training schedules (total_timesteps, num_steps)\n")
    results = []

    for total_timesteps, num_steps in schedule_values:
        agent = PPOAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )

        print(f"Training with total_timesteps={total_timesteps}, num_steps={num_steps}...")
        train_PPO(
            agent, 
            env=env, 
            print_logs=False, 
            save_model=False, 
            total_timesteps=total_timesteps, 
            num_steps=num_steps,
            increment_seed=True
        )

        total_reward = 0
        for i in range(args.num_runs):
            # Run tests against the same env (use i as seed for each (total_timesteps, num_steps))
            total_reward += run_PPO(agent, env, i)

        avg_reward = total_reward / args.num_runs
        results.append({
            "total_timesteps": total_timesteps,
            "num_steps": num_steps,
            "average_reward": avg_reward
        })

    print(f"Average rewards for training schedules over {args.num_runs} runs")
    for entry in results:
        print(
            f"total_timesteps={entry['total_timesteps']}, "
            f"num_steps={entry['num_steps']}: "
            f"{entry['average_reward']}"
        )

    save_results(results, TRAINING_STEPS_CSV, args.num_runs)



def save_results(results: list[dict[str, Any]], path: str, num_runs: int) -> None:
    """
    Save experiment results to a CSV file.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of results dictionaries.
    path : str
        Path to save file
    num_runs : int
        Number of runs (for computing average reward)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    for r in results:
        r["num_runs"] = num_runs

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
        f.write("\n")


# Check various env and/or policy parameters across different values
# Methodology: For parameter p and value v, train with p = v on random environments (seeds)
#   Then run N trial runs across envs with specific seeds and take average reward
#   Repeat for different values, use same environment (and NN) seeds
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO with varying parameters.")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs per parameter option")
    args = parser.parse_args()

    # Hyperparameter value ranges for testing
    lr_values = [7e-4, 9e-4, 1e-3, 3e-3]
    gamma_values = [0.95, 0.97, 0.99, 0.995]
    clip_eps_values = [0.1, 0.15, 0.2]
    update_epochs_values = [1, 2, 4]
    num_minibatches_values = [1, 2, 4]
    entropy_coefficient_values = [0.2, 0.3, 0.5, 0.8]
    kl_threshold_values = [0.2, 0.3, 0.4, 0.5]

    check_hyperparameter(args, "lr", lr_values)
    # check_hyperparameter(args, "gamma", gamma_values)
    check_hyperparameter(args, "clip_eps", clip_eps_values)
    check_hyperparameter(args, "update_epochs", update_epochs_values)
    check_hyperparameter(args, "num_minibatches", num_minibatches_values)
    check_hyperparameter(args, "entropy_coefficient", entropy_coefficient_values)
    check_hyperparameter(args, "kl_threshold", kl_threshold_values)

    # # Environment hyperparameters
    # max_steps_values = [100, 200, 300, 400]
    # check_env_max_steps(args, max_steps_values)

    # # Training schedule values
    # schedule_values = [
    #     (50_000, 500),
    #     (100_000, 1000),
    #     (150_000, 1500),
    #     (200_000, 2000),
    # ]
    # check_training_schedule(args, schedule_values)