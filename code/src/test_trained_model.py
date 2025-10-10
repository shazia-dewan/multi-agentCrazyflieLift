# Run a trained model against a series of target positions and track returns

import argparse
import os
import numpy as np
from tabulate import tabulate

from constants import SCENE_PATH
from hover_target_env_RL import CrazyflieEnv
from PPO_agent import PPOAgentVec

def evaluate(agent: PPOAgentVec, env: CrazyflieEnv) -> float:
    """
    Runs a deterministic (mean-greedy) PPO policy on a single environment instance.

    Parameters
    ----------
    agent : PPOAgentVec
        The PPO agent
    env_fn : Callable[[], CrazyflieEnv]
        A callable returning a CrazyflieEnv instance

    Returns
    -------
    float
        Total episode return
    """
    obs, _ = env.reset(seed=42)
    done = False
    ep_reward = 0.0

    while not done:
        action, _, _ = agent.sample_action(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_reward += reward

    return ep_reward


def main():
    parser = argparse.ArgumentParser(description="Train or render PPO")
    parser.add_argument(
        "--load_model",
        nargs="?",
        const=True,
        default=False,
        help="Optionally provide a model path. If omitted, uses default PPO save path."
    )
    args = parser.parse_args()

    if not args.load_model:
        print("No model provided, attempting default model 'ppo_model.pt'.")
        model_name = "ppo_model"
    else:
        model_name = args.load_model
    
    env = CrazyflieEnv(
        xml_path=SCENE_PATH,
        num_drones=1,
        max_steps=1500,
        random_initialization=False,
        debug=True
    )
    agent = PPOAgentVec(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "rl_models", 
        f"{model_name}.pt"
    )
    agent.load(model_path)

    target_positions = [
        # Z hover
        np.array([0.0, 0.0, 0.2], dtype=np.float32),
        np.array([0.0, 0.0, 0.8], dtype=np.float32),

        # XZ hover
        np.array([1.0, 0.0, 0.5], dtype=np.float32),
        np.array([-1.0, 0.0, 0.5], dtype=np.float32),

        # YZ hover
        np.array([0.0, 1.0, 0.5], dtype=np.float32),
        np.array([0.0, -1.0, 0.5], dtype=np.float32),

        # XYZ hover close
        np.array([0.2, 0.2, 0.2], dtype=np.float32),
        np.array([1.0, 1.0, 1.0], dtype=np.float32),

        # XYZ hover far
        np.array([3.0, 3.0, 0.2], dtype=np.float32),
        np.array([5.0, 5.0, 3.0], dtype=np.float32),
        np.array([-1.5, 1.5, 1.5], dtype=np.float32),
        np.array([7.0, -7.0, 2.5], dtype=np.float32),
    ]

    results = []
    for i, target in enumerate(target_positions):
        env.target_pos = target
        episode_return = evaluate(agent, env)
        results.append((i, target.tolist(), episode_return))

    avg_return = np.mean([r[2] for r in results])

    table = [
        (idx, str([round(p, 1) for p in pos]), f"{ret:.2f}")
        for idx, pos, ret in results
    ]
    print("\nHover Evaluation Results")
    print(tabulate(table, headers=["Target #", "Target Position [x, y, z]", "Return"]))

if __name__ == "__main__":
    main()
