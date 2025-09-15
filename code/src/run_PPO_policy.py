import os
import time
import mujoco.viewer
from PPO_agent import PPOAgent
from hover_target_environment import CrazyflieEnv
import numpy as np
import argparse

if __name__ == "__main__":
    # Run script args
    parser = argparse.ArgumentParser(description="Train or load a model")
    parser.add_argument(
        "--load_model",
        nargs="?",
        const=True,
        default=False,
        help="Optionally provide a model path. If omitted, uses default PPO save path."
    )
    args = parser.parse_args()

    # Path to crazyflie scene
    scenePath = os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets", 
        "bitcraze_crazyflie_2", 
        "scene.xml"
    )

    # Path to save trained model
    save_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models", 
        f"ppo_model.pt"
    )

    # Training and evaluation env
    env = CrazyflieEnv(
        xml_path=scenePath,
        num_drones=1,
        target_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )

    agent = PPOAgent(
        obs_dim = env.observation_space.shape[0],
        action_dim = env.action_space.shape[0]
    )

    # Load existing PPO model or train new one
    if args.load_model:
        # Use user-provided model if present (as string), otherwise use default stored model
        if args.load_model is True:
            model_path = save_path 
        else:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "models", 
                f"{args.load_model}.pt"
            )
        print(f"Loading model from: {model_path}")
        agent.load(model_path)
    else:
        # Number of steps before a policy update (can span multiple episodes)
        total_timesteps = 150_000
        num_steps = 1500
        num_updates = int(total_timesteps // num_steps)

        # Reset environment
        obs, _ = env.reset()

        update_counter = 0
        episode_counter = 1

        # Iterate over a number of steps rather than episodes which may terminate early
        for update in range(num_updates):

            update_counter += 1
            episode_return = 0

            for step in range(num_steps):
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
                    obs, _ = env.reset()
                    print(f"Episode {episode_counter} (update {update_counter}/{num_updates}) finished with return: {episode_return:.2f}")
                    episode_counter += 1
                    episode_return = 0

            # Update policy/value networks after every rollout of num_steps
            agent.update_policy()

            # Anneal lr param for policy and value optimizers (decreases lr over time based on number of updates)
            lr_now = agent.lr * (1.0 - update / num_updates)
            for optimizer in [agent.policy_optimizer, agent.value_optimizer]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now
                
        agent.save(save_path)


    # Evaluation / Visualization - run learned policy in the sim
    obs, _ = env.reset()
    num_sim_steps = 5000
    env.max_steps = num_sim_steps
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        print("\nRunning final visualization based on learned policy")
        for step in range(num_sim_steps):
            # Sample action from trained PPO policy
            action, _, _ = agent.sample_action(obs)

            # Step the environment
            obs, reward, done, _, _ = env.step(action, log_info = True)

            # Render with sleep for real-time visualization
            viewer.sync()
            time.sleep(1/60)

            if done:
                print(f"Simulation ended at step {step+1}/{num_sim_steps}")
                break
