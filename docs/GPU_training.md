# Using Jax for speedy reinforcement learning

Motivation: After developing our environment and training schedule, we ran into time and compute constraints
- Could not run more than ~40 envs on our computer, which for many timesteps took hours

Solution: Use Jax (via MuJoCo Playground environments) to speed things up
- Run in Google collab to get around Jax GPU OS constraints (Jax GPU is meant to run on Linux)
- **NOTE**: Work on the MAPPO side of this project was continued in the notebook, elements of the non-jax MAPPO agent and training code in this repository may not reflect the final approach that we took.

# Background

Jax compiles Python code into optimized machine code (usually via XLA)
- Traces Python functions, builds computation graph, generates machine code for device (GPU in this case)
- Jax can then reuse compiled code, which (combined with GPU speed), makes computations very fast

Since Jax/JIT traces operations and builds a computation graph, we want to keep things relatively simple to avoid long computations
- Created a new Crazyflie XMl without all the collision geoms to speed things up
  - Previously, the physics integrator had to consider all collisions even if we never actually collided with anything
  - We weren't using these geoms anyway, as we end training episodes if drone(s) get to close to the ground or one another ("crash" but no collision)

# Google Collab Notebook

See notebook in `.../gpu_training`, it builds a MuJoCo playground environment and provides options to run single agent PPO with Brax or 2-drone MAPPO with a custom implementation. Both methods can be run with 1 or 2 drones but 2-drone PPO is essentially just single-agent RL with a centralized "dispatcher" controlling both drones, and 1-drone MAPPO reduces to PPO. The notebook relies on the MuJoCo 1 and/or 2 drone scenes, which can also be found under the `.../gpu_training` folder `assets_mjx`. This folder can be uploaded to Google drive which can then be mounted in the notebook (avoids re-uploading each time) or the folder can compressed to a zip folder and uploaded manually in the notebook, both options are available and outlined.

# Resources

[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) with example [Cartpole Balance RL Notebook](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)

[Jax](https://github.com/jax-ml/jax), [Flax (Jax neural networks)](https://flax.readthedocs.io/en/v0.6.11/index.html), and [Brax (Jax single-agent training)](https://github.com/google/brax)


<hr>

## More Context: Vectorizing multiple training environments

Instead of training PPO on a single simulation, we run multiple instances of the environment in parallel and treat them like a batch. The same concept applied later to our Jax notebook on a much larger scale.

We discovered this early on in CPU training and applied it using the SB3 vectorized wrapper (but later transitioned away from this into our Jax notebook). The initial vectorization methodology and implementation is described below.

## Methodology

Each env runs independently (different seeds) and at every step PPO collects a batch of experience from all envs at once.
- Speed (for N envs, N steps per clock cycle vs 1, depends on #CPU cores tho)
    - E.g. ~8x speed on an 8 core CPU (can/should use >8 envs tho, just means more context-switching)
- Stability (more samples across diverse envs, results are less variant)
- Exploration (each env explores their own state space)

## Implementation

Wrap existing environment in [SB3 Vectorized Wrapper](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)
- Essentially pass multiple instances of the env with different seeds to the wrapper

Run standard PPO policy, this time using batches of observations & actions across all envs
- Pass vectorized environments to PPO (rollout buffer now stores info across all envs)
    - Observations now have shape (num_envs, obs_dim)
    - Actions now have shape (num_envs, action_dim)
    - Returns, advantages, log-probs, and values flatten to (1600,)
        - (1600,) = 1D tensor w/ 1600 elements
    - Everything gets batched: The policy collects batched observations from all envs and returns batched actions
- Rollout buffer has (num_envs * num_steps) transitions
    - E.g. transitions are indexed by time steps
    - PPO optimizes all transitions from all envs across the rollout horizon

Example: 200 steps × 8 envs = 1600 transitions
- All 1600 transitions are merged into one big batch in the rollout buffer
- Transitions are grouped by timestep: [env0_t0, env1_t0, env2_t0..., env0_t1...]
    - Observations flatten to (1600, obs_dim)
    - Actions flatten to (1600, action_dim)
    - Returns, advantages, log-probs, and values flatten to (1600,)
        - (1600,) = 1D tensor w/ 1600 elements
- Once flattened, PPO no longer distinguishes which env each transition came from
  - Then applies standard PPO steps to the batch e.g. shuffling and minibatching
- Aside from batching, everything else is the same as single-env PPO.