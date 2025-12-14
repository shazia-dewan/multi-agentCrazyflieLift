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

Attached in the repo as `MARL.ipynb`, builds a MuJoCo playground environment and provides options to run single agent PPO with Brax or 2-drone MAPPO with a custom implementation. Both methods can be run with 1/2 drones but 2-drone PPO is essentially just single-agent RL with a centralized "dispatcher" controlling both drones, and 1-drone MAPPO reduces to PPO.

# Resources

[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) with example [Cartpole Balance RL Notebook](https://colab.research.google.com/github/google-deepmind/mujoco_playground/blob/main/learning/notebooks/dm_control_suite.ipynb)

[Brax](https://github.com/google/brax)
- Brax's PPO implementation was referenced in the construction of our custom Jax MAPPO implementation


[Jax](https://github.com/jax-ml/jax) and [Flax (Jax neural networks)](https://flax.readthedocs.io/en/v0.6.11/index.html)