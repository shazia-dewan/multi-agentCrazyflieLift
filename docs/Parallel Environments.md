# Optimization: Vectorize multiple training environments

Instead of training PPO on a single simulation, we run multiple copies of the environment in parallel and treat them like a batch.

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