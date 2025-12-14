# Proximal Policy Optimization (PPO)

For our single-agent testing, we use PPO (actor-critic, gaussian).
- Actor (policy) network
  - Receives observations (positions, velocity...) as input
  - Learns to output a continuous action distribution (μ, σ) for [thrust, roll, pitch, yaw]
    - If we are acting greedily (evaluation) the action is selected as μ
- Critic (state estimator)
  - Receives observations / state as input
  - Learns to output a scalar value for the current state, used to adjust the actor's policy

# Multi-Agent Proximal Policy Optimization (MAPPO)

Each agent shares the same policy but acts independently
- Actor (policy) network
  - Receives each drone's observation individually and outputs an individual action for each drone
    - E.g. For drone A and B, policy receives obs(A) and outputs action(A), repeat for B
- Critic (state estimator)
  - Receives the observations of all drones and outputs a scalar value for the global state
    - This global value is used to adjust the policy, rather than a per-drone state value
  - This is a "centralized" critic
    - I believe some MAPPO variations use a per-drone state estimate rather than a global one