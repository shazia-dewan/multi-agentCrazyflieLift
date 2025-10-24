# MAPPO: Centralized Trainined, Decentralized Execution CTDE
Each agent (drone) learns its own policy based on its own observations (does not see other drone state(s)).
  - All drones share a value function (critic) that sees the joint state of all drones