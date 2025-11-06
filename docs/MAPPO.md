# MAPPO: Centralized Trainined, Decentralized Execution CTDE
Each agent (drone) learns its own policy based on its own observations (does not see other drone state(s)).
  - All drones share a value function (critic) that sees the joint state of all drones

We are currently using the shared variant version of MAPPO - one policy for all drones, but pass each drones observations sequentially and get the corresponding action per drone (decentralized).
  - Applicable for our use case (hovering at a target without colliding) since all drones are of the same type and share the same role (every observation --> action is applicable training data for each drone --> one policy)