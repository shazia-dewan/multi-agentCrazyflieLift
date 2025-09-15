# Setup

Install dependencies with `pip install -r requirements.txt`

On Windows / Linux, running `<PYTHON_CMD> <RUN_FILE>.py` should work
- E.g. `python run_PPO_policy.py`
- On macOS, you can use `mjpython`
  - E.g `mjpython run_PPO_policy.py`
  - mjpython installs as part of MuJoCo and wraps your system python version

# Environment files

### `hover_target_environment.py`

Training environment to learn how to hover at a specified point
- Reward is based on (previous distance to target - current distance to target)
  - E.g. reward moving closer to the target

# Training + Evaluation files

Generated models are stored under `.../models`
- Outputs a model with a default name e.g. `ppo_model.pt` after each training + evaluation run
- Rename / move / copy the output model if you would like to retain it (so it is not overwritten on the next run)

### `run_PPO_policy.py`

Single agent PPO-Clip policy

Run options
- `--load_model <MODEL_NAME>`
  - Instead of training a new model, load an existing one
  - If no model name is provided, a default name will be used e.g. `ppo_hover`
  - Example: `mjpython run_PPO_policy.py --load_model ppo_model_hover_v1`

# Environment and Training

Simulation & Physics Engine: **[MuJoCo](https://mujoco.org/)**
- Handles all physical dynamics of the Crazyflie drone
- Provides 3D simulation including forces, torques, collisions, and drone kinematics
- Real-time visualization with the mujoco viewer

Environment Definition: **[Gymnasium API](https://gymnasium.farama.org/index.html)**
- Wraps the MuJoCo model into a reinforcement learning environment
  - Observation space (drone position, orientation, linear velocities, and angular velocities)
  - Action space (thrust and rotation)
  - Stepping: action --> simulation step --> calculate reward --> check termination
