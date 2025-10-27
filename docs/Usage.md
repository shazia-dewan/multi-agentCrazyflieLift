# Setup

Install dependencies with `pip install -r requirements.txt`
- Dependencies ran based on `Python 3.11.7`, other version may work
- If you would like to use an NVIDIA GPU for tensor computations (`--device cuda`), run the following:
  - `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall`

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

### `run_PPO_policy_vec.py`

Single agent PPO-Clip policy, trains across multiple parallel environments

Run options (all optional with default values)
- `--load_model <MODEL_NAME>`
  - Instead of training a new model, load an existing one
  - If no model name is provided, a default name will be used e.g. `ppo_hover`
- `--num_envs <NUM_ENVS>`
  - Number of parallel environments
- `--total_timesteps <TOTAL_STEPS>`
  - Total training time steps
- `--num_steps <NUM_STEPS>`
  - Time steps before a policy update
- `--device <DEVICE>`
  - Device used for tensor computations in the vectorized PPO agent (e.g. cpu, cuda, mps)

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
