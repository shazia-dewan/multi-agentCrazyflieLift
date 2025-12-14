# Explanation of `.../initial_cpu_training` folder

Early on in this project, we ran training, hyperparameter tests, etc... for a CPU-based environment `hover_target_env.py`. We later switched to GPU-based training (`.../gpu_training`) and stopped working on the CPU side of things. As such, the CPU side of things is still in a working state and does not reflect the latest environment and training changes in the GPU training notebook (e.g. environment observations and rewards). Since some of the files may still be helpful, they have not been deleted, and are explained here.

## Training & Evaluation

`run_MAPPO_policy.py`
- Train a MAPPO agent or load/visualize one with `--load_model`

`run_PPO_policy.py`
- Train a PPO agent or load/visualize one with `--load_model`

## Agents

`MAPPO_agent.py`
- Custom MAPPO implementation

`PPO_agent.py`
- Custom PPO implementation

`neural_network.py`
- Neural networks for the Actor and Critic

## Environment

`hover_target_env.py`
- Training environment, used for both PPO and MAPPO

## Utility

`run_logs_parser.py`
- Parse training logs to view reward x update graph

`run_manual_control.py`
- Run a MuJoCo simulation that receives manual keyboard input to inspect drone dynamics

`test_PPO_parameters.py`
- Train and evaluate an agent with varying hyperparameters

`test_trained_model.py`
- Test a trained model against varying targets


## Config

`constants.py`
- Constants file


# Detailed Setup and Example Training

Install dependencies with `pip install -r requirements.txt`
- Dependencies ran based on `Python 3.11.7`, other version may work
- If you would like to use an NVIDIA GPU for tensor computations (`--device cuda`), run the following:
  - `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall`

On Windows / Linux, running `<PYTHON_CMD> <RUN_FILE>.py` should work
- E.g. `python run_PPO_policy.py`
- On macOS, you can use `mjpython`
  - E.g `mjpython run_PPO_policy.py`
  - mjpython installs as part of MuJoCo and wraps your system python version

### Training

(Single-agent example): `run_PPO_policy_vec.py`
- Single agent PPO-Clip policy, trains across multiple parallel environments

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

## Environment and Training Resources

Simulation & Physics Engine: **[MuJoCo](https://mujoco.org/)**
- Handles all physical dynamics of the Crazyflie drone
- Provides 3D simulation including forces, torques, collisions, and drone kinematics
- Real-time visualization with the mujoco viewer

Environment Definition: **[Gymnasium API](https://gymnasium.farama.org/index.html)**
- Wraps the MuJoCo model into a reinforcement learning environment
  - Observation space (drone position, orientation, linear velocities, and angular velocities)
  - Action space (thrust and rotation)
  - Stepping: action --> simulation step --> calculate reward --> check termination

