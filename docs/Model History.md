# Step 1: Initial Testing 
### Trained on a single environment, basic position and velocity observation space torch.Size([64, 13])

**Model: z_hover_v1.pt**
- Goal: Hover at some [0, 0, Z]
- Somewhat successful, lots of oscillation
- Implemented with single-env PPO

**Model: z_hover_v2.pt**
- Same as v1, some parameter differences

# Step 2: Parallel Training Environments

**Model: z_hover_24env.pt**
- Goal: Hover at some [0, 0, Z]
- Somewhat successful, hovers perfectly at [0, 0, 0.5]
    - Need to investigate how to adapt for any Z (hovers at incorrect height or very elliptical)
- Implemented with multiple-env PPO (24 parallel envs during training)

# Step 3: Enhance Observation Space (Engineered Features)

**Model: z_hover_24env_v2.pt**
- Goal: Hover at some [0, 0, Z]
-  Quite successful, hovers well at most [0, 0, Z] (experimentally)
  - Some minor elliptical behaviour after stabilization (+/- 0.02m)
  - For high/low Z, hovers slightly below/above the target
- Implemented with multiple-env PPO (24 parallel envs during training)
- Added previous action, relative pos to target, Z pos error, and relative thrust (thrust / base hover thrust) to observation space: torch.Size([64, 23])
- Added training curriculum to environment to gradually increase random drone and target positions on env reset

**Model: z_hover_24env_v3.pt**
- Goal: Hover at some [0, 0, Z]
- Experimentally optimal within low Z [0.2, ~4] but hovers just short of high Z e.g. 5.5
- Implemented with multiple-env PPO (24 parallel envs during training)
  - Higher env max steps at 1200 (avoids early truncation, helps drone not overshoot target)
- Added more engineered features (pos error 1st & 2nd derivative, pos error integral, velocity features, more normalization, etc...)
- No curriculum

# Step 4: Modifying Features and Rewards for YZ Hover + GAE

**Model: yz_hover_8env_750k_step_gae.pt**
- Extending to [0, Y, Z]
  - Added rewards for roll movement
    - +/- for rolling towards/away from target Y position
    - +/- for rolling away from/towards from current Y velocity
  - Feature engineering for rotation, largely using the drone rotation matrix
  - Switch to GAE for critic advantage estimation (was using MC estimation before)
- Result
  - Works for very small Y range +/- 0.02 if on same Z, oversteers for anything greater


**Model: yz_hover_8env_900k_step_curriculum**
- Changes since last model
  - Added emphasis (reward) for rolling to counter angular velocity on roll axis
  - Re-added curriculum: gradually scales training ranges (e.g. starting pos) but also navigation rewards
    - Removed LR annealing so that later curriculum stages are not diminished by low step size
    - Motive: Learn stable YZ hovering, then emphasize navigation to target once stable later in the training
- Result: Stable Y rolling, but rolls past target on Y axis (also occasionally rolls the wrong way)
  - May need to train later curriculum portion (navigation) for longer

**Model: yz_hover_10_mujoco_steps**
- After struggling with roll (rotation in general) for a while, I realized the problem was that having 1 env step = 1 MuJoCo physics step leads to "invisible" cause and effect for rotating actions since roll/pitch/yaw take a longer time to propagate than thrust. Because of this, we now have 10 MuJoCo physics steps per 1 env step (agent takes action based on observation, take 10 physics steps in MuJoCo, report new observation back to agent)
  - Rewards are now purely based on state rather than action (e.g. no more rewarding rolling to counteract velocity)
    - Rewards for survival, moving towards target, and proximity to target
  - Results
    - Successful YZ hover, some oscillation / instability but the agent always orients towards and stays near the target
  - Agent was trained with 8 parallel envs and 600k timesteps

**Model: xyz_hover**
- yz_hover model extended to XYZ, still some oscillation but the agent has learned to hover at a target successfuly in 3D
  - 8 parallel envs, 600k timesteps, curriculum that progresses the range of random initial pos/rot/vel/ang_vel
- Next steps:
  - Test some other reward metrics for stability such as rewarding low velocity, rotation, etc...

**Model: xyz_hover**
- xyz_hover model extended to work in *any* XYZ
  - No curriculum to progress initial state ranges (e.g. pos), instead, sample from a distribution
  - Modify features to prefer body coordinates (e.g. instead of world pos_error, use body frame pos_error)
  - Trained with 24 parallel envs with 600k steps each
- Results
  - Works well experimentally (14 tests for varying target positions in XYZ within ~5m)
    - Not perfect (fails one out of 14 tests), perhaps due to insufficient training or unoptimized features, rewards...

# Step 5: Multi-Agent Coordination

**Model: multi_xyz_hover**
- xyz_hover with two drones using multi-agent PPO (MAPPO)
  - MAPPO: Centralized Critic (sees all drone states, computes one reward per step) Decentralized Execution (drone's only see their own state)
- Using separate agent (MAPPO agent) but same environment with minor changes
  - Adapted for MARL --> E.g. iterate over each drone, compute observations, action, per-drone rewards (later summed for total step reward)
  - New rewards/penalties
    - Penalty for proximity to other drones and penalty for collision with other drone
- Results (24 envs, 900k steps per env)
  - Semi-successful XYZ hover, tested against ~20 targets, eventually fails after some time on the vast majority
    - Drones have learned to hover around target while avoiding one another reasonable well, but still eventually crash/fail for most targets

**Model: multi_xyz_hover_40env**
- Same setup as multi_xyz_hover but with 40 parallel envs, 600k steps per env
  - Better results (same ~20 target testing, more truncation)
    - Getting successful multi-XYZ hover might just be a compute issue at the moment, will explore more
  - Model **multi_xyz_hover_40env_2m400k** also uses 40 envs but was trained with 2.4 mil steps per env
    - Better results, but still not ideal
