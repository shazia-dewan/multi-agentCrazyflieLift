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
