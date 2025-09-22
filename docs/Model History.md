# Step 1: Initial Testing 
### Trained on a single environment, basic position and velocity observation space torch.Size([64, 13])

**Model: ppo_model_hover_v1.pt**
- Goal: Hover at some [0, 0, Z]
- Somewhat successful, lots of oscillation
- Implemented with single-env PPO

**Model: ppo_model_hover_v2.pt**
- Same as v1, some parameter differences

# Step 2: Parallel Training Environments

**Model: ppo_model_hover_env24.pt**
- Goal: Hover at some [0, 0, Z]
- Somewhat successful, hovers perfectly at [0, 0, 0.5]
    - Need to investigate how to adapt for any Z (hovers at incorrect height or very elliptical)
- Implemented with multiple-env PPO (24 parallel envs during training)

# Step 3: Enhance Observation Space (Engineered Features)

**Model: ppo_model_hover_env24_v2.pt**
- Goal: Hover at some [0, 0, Z]
-  Quite successful, hovers well at most [0, 0, Z] (experimentally)
  - Some minor elliptical behaviour after stabilization (+/- 0.02m)
  - For high/low Z, hovers slightly below/above the target
- Implemented with multiple-env PPO (24 parallel envs during training)
- Added previous action, relative pos to target, Z pos error, and relative thrust (thrust / base hover thrust) to observation space: torch.Size([64, 23])
- Added training curriculum to environment to gradually increase random drone and target positions on env reset

**Model: ppo_model_hover_env24_v3.pt**
- Goal: Hover at some [0, 0, Z]
- Experimentally optimal within low Z [0.2, ~4] but hovers just short of high Z e.g. 5.5
- Implemented with multiple-env PPO (24 parallel envs during training)
  - Higher env max steps at 1200 (avoids early truncation, helps drone not overshoot target)
- Added more engineered features (pos error 1st & 2nd derivative, pos error integral, velocity features, more normalization, etc...)
- No curriculum
