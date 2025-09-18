### 1. ppo_model_hover_v1.pt
- Hover at some [0, 0, Z]
- Somewhat successful, lots of oscillation
- Implemented with single-env PPO

### 2. ppo_model_hover_v2.pt
- Same as v1, some parameter differences

### 3. ppo_model_hover_env24.pt
- Hover at some [0, 0, Z]
- Somewhat successful, hovers perfectly at [0, 0, 0.5]
    - Need to investigate how to adapt for any Z
- Implemented with multiple-env (parrallel) PPO (24 parallel envs)
