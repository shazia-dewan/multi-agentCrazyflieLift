import os
import numpy as np

SCENE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets", 
    "bitcraze_crazyflie_2", 
    "scene.xml"
)

MULTI_SCENE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets", 
    "bitcraze_crazyflie_2", 
    "scene_multi.xml"
)

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "rl_models", 
    "ppo_model.pt"
)

MAPPO_MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "rl_models", 
    "mappo_model.pt"
)

LOG_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "run_output", 
    "run_logs_PPO_training.log"
)