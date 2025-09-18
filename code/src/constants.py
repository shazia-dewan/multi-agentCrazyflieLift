import os

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
    "models", 
    f"ppo_model.pt"
)