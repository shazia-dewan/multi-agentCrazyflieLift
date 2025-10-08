import argparse
import time
import matplotlib.pyplot as plt
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

from PID_controller import NeutralHoverController
from constants import SCENE_PATH
from PID_test_env import PIDTestEnv

# Run the PID in the test env, track oscillation of roll/pitch/yaw via history
def test_hover(env: PIDTestEnv, controller: NeutralHoverController, steps: int=10000, render: bool=False):
    env.reset()
    t_hist, roll_hist, pitch_hist, yaw_hist = [], [], [], []

    for step_i in range(steps):
        pos = env.state[0][0 : 3]
        quat = env.state[0][3 : 7]
        vel = env.state[0][7 : 10]
        ang_vel = env.state[0][10 : 13]

        quat_xyzw = np.roll(quat, -1)
        # roll/pitch/yaw = euler
        rpy = R.from_quat(quat_xyzw).as_euler('xyz')
            
        # Sample the controller action
        cmd = controller.update(pos, vel, rpy, ang_vel)
        action = np.array([cmd["thrust"], cmd["roll"], cmd["pitch"], cmd["yaw"]], dtype=np.float32)
        _, _, done, _, _ = env.step(action)

        roll_hist.append(rpy[0])
        pitch_hist.append(rpy[1])
        yaw_hist.append(rpy[2])
        t_hist.append(step_i / steps)

        if render and viewer is not None:
            viewer.sync()
            time.sleep(1 / 1000)

        if done:
            break

    return np.array(t_hist), np.array(roll_hist), np.array(pitch_hist), np.array(yaw_hist)


def plot_oscillations(t_hist, roll_hist, pitch_hist, yaw_hist, title="Attitude Oscillations"):
    """
    Plot rotation control history to inspect oscillation (goal is to reduce oscillation and maintain steadiness)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t_hist, np.degrees(roll_hist), label="Roll (deg)")
    plt.plot(t_hist, np.degrees(pitch_hist), label="Pitch (deg)")
    plt.plot(t_hist, np.degrees(yaw_hist), label="Yaw (deg)")
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PID K values")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the PID control test, requires script to run via mjpython on macOS"
    )
    args = parser.parse_args()


    # Env randomly initializes a rotation axis with a rotation to test PID return to stability
    # Can set specific starting rotation and rotation axis in PID_test_env.py
    env = PIDTestEnv(
        xml_path=SCENE_PATH,
        num_drones=1,
        debug=True
    )
    controller = NeutralHoverController()

    print("Running hover test for gain visualization...")

    if args.render:
        with mujoco.viewer.launch_passive(env.mujoco_scene, env.data) as viewer:
            test_hover(env, controller, render=True)
    else:
        timestep_history, roll_history, pitch_history, yaw_history = test_hover(env, controller, render=False)
        plot_oscillations(timestep_history, roll_history, pitch_history, yaw_history)