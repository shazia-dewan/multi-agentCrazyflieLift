import os
import re
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_log_file(path):
    # Regex to capture Pos=[x y z]
    pos_pattern = re.compile(r"Pos=\[([^\]]+)\]")

    xs, ys, zs = [], [], []

    with open(path, "r") as f:
        for line in f:
            match = pos_pattern.search(line)
            if match:
                # Extract numbers inside Pos=[ ... ]
                nums = match.group(1).strip().split()
                # Convert sci-notation strings to floats
                if len(nums) == 3:
                    x, y, z = map(float, nums)
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

    return xs, ys, zs


def plot_positions(xs, ys, zs):
    steps = list(range(len(xs)))

    # 3D trajectory plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs)
    ax.set_title("Drone 3D Position Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)

    # Separate XYZ vs step plot
    fig2 = plt.figure(figsize=(10, 6))
    plt.plot(steps, xs, label="X")
    plt.plot(steps, ys, label="Y")
    plt.plot(steps, zs, label="Z")
    plt.title("Drone Position Over Time")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    log_path = os.path.join(
        os.path.dirname(__file__),
        "drone_obs.log"
    )
    xs, ys, zs = parse_log_file(log_path)
    
    if not xs:
        print("No position data found! Check log formatting.")
        sys.exit(1)

    plot_positions(xs, ys, zs)
