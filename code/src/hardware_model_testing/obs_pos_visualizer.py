import os
import re
import matplotlib.pyplot as plt
import argparse

# Plot hardware run positional data from log file

import re

def parse_log_file(path, pattern_key, num_drones):
    """
    Returns a dict mapping drone index -> (xs, ys, zs)
    """
    # Initialize storage per drone
    data = {i: ([], [], []) for i in range(1, num_drones+1)}

    # Regex: match 'Step <num>, Drone <num>, Pos=[...]'
    pattern = re.compile(
        rf"Step \d+, Drone (\d+), {re.escape(pattern_key)}=\[([^\]]+)\]"
    )

    with open(path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                drone_idx = int(match.group(1))
                nums = match.group(2).strip().split()
                if len(nums) == 3:
                    x, y, z = map(float, nums)
                    data[drone_idx][0].append(x)
                    data[drone_idx][1].append(y)
                    data[drone_idx][2].append(z)
    return data



def plot_positions(xs, ys, zs, pos_label):
    steps = list(range(len(xs)))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(xs, ys, zs, label="Trajectory")

    # Start point
    ax.scatter(xs[0], ys[0], zs[0], color="green", s=80, marker="o", label="Start")

    # End point
    ax.scatter(xs[-1], ys[-1], zs[-1], color="red", s=80, marker="X", label="End")

    # Origin (0,0,0)
    ax.scatter(0, 0, 0, color="black", s=60, marker="+", label="Origin (0,0,0)")

    # Annotations
    ax.text(xs[0], ys[0], zs[0], " Start", color="green")
    ax.text(xs[-1], ys[-1], zs[-1], " End", color="red")
    ax.text(0, 0, 0, " Origin", color="black")

    ax.set_title(f"{pos_label} Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)

    # XYZ vs step plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, xs, label="X")
    plt.plot(steps, ys, label="Y")
    plt.plot(steps, zs, label="Z")
    plt.title(f"{pos_label} Over Time")
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)

    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize positional data from the observaiton log file.",
    )
    parser.add_argument(
        "--num_drones",
        type=int,
        default=2,
        help="Number of drones tesetd (and therefore in the log file).",
    )
    args = parser.parse_args()
        
    log_path = os.path.join(
        os.path.dirname(__file__),
        "drone_obs.log"
    )

    # Inspect drone positions
    drone_positions = parse_log_file(log_path, "Pos", args.num_drones)
    for drone_idx, (xs, ys, zs) in drone_positions.items():
        plot_positions(xs, ys, zs, f"Drone {drone_idx} Position")