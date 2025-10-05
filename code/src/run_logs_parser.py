# Script to parse training log file to examine average returns per policy update

import re
import matplotlib.pyplot as plt
from collections import defaultdict

LOG_FILE = "run_logs_PPO_training.log"

# Regex pattern to extract update and return value
line_pattern = re.compile(
    r"Update\s+(\d+)/\d+:\s+env\s+\d+\s+terminated an episode with return:\s+([-\d\.]+)"
)

# Read the log file, track returns per update
returns_by_update = defaultdict(list)
with open(LOG_FILE, "r") as f:
    for line in f:
        match = line_pattern.search(line)
        if match:
            update = int(match.group(1))
            episode_return = float(match.group(2))
            returns_by_update[update].append(episode_return)

# Compute average return per update
updates = sorted(returns_by_update.keys())
avg_returns = [
    sum(returns_by_update[u]) / len(returns_by_update[u]) for u in updates
]

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(updates, avg_returns, marker=".")
plt.title("Average Return per Update")
plt.xlabel("Update")
plt.ylabel("Average Return")
plt.grid(True)
plt.tight_layout()
plt.show()
