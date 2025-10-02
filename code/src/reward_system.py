from collections import defaultdict

class RewardTracker:
    """
    Utility class for tracking environment rewards 
    """
    def __init__(self):
        self.rewards_and_penalties = defaultdict(lambda: {"value": 0.0, "sum": 0.0, "abs_sum": 0.0})

    def update(self, name, value):
        """Update current value and accumulate abs_sum."""
        self.rewards_and_penalties[name]["value"] = value
        self.rewards_and_penalties[name]["sum"] += value
        self.rewards_and_penalties[name]["abs_sum"] += abs(value)

    def total(self):
        return sum(r["sum"] for r in self.rewards_and_penalties.values())
    
    def step_total(self):
        return sum(r["value"] for r in self.rewards_and_penalties.values())

    def summary(self):
        """Return a snapshot for logging/inspection."""
        return {
            "reward_summary": dict(self.rewards_and_penalties),
            "total": self.total()
        }
