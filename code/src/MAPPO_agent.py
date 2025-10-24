import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from neural_network import PolicyNetwork, ValueNetwork

class RolloutBufferMAPPO:
    """
    Multi-agent rollout buffer for PPO, similar to single-agent RolloutBuffer but adapted for multiple agents.
    Handles data of shape (num_envs, num_agents, ...) instead of (num_envs, ...).
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminateds = []
        self.log_probs = []
        self.state_values = []

    def store(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        log_probs: torch.Tensor,
        values: torch.Tensor
    ):
        """
        Store one timestep of data for all envs and all agents.

        observations: (num_envs, obs_dim_total)
        actions: (num_envs, action_dim_total)
        rewards: (num_envs,)
        terminateds: (num_envs,)
        log_probs: (num_envs, num_agents)
        values: (num_envs,)
        """
        self.observations.append(np.asarray(observations))
        self.actions.append(np.asarray(actions))
        self.rewards.append(np.asarray(rewards))
        self.terminateds.append(np.asarray(terminateds))

        self.log_probs.append(log_probs.detach().cpu())
        self.state_values.append(values.detach().cpu())

    def clear(self):
        self.__init__()

    def num_steps(self):
        return len(self.observations)

    def num_envs(self):
        if not self.observations:
            return 0
        return self.observations[0].shape[0]



class MAPPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_drones: int = 1,
        lr: float = 3e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        update_epochs: int = 2,
        num_minibatches: int = 2,
        entropy_coefficient: float = 0.001,
        value_loss_coefficient: float = 0.5,
        kl_threshold: float = 0.05,
        device: str = "cpu"
    ):
        """
        Initialize values for the vectorized PPO agent

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space.
        action_dim : int
            Dimension of the action space.
        num_drones : int
            Number of drones (agents) in the environment.
        lr : float, default=3e-4
            Learning rate (step size) for both policy and value networks.
        gamma : float
            Discount factor for returns.
        clip_eps : float
            Clipping parameter epsilon for PPO objective.
        update_epochs : int
            Number of epochs to update policy per batch of rollouts.
        num_minibatches : float
            Number of minibatches for policy updates.
        entropy_coefficient : float
            Coefficient used for entropy bonus in policy loss
        value_loss_coefficient : float
            Coefficient for the value function loss used by the critic
        kl_threshold : float
            Threshold for KL-divergence
        device : str
            The device to use for PyTorch computations
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_drones = num_drones
        self.buffer = RolloutBufferMAPPO()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.lr = lr
        self.num_minibatches = num_minibatches
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.kl_threshold = kl_threshold
        self.device = torch.device(device)

        self.obs_names = [f"obs_{i}" for i in range(obs_dim)]
        self.obs_importance = np.zeros((action_dim, obs_dim), dtype=np.float32)
        self.obs_counts = np.zeros((action_dim, obs_dim), dtype=np.int32)

        # Initialize networks and optimizes, critic (value) takes full obs_dim, actor (policy) takes per-drone obs_dim
        self.policy_network = PolicyNetwork(obs_dim // num_drones, action_dim // num_drones).to(self.device)
        self.value_network = ValueNetwork(obs_dim).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)


    def sample_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Sample an action from the policy given an observation array.

        Parameters
        ----------
        obs : np.ndarray of shape (num_envs, obs_dim) or (obs_dim,)
            Observation from the environment of shape (observation_dimension).
        deterministic : bool
            If true, use the mean action to sample (no variance, greedy)

        Returns
        -------
        action : np.ndarray of shape (num_envs, action_dim) or (action_dim,)
            Sampled action of shape (action_dimension).
        log_prob : torch.Tensor of shape (num_envs,)
            Log probability of the action under the current policy.
        value : torch.Tensor of shaep (num_envs,)
            Estimated state value from the critic.
        """

        obs_per_agent = self.obs_dim // self.num_drones
        act_per_agent = self.action_dim // self.num_drones

        # Determine if we are running a one or multiple envs
        one_env = False
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[np.newaxis, :]
            one_env = True

        num_envs = obs_arr.shape[0]

        # Convert to tensor and reshape to (num_envs, num_agents, obs_per_agent)
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        if obs_tensor.shape[1] != self.obs_dim:
            raise ValueError(f"Unexpected obs_dim. Got {obs_tensor.shape[1]}, expected {self.obs_dim}")
        obs_agents = obs_tensor.view(num_envs, self.num_drones, obs_per_agent)
        obs_agents_flat = obs_agents.reshape(num_envs * self.num_drones, obs_per_agent)
        
        # Policy network (shared) processes per-agent obs
        mean_flat, std_flat = self.policy_network(obs_agents_flat)
        dist = Normal(mean_flat, std_flat)

        if deterministic:
            actions_flat = mean_flat
        else:
            actions_flat = dist.rsample()

        # Per-agent log probs
        log_probs_flat = dist.log_prob(actions_flat).sum(dim=-1)

        # Reshape actions and log_probs back to (num_envs, num_agents, act_per_agent) and (num_envs, num_agents)
        actions = actions_flat.view(num_envs, self.num_drones, act_per_agent)
        log_probs = log_probs_flat.view(num_envs, self.num_drones)

        # Centralized critic: value for joint obs from all agents of shape (num_envs, obs_dim_total) --> shape (num_envs,)
        values = self.value_network(obs_tensor).view(-1)

        # Convert actions to numpy for VecEnv, concatenate agents per env into final action vector
        # From (num_envs, num_agents, act_per_agent) --> (num_envs, action_dim)
        actions_concat = actions.reshape(num_envs, -1)
        
        actions_np = actions_concat.detach().cpu().numpy()
        log_probs_cpu = log_probs.detach().cpu()
        values_cpu = values.detach().cpu()

        if one_env:
            return actions_np[0], log_probs_cpu[0], values_cpu[0]
        else:
            return actions_np, log_probs_cpu, values_cpu
        

    def store_transition(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        log_probs: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        """
        Store batched transition at current timestep.
        All inputs are expected batched over envs.

        Parameters
        ----------
        observations : np.ndarray
        actions : np.ndarray
        rewards : np.ndarray
        terminateds : np.ndarray
        log_probs : torch.Tensor
        values : torch.Tensor
        """
        self.buffer.store(observations, actions, rewards, terminateds, log_probs, values)


    def _compute_gae(self, last_values: np.ndarray | None = None, lamda: float = 0.95):
        """
        Compute GAE advantages and returns for multi-agent PPO, identical to the PPO implementation in PPO_agent.py
        """
        T = self.buffer.num_steps()
        num_envs = self.buffer.num_envs()

        # Empty buffer
        if T == 0:
            return np.array([]), np.array([])

        # Initialize values for bootstrapping truncated episodes (or actual values for terminated episodes)
        if last_values is None:
            next_values = np.zeros(num_envs, dtype=np.float32)
        else:
            next_values = np.asarray(last_values).reshape(-1).astype(np.float32)

        advantages = [None] * T
        returns = [None] * T

        # Running advantage buffer (GAE)
        gae = np.zeros(num_envs, dtype=np.float32)

        # Iterate backward through trajectory, populate per-timestep advantages and returns
        for t in reversed(range(T)):
            # Sample rewards, state values, and end conditions at time t and flatten to shape (num_envs,)
            rewards_t = np.asarray(self.buffer.rewards[t]).reshape(-1).astype(np.float32)
            state_values_t = np.asarray(self.buffer.state_values[t]).reshape(-1).astype(np.float32)
            terminateds_t = np.asarray(self.buffer.terminateds[t]).reshape(-1).astype(np.float32)

            # Bootstrap with critic if the episode is not terminal
            reward_delta = rewards_t + self.gamma * next_values * (1.0 - terminateds_t) - state_values_t

            # Update running advantage estimate, for terminal episodes, gae = reward_delta
            gae = reward_delta + self.gamma * lamda * (1.0 - terminateds_t) * gae

            advantages[t] = gae.copy()
            returns[t] = (advantages[t] + state_values_t).copy()

            # Prepare next value (shift one step back in time for propagation)
            next_values = state_values_t

        # Flatten in time-major order
        advantages_flat = np.concatenate([a.reshape(-1,) for a in advantages], axis=0)
        returns_flat = np.concatenate([r.reshape(-1,) for r in returns], axis=0)

        return advantages_flat, returns_flat


    def _flatten_buffer(self, last_values: np.ndarray | None=None):
        T = self.buffer.num_steps()
        if T == 0:
            return None
        num_envs = self.buffer.num_envs()

        obs = np.array(self.buffer.observations)   # (T, num_envs, full_obs_dim)
        acts = np.array(self.buffer.actions)       # (T, num_envs, full_action_dim)
        logps = torch.stack(self.buffer.log_probs) # (T, num_envs, num_agents)
        device = self.device

        # Compute env-level returns and advantages (shape (T*num_envs,))
        advantages_env, returns_env = self._compute_gae(last_values)

        # Convert to tensors
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)    # (T, num_envs, full_obs_dim)
        acts_t = torch.as_tensor(acts, dtype=torch.float32, device=device)  # (T, num_envs, full_action_dim)
        logp_t = logps.to(device)                                           # (T, num_envs, num_agents)

        # Full obs flat for value net training: shape (T*num_envs, full_obs_dim)
        full_obs_flat = obs_t.reshape(-1, obs_t.shape[-1])

        # env-level advantages/returns as tensors (T*num_envs,)
        adv_env_t = torch.as_tensor(advantages_env, dtype=torch.float32, device=device)
        ret_env_t = torch.as_tensor(returns_env, dtype=torch.float32, device=device)

        return {
            "obs": obs_t,                   # (T, num_envs, full_obs_dim)
            "acts": acts_t,                 # (T, num_envs, full_action_dim)
            "logps": logp_t,                # (T, num_envs, num_agents)
            "adv_env": adv_env_t,           # (T*num_envs,)
            "ret_env": ret_env_t,           # (T*num_envs,)
            "full_obs_flat": full_obs_flat, # (T*num_envs, full_obs_dim)
            "T": T,
            "num_envs": num_envs
        }


    def update_policy(self, last_values: np.ndarray | None = None) -> None:
        data = self._flatten_buffer(last_values)
        if data is None:
            print("Buffer empty: nothing to update.")
            return

        obs_t = data["obs"]
        acts_t = data["acts"]
        logp_t = data["logps"]
        adv_env = data["adv_env"]    # (T*num_envs,)
        ret_env = data["ret_env"]    # (T*num_envs,)
        full_obs_flat = data["full_obs_flat"]
        T = data["T"]
        num_envs = data["num_envs"]
        num_agents = self.num_drones

        obs_per_agent = self.obs_dim // num_agents
        act_per_agent = self.action_dim // num_agents

        # Normalize env-level advantages
        adv_env = (adv_env - adv_env.mean()) / (adv_env.std(unbiased=False) + 1e-9)

        # Reshape for per-agent access:
        # obs_t: (T, num_envs, full_obs_dim) -> reshape last dim to (num_agents, obs_per_agent)
        obs_per_agent_t = obs_t.reshape(T, num_envs, num_agents, obs_per_agent)   # (T, num_envs, num_agents, obs_per_agent)
        acts_per_agent_t = acts_t.reshape(T, num_envs, num_agents, act_per_agent) # (T, num_envs, num_agents, act_per_agent)
        logp_t = logp_t  # (T, num_envs, num_agents)

        # Apply the same (centralized) adv and return to each agent
        adv_per_agent = adv_env.view(T, num_envs).unsqueeze(-1).repeat(1, 1, num_agents)  # (T, num_envs, num_agents)
        ret_per_agent = ret_env.view(T, num_envs).unsqueeze(-1).repeat(1, 1, num_agents)

        N = T * num_envs  # transitions per agent
        num_minibatches = min(self.num_minibatches, N)
        batch_size = max(N // num_minibatches, 1)

        for agent_idx in range(num_agents):
            # Flatten per-agent arrays to length N
            agent_obs = obs_per_agent_t[:, :, agent_idx, :].reshape(-1, obs_per_agent)   # (N, obs_per_agent)
            agent_acts = acts_per_agent_t[:, :, agent_idx, :].reshape(-1, act_per_agent) # (N, act_per_agent)
            agent_adv = adv_per_agent[:, :, agent_idx].reshape(-1)  # (N,)
            agent_ret = ret_per_agent[:, :, agent_idx].reshape(-1)  # (N,)
            agent_old_logp = logp_t[:, :, agent_idx].reshape(-1).to(self.device) # (N,)

            for _ in range(self.update_epochs):
                perm = torch.randperm(N, device=self.device)

                # Compute old mean/std once for KL
                with torch.no_grad():
                    mean_old, std_old = self.policy_network(agent_obs)

                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    idx = perm[start:end]

                    obs_batch = agent_obs[idx]
                    acts_batch = agent_acts[idx]
                    adv_batch = agent_adv[idx]
                    ret_batch = agent_ret[idx]
                    old_logp_batch = agent_old_logp[idx]

                    mean, std = self.policy_network(obs_batch)
                    dist = Normal(mean, std)
                    log_probs = dist.log_prob(acts_batch).sum(dim=-1)

                    # KL check
                    dist_old = Normal(mean_old[idx].detach(), std_old[idx].detach())
                    kl = torch.distributions.kl_divergence(dist_old, dist).sum(dim=-1).mean()
                    if kl > self.kl_threshold:
                        print(f"Early stopping due to KL {kl.item():.4f} > {self.kl_threshold}")
                        break

                    ratio = torch.exp(log_probs - old_logp_batch)
                    unclipped = ratio * adv_batch
                    clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch
                    policy_loss = -(torch.min(unclipped, clipped).mean() + self.entropy_coefficient * dist.entropy().sum(dim=-1).mean())

                    # Value loss uses centralized observation and env-level returns
                    obs_value_batch = full_obs_flat[idx]   # (batch, full_obs_dim)
                    pred_values = self.value_network(obs_value_batch).view(-1)
                    value_loss = self.value_loss_coefficient * nn.MSELoss()(pred_values, ret_batch)

                    # Update policy
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                    self.policy_optimizer.step()

                    # Update value
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
                    self.value_optimizer.step()

        # Clear buffer
        self.buffer.clear()


    def get_obs_importance(self, action_idx: int | None = None) -> dict:
        """
        Returns feature importance for each observation.
        
        Parameters
        ----------
        action_idx : int or None
            If specified, return importance for that action dimension.
            If None, return aggregated importance across all actions.
        """
        importance = {}
        if action_idx is not None:
            counts = self.obs_counts[action_idx]
            scores = self.obs_importance[action_idx] / np.maximum(counts, 1)
        else:
            counts = np.sum(self.obs_counts, axis=0)
            scores = np.sum(self.obs_importance, axis=0) / np.maximum(counts, 1)

        for i, s in enumerate(scores):
            if counts[i] > 0:
                importance[self.obs_names[i]] = s
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))



    def save(self, filepath: str):
        """
        Save the model to a file

        Parameters
        ----------
        filepath : str
            The file path for the model
        """
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }, filepath)
        print(f"Policy saved to {filepath}")

    def load(self, filepath: str):
        """
        Load a model from a file

        Parameters
        ----------
        filepath : str
            The file path for the model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"Policy loaded from {filepath}")
