import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from neural_network import PolicyNetwork, ValueNetwork

# Some context: See Parallel Environments.md
# <TENSOR>.cpu() restore the info in CPU memory (helpful if using GPU)
#   If using NVIDIA GPU e.g. device="cuda", install PyTorch with cuda support (see Usage.md)
# Current setup is suboptimal for full GPU computations (e.g. lots of conversion to CPU storage)
#   For larger batches or networks, perhaps store rollout buffer directly as torch tensors on GPU
#   Only call .cpu().numpy() before env.step
#   Stepping might still be the bottleneck via CPU

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
        rewards: (num_envs,) or (num_envs, num_agents)
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
        Compute GAE advantages and returns for multi-agent PPO (similar to PPO_agent.py but adapted for multi-agent data).
        If rewards are (num_envs, num_agents), the same advantage per env
        is applied to all agents (centralized value function).
        """
        T = self.buffer.num_steps()
        num_envs = self.buffer.num_envs()
        if T == 0:
            return np.array([]), np.array([])

        # Get reward and value shapes
        rewards_0 = np.asarray(self.buffer.rewards[0])
        multi_agent = rewards_0.ndim == 2
        num_agents = rewards_0.shape[1] if multi_agent else 1

        # Bootstrap value for truncations
        if last_values is None:
            next_values = np.zeros(num_envs, dtype=np.float32)
        else:
            next_values = np.asarray(last_values).reshape(-1).astype(np.float32)

        advantages, returns = [None] * T, [None] * T
        gae = np.zeros(num_envs, dtype=np.float32)

        for t in reversed(range(T)):
            rewards_t = np.asarray(self.buffer.rewards[t], dtype=np.float32)
            if not multi_agent:
                rewards_t = rewards_t.reshape(num_envs, 1)  # unify shape (num_envs, num_agents)
            state_values_t = np.asarray(self.buffer.state_values[t]).reshape(num_envs).astype(np.float32)
            terminateds_t = np.asarray(self.buffer.terminateds[t]).reshape(num_envs).astype(np.float32)

            # TD residual (delta)
            reward_delta = rewards_t.mean(axis=1) + self.gamma * next_values * (1.0 - terminateds_t) - state_values_t

            gae = reward_delta + self.gamma * lamda * (1.0 - terminateds_t) * gae

            # For multi-agent, replicate the same env-level advantage to all agents
            adv_per_agent = np.repeat(gae[:, None], num_agents, axis=1)
            ret_per_agent = np.repeat((gae + state_values_t)[:, None], num_agents, axis=1)

            advantages[t] = adv_per_agent
            returns[t] = ret_per_agent

            next_values = state_values_t

        advantages_flat = np.concatenate([a.reshape(-1, num_agents) for a in advantages], axis=0)
        returns_flat = np.concatenate([r.reshape(-1, num_agents) for r in returns], axis=0)

        return advantages_flat, returns_flat


    def _flatten_buffer(self, last_values: np.ndarray | None=None):
        """
        Flatten rollout buffer into tensors for PPO updates.
        Works for multi-agent data: (T, num_envs, num_agents, ...) → (T * num_envs * num_agents, ...)
        """
        T = self.buffer.num_steps()
        if T == 0:
            return None
        num_envs = self.buffer.num_envs()

        obs = np.array(self.buffer.observations)
        acts = np.array(self.buffer.actions)
        logps = torch.stack(self.buffer.log_probs)

        num_agents = logps.shape[-1] if logps.ndim == 3 else 1
        device = self.device

        # Compute returns and advantages per-env
        advantages, returns = self._compute_gae(last_values)

        # Flatten (env, agent) → one batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        acts_t = torch.as_tensor(acts, dtype=torch.float32, device=device)

        # Repeat advantages and returns for each agent if multi-agent
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=device).repeat(1, num_agents)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=device).repeat(1, num_agents)

        logp_t = logps.to(device)

        # If multi-agent, repeat obs for each agent
        if num_agents > 1:
            obs_t = obs_t.repeat_interleave(num_agents, dim=1)  # (T, num_envs * num_agents, obs_dim)
            acts_t = acts_t.repeat_interleave(num_agents, dim=1)
            adv_t = adv_t.view(T, num_envs * num_agents)
            ret_t = ret_t.view(T, num_envs * num_agents)
            logp_t = logp_t.view(T, num_envs * num_agents)
        else:
            obs_t = obs_t
            acts_t = acts_t
            adv_t = adv_t.view(T * num_envs)
            ret_t = ret_t.view(T * num_envs)
            logp_t = logp_t.view(T * num_envs)

        # Final flattening over time
        return (
            obs_t.reshape(-1, obs_t.shape[-1]),
            acts_t.reshape(-1, acts_t.shape[-1]),
            adv_t.reshape(-1),
            ret_t.reshape(-1),
            logp_t.reshape(-1),
        )


    def update_policy(self, last_values: np.ndarray | None = None) -> None:
        """
        Update policy and value networks using PPO-Clip for multiple drones.
        Each drone is updated separately using slices of the flattened rollout buffer.
        Value network uses full centralized observations.
        """
        # Flatten the buffer, retrieve tensors
        flattened = self._flatten_buffer(last_values)
        if flattened is None:
            print("Buffer empty: nothing to update.")
            return

        observations, actions, advantages, returns, old_log_probs = flattened
        T = self.buffer.num_steps()
        num_envs = self.buffer.num_envs()
        num_agents = self.num_drones
        obs_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        # Normalize advantages globally
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-9)
        advantages = advantages.detach()

        # Reshape flattened data to [T, num_envs, num_agents, ...]
        obs_reshaped = observations.view(T, num_envs, num_agents, obs_dim)
        acts_reshaped = actions.view(T, num_envs, num_agents, action_dim)
        adv_reshaped = advantages.view(T, num_envs, num_agents)
        ret_reshaped = returns.view(T, num_envs, num_agents)
        logp_reshaped = old_log_probs.view(T, num_envs, num_agents)

        # Total transitions per agent
        N = T * num_envs
        num_minibatches = min(self.num_minibatches, N)
        batch_size = max(N // num_minibatches, 1)

        # Flatten full obs for value network (centralized)
        full_obs_flat = obs_reshaped.reshape(-1, obs_dim)  # shape [T*num_envs*num_agents, obs_dim]

        for agent_idx in range(num_agents):
            # Slice per-agent observations and actions
            agent_obs = obs_reshaped[:, :, agent_idx, :].reshape(-1, obs_dim // num_agents)
            agent_acts = acts_reshaped[:, :, agent_idx, :].reshape(-1, action_dim // num_agents)
            agent_adv = adv_reshaped[:, :, agent_idx].reshape(-1)
            agent_ret = ret_reshaped[:, :, agent_idx].reshape(-1)
            agent_old_logp = logp_reshaped[:, :, agent_idx].reshape(-1)

            for _ in range(self.update_epochs):
                permutation = torch.randperm(N, device=self.device)

                # Compute old mean/std once for KL check
                with torch.no_grad():
                    mean_old, std_old = self.policy_network(agent_obs)

                for start in range(0, N, batch_size):
                    end = min(start + batch_size, N)
                    batch_idx = permutation[start:end]

                    obs_batch = agent_obs[batch_idx]
                    acts_batch = agent_acts[batch_idx]
                    adv_batch = agent_adv[batch_idx]
                    ret_batch = agent_ret[batch_idx]
                    old_logp_batch = agent_old_logp[batch_idx]

                    # Policy forward pass
                    mean, std = self.policy_network(obs_batch)
                    dist = Normal(mean, std)
                    log_probs = dist.log_prob(acts_batch).sum(dim=-1)

                    # KL divergence check
                    dist_old = Normal(mean_old[batch_idx].detach(), std_old[batch_idx].detach())
                    kl_mean = torch.distributions.kl_divergence(dist_old, dist).sum(dim=-1).mean()
                    if kl_mean > self.kl_threshold:
                        print(f"\nEarly stopping due to KL divergence: {kl_mean.item():.4f} > {self.kl_threshold}\n")
                        break

                    # PPO clipped objective
                    ratio = torch.exp(log_probs - old_logp_batch)
                    unclipped_objective = ratio * adv_batch
                    clipped_objective = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_batch
                    policy_loss = -(torch.min(unclipped_objective, clipped_objective).mean() +
                                    self.entropy_coefficient * dist.entropy().sum(dim=-1).mean())

                    # Value loss (centralized obs)
                    obs_value_batch = full_obs_flat[batch_idx]  # full obs for all agents
                    pred_values = self.value_network(obs_value_batch).view(-1)
                    value_loss = self.value_loss_coefficient * nn.MSELoss()(pred_values, ret_batch)

                    # Policy optimizer step
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                    self.policy_optimizer.step()

                    # Value optimizer step
                    self.value_optimizer.zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
                    self.value_optimizer.step()

        # Clear buffer after update
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
