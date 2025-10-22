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

class RolloutBufferVec:
    """
    Stores batched transitions collected during environment rollouts for PPO updates.
    Each stored item is a list over timesteps where each element is an array/tensor
    of shape (num_envs, ...).
    """
    def __init__(self):
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.terminateds: list[np.ndarray] = []
        self.log_probs: list[torch.Tensor] = []
        self.state_values: list[torch.Tensor] = []

    def store(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        log_probs: torch.Tensor,
        values: torch.Tensor
    ) -> None:
        """
        Store a batched transition (for all envs at current timestep).
        
        Parameters
        ----------
        observations : np.ndarray of shape (num_envs, obs_dim)
        actions : np.ndarray of shape (num_envs, action_dim)
        rewards : np.ndarray of shape (num_envs,)
        terminateds : np.ndarray of shape (num_envs,)
            Terminated plural, termination bools for each env
        log_probs : torch.Tensor of shape (num_envs,)
        values : torch.Tensor of shape (num_envs,) or (num_envs,1)
        """

        # Convert observations/actions/rewards/terminateds to numpy for Gymnasium
        # Rewards and terminateds may be scalars when num_envs is 1, make them 1D arrays
        self.observations.append(np.asarray(observations))
        self.actions.append(np.asarray(actions))
        self.rewards.append(np.asarray(rewards).reshape(-1))
        self.terminateds.append(np.asarray(terminateds).reshape(-1))

        # Log_probs and values must be tensors shaped (num_envs,)
        self.log_probs.append(log_probs.detach().cpu().reshape(-1))
        self.state_values.append(values.detach().cpu().reshape(-1))

    def clear(self) -> None:
        """
        Reset the transition info
        """
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminateds = []
        self.log_probs = []
        self.state_values = []

    def num_steps(self) -> int:
        """
        Returns
        -------
        The number of timesteps per batch based on len(observations) = length of buffer
        """
        return len(self.observations)

    def num_envs(self) -> int:
        """
        Returns
        -------
        The number of environments per timestep (based on the number of observations at step t = 0)
        """
        if self.num_steps() == 0:
            return 0
        return self.observations[0].shape[0]


class PPOAgentVec:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        update_epochs: int = 2,
        num_minibatches: int = 2,
        entropy_coefficient: float = 0.001,
        value_loss_coefficient: float = 0.5,
        kl_threshold: float = 0.05,
        device: str = "cpu",
        track_obs_gradient: bool = False
    ):
        """
        Initialize values for the vectorized PPO agent

        Parameters
        ----------
        obs_dim : int
            Dimension of the observation space.
        action_dim : int
            Dimension of the action space.
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
        track_obs_gradient : bool
            Whether to track the gradient wrt observations (e.g. logs)
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.buffer = RolloutBufferVec()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.lr = lr
        self.num_minibatches = num_minibatches
        self.entropy_coefficient = entropy_coefficient
        self.value_loss_coefficient = value_loss_coefficient
        self.kl_threshold = kl_threshold
        self.device = torch.device(device)

        self.track_obs_gradient = track_obs_gradient
        self.obs_names = [f"obs_{i}" for i in range(obs_dim)]
        self.obs_importance = np.zeros((action_dim, obs_dim), dtype=np.float32)
        self.obs_counts = np.zeros((action_dim, obs_dim), dtype=np.int32)

        self.policy_network = PolicyNetwork(obs_dim, action_dim).to(self.device)
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

        # Determine if we are running a one or multiple envs
        one_env = False
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr[np.newaxis, :]
            one_env = True

        # Obs tensor shape: (num_envs, obs_dim), mean & std shape: (num_envs, action_dim)
        obs_tensor = torch.FloatTensor(obs_arr).to(self.device)
        if self.track_obs_gradient:
            obs_tensor.requires_grad_(True)
        mean, std = self.policy_network(obs_tensor)
        dist = Normal(mean, std)

        if deterministic:
            action_tensor = mean
        else:
            action_tensor = dist.rsample()


        # Observation gradients w.r.t. each action (Track importance of features for each action, used during eval)
        if self.track_obs_gradient:
            for a in range(self.action_dim):
                self.policy_network.zero_grad(set_to_none=True)
                self.value_network.zero_grad(set_to_none=True)
                obs_tensor.grad = None
                
                # Compute observation gradients for current action w.r.t. the input observation vector
                # Idea: Gain an understanding of which features are impactful for which actions by
                #   looking all the way back to the input layer, not just the final hidden layer.
                #   For action a, how sensitive is the action mean to changes in feature i?
                mean[0, a].backward(retain_graph=True)
                gradients = obs_tensor.grad.detach().cpu().numpy()[0]
                
                for i, g in enumerate(gradients):
                    self.obs_importance[a, i] += abs(g)
                    self.obs_counts[a, i] += 1
                
                obs_tensor.grad.zero_()

        # Sum log_prob per env across action dimensions, log_prob & value shape: (num_envs,)
        log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        
        # Critic (value network) value estimate
        value = self.value_network(obs_tensor).view(-1)

        # Convert actions to numpy for env.step (for Gymnasium)
        actions_np = action_tensor.detach().cpu().numpy()
        if one_env:
            return actions_np[0], log_prob.detach().cpu(), value.detach().cpu()
        else:
            return actions_np, log_prob.detach().cpu(), value.detach().cpu()

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


    def compute_gae(self, last_values: np.ndarray | None = None, lamda: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE (Generalized Advantage Estimation) advantages and returns.
        GAE computes a lower variance advantage estimate when compared to discounted Monte Carlo by
        adding the lambda factor which interpolates MC and TD-style estimation (balance variance with bias).

        Parameters
        ----------
        last_values : None or np.ndarray of shape (num_envs,)
            Critic value estimates for the final states across envs (used for bootstrapping if episode was truncated).
        lamda : float
            GAE smoothing parameter
            - λ=1: high variance and unbiased (Monte Carlo-like), λ=0: low variance and biased (TD-like)

        Returns
        -------
        advantages_flat : np.ndarray of shape (T * num_envs,)
            Advantage estimates for each timestep in the trajectory (used for actor updates).
            - Example: [t0_env0_advantage, t0_env1_advantage, ..., t1_env0_advantage, ...]
        returns_flat : np.ndarray of shape (T * num_envs,)
            Discounted returns for each timestep in the trajectory (used for critic updates).
            - Example: [t0_env0_return, t0_env1_return, ..., t1_env0_return, ...]

        Returns
        -------
        returns_flat : np.ndarray of shape (num_steps (T) * num_envs,) in time-major order
            Discounted returns for each timestep in the trajectory
            Example: [t0_env0_return, t0_env1_return, ..., t1_env0_return, ...]
        """
        T = self.buffer.num_steps()
        num_envs = self.buffer.num_envs()

        # Empty buffer
        if T == 0:
            return np.array([]), np.array([])

        # Initialize values for bootstrapping
        # If episode ended naturally (termination)
        #   -> returns are fully observed, so no need to bootstrap
        #   -> initialize next_values = [0, 0, ...] so backward loop just propagates actual rewards
        # If episode ended "early" after a fixed number of steps (truncation)
        #   -> bootstrap the missing future rewards with critic’s estimate of final state values
        #   -> initialize next_values to [pred_value_1, pred_value_2...]
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
            # For terminal episodes, 1.0 - terminateds_t = 0 -> reward_delta = rewards_t - state_values_t
            # For truncated episodes, reward_delta = rewards_t + self.gamma * next_values - state_values_t
            #    Propagates discounted (gamma) returns from the critic estimate rather than actual return
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


    def _flatten_buffer(self, last_values: np.ndarray | None = None):
        """
        Convert lists of timestep batched arrays to flattened tensors for training.

        Returns
        -------
        observations : Tensor of shape (T * num_envs, obs_dim)
        actions : Tensor of shape (T * num_envs, action_dim)
        advantages : Tensor of shape (T * num_envs, action_dim)
        returns : Tensor of shape (T * num_envs,)
        old_log_probs : Tensor of shape (T * num_envs,)
        """
        T = self.buffer.num_steps()
        if T == 0:
            return None

        # Observations: list of T arrays each (num_envs, obs_dim), flattens to (T * num_envs, obs_dim)
        # Actions: list of T arrays each (num_envs, action_dim), flattens to (T * num_envs, obs_dim)
        # Returns and advantages already flattened from compute_gae() (T * num_envs,)
        observations_np = np.concatenate(self.buffer.observations, axis=0)
        actions_np = np.concatenate(self.buffer.actions, axis=0)
        advantages_np, returns_np = self.compute_gae(last_values)

        # log_probs are of shape (num_envs,), flattens to shape (T*num_envs,)
        log_probs_t = torch.cat(self.buffer.log_probs, dim=0)

        # Convert numpy to tensors on device
        observations_t = torch.FloatTensor(observations_np).to(self.device)
        actions_t = torch.FloatTensor(actions_np).to(self.device)
        advantages_t = torch.FloatTensor(advantages_np).to(self.device)
        returns_t = torch.FloatTensor(returns_np).to(self.device)
        old_log_probs_t = log_probs_t.to(self.device)

        return observations_t, actions_t, advantages_t, returns_t, old_log_probs_t

    def update_policy(self, last_values: np.ndarray | None = None) -> None:
        """
        Update policy and value networks using PPO-Clip.
        Works on the flattened rollout buffer arrays with shapes T * num_envs.

        Steps
        -----
        1. Compute returns and advantages.
        2. Shuffle and split data into minibatches.
        3. For each minibatch:
            a. Compute current policy and ratio of new policy vs old policy log_prob ~ r(𝜃).
            b. Compute clipped surrogate objective and value loss ~ L_CLIP & MSE.
            c. Perform gradient updates for both policy and value networks.
        4. Repeat for multiple epochs over the entire buffer.
        5. Clear rollout buffer after update.
        """

        # Flatten the buffer, retrieve tensors
        flattened = self._flatten_buffer(last_values)
        if flattened is None:
            print("Buffer empty: nothing to update.")
            return
        observations, actions, advantages, returns, old_log_probs = flattened

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-9)
        advantages = advantages.detach()

        # Total number of collected transitions
        N = observations.shape[0]

        # Minibatching
        num_minibatches = min(self.num_minibatches, N)
        batch_size = max(N // num_minibatches, 1)

        for _ in range(self.update_epochs):

            # Shuffle indices into a random permutation
            permutation = torch.randperm(N, device=self.device)

            # Track old mean & std over entire previous buffer observations (for KL-divergence)
            with torch.no_grad():
                mean_old, std_old = self.policy_network(observations)

            for start in range(0, N, batch_size):
                ###################
                # Gather minibatch
                ###################

                # Take a slice of indices starting at a transition index (don't exceed N)
                end = min(start + batch_size, N)
                batch_idx = permutation[start:end]

                # Use the slice of indices to index minibatch values (e.g. observations)
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                old_logprobs_batch = old_log_probs[batch_idx]
                returns_batch = returns[batch_idx]
                advantages_batch = advantages[batch_idx]

                # Compute current policy distribution for minibatch (forward pass)
                mean, std = self.policy_network(obs_batch)
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions_batch).sum(dim=-1)


                ######################
                # KL-Divergence check
                ######################

                # KL divergence with old policy (avoid diverging too much from old policy)
                dist_old = Normal(mean_old[batch_idx].detach(), std_old[batch_idx].detach())
                kl_mean = torch.distributions.kl_divergence(dist_old, dist).sum(dim=-1).mean()

                if kl_mean > self.kl_threshold:
                    print(f"\nEarly stopping due to KL divergence: {kl_mean.item():.4f} > {self.kl_threshold}\n")
                    break


                ####################################
                # Importance sampling and clipping
                ####################################

                # Importance sampling (probability) ratio
                ratio = torch.exp(log_probs - old_logprobs_batch)

                # PPO clipping step
                unclipped_objective = ratio * advantages_batch
                clipped_objective = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_batch
                final_objective = torch.min(unclipped_objective, clipped_objective)

                # PyTorch optimizers perform gradient descent (minimize loss)
                #   So take  -(mean of L_CLIP objective) and subtract entropy sum of action dimension.
                #   Variant action distributions have high entropy (uncertainty), so policy_loss is minimized
                #   more and the policy is motivated to stay exploratory rather than collapsing too early
                entropy = dist.entropy().sum(dim=-1).mean()
                policy_loss = -final_objective.mean() - self.entropy_coefficient * entropy

                # Value function loss (MSE)
                pred_values = self.value_network(obs_batch).view(-1)
                value_loss = self.value_loss_coefficient * nn.MSELoss()(pred_values, returns_batch)


                ######################################
                # Gradient descent and backpropagation
                # General idea: https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/
                ######################################

                # Update policy network: Clear gradients and backpropagate
                # Clip gradient of policy network to prevent unstable updates
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                self.policy_optimizer.step()

                # Update the value network: Clear gradients and backpropagate 
                # Clip gradient of value network to prevent unstable updates
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
