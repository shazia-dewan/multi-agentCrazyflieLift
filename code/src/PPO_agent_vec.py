import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from neural_network import PolicyNetwork, ValueNetwork

# Some context: See Parallel Environments.md
# <TENSOR>.cpu() restore the info in CPU memory (helpful if using GPU)
#   Currently we are using CPU, but it's in place if we switch e.g. device="cuda"

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
        lr: float = 7e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        entropy_coefficient: float = 0.01,
        kl_threshold: float = 0.5,
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
        kl_threshold : float
            Threshold for KL-divergence
        device : str
            The device to use for PyTorch computations
        """
        self.buffer = RolloutBufferVec()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.lr = lr
        self.num_minibatches = num_minibatches
        self.entropy_coefficient = entropy_coefficient
        self.kl_threshold = kl_threshold
        self.device = torch.device(device)

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
        mean, std = self.policy_network(obs_tensor)
        dist = Normal(mean, std)

        if deterministic:
            action_tensor = mean
        else:
            action_tensor = dist.rsample()

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

    def compute_returns(self, last_values: np.ndarray | None = None) -> np.ndarray:
        """
        Compute discounted Monte Carlo returns for the stored trajectory.

        Parameters
        ----------
        last_values : None or np.ndarray of shape (num_envs,)
            Value estimates for the final states (used to bootstrap values from critic).

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
            return np.array([])

        # Episode ended naturally (termination)
        #   -> returns are fully observed, so no need to bootstrap
        #   -> initialize R = [0, 0, ...] so backward loop just propagates actual rewards
        # Episode ended "early" after a fixed number of steps (truncation)
        #   -> bootstrap the missing future rewards with critic’s estimate of final state values
        #   -> initialize R to [pred_value_1, pred_value_2...]
        if last_values is None:
            R = np.zeros(num_envs, dtype=np.float32)
        else:
            R = np.asarray(last_values).reshape(-1).astype(np.float32)

        # Iterate backward over timesteps, populate per-timestep returns
        returns = [None] * T
        for t in reversed(range(T)):
            # Sample rewards and end conditions at time t and flatten to shape (num_envs,)
            rewards_t = np.asarray(self.buffer.rewards[t]).reshape(-1).astype(np.float32)
            terminateds_t = np.asarray(self.buffer.terminateds[t]).reshape(-1).astype(np.float32)

            # Terminated episodes (end with terminated = True = 1) --> Use actual returns
            #   For timesteps where terminated = True, just use actual return (1.0 - terminateds_t = 0 -> R = rewards_t)
            #   For timesteps where terminated != True, propagate discounted (gamma) return values from real final states
            # Truncated episodes (end with terminated = False = 0) --> Bootstrap using critic predicted value
            #   For every R, propagate the discounted (gamma) predicted returns from the critic estimate
            R = rewards_t + self.gamma * R * (1.0 - terminateds_t)
            returns[t] = R.copy()

        # Flatten returns in time-major order of shape (T * num_envs,)
        returns_flat = np.concatenate([r.reshape(-1, ) for r in returns], axis=0)
        return returns_flat


    def _flatten_buffer(self, last_values: np.ndarray | None = None):
        """
        Convert lists of timestep batched arrays to flattened tensors for training.

        Returns
        -------
        observations : Tensor of shape (T * num_envs, obs_dim)
        actions : Tensor of shape (T * num_envs, action_dim)
        returns : Tensor of shape (T * num_envs,)
        old_log_probs : Tensor of shape (T * num_envs,)
        state_values : Tensor of shape (T * num_envs,)
        """
        T = self.buffer.num_steps()
        if T == 0:
            return None

        # Observations: list of T arrays each (num_envs, obs_dim) flattens to (T * num_envs, obs_dim)
        # Actions: list of T arrays each (num_envs, action_dim) flattens to (T * num_envs, obs_dim)
        # Returns already flattened from compute_returns() (T * num_envs,)
        observations_np = np.concatenate(self.buffer.observations, axis=0)
        actions_np = np.concatenate(self.buffer.actions, axis=0)
        returns_np = self.compute_returns(last_values)

        # log_probs and values are of shape (num_envs,) and flatten to (T*num_envs,)
        logp_t = torch.cat(self.buffer.log_probs, dim=0)
        vals_t = torch.cat(self.buffer.state_values, dim=0)

        # Convert numpy to tensors on device
        observations_t = torch.FloatTensor(observations_np).to(self.device)
        actions_t = torch.FloatTensor(actions_np).to(self.device)
        returns_t = torch.FloatTensor(returns_np).to(self.device)
        old_log_probs_t = logp_t.to(self.device)
        state_values_t = vals_t.to(self.device)

        return observations_t, actions_t, returns_t, old_log_probs_t, state_values_t

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
        observations, actions, returns, old_log_probs, state_values = flattened

        # Advantage function = returns - value estimates
        advantages = (returns - state_values).detach()

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

                # Compute current policy distribution for minibatch
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
                    print(f"Early stopping due to KL divergence: {kl_mean.item():.4f} > {self.kl_threshold}")
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

                # Add entropy and take -(mean of L_CLIP objective) since PyTorch optimizers perform gradient descent
                entropy = dist.entropy().sum(dim=-1).mean()
                policy_loss = -final_objective.mean() - self.entropy_coefficient * entropy

                # Value function loss (MSE)
                pred_values = self.value_network(obs_batch).view(-1)
                value_loss = nn.MSELoss()(pred_values, returns_batch)


                ######################################
                # Gradient descent and backpropagation
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
