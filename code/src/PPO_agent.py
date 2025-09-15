import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from neural_network import PolicyNetwork, ValueNetwork


class RolloutBuffer:
    """
    Stores transitions collected during environment rollouts for PPO updates.
    Each attribute is a list of observations, actions, rewards, etc...
    """
    def __init__(self):
        """Initialize empty rollout buffers."""
        self.observations: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[torch.Tensor] = []
        self.state_values: list[torch.Tensor] = []

    def store(
        self, 
        obs: torch.Tensor, action: torch.Tensor, reward: float, 
        done: bool, log_prob: torch.Tensor, value: torch.Tensor
    ) -> None:
        """
        Store a single transition in the buffer.

        Parameters
        ----------
        obs : torch.Tensor
            Observed state
        action : torch.Tensor
            Action taken
        reward : float
            Reward received
        done : bool
            Whether the episode terminated
        log_prob : torch.Tensor
            Log-probability of the action under the policy
        value : torch.Tensor
            Estimated value of the state
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.state_values.append(value)

    def clear(self) -> None:
        """
        Clear all stored transitions.
        Useful after a PPO update.
        """
        self.observations, self.actions, self.rewards, self.dones, self.log_probs, self.state_values = [], [], [], [], [], []



class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for continuous action spaces.
    Uses separate Policy (actor) and Value (critic) networks with Gaussian policy.
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        update_epochs: int = 2,
        num_minibatches: int = 2
    ):
        """
        Initialize PPO agent.

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
        """
        self.buffer = RolloutBuffer()
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.lr = lr
        self.num_minibatches = num_minibatches

        # Initialize policy/actor and value/critic neural networks
        # Optimizers update network parameters after computing gradients via backpropagation
        self.policy_network = PolicyNetwork(obs_dim, action_dim)
        self.value_network = ValueNetwork(obs_dim)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

    def sample_action(self, obs: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy given an observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation from the environment of shape (observation_dimension).

        Returns
        -------
        action : np.ndarray
            Sampled action of shape (action_dimension).
        log_prob : torch.Tensor
            Log probability of the action under the current policy.
        value : torch.Tensor
            Estimated state value from the critic.
        """

        # Unsqueeze to shape: [1, observation_dimension] for [batch_size, observation_dimension]
        # No minibatching for now, using entire batch
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        
        # Gaussian parameters and distribution
        mean, std = self.policy_network(obs_tensor)
        dist = Normal(mean, std)

        # Sample action
        action = dist.sample()

        # For importance sampling, we need the prob. ratio rt(𝜃) ~ used in clipped surrogate objective
        # Issue: the action space is multidimensional [thrust, roll, pitch, yaw]
        # Solution: Compute log_prob of each action, take sum (joint log_prob of sampled action)
        # Example action: [0.1, 0.2, 0.3, 0.4] --> log_prob = log_prob(thrust = 0.1) + log_prob(roll = 0.2)...
        log_prob = dist.log_prob(action).sum(dim=-1)

        value = self.value_network(obs_tensor)


        # Detach and return tensors 
        # For action, format to numpy for Gymnasium and remove batch dimension (take only [0])
        return action.detach().numpy()[0], log_prob.detach(), value.detach()

    def store_transition(
        self, 
        obs: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        done: bool, 
        log_prob: torch.Tensor, 
        value: torch.Tensor
    ) -> None:
        """
        Store a single transition in the rollout buffer.

        Parameters
        ----------
        obs : np.ndarray
            Observed state.
        action : np.ndarray
            Action taken.
        reward : float
            Reward received.
        done : bool
            Whether the episode terminated.
        log_prob : torch.Tensor
            Log probability of the action under policy.
        value : torch.Tensor
            Estimated value of the state.
        """
        self.buffer.store(obs, action, reward, done, log_prob, value)

    def compute_returns(self, last_value: float = 0.0) -> list[float]:
        """
        Compute discounted Monte Carlo returns for the stored trajectory.

        Parameters
        ----------
        last_value : float
            Value estimate for the final state (used in bootstrapping).

        Returns
        -------
        returns : List[float]
            Discounted returns for each timestep in the trajectory.
        """
        returns: list[float] = []
        R = last_value

        # Iterate backwards through trajectory to compute discounted sum
        for r, d in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        return returns

    def update_policy(self) -> None:
        """
        Update policy and value networks using the collected rollout buffer
        with multiple epochs, each using minibatches.

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

        # Convert buffer data to tensors (this is the info in each transition)
        observations = torch.FloatTensor(np.array(self.buffer.observations))
        actions = torch.FloatTensor(np.array(self.buffer.actions))
        returns = torch.FloatTensor(np.array(self.compute_returns())).detach()
        old_log_probs = torch.stack(self.buffer.log_probs).detach()
        state_values = torch.stack(self.buffer.state_values).squeeze()

        # Advantage function = returns - value estimates
        advantages = returns - state_values
    
        # Total number of collected transitions
        N = len(observations)

        # Batch size
        batch_size = N // self.num_minibatches

        for _ in range(self.update_epochs):

            # Shuffle indices into a random permutation
            permutation = torch.randperm(N)

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
                old_log_probs_batch = old_log_probs[batch_idx]
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
                dist_old = Normal(mean_old[batch_idx], std_old[batch_idx])
                kl_mean = torch.distributions.kl_divergence(dist_old, dist).sum(dim=-1).mean()

                # Stop if divergence too high
                if kl_mean > 0.03:
                    print(f"Early stopping due to KL divergence: {kl_mean:.4f}")
                    break


                ####################################
                # Importance sampling and clipping
                ####################################

                # Importance sampling (probability) ratio
                ratio = torch.exp(log_probs - old_log_probs_batch)
                
                # Vanilla PPO at step t (unclipped) --> L(𝜃) = Et[rt(𝜃) * At]
                # Clipped PPO at step t (for eps 𝜖) --> L_clipped(𝜃) = Et[CLIP(rt(𝜃), 1 - 𝜖, 1 + 𝜖) * At]
                unclipped_objective = ratio * advantages_batch
                clipped_objective = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_batch

                # PPO-Clip takes min(vanilla, clipped) = L_CLIP(𝜃) = Et[min(rt(𝜃) * At, CLIP(rt(𝜃), 1 - 𝜖, 1 + 𝜖) * At]
                # This ensures we take the smaller of the two steps to prevent large gradient updates
                final_objective = torch.min(unclipped_objective, clipped_objective)

                # Take negative of the average L_CLIP objective because PyTorch optimizers perform gradient descent
                # Note: Minimizing -(L_CLIP objective) is equivalent to maximizing +(L_CLIP objective)
                policy_loss = -final_objective.mean()
                
                # Value function loss (MSE) - view(-1) ensures we have [batch_size]
                pred_values = self.value_network(obs_batch).view(-1)
                value_loss = nn.MSELoss()(pred_values, returns_batch)


                ######################################
                # Gradient descent and backpropagation
                ######################################

                # .backward() is the backpropagation step
                # Traverse tensor computation graph and update original parameters
                
                # Clear policy network gradients, use our new ones
                # Uses the policy gradient = E[Score fn * Advantage fn] (Gaussian score fn in our case)
                # Backpropagation: policy_loss --> final_objective --> network parameters (weights, biases, and log_std)
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # Update value network
                # Uses the standard MSE gradient (ordinary regression)
                # Backpropagation: value_loss --> predicted V(s) --> network parameters (weights and biases)
                self.value_optimizer.zero_grad()
                value_loss.backward()
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
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        print(f"Policy loaded from {filepath}")
