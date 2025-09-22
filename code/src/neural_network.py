import torch
import torch.nn as nn
import numpy as np
import random

# Note: These networks inherit from PyTorch nn.module and its functionality
# As such, methods like forward() are called implicitly e.g. with mean, std = self.policy_network(obs_batch)

# For reproducibility, using specific seed for now
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PolicyNetwork(nn.Module):
    """
    Representation of the policy neural network using PyTorch.
    Outputs the mean and standard deviation of a Gaussian policy for continuous actions.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        """
        Initialize neural network.

        Parameters
        ----------
        obs_dim : int
            Dimension of observation space
        action_dim : int
            Dimension of action space
        hidden_size : int
            Number of neurons in hidden layers
        """

        # Define our policy neural network (NN)
        # Feedforward fully connected MLP with two hidden layers 
        # Input/Observations --> Linear & Tanh --> Linear & Tanh --> Linear --> Output/Actions

        super().__init__()

        self.net = nn.Sequential(
            # Map input layer (observations) into the first hidden layer linearly
            nn.Linear(obs_dim, hidden_size),
            # Apply non-linear tanh activation function per neuron (activation values on [-1, 1])
            nn.Tanh(),
            # Map hidden layer 1 into hidden layer 2 linearly + tanh activation
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # Map hidden layer 2 into output layer (actions, in this case the mean of the Gaussian action distribution)
            nn.Linear(hidden_size, action_dim)
        )

        # Learnable parameter for the NN: log(standard deviation) for Gaussian policy
        # Initialized to 0s, meaning our initial Gaussian policy has std = 1 since log(std) = 0 --> std = 1
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input observation tensor of shape (batch_size, observation_dimension)

        Returns
        -------
        mean : torch.Tensor
            Mean of the Gaussian action distribution
        std : torch.Tensor
            Standard deviation of the Gaussian action distribution
        """

        # Ensure batch dimension exists
        if input_tensor.dim() == 1:
            # Shape [1, obs_dim]
            input_tensor = input_tensor.unsqueeze(0)

        mean = self.net(input_tensor)

        # Expand std to match batch size
        std = torch.exp(self.log_std).expand_as(mean)  # shape [batch_size, action_dim]

        return mean, std


class ValueNetwork(nn.Module):
    """
    Representation of the value function neural network using PyTorch.
    Outputs the scalar value of a state.
    """
    def __init__(self, obs_dim: int, hidden_size: int = 128):
        """
        Initialize the value network.

        Parameters
        ----------
        obs_dim : int
            Dimension of observation space
        hidden_size : int
            Number of neurons in hidden layers
        """

        # Value function NN is similar to policy NN but outputs the state value
        # Feedforward fully connected MLP with two hidden layers 
        # Input/Observations --> Linear & Tanh --> Linear & Tanh --> Linear --> State Value

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_tensor: torch.tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input observation tensor of shape (batch_size, observation_dimension)

        Returns
        -------
        value : torch.Tensor
            Predicted state value of shape (batch_size, 1)
        """
        return self.net(input_tensor)