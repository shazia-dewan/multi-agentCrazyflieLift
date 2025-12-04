# Code taken from notebook for bridging to hardware

from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from dataclasses import dataclass

# ---------------------------
# Hyperparameters / settings
# ---------------------------
NUM_ENVS = 2048
NUM_UPDATES = 1000 # Total #updates during training (analogous to total steps)

LR = 3e-3
LR_FINAL = LR * 0.25
MAX_GRAD_NORM = 0.5

SEED = 42
NUM_DRONES = 2
POLICY_HIDDEN = [128 * NUM_DRONES, 128 * NUM_DRONES]
VF_HIDDEN = [128 * NUM_DRONES, 128 * NUM_DRONES]

ACTION_SIZE_PER_DRONE = 4
PER_AGENT_OBS_DIM = 46
PER_ENV_OBS_DIM = PER_AGENT_OBS_DIM * NUM_DRONES


class MLP(nn.Module):
    hidden_sizes: Tuple[int, ...]
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i, h in enumerate(self.hidden_sizes):
            x = nn.relu(nn.Dense(h)(x))
        if self.activate_final:
            x = nn.relu(x)
        return x

class Actor(nn.Module):
    """Per-agent Gaussian policy: state --> mean, log(std)"""
    hidden_sizes: Tuple[int, ...]
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # x: (..., obs_dim_per_agent)
        h = MLP(self.hidden_sizes)(x)

        # Layers for mean and log_std weights & biases
        mean = nn.Dense(self.action_dim)(h)
        log_std = nn.Dense(self.action_dim)(h)

        # Ensure numerically reasonable range for log_std
        log_std = jnp.clip(log_std, -20.0, 2.0)

        return mean, log_std


class CentralizedCritic(nn.Module):
    """Takes global joint observation (all agents concatenated) -> scalar value."""
    hidden_sizes: Tuple[int, ...]
    @nn.compact
    def __call__(self, x):
        h = MLP(self.hidden_sizes)(x)
        v = nn.Dense(1)(h)
        return jnp.squeeze(v, -1)  # (batch,)


# https://flax.readthedocs.io/en/latest/_modules/flax/training/train_state.html
@jax.tree_util.register_pytree_node_class
@dataclass
class PPOTrainState:
    policy_state: TrainState
    value_state: TrainState
    env_steps: jnp.ndarray  # shape (NUM_ENVS,), int32
    ep_returns: jnp.ndarray  # shape (NUM_ENVS,)

    def tree_flatten(self):
        children = (self.policy_state, self.value_state, self.env_steps, self.ep_returns)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)



def create_train_state(rng, num_updates):
    # initialize actor and critic parameters
    rng1, rng2 = jax.random.split(rng)
    # Policy params: we expect to pass per-agent obs shape (PER_AGENT_OBS_DIM)
    actor_module = Actor(hidden_sizes=tuple(POLICY_HIDDEN), action_dim=ACTION_SIZE_PER_DRONE)
    critic_module = CentralizedCritic(hidden_sizes=tuple(VF_HIDDEN))

    dummy_agent_obs = jnp.zeros((1, PER_AGENT_OBS_DIM))
    dummy_global_obs = jnp.zeros((1, PER_ENV_OBS_DIM))

    actor_params = actor_module.init(rng1, dummy_agent_obs)
    critic_params = critic_module.init(rng2, dummy_global_obs)

    # Anneal LR
    lr_schedule = optax.linear_schedule(
        init_value=LR,
        end_value=LR_FINAL,
        transition_steps=num_updates
    )

    policy_tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(lr_schedule),
    )
    value_tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(lr_schedule),
    )

    policy_state = TrainState.create(apply_fn=actor_module.apply, params=actor_params, tx=policy_tx)
    value_state = TrainState.create(apply_fn=critic_module.apply, params=critic_params, tx=value_tx)
    env_steps = jnp.zeros((NUM_ENVS,), dtype=jnp.int32)
    ep_returns = jnp.zeros((NUM_ENVS,), dtype=jnp.float32)

    return PPOTrainState(
        policy_state=policy_state,
        value_state=value_state,
        env_steps=env_steps,
        ep_returns=ep_returns
    ), actor_module, critic_module