import gymnasium as gym
from typing import Tuple, Optional
from scale_rl.buffers.base_buffer import BaseBuffer, Batch
from scale_rl.buffers.numpy_buffer import NpyUniformBuffer, NpyPrioritizedBuffer, NpyEREBuffer


def create_buffer(
    buffer_class_type: str,
    buffer_type: str,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    n_step: int,
    gamma: float,
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    **kwargs,
) -> BaseBuffer:

    if buffer_class_type == 'numpy':
        if buffer_type == 'uniform':
            buffer = NpyUniformBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length,
                sample_batch_size=sample_batch_size,
                alpha_exponential_sampling=0.0,
            )

        elif buffer_type == 'exponential':
            buffer = NpyUniformBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length,
                sample_batch_size=sample_batch_size,
                alpha_exponential_sampling=kwargs.get('alpha_exponential_sampling', 10.0),
            )

        elif buffer_type == 'ere':
            buffer = NpyEREBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length,
                sample_batch_size=sample_batch_size,
                decay_const=kwargs.get('decay_const', 0.966),
                min_recent_samples=kwargs.get('min_recent_samples', 5000),
            )

        elif buffer_type == 'prioritized':
            buffer = NpyPrioritizedBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length,
                sample_batch_size=sample_batch_size,
                alpha_sampling=kwargs.get('alpha_sampling', 0.6),
                beta_start=kwargs.get('beta_start', 0.4),
                max_steps=kwargs.get('max_steps', 1_000_000),
            )

        else:
            raise NotImplementedError

    elif buffer_class_type == 'jax':
        raise NotImplementedError

    return buffer
