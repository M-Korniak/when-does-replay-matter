import os

from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np

from tensordict import TensorDict
from segment_tree import SegmentTree


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        When playground_mode=True, critic_observations are treated as a concatenation of
        regular observations and privileged observations, and only the privileged part is stored
        to save memory.

        TODO (Younggyo): Refactor to split this into SimpleReplayBuffer and NStepReplayBuffer
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        self.actions = torch.zeros(
            (n_env, buffer_size, n_act), device=device, dtype=torch.float
        )
        self.rewards = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.float
        )
        self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
        self.truncations = torch.zeros(
            (n_env, buffer_size), device=device, dtype=torch.long
        )
        self.next_observations = torch.zeros(
            (n_env, buffer_size, n_obs), device=device, dtype=torch.float
        )
        if asymmetric_obs:
            if self.playground_mode:
                # Only store the privileged part of observations (n_critic_obs - n_obs)
                self.privileged_obs_size = n_critic_obs - n_obs
                self.privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
                self.next_privileged_observations = torch.zeros(
                    (n_env, buffer_size, self.privileged_obs_size),
                    device=device,
                    dtype=torch.float,
                )
            else:
                # Store full critic observations
                self.critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
                self.next_critic_observations = torch.zeros(
                    (n_env, buffer_size, n_critic_obs), device=device, dtype=torch.float
                )
        self.ptr = 0

    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            indices = torch.randint(
                0,
                min(self.buffer_size, self.ptr),
                (self.n_env, batch_size),
                device=self.device,
            )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            effective_n_steps = torch.ones_like(dones)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # Sample base indices
            if self.ptr >= self.buffer_size:
                # When the buffer is full, there is no protection against sampling across different episodes
                # We avoid this by temporarily setting self.pos - 1 to truncated = True if not done
                # https://github.com/DLR-RM/stable-baselines3/blob/b91050ca94f8bce7a0285c91f85da518d5a26223/stable_baselines3/common/buffers.py#L857-L860
                # TODO (Younggyo): Change the reference when this SB3 branch is merged
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )
                indices = torch.randint(
                    0,
                    self.buffer_size,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            else:
                # Buffer not full - ensure n-step sequence doesn't exceed valid data
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(
                    0,
                    max_start_idx,
                    (self.n_env, batch_size),
                    device=self.device,
                )
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.truncations[:, current_pos - 1] = curr_truncations
        return out



class ExponentialSampler:
    def __init__(
        self,
        alpha: float,
        size: int,
        num_tasks: int,
        capacity: int = 1_000_000,
        device=None,
        dtype=torch.float32,
    ):
        self.alpha = alpha
        self.size = size
        self.num_tasks = num_tasks
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # exponent step
        self.k = torch.tensor(
            alpha / (capacity - 1),
            device=self.device,
            dtype=self.dtype,
        )

        # normalization constant Z
        self.update_Z(size)

    def update_Z(self, x: int):
        k = self.k
        base = torch.pow(torch.tensor(2.0, device=self.device, dtype=self.dtype), k)

        # geometric series sum: (2^(k x) - 1) / (2^k - 1)
        Z = (torch.pow(base, x) - 1.0) / (base - 1.0)
        self.Z = Z

    @torch.no_grad()
    def sample_indices(self, num_samples: int):
        """
        Vectorized exponential-weighted sampling.
        Returns indices in [0, size-1].
        """
        k = self.k
        base = torch.pow(torch.tensor(2.0, device=self.device, dtype=self.dtype), k)
        denom = base - 1.0

        # sample U(0, Z)
        r = torch.rand(
            num_samples,
            device=self.device,
            dtype=self.dtype,
        ) * self.Z

        # inverse CDF
        # i = log2(1 + r * (2^k - 1)) / k - 1
        inside = 1.0 + r * denom
        i = torch.log2(inside) / k - 1.0

        i = torch.floor(i).to(torch.long)
        return torch.clamp(i, 0, self.size - 1)

    @torch.no_grad()
    def sample(self, shape: tuple, offset: int = 0):
        num_samples = int(torch.tensor(shape).prod().item())

        sample_idx = self.sample_indices(num_samples)

        if offset != 0:
            sample_idx = (sample_idx + offset) % self.capacity

        return sample_idx.view(shape)


class GeomReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
        alpha_sampling: float = 10.0
    ):
        super().__init__(
            n_env,
            buffer_size,
            n_obs,
            n_act,
            n_critic_obs,
            asymmetric_obs,
            playground_mode,
            n_steps,
            gamma,
            device,
        )

        self.size = 0
        self.alpha_sampling = alpha_sampling
        self.sampler = ExponentialSampler(
            alpha=alpha_sampling,
            size=buffer_size,
            num_tasks=n_env,
            capacity=buffer_size,
            device=device
        )
    
    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
        self.sampler.update_Z(self.size)
        self.sampler.size = self.size

    @torch.no_grad()
    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            offset = 0 if self.size < self.buffer_size else self.ptr % self.buffer_size
            indices = self.sampler.sample(shape=(self.n_env, batch_size), offset=offset)
            # print(f"Sampled indices mean: {indices.float().mean().item():.2f}")
            # print(f"Buffer size: {self.size}, ptr: {self.ptr}")

            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            effective_n_steps = torch.ones_like(dones)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # GEOM sampling for n-step returns: restrict pool to valid starting positions
            # (size - n_steps + 1) to avoid sequences extending beyond valid data,
            # then sample with exponential weighting towards most recent.
            if self.ptr >= self.buffer_size:
                # When the buffer is full, there is no protection against sampling across different episodes
                # We avoid this by temporarily setting self.pos - 1 to truncated = True if not done
                # https://github.com/DLR-RM/stable-baselines3/blob/b91050ca94f8bce7a0285c91f85da518d5a26223/stable_baselines3/common/buffers.py#L857-L860
                # TODO (Younggyo): Change the reference when this SB3 branch is merged
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )
                # Valid pool: buffer_size - n_steps + 1 starting positions.
                # Sampler index 0 -> oldest (ptr % buffer_size),
                # sampler index valid_pool-1 -> newest valid start ((ptr - n_steps) % buffer_size).
                valid_pool = max(1, self.buffer_size - self.n_steps + 1)
                orig_size = self.sampler.size
                orig_Z = self.sampler.Z
                self.sampler.size = valid_pool
                self.sampler.update_Z(valid_pool)
                indices = self.sampler.sample(
                    shape=(self.n_env, batch_size), offset=self.ptr % self.buffer_size
                )
                self.sampler.size = orig_size
                self.sampler.Z = orig_Z
            else:
                # Buffer not full - valid starts are [0, ptr - n_steps].
                # Sampler index 0 -> oldest, valid_pool-1 -> newest valid start.
                valid_pool = max(1, self.ptr - self.n_steps + 1)
                orig_size = self.sampler.size
                orig_Z = self.sampler.Z
                self.sampler.size = valid_pool
                self.sampler.update_Z(valid_pool)
                indices = self.sampler.sample(shape=(self.n_env, batch_size), offset=0)
                self.sampler.size = orig_size
                self.sampler.Z = orig_Z
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.truncations[:, current_pos - 1] = curr_truncations
        return out


class EREReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
        decay_constant: float = 0.996,
        min_recent_samples: int = 1000,
        total_updates: int = 2000
    ):
        super().__init__(
            n_env,
            buffer_size,
            n_obs,
            n_act,
            n_critic_obs,
            asymmetric_obs,
            playground_mode,
            n_steps,
            gamma,
            device,
        )

        self.size = 0
        self.decay_constant = decay_constant
        self.min_recent_samples = min_recent_samples
        self.total_updates = total_updates
    
    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    @torch.no_grad()
    def sample(self, batch_size: int, update_step: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            k_most_recent = max(
                self.min_recent_samples,
                int(self.buffer_size * (self.decay_constant ** (1000 * update_step / self.total_updates))),
            )
            recent_size = min(k_most_recent, self.size)
            start = (self.ptr - recent_size) % self.buffer_size
            offset = torch.randint(0, recent_size, size=(self.n_env, batch_size), device=self.device)
            indices = (start + offset) % self.buffer_size

            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            effective_n_steps = torch.ones_like(dones)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # ERE sampling for n-step returns: restrict recent window to valid starting
            # positions (size - n_steps + 1) so sequences don't extend beyond valid data.
            k_most_recent = max(
                self.min_recent_samples,
                int(self.buffer_size * (self.decay_constant ** (1000 * update_step / self.total_updates))),
            )
            if self.ptr >= self.buffer_size:
                # When the buffer is full, there is no protection against sampling across different episodes
                # We avoid this by temporarily setting self.pos - 1 to truncated = True if not done
                # https://github.com/DLR-RM/stable-baselines3/blob/b91050ca94f8bce7a0285c91f85da518d5a26223/stable_baselines3/common/buffers.py#L857-L860
                # TODO (Younggyo): Change the reference when this SB3 branch is merged
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )
                # Valid pool: buffer_size - n_steps + 1 starting positions.
                # start + offset maps: offset 0 -> oldest start, offset recent_size-1 -> newest valid start.
                valid_pool = max(1, self.buffer_size - self.n_steps + 1)
                recent_size = min(k_most_recent, valid_pool)
                start = (self.ptr - self.n_steps + 1 - recent_size) % self.buffer_size
                offset = torch.randint(0, recent_size, size=(self.n_env, batch_size), device=self.device)
                indices = (start + offset) % self.buffer_size
            else:
                # Buffer not full - valid starts are [0, ptr - n_steps].
                valid_pool = max(1, self.ptr - self.n_steps + 1)
                recent_size = min(k_most_recent, valid_pool)
                start = valid_pool - recent_size
                offset = torch.randint(0, recent_size, size=(self.n_env, batch_size), device=self.device)
                indices = start + offset
            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.truncations[:, current_pos - 1] = curr_truncations
        return out

class PERReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
        alpha: float = 0.6,
        beta: float = 0.4,
        max_steps: int = 100000,
    ):
        super().__init__(
            n_env,
            buffer_size,
            n_obs,
            n_act,
            n_critic_obs,
            asymmetric_obs,
            playground_mode,
            n_steps,
            gamma,
            device,
        )

        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.max_steps = max_steps
        self.size = 0

        self.priority_trees = [SegmentTree(buffer_size) for _ in range(n_env)]

    def anneal_priority_beta(self, current_step: int):
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * current_step / self.max_steps)
        
    @torch.no_grad()
    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self.buffer_size
        self.observations[:, ptr] = observations
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        self.next_observations[:, ptr] = next_observations
        if self.asymmetric_obs:
            critic_observations = tensor_dict["critic_observations"]
            next_critic_observations = tensor_dict["next"]["critic_observations"]

            if self.playground_mode:
                # Extract and store only the privileged part
                privileged_observations = critic_observations[:, self.n_obs :]
                next_privileged_observations = next_critic_observations[:, self.n_obs :]
                self.privileged_observations[:, ptr] = privileged_observations
                self.next_privileged_observations[:, ptr] = next_privileged_observations
            else:
                # Store full critic observations
                self.critic_observations[:, ptr] = critic_observations
                self.next_critic_observations[:, ptr] = next_critic_observations
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)

        for env_idx in range(self.n_env):
            tree = self.priority_trees[env_idx]
            if self.n_steps == 1:
                tree.add(tree.max)
            else:
                # The newly written position is not yet a valid n-step start; set priority 0.
                # The position that just became valid (it now has n_steps future transitions)
                # is (ptr - n_steps) % buffer_size (ptr already incremented).
                tree.add(0)
                if self.ptr >= self.n_steps:
                    valid_start = (self.ptr - self.n_steps) % self.buffer_size
                    tree._update_index(valid_start + tree._tree_start_idx, tree.max)
    
    def _sample_idx_from_priority_tree(self, env_id, samples_size):
        p_total = self.priority_trees[env_id].total  # sum of the priorities
        segment_length = p_total / samples_size
        segment_starts = np.arange(samples_size) * segment_length
        valid = False

        while not valid:
            # Uniformly sample from within all segments
            samples = (
                np.random.uniform(0.0, segment_length, [samples_size])
                + segment_starts
            )
            # Retrieve samples from tree with un-normalised probability
            buffer_idxs, tree_idxs, sample_probs = self.priority_trees[env_id].find(samples)
            if np.all(sample_probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        return buffer_idxs, tree_idxs, sample_probs
    
    @torch.no_grad()
    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions
        tree_idxs = np.zeros((self.n_env, batch_size), dtype=np.int32)
        is_weights = np.zeros((self.n_env, batch_size), dtype=np.float32)

        if self.n_steps == 1:
            indices_np = np.zeros((self.n_env, batch_size), dtype=np.int64)
            for env_idx in range(self.n_env):
                buffer_idxs, _tree_idxs, sample_priorities = (
                    self._sample_idx_from_priority_tree(
                        env_id=env_idx,
                        samples_size=batch_size,
                    )
                )
                indices_np[env_idx] = buffer_idxs
                tree_idxs[env_idx] = _tree_idxs

                p_total = self.priority_trees[env_idx].total
                P = sample_priorities / p_total

                N = self.size
                weights = (N * P) ** (-self.beta)

                is_weights[env_idx] = weights / weights.max()
            
            self.last_indices = tree_idxs

            indices = torch.tensor(indices_np, device=self.device)
            is_weights = torch.tensor(is_weights, device=self.device, dtype=torch.float32)

            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            next_observations = torch.gather(
                self.next_observations, 1, obs_indices
            ).reshape(self.n_env * batch_size, self.n_obs)
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )

            rewards = torch.gather(self.rewards, 1, indices).reshape(
                self.n_env * batch_size
            )
            dones = torch.gather(self.dones, 1, indices).reshape(
                self.n_env * batch_size
            )
            truncations = torch.gather(self.truncations, 1, indices).reshape(
                self.n_env * batch_size
            )
            effective_n_steps = torch.ones_like(dones)
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)
                    next_privileged_observations = torch.gather(
                        self.next_privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                    next_critic_observations = torch.cat(
                        [next_observations, next_privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
                    next_critic_observations = torch.gather(
                        self.next_critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)
        else:
            # PER sampling for n-step returns.
            # Valid pool: positions with complete n-step sequences (priorities were set to 0
            # for invalid positions in extend(), so tree.find() will never return them).
            valid_pool = max(1, self.size - self.n_steps + 1)

            if self.ptr >= self.buffer_size:
                # When the buffer is full, there is no protection against sampling across different episodes
                # We avoid this by temporarily setting self.pos - 1 to truncated = True if not done
                # https://github.com/DLR-RM/stable-baselines3/blob/b91050ca94f8bce7a0285c91f85da518d5a26223/stable_baselines3/common/buffers.py#L857-L860
                # TODO (Younggyo): Change the reference when this SB3 branch is merged
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(
                    self.dones[:, current_pos - 1]
                )

            indices_np = np.zeros((self.n_env, batch_size), dtype=np.int64)
            for env_idx in range(self.n_env):
                buffer_idxs, _tree_idxs, sample_priorities = (
                    self._sample_idx_from_priority_tree(
                        env_id=env_idx,
                        samples_size=batch_size,
                    )
                )
                indices_np[env_idx] = buffer_idxs
                tree_idxs[env_idx] = _tree_idxs

                p_total = self.priority_trees[env_idx].total
                P = sample_priorities / p_total
                N = valid_pool
                weights = (N * P) ** (-self.beta)
                is_weights[env_idx] = weights / weights.max()

            self.last_indices = tree_idxs
            indices = torch.tensor(indices_np, device=self.device)
            is_weights = torch.tensor(is_weights, device=self.device, dtype=torch.float32)

            obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

            # Get base transitions
            observations = torch.gather(self.observations, 1, obs_indices).reshape(
                self.n_env * batch_size, self.n_obs
            )
            actions = torch.gather(self.actions, 1, act_indices).reshape(
                self.n_env * batch_size, self.n_act
            )
            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather privileged observations
                    priv_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.privileged_obs_size
                    )
                    privileged_observations = torch.gather(
                        self.privileged_observations, 1, priv_obs_indices
                    ).reshape(self.n_env * batch_size, self.privileged_obs_size)

                    # Concatenate with regular observations to form full critic observations
                    critic_observations = torch.cat(
                        [observations, privileged_observations], dim=1
                    )
                else:
                    # Gather full critic observations
                    critic_obs_indices = indices.unsqueeze(-1).expand(
                        -1, -1, self.n_critic_obs
                    )
                    critic_observations = torch.gather(
                        self.critic_observations, 1, critic_obs_indices
                    ).reshape(self.n_env * batch_size, self.n_critic_obs)

            # Create sequential indices for each sample
            # This creates a [n_env, batch_size, n_step] tensor of indices
            seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
            all_indices = (
                indices.unsqueeze(-1) + seq_offsets
            ) % self.buffer_size  # [n_env, batch_size, n_step]

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(
                self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_dones = torch.gather(
                self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices
            )
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(
                1.0 - all_dones_shifted, dim=2
            )  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Create discount factors
            discounts = torch.pow(
                self.gamma, torch.arange(self.n_steps, device=self.device)
            )  # [n_steps]

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * discounts.view(
                1, 1, -1
            )  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax(
                (all_dones > 0).float(), dim=2
            )  # [n_env, batch_size]
            first_trunc = torch.argmax(
                (all_truncations > 0).float(), dim=2
            )  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(
                first_done, first_trunc
            )  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(
                all_indices, 2, final_indices.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values
            final_next_observations = self.next_observations.gather(
                1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
            )
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)

            if self.asymmetric_obs:
                if self.playground_mode:
                    # Gather final privileged observations
                    final_next_privileged_observations = (
                        self.next_privileged_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.privileged_obs_size
                            ),
                        )
                    )

                    # Reshape for output
                    next_privileged_observations = (
                        final_next_privileged_observations.reshape(
                            self.n_env * batch_size, self.privileged_obs_size
                        )
                    )

                    # Concatenate with next observations to form full next critic observations
                    next_observations_reshaped = final_next_observations.reshape(
                        self.n_env * batch_size, self.n_obs
                    )
                    next_critic_observations = torch.cat(
                        [next_observations_reshaped, next_privileged_observations],
                        dim=1,
                    )
                else:
                    # Gather final next critic observations directly
                    final_next_critic_observations = (
                        self.next_critic_observations.gather(
                            1,
                            final_next_obs_indices.unsqueeze(-1).expand(
                                -1, -1, self.n_critic_obs
                            ),
                        )
                    )
                    next_critic_observations = final_next_critic_observations.reshape(
                        self.n_env * batch_size, self.n_critic_obs
                    )

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)
            next_observations = final_next_observations.reshape(
                self.n_env * batch_size, self.n_obs
            )

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        if self.asymmetric_obs:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = next_critic_observations

        if self.n_steps > 1 and self.ptr >= self.buffer_size:
            # Roll back the truncation flags introduced for safe sampling
            self.truncations[:, current_pos - 1] = curr_truncations
        return out, is_weights.reshape(-1)

    def update_priorities(self, priorites: np.ndarray):
        new_priorites = np.power(priorites, self.alpha).reshape(self.n_env, -1)
        for env_idx in range(self.n_env):
            self.priority_trees[env_idx].update(self.last_indices[env_idx], new_priorites[env_idx])
        

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, center: bool = True, update: bool = True
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            # Calculate global batch size arithmetically
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            # Calculate the stats
            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            # Sync the stats across all processes
            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            # Calculate the mean and variance of the global batch
            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = (
                global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            )
            batch_mean = batch_mean_shifted + self._mean

        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        # Update variance
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RewardNormalizer(nn.Module):
    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer(
            "G", torch.zeros(1, device=device)
        )  # running estimate of the discounted return
        self.register_buffer("G_r_max", torch.zeros(1, device=device))  # running-max
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        return rewards / denominator

    def update_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))

        local_max = torch.max(torch.abs(self.G))

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)

        self.G_r_max = max(self.G_r_max, local_max)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards)


class PerTaskEmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values for each task."""

    def __init__(
        self,
        num_tasks: int,
        shape: tuple,
        device: torch.device,
        eps: float = 1e-2,
        until: int = None,
    ):
        """
        Initialize PerTaskEmpiricalNormalization module.

        Args:
            num_tasks (int): The total number of tasks.
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If specified, learns until the sum of batch sizes
                                 for a specific task exceeds this value.
        """
        super().__init__()
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.num_tasks = num_tasks
        self.shape = shape
        self.eps = eps
        self.until = until
        self.device = device

        # Buffers now have a leading dimension for tasks
        self.register_buffer("_mean", torch.zeros(num_tasks, *shape).to(device))
        self.register_buffer("_var", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer("_std", torch.ones(num_tasks, *shape).to(device))
        self.register_buffer(
            "count", torch.zeros(num_tasks, dtype=torch.long).to(device)
        )

    def forward(
        self, x: torch.Tensor, task_ids: torch.Tensor, center: bool = True
    ) -> torch.Tensor:
        """
        Normalize the input tensor `x` using statistics for the given `task_ids`.

        Args:
            x (torch.Tensor): Input tensor of shape [num_envs, *shape].
            task_ids (torch.Tensor): Tensor of task indices, shape [num_envs].
            center (bool): If True, center the data by subtracting the mean.
        """
        if x.shape[1:] != self.shape:
            raise ValueError(f"Expected input shape (*, {self.shape}), got {x.shape}")
        if x.shape[0] != task_ids.shape[0]:
            raise ValueError("Batch size of x and task_ids must match.")

        # Gather the stats for the tasks in the current batch
        # Reshape task_ids for broadcasting: [num_envs] -> [num_envs, 1, ...]
        view_shape = (task_ids.shape[0],) + (1,) * len(self.shape)
        task_ids_expanded = task_ids.view(view_shape).expand_as(x)

        mean = self._mean.gather(0, task_ids_expanded)
        std = self._std.gather(0, task_ids_expanded)

        if self.training:
            self.update(x, task_ids)

        if center:
            return (x - mean) / (std + self.eps)
        else:
            return x / (std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor, task_ids: torch.Tensor):
        """Update running statistics for the tasks present in the batch."""
        unique_tasks = torch.unique(task_ids)

        for task_id in unique_tasks:
            if self.until is not None and self.count[task_id] >= self.until:
                continue

            # Create a mask to select data for the current task
            mask = task_ids == task_id
            x_task = x[mask]
            batch_size = x_task.shape[0]

            if batch_size == 0:
                continue

            # Update count for this task
            old_count = self.count[task_id].clone()
            new_count = old_count + batch_size

            # Update mean
            task_mean = self._mean[task_id]
            batch_mean = torch.mean(x_task, dim=0)
            delta = batch_mean - task_mean
            self._mean[task_id].copy_(task_mean + (batch_size / new_count) * delta)

            # Update variance using Chan's parallel algorithm
            if old_count > 0:
                batch_var = torch.var(x_task, dim=0, unbiased=False)
                m_a = self._var[task_id] * old_count
                m_b = batch_var * batch_size
                M2 = m_a + m_b + (delta**2) * (old_count * batch_size / new_count)
                self._var[task_id].copy_(M2 / new_count)
            else:
                # For the first batch of this task
                self._var[task_id].copy_(torch.var(x_task, dim=0, unbiased=False))

            self._std[task_id].copy_(torch.sqrt(self._var[task_id]))
            self.count[task_id].copy_(new_count)


class PerTaskRewardNormalizer(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """
        Per-task reward normalizer, motivation comes from BRC (https://arxiv.org/abs/2505.23150v1)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon
        self.device = device

        # Per-task running estimate of the discounted return
        self.register_buffer("G", torch.zeros(num_tasks, device=device))
        # Per-task running-max of the discounted return
        self.register_buffer("G_r_max", torch.zeros(num_tasks, device=device))
        # Use the new per-task normalizer for the statistics of G
        self.G_rms = PerTaskEmpiricalNormalization(
            num_tasks=num_tasks, shape=(1,), device=device
        )

    def _scale_reward(
        self, rewards: torch.Tensor, task_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Scales rewards using per-task statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        # Gather stats for the tasks in the batch
        std_for_batch = self.G_rms._std.gather(0, task_ids.unsqueeze(-1)).squeeze(-1)
        g_r_max_for_batch = self.G_r_max.gather(0, task_ids)

        var_denominator = std_for_batch + self.epsilon
        min_required_denominator = g_r_max_for_batch / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        # Add a small epsilon to the final denominator to prevent division by zero
        # in case g_r_max is also zero.
        return rewards / (denominator + self.epsilon)

    def update_stats(
        self, rewards: torch.Tensor, dones: torch.Tensor, task_ids: torch.Tensor
    ):
        """
        Updates the running discounted return and its statistics for each task.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            dones (torch.Tensor): Done tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        if not (rewards.shape == dones.shape == task_ids.shape):
            raise ValueError("rewards, dones, and task_ids must have the same shape.")

        # === Update G (running discounted return) ===
        # Gather the previous G values for the tasks in the batch
        prev_G = self.G.gather(0, task_ids)
        # Update G for each environment based on its own reward and done signal
        new_G = self.gamma * (1 - dones.float()) * prev_G + rewards
        # Scatter the updated G values back to the main buffer
        self.G.scatter_(0, task_ids, new_G)

        # === Update G_rms (statistics of G) ===
        # The update function handles the per-task logic internally
        self.G_rms.update(new_G.unsqueeze(-1), task_ids)

        # === Update G_r_max (running max of |G|) ===
        prev_G_r_max = self.G_r_max.gather(0, task_ids)
        # Update the max for each environment
        updated_G_r_max = torch.maximum(prev_G_r_max, torch.abs(new_G))
        # Scatter the new maxes back to the main buffer
        self.G_r_max.scatter_(0, task_ids, updated_G_r_max)

    def forward(self, rewards: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Normalizes rewards. During training, it also updates the running statistics.

        Args:
            rewards (torch.Tensor): Reward tensor, shape [num_envs].
            task_ids (torch.Tensor): Task indices, shape [num_envs].
        """
        return self._scale_reward(rewards, task_ids)


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    obs_normalizer,
    critic_obs_normalizer,
    args,
    save_path,
):
    """Save model parameters and training configuration to disk."""

    def get_ddp_state_dict(model):
        """Get state dict from model, handling DDP wrapper if present."""
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(get_ddp_state_dict(actor)),
        "qnet_state_dict": cpu_state(get_ddp_state_dict(qnet)),
        "qnet_target_state_dict": cpu_state(get_ddp_state_dict(qnet_target)),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")


def get_ddp_state_dict(model):
    """Get state dict from model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_ddp_state_dict(model, state_dict):
    """Load state dict into model, handling DDP wrapper if present."""
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


@torch.no_grad()
def mark_step():
    # call this once per iteration *before* any compiled function
    torch.compiler.cudagraph_mark_step_begin()
