import gymnasium as gym
import numpy as np

import jax
import jax.numpy as jnp

import os
import pickle

from jaxrl.utils import Batch
from jaxrl.segment_tree import SegmentTree


import numpy as np

class ExponentialSampler:
    def __init__(self, alpha: float, size: int, num_tasks: int, max_steps: int = 1_000_000):
        self.alpha = alpha
        self.size = size
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        
        # exponent step
        self.k = alpha / (max_steps - 1)
        
        # normalization constant
        self.Z = self.compute_Z(size)

    def compute_Z(self, x: int):
        if x <= 0:
            return 1.0
        k = self.k
        # geometric series sum: (2^(k x) - 1) / (2^k - 1)
        return (2**(k * x) - 1) / (2**k - 1)

    def sample_indices(self, num_samples: int):
        """
        Vectorized exponential-weighted sampling.
        Returns indices in [0, size-1].
        """
        k = self.k
        base = 2**k
        denom = base - 1
        
        # sample U(0, Z)
        r = np.random.rand(num_samples) * self.Z
        
        # solve closed-form inverse CDF
        # i = log2(1 + r*(2^k -1))/k - 1
        inside = 1 + r * denom
        i = (np.log2(inside) / k) - 1
        
        i = np.floor(i).astype(int)
        return np.clip(i, 0, self.size - 1)

    def sample(self, size: int, shape: tuple, offset: int = 0):
        """
        Same interface as LinearSampler.
        """
        num_samples = np.prod(shape)
        
        # uniform task indices
        task_indx = np.random.randint(self.num_tasks, size=num_samples)
        
        # exponential per-task index sampling
        sample_indx = self.sample_indices(num_samples)

        # apply offset
        sample_indx = (sample_indx + offset) % self.max_steps

        return (
            task_indx.reshape(shape),
            sample_indx.reshape(shape),
        )
    

class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int):
        self.observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=np.float32) # dtype=observation_space.dtype)
        self.actions = np.empty((num_tasks, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=np.float32) # dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 40
        self.num_tasks = num_tasks
        self.last_indxs = (None, None) 

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray, return_overriden: bool = False):
        if return_overriden:
            observation_overriden = self.observations[:, self.insert_index].copy()
            action_overriden = self.actions[:, self.insert_index].copy()
            reward_overriden = self.rewards[:, self.insert_index].copy()
            mask_overriden = self.masks[:, self.insert_index].copy()
            next_observation_overriden = self.next_observations[:, self.insert_index].copy()

        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if return_overriden:
            return Batch(
                observations=observation_overriden,
                actions=action_overriden,
                rewards=reward_overriden,
                masks=mask_overriden,
                next_observations=next_observation_overriden,
                task_ids=np.arange(self.num_tasks)
            )
        else:
            return None

    def sample(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size * self.num_tasks, size=(num_batches, batch_size))
        task_indx, sample_indx = np.divmod(indx, self.size)
        self.last_indxs = (task_indx, sample_indx)
        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations,
                     task_ids=task_indx)    

    def sample_task_batches(self, batch_size_per_task: int | None = None):
        batch_size = 32 if batch_size_per_task is None else batch_size_per_task
        indxs = np.random.randint(self.size, size=batch_size)        
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids)  
        
    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index, self.last_indxs), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index, self.last_indxs = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))
        

class TruncGeomReplayBuffer(ParallelReplayBuffer):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int, trunc_geom_alpha: float = 4):
        self.observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=np.float32) # dtype=observation_space.dtype)
        self.actions = np.empty((num_tasks, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=np.float32) # dtype=observation_space.dtype)
        self.size = 0
        self.offset = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 40
        self.num_tasks = num_tasks
        self.last_indxs = (None, None)
        self.sampler = ExponentialSampler(alpha=trunc_geom_alpha, size=0, num_tasks=num_tasks, max_steps=1_000_000)

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray, return_overriden: bool = False):
        if return_overriden:
            observation_overriden = self.observations[:, self.insert_index].copy()
            action_overriden = self.actions[:, self.insert_index].copy()
            reward_overriden = self.rewards[:, self.insert_index].copy()
            mask_overriden = self.masks[:, self.insert_index].copy()
            next_observation_overriden = self.next_observations[:, self.insert_index].copy()

        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.capacity

        if self.size == self.capacity:
            self.offset = (self.offset + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)
        
        self.sampler.size = self.size
        self.sampler.Z = self.sampler.compute_Z(self.size)
        
        if return_overriden:
            return Batch(
                observations=observation_overriden,
                actions=action_overriden,
                rewards=reward_overriden,
                masks=mask_overriden,
                next_observations=next_observation_overriden,
                task_ids=np.arange(self.num_tasks)
            )
        else:
            return None

    def sample(self, batch_size: int, num_batches: int):
        task_indx, sample_indx = self.sampler.sample(self.size * self.num_tasks, (num_batches, batch_size), self.offset)
        self.last_indxs = (task_indx, sample_indx)
        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations,
                     task_ids=task_indx)    

    def sample_task_batches(self, batch_size_per_task: int | None = None):
        batch_size = 32 if batch_size_per_task is None else batch_size_per_task
        indxs = np.random.randint(self.size, size=batch_size)        
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids)  
        
    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index, self.last_indxs), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index, self.last_indxs = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))


class PriorityReplayBuffer(ParallelReplayBuffer):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int, replaceable_fraction: float = 0.0):
        super().__init__(observation_space, action_dim, capacity, num_tasks)
        self.num_tasks = num_tasks
        self.priorities = np.ones((num_tasks, capacity), dtype=np.float32) * np.finfo(np.float32).max
        self.insertion_order = np.zeros((num_tasks, capacity), dtype=np.int32)
        self.insert_index = np.zeros((num_tasks, ), dtype=np.int32)
        self.replace_treshold = int(capacity * replaceable_fraction)
    
    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray):
        # print(self.insert_index)

        self.observations[np.arange(self.num_tasks), self.insert_index] = observation
        self.actions[np.arange(self.num_tasks), self.insert_index] = action
        self.rewards[np.arange(self.num_tasks), self.insert_index] = reward
        self.masks[np.arange(self.num_tasks), self.insert_index] = mask
        self.next_observations[np.arange(self.num_tasks), self.insert_index] = next_observation
        self.priorities[np.arange(self.num_tasks), self.insert_index] = np.finfo(np.float32).max

        self.size = min(self.size + 1, self.capacity)
        if self.size < self.capacity:
            self.insertion_order[np.arange(self.num_tasks), self.insert_index] = self.insert_index 
            self.insert_index = self.insert_index + 1
        else:
            replaced_positions = self.insertion_order[np.arange(self.num_tasks), self.insert_index]
            mask = self.insertion_order > replaced_positions[:, None]
            self.insertion_order = self.insertion_order - mask.astype(np.int32)
            self.insertion_order[np.arange(self.num_tasks), self.insert_index] = self.capacity - 1

            adjusted_priorities = self.priorities.copy()
            adjusted_priorities[self.insertion_order >= self.replace_treshold] = np.finfo(np.float32).max
            self.insert_index = np.argmin(adjusted_priorities, axis=1)
    
    def update_last_batch_priorities(self, new_priorities: jnp.ndarray):
        task_indx, sample_indx = self.last_indxs          
        self.priorities[task_indx, sample_indx] = new_priorities


class ForgettingParallelReplayBuffer(ParallelReplayBuffer):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int, num_bins: int = 101):
        super().__init__(observation_space, action_dim, capacity, num_tasks)
        self.num_tasks = num_tasks
        self.q_probabilities = np.empty((num_tasks, capacity, num_bins), dtype=np.float32)

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray, q_probability: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.q_probabilities[:, self.insert_index] = q_probability
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_task_batches(self, batch_size_per_task: int | None = None):
        batch_size = 32 if batch_size_per_task is None else batch_size_per_task
        indxs = np.random.randint(self.size, size=batch_size)        
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids), self.q_probabilities[:, indxs]

    def is_full(self):
        return self.size >= self.capacity
    
    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size],
                self.q_probabilities[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index, self.last_indxs), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.q_probabilities[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index, self.last_indxs = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))
        

class PrioritySamplingReplayBuffer(ParallelReplayBuffer):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int, alpha_sampling: float = 0.6, beta: float = 0.4, max_steps: int = 1000000):
        super().__init__(observation_space, action_dim, capacity, num_tasks)
        self.num_tasks = num_tasks
        self.priotity_trees = [SegmentTree(capacity) for _ in range(num_tasks)]
        self.alpha_sampling = alpha_sampling
        self.beta_start = beta
        self.beta = beta
        self.max_steps = max_steps
    
    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray, return_overriden: bool = False):
        if return_overriden:
            observation_overriden = self.observations[:, self.insert_index].copy()
            action_overriden = self.actions[:, self.insert_index].copy()
            reward_overriden = self.rewards[:, self.insert_index].copy()
            mask_overriden = self.masks[:, self.insert_index].copy()
            next_observation_overriden = self.next_observations[:, self.insert_index].copy()

        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation

        for task_id in range(self.num_tasks):
            self.priotity_trees[task_id].add(self.priotity_trees[task_id].max)

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        if return_overriden:
            return Batch(
                observations=observation_overriden,
                actions=action_overriden,
                rewards=reward_overriden,
                masks=mask_overriden,
                next_observations=next_observation_overriden,
                task_ids=np.arange(self.num_tasks)
            )
        else:
            return None
    
    def _sample_idx_from_priority_tree(self, task_id, samples_size):
        p_total = self.priotity_trees[task_id].total  # sum of the priorities
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
            buffer_idxs, tree_idxs, sample_probs = self.priotity_trees[task_id].find(samples)
            if np.all(sample_probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        return buffer_idxs, tree_idxs, sample_probs

    def sample(self, batch_size: int, num_batches: int):
        task_indx = np.zeros((num_batches, batch_size), dtype=np.int32)
        sample_indx = np.zeros((num_batches, batch_size), dtype=np.int32)
        is_weights = np.zeros((num_batches, batch_size), dtype=np.float32)
        tree_idxs = np.zeros((num_batches, batch_size), dtype=np.int32)

        for b in range(num_batches):
            sampled_tasks = np.random.randint(0, self.num_tasks, size=batch_size)
            task_indx[b] = sampled_tasks

            for task_id in np.unique(sampled_tasks):
                positions = np.where(sampled_tasks == task_id)[0]
                count = len(positions)

                buffer_idxs, _tree_idxs, sample_priorities = (
                    self._sample_idx_from_priority_tree(
                        task_id=task_id,
                        samples_size=count,
                    )
                )

                sample_indx[b, positions] = buffer_idxs
                tree_idxs[b, positions] = _tree_idxs

                p_total = self.priotity_trees[task_id].total 
                P = sample_priorities / p_total  

                N = self.size
                weights = (N * P) ** (-self.beta)

                is_weights[b, positions] = weights

        is_weights /= is_weights.max()

        self.last_indxs = (task_indx, sample_indx)
        self.last_tree_indxs = tree_idxs

        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]

        batch = Batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            next_observations=next_observations,
            task_ids=task_indx,
        )

        return batch, is_weights
    
    def update_last_batch_priorities(self, new_priorities: jnp.ndarray):
  
        task_indx = self.last_indxs[0]          # shape (num_batches, batch_size)
        tree_indx = self.last_tree_indxs         # shape (num_batches, batch_size)

        new_priorities = np.asarray(new_priorities) ** self.alpha_sampling

        for task_id in range(self.num_tasks):
            mask = task_indx == task_id
            if not np.any(mask):
                continue

            indices = tree_indx[mask]         
            values = new_priorities[mask]      

            self.priotity_trees[task_id].update(indices, values)
    
    def anneal_priority_beta(self, current_step: int):
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * current_step / self.max_steps)
    
    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size],
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))

        # Save trees
        tree_dir = os.path.join(save_dir, "priority_trees")
        os.makedirs(tree_dir, exist_ok=True)

        for task_id, tree in enumerate(self.priotity_trees):
            tree_path = os.path.join(tree_dir, f"tree_{task_id:03d}.pkl")
            with open(tree_path, "wb") as f:
                pickle.dump(tree.state_dict(), f)

        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index, self.last_indxs, self.last_tree_indxs, self.beta), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size], \
            = data_chunk

        # Load trees
        tree_dir = os.path.join(save_dir, "priority_trees")

        self.priotity_trees = [SegmentTree(self.capacity) for _ in range(self.num_tasks)]

        for task_id in range(self.num_tasks):
            tree_path = os.path.join(tree_dir, f"tree_{task_id:03d}.pkl")
            with open(tree_path, "rb") as f:
                state = pickle.load(f)
            self.priotity_trees[task_id].load_state_dict(state)

        self.size, self.insert_index, self.last_indxs, self.last_tree_indxs, self.beta = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))


class EREBuffer(ParallelReplayBuffer):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int, total_updates: int,decay_const: float = 0.996, min_recent_samples: int = 5000):
        super().__init__(observation_space, action_dim, capacity, num_tasks)
        self.num_tasks = num_tasks
        self.total_updates = total_updates
        self.decay_const = decay_const
        self.min_recent_samples = min_recent_samples

    def sample(self, batch_size: int, num_batches: int, update_count: int):
        k_most_recent = max(
            self.min_recent_samples,
            int(self.capacity * (self.decay_const ** (1000 * update_count / self.total_updates)))
        )

        recent_size = min(self.size, k_most_recent)
        start = (self.insert_index - recent_size) % self.capacity
        offsets = np.random.randint(recent_size, size=(num_batches, batch_size))
        sample_indx = (start + offsets) % self.capacity
        task_indx = np.random.randint(self.num_tasks, size=(num_batches, batch_size))

        self.last_indxs = (task_indx, sample_indx)

        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]

        return Batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            masks=masks,
            next_observations=next_observations,
            task_ids=task_indx
        )