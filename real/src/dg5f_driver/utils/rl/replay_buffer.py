import collections
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from .dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
          data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys(), (
            dataset_dict.keys(),
            data_dict.keys(),
        )
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        obs_dim,
        action_dim: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        # if next_observation_space is None:
        #     next_observation_space = observation_space

        dataset_dict = dict(
            observations=np.empty((capacity, obs_dim), dtype=np.float32),
            next_observations=np.empty((capacity, obs_dim), dtype=np.float32),
            actions=np.empty((capacity, action_dim), dtype=np.float32),
            rewards=np.empty((capacity, 2), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self.n_inserts = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict, avg_reward: Optional[float] = None):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        self.n_inserts += 1

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device='cuda'):
        queue = collections.deque()

        def _to_torch(data):
            if isinstance(data, dict) or hasattr(data, "items"):
                return {k: _to_torch(v) for k, v in data.items()}
            elif isinstance(data, np.ndarray):
                return torch.as_tensor(data, device=device)
            else:
                return data

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(_to_torch(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

