# This code is shamelessly stolen from https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
import numpy as np
import random
from joblib import Parallel, delayed

def get_sample(storage, storage_row_index, obses_t, actions, rewards, obses_tp1, dones):
    #print(storage_row_index)
    data = storage[storage_row_index]
    obs_t, action, reward, obs_tp1, done = data
    obses_t.append(np.array(obs_t, copy=False))
    actions.append(np.array(action, copy=False))
    rewards.append(reward)
    obses_tp1.append(np.array(obs_tp1, copy=False))
    dones.append(done)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []

        #with Parallel(n_jobs=100, backend='threading') as parallel:
        #    results = parallel(delayed(get_sample)(self._storage, i, obses_t, actions, rewards, obses_tp1, dones) for i in idxes)

        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        #idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        idxes = np.random.randint(len(self._storage), size=batch_size)
        return self._encode_sample(idxes)
