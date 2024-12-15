import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # Add 1 for the first (0th) frame.
    return next(iter(episode.values())).shape[0]

class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
                self._current_episode[spec.name] = []
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_path = self._replay_dir / eps_fn
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **episode)
            buffer.seek(0)
            with save_path.open('wb') as f:
                f.write(buffer.read())

class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, multistep,
                 discount, fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._multistep = multistep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = np.load(eps_fn)
            episode = {k: episode[k] for k in episode.keys()}
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

            if not self._save_snapshot:
                eps_fn.unlink(missing_ok=True)
        except:
            print(f'Could not load episode: {eps_fn}')
            print(traceback.format_exc())

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            self._store_episode(eps_fn)

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        
        # Add +1 for first dummy transition and ensure room for next state/action
        idx = np.random.randint(0, episode_len(episode) - 2) + 1
        
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + 1]
        reward = episode['reward'][idx]
        discount = episode['discount'][idx]

        # States and actions we need for the SASA sequence
        states_seq = np.stack([episode['observation'][idx-1], episode['observation'][idx]])
        actions_seq = np.stack([episode['action'][idx], episode['action'][idx+1]])

        return (obs, action, action, states_seq, actions_seq, 
                reward, discount, next_obs, episode['observation'][idx + 1])

    def __iter__(self):
        while True:
            yield self._sample()

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                      save_snapshot, nstep, multistep, discount):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                          max_size_per_worker,
                          num_workers,
                          nstep,
                          multistep,
                          discount,
                          fetch_every=1000,
                          save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True,
                                       worker_init_fn=_worker_init_fn)
    return loader