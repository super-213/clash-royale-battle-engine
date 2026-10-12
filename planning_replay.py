"""Episode-aware sequence replay. Terminal successors are never reset states."""
from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np

from .planning_rules import ACTION_COUNT, canonical_mask


@dataclass
class PlanningStep:
    observation: dict
    next_observation: dict
    action: int
    opponent_action: int
    accepted: bool
    reward: float
    done: bool
    ticks: int
    search: dict
    model_version: int = 0
    opponent_id: str = 'random'
    seed: int = -1
    role: int = -1

    def validate(self):
        if not 0 <= self.action < ACTION_COUNT or not 0 <= self.opponent_action < ACTION_COUNT:
            raise ValueError('replay action out of range')
        if not np.isfinite(self.reward) or self.ticks < 0:
            raise ValueError('invalid replay reward or ticks')
        for obs in (self.observation, self.next_observation):
            if not all(np.isfinite(value).all() for value in obs.values()):
                raise ValueError('nonfinite replay observation')
        candidates = np.asarray(self.search['candidates'])
        probabilities = np.asarray(self.search['probabilities'])
        if (candidates.ndim != 1 or probabilities.shape != candidates.shape or len(candidates) < 1
                or len(np.unique(candidates)) != len(candidates) or np.any(candidates < 0)
                or np.any(candidates >= ACTION_COUNT) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
            raise ValueError('invalid replay search target')
        legal = canonical_mask({key: value[None] for key, value in self.observation.items()})[0]
        if not legal[self.action] or not legal[candidates.astype(np.int64)].all():
            raise ValueError('replay contains an illegal root action or candidate')


class SequenceReplay:
    def __init__(self, capacity=32768):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('replay capacity must be positive')
        self.capacity = capacity
        self.episodes = []
        self.active = {}

    def __len__(self):
        return sum(len(episode) for episode in self.episodes)

    def append(self, slot, step):
        step.validate()
        if slot not in self.active:
            episode = []
            self.active[slot] = episode
            self.episodes.append(episode)
        episode = self.active[slot]
        if episode:
            for key in step.observation:
                if not np.array_equal(episode[-1].next_observation[key], step.observation[key]):
                    raise ValueError('replay sequence is not contiguous')
        episode.append(copy.deepcopy(step))
        if step.done:
            del self.active[slot]
        while len(self) > self.capacity:
            removable = next((index for index, item in enumerate(self.episodes)
                              if all(item is not active for active in self.active.values())), None)
            if removable is not None:
                self.episodes.pop(removable)
            else:
                # Bound storage even for exceptionally long unfinished games.
                # The remaining prefix becomes an explicit truncated history.
                longest = max(self.episodes, key=len)
                del longest[0]

    def cut(self):
        """Checkpoint/resume or external reset starts new trajectories."""
        self.active.clear()

    def sample(self, rng, batch_size, unroll, burn_in):
        lengths = np.asarray([len(episode) for episode in self.episodes], dtype=np.int64)
        if not lengths.sum() or batch_size < 1 or unroll < 1 or burn_in < 0:
            raise ValueError('invalid sequence replay sample')
        cumulative = lengths.cumsum()
        samples = []
        for position in rng.integers(int(cumulative[-1]), size=batch_size):
            index = int(np.searchsorted(cumulative, position, side='right'))
            start = int(position - (cumulative[index - 1] if index else 0))
            episode = self.episodes[index]
            samples.append((episode[max(0, start - burn_in):start], episode[start:start + unroll]))
        return samples

    def state_dict(self):
        # Episodes are immutable past data after loading; native games restart.
        return dict(capacity=self.capacity, episodes=self.episodes)

    @classmethod
    def from_state_dict(cls, payload):
        replay = cls(payload['capacity'])
        replay.episodes = payload['episodes']
        for episode in replay.episodes:
            for step in episode:
                step.validate()
            if any(step.done for step in episode[:-1]):
                raise ValueError('replay episode crosses terminal boundary')
        if len(replay) > replay.capacity:
            raise ValueError('replay exceeds capacity')
        return replay
