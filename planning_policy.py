"""Checkpointable planner with the project's predict/state contract."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
import zlib

import numpy as np
import torch

from .native_search import SampledSearch, SearchSettings
from .planning_model import PlanningModel
from .planning_rules import canonical_mask, public_actions


class NativePlanningAgent:
    """Not a PPO algorithm: search targets are trained by planning_training."""
    def __init__(self, model_settings=None, search_settings=None, device='cpu', seed=0):
        self.device = torch.device(device)
        self.policy = PlanningModel(**(model_settings or {})).to(self.device)
        self.search_settings = SearchSettings(**(search_settings or {}))
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.num_timesteps = 0
        self.version = 0
        self.search_enabled = True
        self.last_search = []
        self.training_env = None

    def get_env(self):
        return self.training_env

    def get_vec_normalize_env(self):
        return None

    @torch.no_grad()
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        self.policy.eval()
        count = len(observation['hand'])
        previous = None if state is None else torch.as_tensor(state[0][0], device=self.device, dtype=torch.float32)
        if previous is not None and previous.shape != (count, self.policy.hidden_dim):
            raise ValueError('planning recurrent state has incompatible shape')
        root = self.policy.encode(observation, previous, episode_start)
        masks = torch.as_tensor(canonical_mask(observation), device=self.device)
        # Stable per-observation RNG makes evaluation independent of batching and
        # does not consume the agent's training generator.
        if deterministic:
            rngs = [np.random.default_rng((self.seed + zlib.crc32(b''.join(
                np.asarray(observation[key][row]).tobytes() for key in sorted(observation)))) % (2**32))
                    for row in range(count)]
        else:
            rngs = [np.random.default_rng(int(self.rng.integers(2**32))) for _ in range(count)]
        settings = copy.copy(self.search_settings)
        if not self.search_enabled:
            settings.budget = 0
        identities, self.last_search = SampledSearch(self.policy, settings).run(root, masks, rngs, deterministic)
        actions = public_actions(observation['hand'], identities)
        self._last_root = root.detach()
        return actions, (root.memory.cpu().numpy()[None],)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(schema='native_planning_agent_v1', model_settings=self.policy.settings,
                       search_settings=asdict(self.search_settings), state_dict=self.policy.state_dict(),
                       num_timesteps=self.num_timesteps, version=self.version, seed=self.seed,
                       rng=copy.deepcopy(self.rng.bit_generator.state))
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device='cpu', **kwargs):
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get('schema') != 'native_planning_agent_v1':
            raise ValueError('unsupported planning checkpoint')
        agent = cls(payload['model_settings'], payload['search_settings'], device, payload['seed'])
        agent.policy.load_state_dict(payload['state_dict'])
        agent.policy.eval()
        agent.num_timesteps = int(payload['num_timesteps'])
        agent.version = int(payload['version'])
        agent.rng.bit_generator.state = payload['rng']
        return agent
