"""Recurrent public-state stochastic model for sampled multi-step planning.

The prior predicts public outcomes, not an oracle opponent hand. Training uses
a posterior over outcome codes; online search only samples the prior. All tree
predictions retain spatial features and can be fed into dynamics again.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from .native_spatial_model import NativeSpatialHeatmapExtractor
from .native_vec_env import native_observation_space
from .planning_rules import ACTION_COUNT, TILES, CARD_COSTS, SPELL_TOKENS, action_features


@dataclass
class PlanningState:
    memory: torch.Tensor
    board: torch.Tensor
    hand: torch.Tensor
    next_card: torch.Tensor
    scalar: torch.Tensor
    belief: torch.Tensor

    def select(self, indices):
        return type(self)(*(getattr(self, field.name)[indices] for field in fields(self)))

    def detach(self):
        return type(self)(*(getattr(self, field.name).detach() for field in fields(self)))

    @classmethod
    def stack(cls, states):
        return cls(*(torch.cat([getattr(state, field.name) for state in states]) for field in fields(cls)))


class PlanningModel(nn.Module):
    architecture_version = 'public_stochastic_planning_v1'
    observation_schema = 'native_model_v3'

    def __init__(self, hidden_dim=256, spatial_dim=64, chance_codes=16,
                 entity_layers=2, attention_heads=4, decision_ticks=10):
        super().__init__()
        if (hidden_dim <= 0 or spatial_dim <= 0 or attention_heads <= 0 or entity_layers < 1
                or spatial_dim % 8 or spatial_dim % attention_heads):
            raise ValueError('invalid planning dimensions')
        if chance_codes < 2 or decision_ticks <= 0:
            raise ValueError('planning needs positive ticks and at least two outcome codes')
        self.settings = dict(hidden_dim=hidden_dim, spatial_dim=spatial_dim,
                             chance_codes=chance_codes, entity_layers=entity_layers,
                             attention_heads=attention_heads, decision_ticks=decision_ticks)
        self.hidden_dim, self.spatial_dim = hidden_dim, spatial_dim
        self.chance_codes, self.decision_ticks = chance_codes, decision_ticks
        self.encoder = NativeSpatialHeatmapExtractor(
            native_observation_space(), features_dim=hidden_dim, spatial_channels=spatial_dim,
            cross_dim=spatial_dim, entity_layers=entity_layers, attention_heads=attention_heads)
        self.filter = nn.GRUCell(hidden_dim, hidden_dim)
        context_dim = hidden_dim + spatial_dim + 8 + 9 + 12 + 20
        self.context = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.own_card = nn.Linear(hidden_dim, 9)
        self.enemy_card = nn.Linear(hidden_dim, 9)
        self.own_query = nn.Linear(hidden_dim, spatial_dim * 8)
        self.enemy_query = nn.Linear(hidden_dim, spatial_dim * 8)
        self.keys = nn.Conv2d(spatial_dim, spatial_dim, 1)
        self.value = nn.Linear(hidden_dim, 1)
        self.action_encoder = nn.Sequential(nn.Linear(22 + 2 * spatial_dim, hidden_dim), nn.SiLU())
        self.prior = nn.Linear(hidden_dim * 2, chance_codes)
        self.posterior = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, chance_codes))
        self.outcome_embedding = nn.Linear(chance_codes, hidden_dim, bias=False)
        self.transition = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.board_context = nn.Linear(hidden_dim, spatial_dim)
        self.board_transition = nn.Sequential(
            nn.Conv2d(spatial_dim * 2 + 2, spatial_dim, 3, padding=1), nn.SiLU(),
            nn.Conv2d(spatial_dim, spatial_dim, 3, padding=1))
        self.board_norm = nn.GroupNorm(8, spatial_dim)
        self.scalar_head = nn.Linear(hidden_dim, 12)
        self.belief_head = nn.Linear(hidden_dim, 20)
        self.next_card_head = nn.Linear(hidden_dim, 9)
        self.success = nn.Linear(hidden_dim, 1)
        self.reward = nn.Linear(hidden_dim, 1)
        self.terminal = nn.Linear(hidden_dim, 1)
        self.mask_head = nn.Conv2d(spatial_dim, 8, 1)
        self.spatial_decoder = nn.Conv2d(spatial_dim, 16, 1)
        self.register_buffer('costs', torch.tensor(CARD_COSTS) / 10.)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_training_mode(self, mode):
        self.train(mode)

    def observations(self, observations):
        result = {key: torch.as_tensor(value, device=self.device) for key, value in observations.items()}
        return {key: value.float() if value.is_floating_point() else value for key, value in result.items()}

    def encode(self, observations, previous=None, starts=None):
        obs = self.observations(observations)
        features = self.encoder(obs)
        count = len(features)
        if previous is None:
            previous = features.new_zeros((count, self.hidden_dim))
        if starts is not None:
            previous = previous * (1 - torch.as_tensor(starts, device=self.device).float()).unsqueeze(-1)
        memory = self.filter(features, previous)
        # Full-resolution contextual spatial tokens from the existing encoder.
        board = self.encoder.action_tokens()[1].transpose(1, 2).reshape(count, self.spatial_dim, 32, 18)
        cards = F.one_hot(obs['hand'][:, :4].long(), 9)[..., 1:].sum(1).float().clamp_max(1)
        next_card = F.one_hot(obs['hand'][:, 4].long(), 9).float()
        return PlanningState(memory, board, cards, next_card, obs['scalar'].float(), obs['public_belief'].float())

    def features(self, state):
        return self.context(torch.cat((state.memory, state.board.mean((-2, -1)), state.hand,
                                       state.next_card, state.scalar, state.belief), -1))

    def evaluate(self, state):
        context = self.features(state)
        keys = self.keys(state.board).flatten(2)
        def policy(card_head, query_head, opponent=False):
            card_logits = card_head(context)
            if opponent:
                possible = state.belief[:, :8] > 1e-6
                possible = possible | ~possible.any(-1, keepdim=True)
                card_logits = torch.cat((card_logits[:, :1], card_logits[:, 1:].masked_fill(~possible, -1e9)), -1)
            card = F.log_softmax(card_logits, -1)
            queries = query_head(context).reshape(-1, 8, self.spatial_dim)
            tiles = F.log_softmax(torch.bmm(queries, keys) / self.spatial_dim ** .5, -1)
            return torch.cat((card[:, :1], (card[:, 1:, None] + tiles).flatten(1)), -1)
        return dict(policy=policy(self.own_card, self.own_query),
                    opponent=policy(self.enemy_card, self.enemy_query, True), value=self.value(context).squeeze(-1))

    def predicted_mask(self, state):
        placement = self.mask_head(state.board).flatten(2) > 0
        for card in SPELL_TOKENS:
            placement[:, card - 1] = True
        affordable = state.scalar[:, :1] + 1e-6 >= self.costs
        placement = placement & (state.hand > .5).unsqueeze(-1) & affordable.unsqueeze(-1)
        return torch.cat((torch.ones((len(state.memory), 1), device=self.device, dtype=torch.bool),
                          placement.flatten(1)), -1)

    def chance_distribution(self, state, own_actions, enemy_actions):
        context = self.features(state)
        action = self.joint_action_context(state, own_actions, enemy_actions)
        return self.prior(torch.cat((context, action), -1))

    def joint_action_context(self, state, own_actions, enemy_actions):
        pieces = [action_features(own_actions, self.device), action_features(enemy_actions, self.device)]
        board = state.board.flatten(2).transpose(1, 2)
        for actions in (own_actions, enemy_actions):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.long)
            tiles = (actions - 1).clamp_min(0) % TILES
            local = board[torch.arange(len(actions), device=self.device), tiles]
            pieces.append(local * (actions != 0)[:, None])
        return self.action_encoder(torch.cat(pieces, -1))

    def dynamics(self, state, own_actions, enemy_actions, outcomes=None, target=None, ticks=None):
        own_actions = torch.as_tensor(own_actions, device=self.device, dtype=torch.long)
        enemy_actions = torch.as_tensor(enemy_actions, device=self.device, dtype=torch.long)
        context = self.features(state)
        action = self.joint_action_context(state, own_actions, enemy_actions)
        prior = self.prior(torch.cat((context, action), -1))
        posterior = None
        if target is not None:
            posterior = self.posterior(torch.cat((context, action, self.features(target).detach()), -1))
            # Straight-through outcome samples train decoder and posterior.
            outcomes = F.gumbel_softmax(posterior, tau=1., hard=True)
        elif outcomes is None:
            raise ValueError('search must supply prior-sampled outcome codes')
        else:
            outcomes = F.one_hot(torch.as_tensor(outcomes, device=self.device).long(), self.chance_codes).float()
        memory = self.transition(torch.cat((action, self.outcome_embedding(outcomes)), -1), state.memory)
        stamps = state.board.new_zeros((len(memory), 2, TILES))
        for side, actions in enumerate((own_actions, enemy_actions)):
            tiles = (actions - 1).clamp_min(0) % TILES
            stamps[:, side].scatter_(1, tiles[:, None], (actions != 0).float()[:, None])
        board_context = self.board_context(memory)[:, :, None, None].expand_as(state.board)
        board = self.board_norm(state.board + self.board_transition(torch.cat(
            (state.board, board_context, stamps.reshape(-1, 2, 32, 18)), 1)))
        scalar = self.scalar_head(memory).sigmoid()
        belief = self.belief_head(memory).sigmoid()
        success_logit = self.success(memory).squeeze(-1)
        success = (success_logit > 0).float()
        # Known own card identity and next draw are updated by rules. Unknown
        # subsequent draws are predicted from the outcome-conditioned memory.
        played_card = ((own_actions - 1).clamp_min(0) // TILES + 1)
        played = F.one_hot(played_card, 9)[:, 1:].float() * (own_actions != 0)[:, None]
        accepted = success * (own_actions != 0).float()
        hand = (state.hand - played * accepted[:, None]
                + state.next_card[:, 1:] * accepted[:, None]).clamp(0, 1)
        next_logits = self.next_card_head(memory)
        raw_next_logits = next_logits
        allowed = torch.cat((torch.zeros((len(memory), 1), device=self.device, dtype=torch.bool), hand < .5), -1)
        # The just-played card enters the tail of the four-card draw queue,
        # so it cannot be the immediate next preview after a successful play.
        allowed[:, 1:] &= ~((played > .5) & (accepted[:, None] > .5))
        next_logits = next_logits.masked_fill(~allowed, -1e9)
        predicted_next = F.one_hot(next_logits.argmax(-1), 9).float()
        next_card = torch.where(accepted[:, None] > .5, predicted_next, state.next_card)
        duration = state.scalar.new_full((len(memory),), self.decision_ticks) if ticks is None else torch.as_tensor(
            ticks, device=self.device, dtype=state.scalar.dtype)
        begin = state.scalar[:, 4] * 300.
        end = begin + duration * .05
        regen = torch.zeros_like(begin)
        for left, right, rate in ((0., 120., 2.8), (120., 240., 1.4), (240., 1e8, 2.8 / 3)):
            regen = regen + (end.clamp_max(right) - begin.clamp_min(left)).clamp_min(0) / rate / 10.
        own_elixir = (state.scalar[:, 0] - (played * self.costs).sum(-1) * accepted + regen).clamp(0, 1)
        # Avoid in-place writes on tensors needed by sigmoid backward.
        scalar = torch.cat((own_elixir[:, None], scalar[:, 1:4], (end / 300.).clamp_max(1)[:, None],
                            torch.stack((end < 120, (end >= 120) & (end < 240), end >= 240), -1).float(),
                            state.scalar[:, 8:9], scalar[:, 9:]), -1)
        next_state = PlanningState(memory, board, hand, next_card, scalar, belief)
        return next_state, dict(prior=prior, posterior=posterior, reward=self.reward(memory).squeeze(-1),
                                terminal=self.terminal(memory).squeeze(-1), success=success_logit,
                                next_card=raw_next_logits, spatial=self.spatial_decoder(board),
                                mask=self.mask_head(board).flatten(1))

    def initialize_encoder(self, teacher):
        """Copy compatible encoder tensors only; dynamics is never fabricated."""
        source = teacher.policy.features_extractor.state_dict()
        destination = self.encoder.state_dict()
        copied = {key: value for key, value in source.items()
                  if key in destination and destination[key].shape == value.shape}
        self.encoder.load_state_dict(copied, strict=False)
        return sorted(copied)
