"""Auxiliary world-model heads shared by recurrent PPO encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .native_vec_env import EXECUTED_ACTION_FEATURES


DAMAGE_CLASSES = 5
ELIXIR_BINS = 5
CARD_CLASSES = 8


class AuxiliaryWorldModelHead(nn.Module):
    """Predict public next-state dynamics without replacing the PPO heads."""

    def __init__(self, actor_dim, entity_dim, hidden_dim=128, action_conditioned=False):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.action_conditioned = bool(action_conditioned)
        self.global_encoder = nn.Sequential(
            nn.Linear(actor_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        if self.action_conditioned:
            self.action_encoder = nn.Sequential(
                nn.Linear(EXECUTED_ACTION_FEATURES, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            self.dynamics_encoder = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
        self.tower_delta = nn.Linear(hidden_dim, 2)
        self.damage_source = nn.Linear(hidden_dim, DAMAGE_CLASSES)
        self.opponent_elixir = nn.Linear(hidden_dim, ELIXIR_BINS)
        self.opponent_hand = nn.Linear(hidden_dim, CARD_CLASSES)
        self.opponent_next_card = nn.Linear(hidden_dim, CARD_CLASSES)
        self.future_value = nn.Linear(hidden_dim, 1)

        self.entity_projection = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.entity_context = nn.Linear(hidden_dim, hidden_dim)
        self.entity_norm = nn.LayerNorm(hidden_dim)
        self.entity_death = nn.Linear(hidden_dim, 1)
        self.entity_position_delta = nn.Linear(hidden_dim, 2)

    def forward(self, actor, entity_tokens, executed_actions=None):
        global_features = self.global_encoder(actor)
        dynamics_features = global_features
        if self.action_conditioned:
            if executed_actions is None:
                raise ValueError("action-conditioned prediction requires executed_actions")
            executed_actions = torch.as_tensor(executed_actions, device=actor.device, dtype=actor.dtype)
            if executed_actions.shape != (actor.shape[0], EXECUTED_ACTION_FEATURES):
                raise ValueError("invalid executed_actions shape")
            action_features = self.action_encoder(executed_actions)
            dynamics_features = self.dynamics_encoder(torch.cat(
                (global_features, action_features), dim=-1
            ))
        entity_features = self.entity_norm(
            self.entity_projection(entity_tokens)
            + self.entity_context(dynamics_features).unsqueeze(1)
        )
        entity_features = functional.silu(entity_features)
        return {
            "tower_delta": self.tower_delta(dynamics_features),
            "damage_source": self.damage_source(dynamics_features),
            "opponent_elixir": self.opponent_elixir(global_features),
            "opponent_hand": self.opponent_hand(global_features),
            "opponent_next_card": self.opponent_next_card(global_features),
            "future_value": self.future_value(dynamics_features).squeeze(-1),
            "entity_death": self.entity_death(entity_features).squeeze(-1),
            "entity_position_delta": torch.tanh(
                self.entity_position_delta(entity_features)
            ),
        }


def masked_mean(values, mask):
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def damage_source_targets(temporal):
    """Map the next interval's public damage deltas to five source classes."""
    scores = torch.stack((
        torch.zeros_like(temporal[:, 0]),
        (-temporal[:, 2]).clamp_min(0),
        (-temporal[:, 3]).clamp_min(0),
        torch.maximum((-temporal[:, 10]).clamp_min(0), temporal[:, 8]),
        torch.maximum((-temporal[:, 11]).clamp_min(0), temporal[:, 9]),
    ), dim=1)
    return scores.argmax(dim=1)


__all__ = [
    "AuxiliaryWorldModelHead",
    "damage_source_targets",
    "masked_mean",
]
