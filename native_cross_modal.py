"""Cross-modal arena, entity, history, belief and card reasoning modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .native_model import ResidualBlock
from .native_vec_env import ARENA_HEIGHT, ARENA_WIDTH, CARD_IDS


TILES = ARENA_HEIGHT * ARENA_WIDTH


def _coordinate_features(height, width):
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((
        grid_x,
        grid_y,
        torch.sin(math.pi * grid_x),
        torch.cos(math.pi * grid_x),
        torch.sin(math.pi * grid_y),
        torch.cos(math.pi * grid_y),
    ), dim=-1).reshape(height * width, 6)


class NativeCrossModalExtractor(BaseFeaturesExtractor):
    """Perform explicit cross-attention before producing the recurrent input."""

    def __init__(
        self,
        observation_space,
        features_dim=256,
        spatial_channels=64,
        cross_dim=64,
        entity_layers=2,
        attention_heads=4,
    ):
        super().__init__(observation_space, features_dim=int(features_dim))
        if spatial_channels % 8:
            raise ValueError("spatial_channels must be divisible by 8")
        if cross_dim % attention_heads:
            raise ValueError("cross_dim must be divisible by attention_heads")
        self.battle_backend = "native_batch_v1"
        self.observation_schema = (
            "native_model_v4" if observation_space["spatial"].shape[0] == 20
            else "native_model_v3"
        )
        self.architecture_version = "cross_modal_stateful_v3"
        self.cross_dim = int(cross_dim)

        spatial_inputs = observation_space["spatial"].shape[0]
        entity_inputs = observation_space["entities"].shape[-1]
        history_inputs = observation_space["action_history"].shape[-1]
        scalar_inputs = observation_space["scalar"].shape[-1]
        belief_inputs = observation_space["public_belief"].shape[-1]
        temporal_inputs = observation_space["temporal"].shape[-1]

        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(spatial_inputs, spatial_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, spatial_channels),
            nn.SiLU(),
            ResidualBlock(spatial_channels),
            nn.Conv2d(spatial_channels, spatial_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(spatial_channels),
            nn.Conv2d(spatial_channels, cross_dim, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        spatial_height = (ARENA_HEIGHT + 3) // 4
        spatial_width = (ARENA_WIDTH + 3) // 4
        self.register_buffer(
            "spatial_coordinates",
            _coordinate_features(spatial_height, spatial_width),
            persistent=False,
        )
        self.register_buffer(
            "placement_coordinates",
            _coordinate_features(ARENA_HEIGHT, ARENA_WIDTH),
            persistent=False,
        )
        self.spatial_position = nn.Linear(6, cross_dim)
        self.placement_position = nn.Linear(6, cross_dim)
        self.placement_attention = nn.MultiheadAttention(
            cross_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.placement_norm = nn.LayerNorm(cross_dim)

        self.entity_projection = nn.Sequential(
            nn.Linear(entity_inputs, cross_dim),
            nn.LayerNorm(cross_dim),
            nn.SiLU(),
        )
        entity_layer = nn.TransformerEncoderLayer(
            d_model=cross_dim,
            nhead=attention_heads,
            dim_feedforward=cross_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.entity_transformer = nn.TransformerEncoder(
            entity_layer,
            num_layers=int(entity_layers),
            norm=nn.LayerNorm(cross_dim),
            enable_nested_tensor=False,
        )
        self.entity_pool_query = nn.Parameter(torch.zeros(cross_dim))

        self.history_projection = nn.Sequential(
            nn.Linear(history_inputs, cross_dim),
            nn.LayerNorm(cross_dim),
            nn.SiLU(),
        )
        self.history_entity_attention = nn.MultiheadAttention(
            cross_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.history_norm = nn.LayerNorm(cross_dim)
        self.history_lstm = nn.LSTM(
            input_size=cross_dim,
            hidden_size=cross_dim,
            num_layers=1,
            batch_first=True,
        )

        self.belief_projection = nn.Sequential(
            nn.Linear(belief_inputs, cross_dim),
            nn.LayerNorm(cross_dim),
            nn.SiLU(),
        )
        self.belief_attention = nn.MultiheadAttention(
            cross_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.belief_norm = nn.LayerNorm(cross_dim)

        self.card_embedding = nn.Embedding(
            len(CARD_IDS) + 1, cross_dim, padding_idx=0
        )
        self.card_context_attention = nn.MultiheadAttention(
            cross_dim, attention_heads, dropout=0.0, batch_first=True
        )
        self.card_norm = nn.LayerNorm(cross_dim)

        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_inputs, 32), nn.LayerNorm(32), nn.SiLU()
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_inputs, 32), nn.LayerNorm(32), nn.SiLU()
        )
        fused_inputs = cross_dim * 5 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fused_inputs, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )
        self._action_card_tokens = None
        self._action_placement_tokens = None
        self._auxiliary_entity_tokens = None
        self._auxiliary_entity_mask = None
        nn.init.normal_(self.entity_pool_query, std=0.02)

    def _entity_tokens(self, observations):
        valid = observations["entity_mask"] > 0.5
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        safe_valid[:, 0] |= empty
        tokens = self.entity_projection(observations["entities"])
        tokens = self.entity_transformer(
            tokens, src_key_padding_mask=~safe_valid
        )
        # Keep one unmasked zero key for fully empty scenes so MHA stays finite.
        # The synthetic key must not invent an entity representation.
        tokens = tokens * valid.unsqueeze(-1)
        scores = torch.einsum("bnd,d->bn", tokens, self.entity_pool_query)
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bn,bnd->bd", weights, tokens)
        return tokens, safe_valid, pooled * (~empty).unsqueeze(1)

    def _history_context(self, observations, entity_tokens, entity_valid):
        history = observations["action_history"]
        valid = history[..., 0] > 0.5
        projected = self.history_projection(history)
        attended, _ = self.history_entity_attention(
            projected,
            entity_tokens,
            entity_tokens,
            key_padding_mask=~entity_valid,
            need_weights=False,
        )
        enriched = self.history_norm(projected + attended) * valid.unsqueeze(-1)
        outputs, _ = self.history_lstm(enriched)
        lengths = valid.sum(dim=1).long()
        last = (lengths - 1).clamp_min(0)
        rows = torch.arange(history.shape[0], device=history.device)
        summary = outputs[rows, last] * (lengths > 0).unsqueeze(1)
        return enriched, summary

    def _belief_context(self, observations, entity_tokens, entity_valid, history):
        query = self.belief_projection(observations["public_belief"]).unsqueeze(1)
        context = torch.cat((entity_tokens, history.unsqueeze(1)), dim=1)
        context_valid = torch.cat((
            entity_valid,
            torch.ones((entity_valid.shape[0], 1), dtype=torch.bool, device=entity_valid.device),
        ), dim=1)
        attended, _ = self.belief_attention(
            query,
            context,
            context,
            key_padding_mask=~context_valid,
            need_weights=False,
        )
        return self.belief_norm(query + attended).squeeze(1)

    def _card_context(
        self,
        observations,
        entity_tokens,
        entity_valid,
        history,
        belief,
    ):
        hand = observations["hand"].long()
        valid = hand > 0
        queries = self._card_queries(hand)
        context = torch.cat((
            entity_tokens,
            history.unsqueeze(1),
            belief.unsqueeze(1),
        ), dim=1)
        context_valid = torch.cat((
            entity_valid,
            torch.ones((entity_valid.shape[0], 2), dtype=torch.bool, device=entity_valid.device),
        ), dim=1)
        attended, _ = self.card_context_attention(
            queries,
            context,
            context,
            key_padding_mask=~context_valid,
            need_weights=False,
        )
        return self.card_norm(queries + attended) * valid.unsqueeze(-1)

    def _card_queries(self, hand):
        return self.card_embedding(hand)

    def _card_summary(self, cards, hand):
        valid = (hand > 0).unsqueeze(-1)
        return cards.sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    def _spatial_context(self, observations):
        feature_map = self.spatial_backbone(observations["spatial"])
        batch, channels, height, width = feature_map.shape
        spatial_tokens = feature_map.flatten(2).transpose(1, 2)
        spatial_tokens = spatial_tokens + self.spatial_position(
            self.spatial_coordinates
        ).unsqueeze(0)
        local = functional.interpolate(
            feature_map,
            size=(ARENA_HEIGHT, ARENA_WIDTH),
            mode="bilinear",
            align_corners=False,
        ).flatten(2).transpose(1, 2)
        placement_queries = local + self.placement_position(
            self.placement_coordinates
        ).unsqueeze(0)
        attended, _ = self.placement_attention(
            placement_queries,
            spatial_tokens,
            spatial_tokens,
            need_weights=False,
        )
        placements = self.placement_norm(placement_queries + attended)
        return spatial_tokens.mean(dim=1), placements

    def forward(self, observations):
        spatial, placements = self._spatial_context(observations)
        entities, entity_valid, entity_summary = self._entity_tokens(observations)
        _, history = self._history_context(
            observations, entities, entity_valid
        )
        belief = self._belief_context(
            observations, entities, entity_valid, history
        )
        cards = self._card_context(
            observations, entities, entity_valid, history, belief
        )
        scalar = self.scalar_encoder(observations["scalar"])
        temporal = self.temporal_encoder(observations["temporal"])
        self._action_card_tokens = cards[:, :4]
        self._action_placement_tokens = placements
        self._auxiliary_entity_tokens = entities
        self._auxiliary_entity_mask = entity_valid
        card_summary = self._card_summary(cards, observations["hand"])
        return self.fusion(torch.cat((
            spatial,
            entity_summary,
            history,
            belief,
            card_summary,
            scalar,
            temporal,
        ), dim=1))

    def action_tokens(self):
        if self._action_card_tokens is None or self._action_placement_tokens is None:
            raise RuntimeError("extract features before requesting action tokens")
        return self._action_card_tokens, self._action_placement_tokens

    def auxiliary_entity_tokens(self):
        if self._auxiliary_entity_tokens is None:
            raise RuntimeError("extract features before requesting entity tokens")
        return self._auxiliary_entity_tokens, self._auxiliary_entity_mask


class CrossModalActionHead(nn.Module):
    """Produce card-conditioned placement logits from contextual tokens."""

    def __init__(self, actor_dim, cross_dim):
        super().__init__()
        self.wait_head = nn.Linear(actor_dim, 1)
        self.actor_to_card = nn.Linear(actor_dim, cross_dim)
        self.card_projection = nn.Linear(cross_dim, cross_dim)
        self.card_slot_score = nn.Linear(cross_dim, 1)
        self.placement_projection = nn.Linear(cross_dim, cross_dim)
        self.scale = math.sqrt(cross_dim)

    def forward(self, actor, cards, placements):
        card_queries = torch.tanh(
            self.card_projection(cards) + self.actor_to_card(actor).unsqueeze(1)
        )
        placement_keys = self.placement_projection(placements)
        position_logits = torch.einsum(
            "bcd,btd->bct", card_queries, placement_keys
        ) / self.scale
        slot_logits = torch.cat((
            self.wait_head(actor),
            self.card_slot_score(card_queries).squeeze(-1),
        ), dim=1)
        return torch.cat((
            slot_logits,
            position_logits.reshape(actor.shape[0], 4 * TILES),
        ), dim=1)


__all__ = ["CrossModalActionHead", "NativeCrossModalExtractor"]
