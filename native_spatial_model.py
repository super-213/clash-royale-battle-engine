"""Full-resolution, card-conditioned spatial action modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .native_cross_modal import NativeCrossModalExtractor
from .native_model import ResidualBlock
from .native_vec_env import ARENA_HEIGHT, ARENA_WIDTH


class NativeSpatialHeatmapExtractor(NativeCrossModalExtractor):
    """Keep a 32x18 CNN map for the actor instead of upsampling coarse tokens."""

    def __init__(
        self,
        observation_space,
        features_dim=256,
        spatial_channels=64,
        cross_dim=64,
        entity_layers=2,
        attention_heads=4,
    ):
        super().__init__(
            observation_space,
            features_dim=features_dim,
            spatial_channels=spatial_channels,
            cross_dim=cross_dim,
            entity_layers=entity_layers,
            attention_heads=attention_heads,
        )
        spatial_channels = int(spatial_channels)
        spatial_inputs = observation_space["spatial"].shape[0]

        # Replace the v3 downsample/query path.  The entity/history/belief/card
        # cross-attention modules stay unchanged and old v3 checkpoints retain
        # their original policy class.
        del self.spatial_backbone
        del self.spatial_position
        del self.placement_position
        del self.placement_attention
        del self.placement_norm
        del self.spatial_coordinates

        self.spatial_backbone = nn.Sequential(
            nn.Conv2d(
                spatial_inputs, spatial_channels, 3, padding=1, bias=False
            ),
            nn.GroupNorm(8, spatial_channels),
            nn.SiLU(),
            ResidualBlock(spatial_channels),
            ResidualBlock(spatial_channels),
            nn.Conv2d(spatial_channels, self.cross_dim, 1, bias=False),
        )
        self.placement_position = nn.Linear(6, self.cross_dim)
        self.placement_norm = nn.LayerNorm(self.cross_dim)
        self.spatial_feature_shape = (ARENA_HEIGHT, ARENA_WIDTH)
        self.architecture_version = "spatial_heatmap_stateful_v4"

    def _spatial_context(self, observations):
        feature_map = self.spatial_backbone(observations["spatial"])
        if tuple(feature_map.shape[-2:]) != self.spatial_feature_shape:
            raise RuntimeError(
                "spatial action map must preserve the 32x18 arena resolution"
            )
        spatial_tokens = feature_map.flatten(2).transpose(1, 2)
        spatial_tokens = self.placement_norm(
            spatial_tokens
            + self.placement_position(self.placement_coordinates).unsqueeze(0)
        )
        return spatial_tokens.mean(dim=1), spatial_tokens


class NativeRoleAwareSpatialHeatmapExtractor(NativeSpatialHeatmapExtractor):
    """v7: distinguish playable cards from the next draw in every branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.card_role_embedding = nn.Embedding(2, self.cross_dim)
        self.register_buffer("card_roles", torch.tensor([0, 0, 0, 0, 1]),
                             persistent=False)
        self.fusion[0] = nn.Linear(self.cross_dim * 6 + 64, self.features_dim)
        self.architecture_version = "role_aware_spatial_stateful_v7"

    def _card_queries(self, hand):
        return self.card_embedding(hand) + self.card_role_embedding(self.card_roles).unsqueeze(0)

    def _card_summary(self, cards, hand):
        valid = (hand[:, :4] > 0).unsqueeze(-1)
        current = cards[:, :4].sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return torch.cat((current, cards[:, 4]), dim=-1)


class SpatialHeatmapActionHead(nn.Module):
    """Implement P(card) then P(tile | card, full-resolution CNN map)."""

    def __init__(self, actor_dim, cross_dim):
        super().__init__()
        self.wait_head = nn.Linear(actor_dim, 1)
        self.actor_to_card = nn.Linear(actor_dim, cross_dim)
        self.card_projection = nn.Linear(cross_dim, cross_dim)
        self.card_slot_score = nn.Linear(cross_dim, 1)
        self.spatial_key_projection = nn.Linear(cross_dim, cross_dim)
        self.spatial_bias = nn.Linear(cross_dim, 1)
        self.scale = math.sqrt(cross_dim)
        self._last_heatmaps = None

    def forward(self, actor, cards, spatial_tokens):
        # The recurrent actor state supplies temporal context, while every card
        # keeps its own query and therefore its own placement preference.
        card_queries = torch.tanh(
            self.card_projection(cards) + self.actor_to_card(actor).unsqueeze(1)
        )
        spatial_keys = self.spatial_key_projection(spatial_tokens)
        heatmaps = torch.einsum(
            "bcd,btd->bct", card_queries, spatial_keys
        ) / self.scale
        heatmaps = heatmaps + self.spatial_bias(spatial_tokens).transpose(1, 2)
        self._last_heatmaps = heatmaps.detach().reshape(
            actor.shape[0], 4, ARENA_HEIGHT, ARENA_WIDTH
        )
        slot_logits = torch.cat((
            self.wait_head(actor),
            self.card_slot_score(card_queries).squeeze(-1),
        ), dim=1)
        return torch.cat((slot_logits, heatmaps.flatten(1)), dim=1)

    def heatmaps(self):
        if self._last_heatmaps is None:
            raise RuntimeError("run the spatial action head before reading heatmaps")
        return self._last_heatmaps


__all__ = ["NativeSpatialHeatmapExtractor", "NativeRoleAwareSpatialHeatmapExtractor", "SpatialHeatmapActionHead"]
