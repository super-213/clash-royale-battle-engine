"""CNN + entity Transformer + LSTM policy for ``NativeBatchVecEnv``.

The three encoders have deliberately separate responsibilities.  The CNN sees
arena occupancy and local combat state, self-attention reasons over public
entities regardless of their grid collisions, and the LSTM summarizes the
ordered successful-play stream.  Their outputs are fused with the hand and
current public context before the actor and critic branches split.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .native_vec_env import ARENA_HEIGHT, ARENA_WIDTH, CARD_IDS


TILES = ARENA_HEIGHT * ARENA_WIDTH


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, inputs):
        return self.activation(inputs + self.layers(inputs))


class MPSCompatibleAdaptiveAvgPool2d(nn.Module):
    """Adaptive pooling expressed as reductions supported by Apple MPS."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = tuple(output_size)

    def forward(self, features):
        input_height, input_width = features.shape[-2:]
        output_height, output_width = self.output_size
        rows = []
        for output_y in range(output_height):
            start_y = output_y * input_height // output_height
            end_y = (
                (output_y + 1) * input_height + output_height - 1
            ) // output_height
            columns = []
            for output_x in range(output_width):
                start_x = output_x * input_width // output_width
                end_x = (
                    (output_x + 1) * input_width + output_width - 1
                ) // output_width
                columns.append(
                    features[..., start_y:end_y, start_x:end_x].mean(
                        dim=(-2, -1)
                    )
                )
            rows.append(torch.stack(columns, dim=-1))
        return torch.stack(rows, dim=-2)


class NativeHybridExtractor(BaseFeaturesExtractor):
    """Fuse spatial, set-like entity, temporal and current-context encoders."""

    def __init__(
        self,
        observation_space,
        features_dim=256,
        spatial_channels=64,
        entity_dim=64,
        entity_layers=2,
        entity_heads=4,
        history_dim=64,
        card_dim=16,
    ):
        super().__init__(observation_space, features_dim=int(features_dim))
        self.battle_backend = "native_batch_v1"
        self.observation_schema = "native_model_v3"
        self.architecture_version = "cnn_entity_transformer_event_lstm_v2"
        if entity_dim % entity_heads:
            raise ValueError("entity_dim must be divisible by entity_heads")
        if spatial_channels % 8:
            raise ValueError("spatial_channels must be divisible by 8")

        spatial_inputs = observation_space["spatial"].shape[0]
        entity_inputs = observation_space["entities"].shape[-1]
        history_inputs = observation_space["action_history"].shape[-1]
        scalar_inputs = observation_space["scalar"].shape[-1]
        belief_inputs = observation_space["public_belief"].shape[-1]
        temporal_inputs = observation_space["temporal"].shape[-1]

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(spatial_inputs, spatial_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, spatial_channels),
            nn.SiLU(),
            ResidualBlock(spatial_channels),
            nn.Conv2d(spatial_channels, spatial_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(spatial_channels),
            nn.Conv2d(spatial_channels, 96, 3, stride=2, padding=1),
            nn.SiLU(),
            MPSCompatibleAdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(96 * 2 * 2, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

        self.entity_projection = nn.Sequential(
            nn.Linear(entity_inputs, entity_dim),
            nn.LayerNorm(entity_dim),
            nn.SiLU(),
        )
        entity_layer = nn.TransformerEncoderLayer(
            d_model=entity_dim,
            nhead=entity_heads,
            dim_feedforward=entity_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.entity_transformer = nn.TransformerEncoder(
            entity_layer,
            num_layers=int(entity_layers),
            norm=nn.LayerNorm(entity_dim),
            enable_nested_tensor=False,
        )
        self.entity_pool_query = nn.Parameter(torch.zeros(entity_dim))

        self.history_projection = nn.Sequential(
            nn.Linear(history_inputs, history_dim),
            nn.LayerNorm(history_dim),
            nn.SiLU(),
        )
        self.history_lstm = nn.LSTM(
            input_size=history_dim,
            hidden_size=history_dim,
            num_layers=1,
            batch_first=True,
        )

        self.card_embedding = nn.Embedding(
            len(CARD_IDS) + 1,
            card_dim,
            padding_idx=0,
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_inputs, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.belief_encoder = nn.Sequential(
            nn.Linear(belief_inputs, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_inputs, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
        )
        fused_inputs = 128 + entity_dim + history_dim + 5 * card_dim + 32 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fused_inputs, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )
        nn.init.normal_(self.entity_pool_query, std=0.02)

    def _entity_features(self, observations):
        valid = observations["entity_mask"] > 0.5
        # TransformerEncoder cannot accept an all-padding row.  Native states
        # contain towers, but keep the encoder well-defined for synthetic tests
        # and terminal edge cases.
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
        tokens = self.entity_projection(observations["entities"])
        tokens = self.entity_transformer(
            tokens,
            src_key_padding_mask=~safe_valid,
        )
        scores = torch.einsum("bnd,d->bn", tokens, self.entity_pool_query)
        scores = scores.masked_fill(~safe_valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bn,bnd->bd", weights, tokens)
        return pooled * (~empty).unsqueeze(1)

    def _history_features(self, observations):
        history = observations["action_history"]
        valid = history[..., 0] > 0.5
        lengths = valid.sum(dim=1).long()
        encoded = self.history_projection(history)
        outputs, _ = self.history_lstm(encoded)
        last_indices = (lengths - 1).clamp_min(0)
        batch_indices = torch.arange(history.shape[0], device=history.device)
        features = outputs[batch_indices, last_indices]
        return features * (lengths > 0).unsqueeze(1)

    def forward(self, observations):
        spatial = self.spatial_encoder(observations["spatial"])
        entities = self._entity_features(observations)
        history = self._history_features(observations)
        hand = self.card_embedding(observations["hand"].long()).flatten(1)
        scalar = self.scalar_encoder(observations["scalar"])
        belief = self.belief_encoder(observations["public_belief"])
        temporal = self.temporal_encoder(observations["temporal"])
        return self.fusion(torch.cat(
            (spatial, entities, history, hand, scalar, belief, temporal), dim=1
        ))


class HierarchicalMaskedDistribution(Distribution):
    """Exact ``P(card) * P(tile | card)`` distribution with public masks."""

    def __init__(self, slot_entropy_coef=1.0, position_entropy_coef=1.0):
        super().__init__()
        self.slot_entropy_coef = float(slot_entropy_coef)
        self.position_entropy_coef = float(position_entropy_coef)
        self.slot_distribution = None
        self.position_distributions = []
        self.action_masks = None
        self.slot_valid = None

    def proba_distribution_net(self, latent_dim):
        return nn.Linear(latent_dim, 5 + 4 * TILES)

    def proba_distribution(self, action_logits, action_masks):
        masks = action_masks.reshape(-1, 5, TILES).bool().clone()
        # Recurrent minibatches pad observations with zeros. Give those rows a
        # finite, deterministic wait distribution; they are excluded from loss.
        masks[:, 0, 0] |= ~masks.flatten(1).any(dim=1)
        slot_logits = action_logits[:, :5]
        position_logits = action_logits[:, 5:].reshape(-1, 4, TILES)
        minimum = torch.finfo(action_logits.dtype).min
        self.action_masks = masks
        self.slot_valid = masks.any(dim=-1)
        self.slot_distribution = Categorical(
            logits=slot_logits.masked_fill(~self.slot_valid, minimum)
        )
        self.position_distributions = [
            Categorical(
                logits=position_logits[:, slot - 1].masked_fill(
                    ~masks[:, slot], minimum
                )
            )
            for slot in range(1, 5)
        ]
        return self

    @staticmethod
    def _decode(slots, tiles):
        tiles = torch.where(slots == 0, torch.zeros_like(tiles), tiles)
        return torch.stack((slots, tiles // ARENA_WIDTH, tiles % ARENA_WIDTH), dim=1)

    def _tiles_for_slots(self, slots, deterministic):
        all_probabilities = torch.stack([
            distribution.probs
            for distribution in self.position_distributions
        ], dim=1)
        rows = torch.arange(slots.shape[0], device=slots.device)
        probabilities = all_probabilities[rows, (slots - 1).clamp(0, 3)]
        wait_probabilities = torch.zeros_like(probabilities)
        wait_probabilities[:, 0] = 1
        probabilities = torch.where(
            (slots == 0).unsqueeze(1), wait_probabilities, probabilities
        )
        return (probabilities.argmax(dim=1) if deterministic
                else torch.multinomial(probabilities, 1).squeeze(1))

    def _sanitize(self, slots, tiles):
        rows = torch.arange(slots.shape[0], device=slots.device)
        legal = self.action_masks[rows, slots, tiles]
        slots = torch.where(legal, slots, torch.zeros_like(slots))
        tiles = torch.where(legal, tiles, torch.zeros_like(tiles))
        return slots, tiles

    def sample(self):
        slots = self.slot_distribution.sample()
        slots, tiles = self._sanitize(slots, self._tiles_for_slots(slots, False))
        return self._decode(slots, tiles)

    def mode(self):
        slots = torch.argmax(self.slot_distribution.probs, dim=1)
        slots, tiles = self._sanitize(slots, self._tiles_for_slots(slots, True))
        return self._decode(slots, tiles)

    def log_prob(self, actions):
        count = self.action_masks.shape[0]
        if actions.shape != (count, 3) or actions.is_complex() or actions.dtype == torch.bool:
            raise ValueError("actions must be a numeric tensor of shape (batch, 3)")
        supported = torch.isfinite(actions).all(dim=1) & (actions == actions.round()).all(dim=1)
        supported &= (
            (actions[:, 0] >= 0) & (actions[:, 0] < 5)
            & (actions[:, 1] >= 0) & (actions[:, 1] < ARENA_HEIGHT)
            & (actions[:, 2] >= 0) & (actions[:, 2] < ARENA_WIDTH)
        )
        # Clamp before conversion/indexing so unsupported encodings yield -inf
        # rather than aliasing a tile, truncating a fraction or raising an index error.
        safe = torch.nan_to_num(actions)
        slots = safe[:, 0].clamp(0, 4).long()
        tiles = (safe[:, 1].clamp(0, ARENA_HEIGHT - 1).long() * ARENA_WIDTH
                 + safe[:, 2].clamp(0, ARENA_WIDTH - 1).long())
        rows = torch.arange(count, device=slots.device)
        supported &= self.action_masks[rows, slots, tiles]
        supported &= (slots != 0) | ((actions[:, 1] == 0) & (actions[:, 2] == 0))
        slot_log_prob = self.slot_distribution.log_prob(slots)
        position_log_probs = torch.stack([
            distribution.log_prob(tiles)
            for distribution in self.position_distributions
        ], dim=1)
        selected = position_log_probs.gather(1, (slots - 1).clamp(0, 3).unsqueeze(1)).squeeze(1)
        result = slot_log_prob + torch.where(
            slots == 0, torch.zeros_like(selected), selected
        )
        return result.masked_fill(~supported, -torch.inf)

    def entropy_components(self):
        per_slot = torch.stack([
            distribution.entropy()
            for distribution in self.position_distributions
        ], dim=1)
        play_probabilities = self.slot_distribution.probs[:, 1:]
        weighted_position = (play_probabilities * per_slot).sum(dim=1)
        return self.slot_distribution.entropy(), weighted_position

    def entropy(self):
        slot_entropy, position_entropy = self.entropy_components()
        return (
            self.slot_entropy_coef * slot_entropy
            + self.position_entropy_coef * position_entropy
        )

    def actions_from_params(self, action_logits, action_masks, deterministic=False):
        self.proba_distribution(action_logits, action_masks)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits, action_masks):
        actions = self.actions_from_params(action_logits, action_masks)
        return actions, self.log_prob(actions)


class NativeHybridMaskedPolicy(MultiInputActorCriticPolicy):
    """SB3 actor-critic using the hybrid encoder and hierarchical action mask."""

    def __init__(
        self,
        *args,
        slot_entropy_coef=0.001,
        position_entropy_coef=0.00025,
        **kwargs,
    ):
        kwargs.setdefault("features_extractor_class", NativeHybridExtractor)
        kwargs.setdefault("net_arch", {"pi": [256], "vf": [256]})
        kwargs.setdefault("normalize_images", False)
        kwargs.setdefault("ortho_init", False)
        self.slot_entropy_coef = float(slot_entropy_coef)
        self.position_entropy_coef = float(position_entropy_coef)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = HierarchicalMaskedDistribution(
            slot_entropy_coef=self.slot_entropy_coef,
            position_entropy_coef=self.position_entropy_coef,
        )
        self.action_net = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _distribution(self, latent, observation):
        logits = self.action_net(latent)
        return self.action_dist.proba_distribution(
            logits, observation["action_mask"]
        )

    def forward(self, observation, deterministic=False):
        features = self.extract_features(observation)
        actor, critic = self.mlp_extractor(features)
        distribution = self._distribution(actor, observation)
        actions = distribution.get_actions(deterministic=deterministic)
        return (
            actions.reshape((-1, *self.action_space.shape)),
            self.value_net(critic),
            distribution.log_prob(actions),
        )

    def evaluate_actions(self, observation, actions):
        features = self.extract_features(observation)
        actor, critic = self.mlp_extractor(features)
        distribution = self._distribution(actor, observation)
        return (
            self.value_net(critic),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, observation):
        features = self.extract_features(observation, self.pi_features_extractor)
        actor = self.mlp_extractor.forward_actor(features)
        return self._distribution(actor, observation)

    def predict_values(self, observation):
        features = self.extract_features(observation, self.vf_features_extractor)
        critic = self.mlp_extractor.forward_critic(features)
        return self.value_net(critic)

    def _predict(self, observation, deterministic=False):
        return self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )


__all__ = [
    "HierarchicalMaskedDistribution",
    "NativeHybridExtractor",
    "NativeHybridMaskedPolicy",
]
