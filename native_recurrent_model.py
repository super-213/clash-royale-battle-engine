"""Stateful masked RecurrentPPO policy with explicitly truncated BPTT."""

from __future__ import annotations

from collections import namedtuple
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.policies import (
    RecurrentMultiInputActorCriticPolicy,
)
from sb3_contrib.common.recurrent.type_aliases import (
    RNNStates,
    RecurrentDictRolloutBufferSamples,
)
from stable_baselines3.common.utils import explained_variance

from .native_model import (
    HierarchicalMaskedDistribution,
    NativeHybridExtractor,
)
from .native_cross_modal import CrossModalActionHead, NativeCrossModalExtractor
from .native_spatial_model import (
    NativeSpatialHeatmapExtractor,
    SpatialHeatmapActionHead,
)
from .native_world_model import (
    AuxiliaryWorldModelHead,
    masked_mean,
)
from .native_vec_env import (
    ENTITY_TARGET_SCHEMA, MAX_ENTITIES,
    EXECUTED_ACTION_SCHEMA, EXECUTED_ACTION_FEATURES,
    TRANSITION_TARGET_SCHEMA, NEXT_SCENE_TARGET_SCHEMA, scene_value,
)


class TruncatedRecurrentDictRolloutBuffer(RecurrentDictRolloutBuffer):
    """Recurrent Dict buffer with hard sequence boundaries for TBPTT."""

    def __init__(self, *args, tbptt_steps=32, **kwargs):
        self.tbptt_steps = int(tbptt_steps)
        if self.tbptt_steps <= 0:
            raise ValueError("tbptt_steps must be positive")
        self.max_sampled_sequence_length = 0
        super().__init__(*args, **kwargs)

    def get(self, batch_size=None):
        assert self.full, "Rollout buffer must be full before sampling from it"
        if not self.generator_ready:
            for tensor in (
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
            ):
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)
            for key, observation in self.observations.items():
                self.observations[key] = self.swap_and_flatten(observation)
            for tensor in (
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "hidden_states_pi",
                "cell_states_pi",
                "hidden_states_vf",
                "cell_states_vf",
                "episode_starts",
            ):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        total = self.buffer_size * self.n_envs
        batch_size = total if batch_size is None else int(batch_size)
        split_index = np.random.randint(total)
        indices = np.arange(total)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        # swap_and_flatten produces environment-major sequences. Mark every
        # tbptt_steps as a sequence boundary without setting episode_starts;
        # _get_samples retrieves the stored hidden state at each such boundary.
        env_change = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        env_change[:: self.tbptt_steps, :] = 1.0
        env_change = self.swap_and_flatten(env_change)
        for start in range(0, total, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield self._get_samples(batch_indices, env_change)

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)
        sequence_count = int(samples.lstm_states.pi[0].shape[1])
        if sequence_count:
            sequence_length = int(samples.mask.numel() // sequence_count)
            self.max_sampled_sequence_length = max(
                self.max_sampled_sequence_length, sequence_length
            )
            if sequence_length > self.tbptt_steps:
                raise RuntimeError(
                    f"TBPTT sequence length {sequence_length} exceeds "
                    f"configured limit {self.tbptt_steps}"
                )
        return samples


EntityTargetRolloutSamples = namedtuple(
    "EntityTargetRolloutSamples",
    (*RecurrentDictRolloutBufferSamples._fields, "entity_targets", "transition_targets", "future_targets"),
)


class EntityTargetRecurrentDictRolloutBuffer(TruncatedRecurrentDictRolloutBuffer):
    """Store native-ID labels outside policy observations, with identical padding."""

    def __init__(self, *args, auxiliary_horizon=4, **kwargs):
        self.auxiliary_horizon = int(auxiliary_horizon)
        if self.auxiliary_horizon <= 0:
            raise ValueError("auxiliary_horizon must be positive")
        super().__init__(*args, **kwargs)

    def reset(self):
        super().reset()
        self.entity_targets = np.zeros(
            (self.buffer_size, self.n_envs, MAX_ENTITIES, 4), dtype=np.float32
        )
        self._pending_entity_targets = None
        self.transition_targets = np.zeros((self.buffer_size, self.n_envs, 8), dtype=np.float32)
        self.successor_values = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
        self._pending_transition_targets = None
        self._pending_successor_values = None

    def set_entity_targets(self, infos):
        self._pending_entity_targets = self._pending_transition_targets = self._pending_successor_values = None
        if len(infos) != self.n_envs or any(
            info.get("entity_target_schema") != ENTITY_TARGET_SCHEMA
            for info in infos
        ):
            raise ValueError(
                "world-model training requires native_entity_id_v1 entity labels"
            )
        targets = np.stack([info["entity_targets"] for info in infos])
        if targets.shape != (self.n_envs, MAX_ENTITIES, 4):
            raise ValueError("invalid native entity target shape")
        if not np.isfinite(targets).all():
            raise ValueError("native entity targets must be finite")
        if (not np.isin(targets[..., (0, 3)], (0, 1)).all()
                or np.any(np.abs(targets[..., 1:3]) > 1)):
            raise ValueError("invalid native entity target flags or displacement")
        if any(info.get("transition_target_schema") != TRANSITION_TARGET_SCHEMA for info in infos):
            raise ValueError("world-model training requires public_damage_result_v2 labels")
        transitions = np.stack([info["transition_targets"] for info in infos])
        if transitions.shape != (self.n_envs, 8) or not np.isfinite(transitions).all():
            raise ValueError("invalid public transition targets")
        if (not np.isin(transitions[:, 7], (0, 1)).all() or np.any(np.abs(transitions[:, :2]) > 1)
                or np.any((transitions[:, 2:7] < 0) | (transitions[:, 2:7] > 1))):
            raise ValueError("invalid public transition target flags or range")
        successors = np.zeros((self.n_envs, 2), dtype=np.float32)
        for index, info in enumerate(infos):
            if "next_scene_value" in info:
                value = info["next_scene_value"]
                if (info.get("next_scene_target_schema") != NEXT_SCENE_TARGET_SCHEMA
                        or not np.isscalar(value) or not np.isfinite(value) or abs(value) > 1.25):
                    raise ValueError("invalid public successor scene target")
                successors[index] = (value, 1)
            elif "terminal_observation" in info:
                scalar = np.asarray(info["terminal_observation"]["scalar"])
                if scalar.shape != self.observation_space['scalar'].shape or not np.isfinite(scalar).all():
                    raise ValueError("invalid terminal scalar target")
                successors[index] = (scene_value(scalar), 1)
        self._pending_entity_targets = targets
        self._pending_transition_targets = transitions
        self._pending_successor_values = successors

    def add(self, *args, **kwargs):
        if self._pending_entity_targets is None:
            raise RuntimeError("missing native entity targets for rollout step")
        self.entity_targets[self.pos] = self._pending_entity_targets
        self.transition_targets[self.pos] = self._pending_transition_targets
        self.successor_values[self.pos] = self._pending_successor_values
        self._pending_entity_targets = None
        self._pending_transition_targets = self._pending_successor_values = None
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.generator_ready:
            # Labels can cross TBPTT/minibatch boundaries; gradients still cannot.
            # Store the observed successor, including the last nonterminal step.
            # Never substitute an auto-reset observation for a terminal state.
            future = np.zeros((self.buffer_size, self.n_envs, 2), dtype=np.float32)
            scalar = self.observations["scalar"]
            for step in range(self.buffer_size):
                endpoint = step + self.auxiliary_horizon
                if endpoint < self.buffer_size:
                    valid = ~self.episode_starts[step + 1:endpoint + 1].astype(bool).any(axis=0)
                    values = scene_value(scalar[endpoint])
                    future[step, :, 0] = values
                    future[step, :, 1] = valid
                if endpoint - 1 < self.buffer_size:
                    successor = self.successor_values[endpoint - 1]
                    valid = (successor[:, 1] > 0.5) & ~self.episode_starts[
                        step + 1:endpoint].astype(bool).any(axis=0)
                    future[step, valid] = successor[valid]
            self.future_targets = self.swap_and_flatten(future)
            self.transition_targets = self.swap_and_flatten(self.transition_targets)
            self.entity_targets = self.swap_and_flatten(self.entity_targets)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)
        targets = self.pad(self.entity_targets[batch_inds]).reshape(
            -1, MAX_ENTITIES, 4
        )
        transitions = self.pad(self.transition_targets[batch_inds]).reshape(-1, 8)
        future = self.pad(self.future_targets[batch_inds]).reshape(-1, 2)
        return EntityTargetRolloutSamples(*samples, targets, transitions, future)


ActionTargetRolloutSamples = namedtuple(
    "ActionTargetRolloutSamples",
    (*EntityTargetRolloutSamples._fields, "executed_actions"),
)


class ActionTargetRecurrentDictRolloutBuffer(EntityTargetRecurrentDictRolloutBuffer):
    """Keep executed actions aligned with state t, including sequence padding."""

    def reset(self):
        super().reset()
        self.executed_actions = np.zeros(
            (self.buffer_size, self.n_envs, EXECUTED_ACTION_FEATURES), dtype=np.float32
        )
        self._pending_executed_actions = None

    def set_entity_targets(self, infos):
        self._pending_executed_actions = None
        self._pending_entity_targets = self._pending_transition_targets = self._pending_successor_values = None
        if len(infos) != self.n_envs or any(
            info.get("executed_action_schema") != EXECUTED_ACTION_SCHEMA
            for info in infos
        ):
            raise ValueError("action-conditioned training requires executed_card_xy_v1 actions")
        actions = np.stack([info["executed_action"] for info in infos])
        if actions.shape != (self.n_envs, EXECUTED_ACTION_FEATURES):
            raise ValueError("invalid executed action shape")
        if not np.isfinite(actions).all() or np.any((actions < 0) | (actions > 1)):
            raise ValueError("executed actions must be finite and in [0, 1]")
        categories = actions[:, :9]
        if not np.all((categories == 0) | (categories == 1)) or not np.all(
            categories.sum(axis=1) == 1
        ):
            raise ValueError("executed action must select wait or exactly one card")
        if np.any(actions[actions[:, 0] == 1, -2:] != 0):
            raise ValueError("waiting must have zero executed coordinates")
        super().set_entity_targets(infos)
        self._pending_executed_actions = actions

    def add(self, *args, **kwargs):
        if self._pending_executed_actions is None:
            raise RuntimeError("missing executed actions for rollout step")
        self.executed_actions[self.pos] = self._pending_executed_actions
        self._pending_executed_actions = None
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.executed_actions = self.swap_and_flatten(self.executed_actions)
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env_change, env=None):
        samples = super()._get_samples(batch_inds, env_change, env)
        actions = self.pad(self.executed_actions[batch_inds]).reshape(
            -1, EXECUTED_ACTION_FEATURES
        )
        return ActionTargetRolloutSamples(*samples, actions)


class NativeRecurrentPPO(RecurrentPPO):
    """Masked recurrent PPO with fixed-length TBPTT and singleton-safe updates."""

    native_rollout_buffer_class = TruncatedRecurrentDictRolloutBuffer
    has_auxiliary_tasks = False

    def __init__(self, *args, tbptt_steps=32, **kwargs):
        self.tbptt_steps = int(tbptt_steps)
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        # The parent allocates a standard recurrent buffer. Release it before
        # constructing our replacement, instead of retaining both allocations.
        self.rollout_buffer = None
        lstm = self.policy.lstm_actor
        hidden_shape = (
            self.n_steps,
            lstm.num_layers,
            self.n_envs,
            lstm.hidden_size,
        )
        self.rollout_buffer = self.native_rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            tbptt_steps=self.tbptt_steps,
            **({"auxiliary_horizon": self.auxiliary_horizon}
               if isinstance(self, NativeWorldModelPPO) else {}),
        )

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        entropy_losses = []
        pg_losses, value_losses, clip_fractions = [], [], []
        auxiliary_totals = []
        auxiliary_tasks = (
            tuple(name for name, weight in self.auxiliary_weights.items() if weight > 0)
            if self.has_auxiliary_tasks and self.auxiliary_coef > 0 else ()
        )
        optimize_auxiliary = bool(auxiliary_tasks)
        auxiliary_records = {name: [] for name in auxiliary_tasks}
        continue_training = True
        approx_kl_divs = []
        loss = torch.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                mask = rollout_data.mask > 1e-8
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                task_losses = {}
                if optimize_auxiliary:
                    if self.action_conditioned:
                        predictions = self.policy.auxiliary_predictions(
                            executed_actions=rollout_data.executed_actions
                        )
                    else:
                        predictions = self.policy.auxiliary_predictions()
                    auxiliary_loss, task_losses = self._auxiliary_losses(
                        rollout_data, predictions
                    )
                values = values.flatten()
                advantages = rollout_data.advantages
                # Direct Python callers may use a minibatch size that leaves
                # one sample. Unbiased std on that singleton is NaN.
                if self.normalize_advantage and advantages[mask].numel() > 1:
                    advantages = (
                        advantages - advantages[mask].mean()
                    ) / (advantages[mask].std() + 1e-8)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.mean(
                    torch.min(policy_loss_1, policy_loss_2)[mask]
                )
                pg_losses.append(policy_loss.item())
                clip_fractions.append(torch.mean(
                    (torch.abs(ratio - 1) > clip_range).float()[mask]
                ).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = torch.mean(
                    ((rollout_data.returns - values_pred) ** 2)[mask]
                )
                value_losses.append(value_loss.item())
                entropy_loss = (
                    -torch.mean(-log_prob[mask])
                    if entropy is None
                    else -torch.mean(entropy[mask])
                )
                entropy_losses.append(entropy_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                if optimize_auxiliary:
                    loss = loss + self.auxiliary_coef * auxiliary_loss
                    auxiliary_totals.append(auxiliary_loss.item())
                for name, value in task_losses.items():
                    auxiliary_records[name].append(value.item())

                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        ((torch.exp(log_ratio) - 1) - log_ratio)[mask]
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to "
                            f"reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if optimize_auxiliary:
            self.logger.record("train/auxiliary_loss", np.mean(auxiliary_totals))
        for name, values in auxiliary_records.items():
            self.logger.record(
                f"train/aux_{name}_loss",
                np.mean(values) if values else 0.0,
            )
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard"
        )
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/tbptt_steps", self.tbptt_steps)
        self.logger.record(
            "train/recurrent_sequence_max",
            self.rollout_buffer.max_sampled_sequence_length,
        )


class NativeWorldModelPPO(NativeRecurrentPPO):
    """Recurrent PPO jointly optimized with leakage-free auxiliary tasks."""

    native_rollout_buffer_class = EntityTargetRecurrentDictRolloutBuffer
    has_auxiliary_tasks = True
    action_conditioned = False

    def _update_info_buffer(self, infos, dones=None):
        # RecurrentPPO invokes this after env.step and before rollout_buffer.add.
        super()._update_info_buffer(infos, dones)
        self.rollout_buffer.set_entity_targets(infos)

    DEFAULT_AUXILIARY_WEIGHTS = {
        "tower_delta": 1.0,
        "entity_death": 0.5,
        "entity_position": 0.5,
        "damage_source": 0.25,
        "opponent_elixir": 0.25,
        "opponent_hand": 0.25,
        "opponent_next_card": 0.25,
        "future_value": 0.5,
    }

    def __init__(
        self,
        *args,
        auxiliary_coef=0.1,
        auxiliary_horizon=4,
        auxiliary_weights=None,
        **kwargs,
    ):
        self.auxiliary_coef = float(auxiliary_coef)
        self.auxiliary_horizon = int(auxiliary_horizon)
        if not np.isfinite(self.auxiliary_coef) or self.auxiliary_coef < 0:
            raise ValueError("auxiliary_coef must be finite and non-negative")
        if self.auxiliary_horizon <= 0:
            raise ValueError("auxiliary_horizon must be positive")
        self.auxiliary_weights = dict(self.DEFAULT_AUXILIARY_WEIGHTS)
        if auxiliary_weights:
            if auxiliary_weights.keys() - self.DEFAULT_AUXILIARY_WEIGHTS.keys():
                raise ValueError("unknown auxiliary task weight")
            self.auxiliary_weights.update({
                key: float(value) for key, value in auxiliary_weights.items()
            })
        if any(not np.isfinite(value) or value < 0 for value in self.auxiliary_weights.values()):
            raise ValueError("auxiliary task weights must be finite and non-negative")
        super().__init__(*args, **kwargs)
        if self.auxiliary_horizon >= self.tbptt_steps:
            raise ValueError(
                "auxiliary_horizon must be smaller than tbptt_steps"
            )

    def _setup_model(self):
        super()._setup_model()
        if not hasattr(self.policy, "auxiliary_predictions"):
            raise TypeError(
                "NativeWorldModelPPO requires a policy with auxiliary_predictions"
            )
        if getattr(self.policy, "action_conditioned", False) != self.action_conditioned:
            raise TypeError("world-model algorithm and policy action conditioning must match")

    @staticmethod
    def _sequences(values, sequence_count):
        return values.reshape(sequence_count, -1, *values.shape[1:])

    def _auxiliary_losses(self, rollout_data, predictions):
        sequence_count = int(rollout_data.lstm_states.pi[0].shape[1])
        mask = self._sequences(rollout_data.mask > 1e-8, sequence_count)
        observations = {
            key: self._sequences(value, sequence_count)
            for key, value in rollout_data.observations.items()
        }
        predicted = {
            key: self._sequences(value, sequence_count)
            for key, value in predictions.items()
        }
        transitions = self._sequences(rollout_data.transition_targets, sequence_count)
        transition_valid = mask & (transitions[..., 7] > 0.5)
        losses = {}
        if self.auxiliary_weights["tower_delta"] > 0:
            tower_error = functional.smooth_l1_loss(
                predicted["tower_delta"], transitions[..., :2], reduction="none"
            ).mean(dim=-1)
            losses["tower_delta"] = masked_mean(tower_error, transition_valid)
        if self.auxiliary_weights["damage_source"] > 0:
            damage_error = functional.cross_entropy(
                predicted["damage_source"].reshape(-1, 5),
                transitions[..., 2:7].argmax(dim=-1).reshape(-1), reduction="none"
            ).reshape_as(mask)
            losses["damage_source"] = masked_mean(damage_error, transition_valid)

        targets = self._sequences(rollout_data.entity_targets, sequence_count)
        death_targets = targets[..., 0]
        entity_valid = (mask.unsqueeze(-1) & (targets[..., 3] > 0.5)
                        & (observations["entity_mask"] > 0.5))
        if self.auxiliary_weights["entity_death"] > 0:
            death_error = functional.binary_cross_entropy_with_logits(
                predicted["entity_death"], death_targets, reduction="none"
            )
            losses["entity_death"] = masked_mean(death_error, entity_valid)
        if self.auxiliary_weights["entity_position"] > 0:
            position_error = functional.smooth_l1_loss(
                predicted["entity_position_delta"], targets[..., 1:3], reduction="none"
            ).mean(dim=-1)
            losses["entity_position"] = masked_mean(
                position_error, entity_valid & (death_targets < 0.5)
            )

        valid = mask
        scalar = observations["scalar"]
        if self.auxiliary_weights["opponent_elixir"] > 0:
            elixir_targets = torch.clamp(
                (scalar[..., 1] * 5).long(), min=0, max=4
            )
            elixir_error = functional.cross_entropy(
                predicted["opponent_elixir"].reshape(-1, 5),
                elixir_targets.reshape(-1),
                reduction="none",
            ).reshape_as(valid)
            losses["opponent_elixir"] = masked_mean(elixir_error, valid)

        belief = observations["public_belief"]
        # An empty rule-consistent posterior is unavailable supervision, not
        # evidence that all eight cards are absent from the opponent's hand.
        if self.auxiliary_weights["opponent_hand"] > 0:
            belief_valid = valid & (belief[..., :8].sum(dim=-1) > 0)
            hand_error = functional.binary_cross_entropy_with_logits(
                predicted["opponent_hand"], belief[..., :8], reduction="none"
            ).mean(dim=-1)
            losses["opponent_hand"] = masked_mean(hand_error, belief_valid)
        if self.auxiliary_weights["opponent_next_card"] > 0:
            next_card_targets = belief[..., 8:16]
            next_valid = valid & (next_card_targets.sum(dim=-1) > 0)
            next_card_targets = next_card_targets / next_card_targets.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            next_card_error = -(
                next_card_targets
                * functional.log_softmax(
                    predicted["opponent_next_card"], dim=-1
                )
            ).sum(dim=-1)
            losses["opponent_next_card"] = masked_mean(next_card_error, next_valid)

        future = self._sequences(rollout_data.future_targets, sequence_count)
        if self.auxiliary_weights["future_value"] > 0:
            future_error = functional.smooth_l1_loss(
                predicted["future_value"], future[..., 0], reduction="none"
            )
            losses["future_value"] = masked_mean(future_error, mask & (future[..., 1] > 0.5))
        total = sum(
            self.auxiliary_weights[name] * value
            for name, value in losses.items()
        )
        return total, losses


class NativeActionConditionedWorldModelPPO(NativeWorldModelPPO):
    """PPO whose auxiliary dynamics learn from native accepted learner actions."""

    native_rollout_buffer_class = ActionTargetRecurrentDictRolloutBuffer
    action_conditioned = True


class NativeRecurrentMaskedPolicy(RecurrentMultiInputActorCriticPolicy):
    """CNN/Transformer encoder followed by persistent actor/critic LSTMs."""

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
        kwargs.setdefault("lstm_hidden_size", 256)
        kwargs.setdefault("n_lstm_layers", 1)
        kwargs.setdefault("shared_lstm", False)
        kwargs.setdefault("enable_critic_lstm", True)
        self.slot_entropy_coef = float(slot_entropy_coef)
        self.position_entropy_coef = float(position_entropy_coef)
        super().__init__(*args, **kwargs)
        extractor = self.features_extractor
        extractor.architecture_version = (
            "cnn_entity_transformer_event_lstm_stateful_v2"
        )

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
        return self.action_dist.proba_distribution(
            self.action_net(latent), observation["action_mask"]
        )

    def forward(
        self,
        observation,
        lstm_states: RNNStates,
        episode_starts,
        deterministic=False,
    ):
        features = self.extract_features(observation)
        if self.share_features_extractor:
            policy_features = value_features = features
        else:
            policy_features, value_features = features
        policy_latent, policy_states = self._process_sequence(
            policy_features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
        )
        if self.lstm_critic is not None:
            value_latent, value_states = self._process_sequence(
                value_features,
                lstm_states.vf,
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            value_latent = policy_latent.detach()
            value_states = (
                policy_states[0].detach(),
                policy_states[1].detach(),
            )
        else:
            value_latent = self.critic(value_features)
            value_states = policy_states
        policy_latent = self.mlp_extractor.forward_actor(policy_latent)
        value_latent = self.mlp_extractor.forward_critic(value_latent)
        distribution = self._distribution(policy_latent, observation)
        actions = distribution.get_actions(deterministic=deterministic)
        slot_entropy, position_entropy = distribution.entropy_components()
        selected_cards = observation["hand"][:, :4].gather(
            1, (actions[:, 0].long() - 1).clamp(0, 3).unsqueeze(1)
        ).squeeze(1)
        self._rollout_diagnostics = torch.stack((
            slot_entropy, position_entropy, distribution.slot_distribution.probs[:, 0],
            (actions[:, 0] == 0).float(),
            *[((actions[:, 0] > 0) & (selected_cards == card)).float() for card in range(1, 9)],
        ), dim=1).detach()
        return (
            actions.reshape((-1, *self.action_space.shape)),
            self.value_net(value_latent),
            distribution.log_prob(actions),
            RNNStates(policy_states, value_states),
        )

    def evaluate_actions(
        self,
        observation,
        actions,
        lstm_states: RNNStates,
        episode_starts,
    ):
        features = self.extract_features(observation)
        if self.share_features_extractor:
            policy_features = value_features = features
        else:
            policy_features, value_features = features
        policy_latent, _ = self._process_sequence(
            policy_features,
            lstm_states.pi,
            episode_starts,
            self.lstm_actor,
        )
        if self.lstm_critic is not None:
            value_latent, _ = self._process_sequence(
                value_features,
                lstm_states.vf,
                episode_starts,
                self.lstm_critic,
            )
        elif self.shared_lstm:
            value_latent = policy_latent.detach()
        else:
            value_latent = self.critic(value_features)
        policy_latent = self.mlp_extractor.forward_actor(policy_latent)
        value_latent = self.mlp_extractor.forward_critic(value_latent)
        distribution = self._distribution(policy_latent, observation)
        return (
            self.value_net(value_latent),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, observation, lstm_states, episode_starts):
        features = self.extract_features(observation, self.pi_features_extractor)
        latent, lstm_states = self._process_sequence(
            features,
            lstm_states,
            episode_starts,
            self.lstm_actor,
        )
        latent = self.mlp_extractor.forward_actor(latent)
        return self._distribution(latent, observation), lstm_states


class NativeCrossModalRecurrentPolicy(NativeRecurrentMaskedPolicy):
    """Stateful policy whose action logits are produced by cross-modal tokens."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("features_extractor_class", NativeCrossModalExtractor)
        super().__init__(*args, **kwargs)
        self.features_extractor.architecture_version = "cross_modal_stateful_v3"

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = HierarchicalMaskedDistribution(
            slot_entropy_coef=self.slot_entropy_coef,
            position_entropy_coef=self.position_entropy_coef,
        )
        cross_dim = self.features_extractor.cross_dim
        self.action_net = CrossModalActionHead(
            self.mlp_extractor.latent_dim_pi, cross_dim
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _distribution(self, latent, observation):
        cards, placements = self.pi_features_extractor.action_tokens()
        logits = self.action_net(latent, cards, placements)
        return self.action_dist.proba_distribution(
            logits, observation["action_mask"]
        )


class NativeSpatialHeatmapRecurrentPolicy(NativeCrossModalRecurrentPolicy):
    """Stateful policy with direct per-card 32x18 placement heatmaps."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "features_extractor_class", NativeSpatialHeatmapExtractor
        )
        super().__init__(*args, **kwargs)
        self.features_extractor.architecture_version = (
            "spatial_heatmap_stateful_v4"
        )

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        self.action_dist = HierarchicalMaskedDistribution(
            slot_entropy_coef=self.slot_entropy_coef,
            position_entropy_coef=self.position_entropy_coef,
        )
        self.action_net = SpatialHeatmapActionHead(
            self.mlp_extractor.latent_dim_pi,
            self.features_extractor.cross_dim,
        )
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


class NativeWorldModelRecurrentPolicy(NativeSpatialHeatmapRecurrentPolicy):
    """Spatial recurrent policy with shared-encoder auxiliary predictions."""

    action_conditioned = False

    def __init__(self, *args, world_model_hidden_dim=128, **kwargs):
        self.world_model_hidden_dim = int(world_model_hidden_dim)
        self._auxiliary_actor_latent = None
        super().__init__(*args, **kwargs)
        self.features_extractor.architecture_version = (
            "auxiliary_world_model_stateful_v5"
        )

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.world_model = AuxiliaryWorldModelHead(
            actor_dim=self.mlp_extractor.latent_dim_pi,
            entity_dim=self.features_extractor.cross_dim,
            hidden_dim=self.world_model_hidden_dim,
            action_conditioned=self.action_conditioned,
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _distribution(self, latent, observation):
        self._auxiliary_actor_latent = latent
        return super()._distribution(latent, observation)

    def auxiliary_predictions(self, executed_actions=None):
        if self._auxiliary_actor_latent is None:
            raise RuntimeError("evaluate the policy before auxiliary prediction")
        entity_tokens, _ = self.pi_features_extractor.auxiliary_entity_tokens()
        return self.world_model(
            self._auxiliary_actor_latent,
            entity_tokens,
            executed_actions=executed_actions,
        )


class NativeActionConditionedWorldModelRecurrentPolicy(NativeWorldModelRecurrentPolicy):
    """v6 action-conditioned auxiliary heads; PPO actor/critic are unchanged."""

    action_conditioned = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features_extractor.architecture_version = (
            "action_conditioned_world_model_stateful_v6"
        )


__all__ = [
    "NativeRecurrentMaskedPolicy",
    "NativeCrossModalRecurrentPolicy",
    "NativeSpatialHeatmapRecurrentPolicy",
    "NativeRecurrentPPO",
    "NativeWorldModelPPO",
    "NativeWorldModelRecurrentPolicy",
    "NativeActionConditionedWorldModelPPO",
    "NativeActionConditionedWorldModelRecurrentPolicy",
    "TruncatedRecurrentDictRolloutBuffer",
]
