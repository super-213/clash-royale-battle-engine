import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "native_engine_test_package"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)

protocol = importlib.import_module(f"{PACKAGE_NAME}.protocol")
vec_module = importlib.import_module(f"{PACKAGE_NAME}.native_vec_env")
model_module = importlib.import_module(f"{PACKAGE_NAME}.native_model")
recurrent_module = importlib.import_module(f"{PACKAGE_NAME}.native_recurrent_model")
train_module = importlib.import_module(f"{PACKAGE_NAME}.train_native")
NativeBatchVecEnv = vec_module.NativeBatchVecEnv


def make_state(slot, generation, tick=90, finalized=False, result=None):
    cards = vec_module.CARD_IDS
    players = []
    for owner in (0, 1):
        hand = tuple(
            protocol.HandCard(index, cards[index], 1, 0) for index in range(4)
        )
        players.append(protocol.Player(owner, 100_000, hand, tuple(cards[4:])))
    entities = (
        protocol.Entity(
            slot * 100 + 1, 0, -1, 9000, 4000, 1000, 1000,
            None, None, "building", None, None, None, None, None,
        ),
        protocol.Entity(
            slot * 100 + 2, 1, -1, 9000, 28000, 1000, 1000,
            None, None, "building", None, None, None, None, None,
        ),
    )
    crowns = (1, 0) if finalized and result == 0 else (0, 1) if finalized else (0, 0)
    return protocol.State(
        generation,
        tick,
        finalized,
        result,
        crowns,
        tuple(players),
        entities,
        (),
        1,
    )


class FakeBatch:
    def __init__(self, battles):
        self.battles = battles
        self.states = ()
        self.last_receipts = tuple(() for _ in range(battles))
        self.step_calls = []
        self.reset_slot_calls = []
        self.generation = 0
        self.finish_slot_zero = False
        self.closed = False

    def reset(self, seed=1):
        self.states = tuple(self._reset_state(slot) for slot in range(self.battles))
        return self.states

    def _reset_state(self, slot):
        self.generation += 1
        return make_state(slot, self.generation)

    def step(self, actions=(), ticks=10):
        self.step_calls.append((tuple(tuple(group) for group in actions), ticks))
        next_states = []
        all_receipts = []
        for slot, (state, plays) in enumerate(zip(self.states, actions)):
            played = []
            receipts = []
            for action_index, play in enumerate(plays):
                player = state.players[play.owner]
                card = next(card for card in player.hand if card.slot == play.slot)
                played.append(protocol.CardPlay(
                    action_index + 1, state.tick + 1, play.owner, card.card, card.cost
                ))
                receipts.append(SimpleNamespace(
                    status=0,
                    owner=play.owner,
                    card_id=card.card,
                ))
            finalized = self.finish_slot_zero and slot == 0
            next_states.append(protocol.State(
                state.generation,
                state.tick + ticks,
                finalized,
                0 if finalized else None,
                (1, 0) if finalized else state.crowns,
                state.players,
                state.entities,
                tuple(played),
                state.next_sequence + len(played),
            ))
            all_receipts.append(tuple(receipts))
        self.states = tuple(next_states)
        self.last_receipts = tuple(all_receipts)
        return self.states

    def reset_slots(self, slots, seeds):
        self.reset_slot_calls.append((tuple(slots), tuple(seeds)))
        values = list(self.states)
        refreshed = []
        for slot in slots:
            values[slot] = self._reset_state(slot)
            refreshed.append(values[slot])
        self.states = tuple(values)
        return tuple(refreshed)

    def close(self):
        self.closed = True


class NativeEntityTargetsTest(unittest.TestCase):
    def troop(self, identity, x=5000, y=10000):
        return replace(
            make_state(0, 1).entities[0], id=identity, card=vec_module.CARD_IDS[0],
            entity_type="troop", x=x, y=y,
        )

    def test_crossing_identical_units_keep_identity_and_orientation(self):
        # Adjacent IDs above float32's exact range must remain distinct.
        first = self.troop(2**24, x=5000)
        second = self.troop(2**24 + 1, x=6000)
        before = replace(make_state(0, 1), entities=(second, first))
        after = replace(before, entities=(
            replace(second, x=4900), replace(first, x=6100),
        ))
        for owner, direction in ((0, 1), (1, -1)):
            labels = vec_module._entity_transition_targets(before, after, owner)
            np.testing.assert_array_equal(labels[:2, 0], [0, 0])
            np.testing.assert_allclose(
                labels[:2, 1], direction * np.array([1100, -1100]) / 18000,
                atol=1e-7,
            )
            np.testing.assert_array_equal(labels[:2, 3], [1, 1])
            self.assertFalse(labels[2:].any())

    def test_replacement_at_same_position_is_death_and_survivor_not_reused(self):
        before = replace(make_state(0, 1), entities=(self.troop(10), self.troop(20)))
        after = replace(before, entities=(self.troop(20), self.troop(30)))
        labels = vec_module._entity_transition_targets(before, after, 0)
        np.testing.assert_array_equal(labels[:2, 0], [1, 0])
        np.testing.assert_array_equal(labels[:2, 1:3], 0)

    def test_survivor_outside_next_token_cap_is_not_dead(self):
        survivor = self.troop(1000)
        before = replace(make_state(0, 1), entities=(survivor,))
        after = replace(before, entities=(
            *(self.troop(index) for index in range(vec_module.MAX_ENTITIES)),
            replace(survivor, y=11000),
        ))
        labels = vec_module._entity_transition_targets(before, after, 0)
        self.assertEqual(labels[0, 0], 0)
        self.assertAlmostEqual(labels[0, 2], 1000 / 32000)

    def test_buildings_padding_and_generation_boundary(self):
        before = make_state(0, 1)
        after = replace(before, entities=())
        self.assertFalse(vec_module._entity_transition_targets(before, after, 0).any())
        with self.assertRaisesRegex(ValueError, "same episode"):
            vec_module._entity_transition_targets(before, replace(after, generation=2), 0)

    def test_terminal_labels_precede_autoreset_and_role_change(self):
        troop = self.troop(50)

        class TerminalBatch(FakeBatch):
            def _reset_state(self, slot):
                state = super()._reset_state(slot)
                return replace(state, entities=(*state.entities, troop))

            def step(self, actions=(), ticks=10):
                states = super().step(actions, ticks)
                self.states = tuple(
                    replace(state, entities=state.entities[:2], finalized=True)
                    for state in states
                )
                return self.states

        fake = TerminalBatch(1)
        env = VecMonitor(NativeBatchVecEnv(
            battles=1, batch_factory=lambda **kwargs: fake,
            opponent_policy=lambda *args: (0, 0, 0),
        ))
        try:
            env.reset()
            observations, _, dones, infos = env.step(np.zeros((1, 3)))
            self.assertTrue(dones[0])
            self.assertEqual(infos[0]["entity_target_schema"], vec_module.ENTITY_TARGET_SCHEMA)
            self.assertEqual(infos[0]["entity_targets"][2, 0], 1)
            self.assertEqual(infos[0]["entity_targets"][2, 3], 1)
            # Same native ID exists again in the reset episode, with flipped role.
            self.assertEqual(observations["entity_mask"][0, 2], 1)
            self.assertEqual(observations["scalar"][0, 8], 1)
            self.assertNotIn("entity_targets", observations)
        finally:
            env.close()

    def test_entity_losses_mask_boundaries_padding_and_dead_positions(self):
        sequences, length, entities = 2, 4, vec_module.MAX_ENTITIES
        count = sequences * length
        observations = {
            key: torch.zeros(count, *space.shape)
            for key, space in vec_module.native_observation_space().spaces.items()
        }
        observations["entity_mask"][:, 0] = 1
        labels = torch.zeros(count, entities, 4)
        labels[:, 0, 3] = 1
        labels[:, 0, 1:3] = 1  # Large errors on excluded transitions.
        valid_rows = [0, 4, 5, 6]
        labels[valid_rows, 0, 1:3] = 0.2
        labels[5, 0, 0] = 1
        labels[5, 0, 1:3] = 1  # Dead entity position must be excluded too.
        predictions = {
            "tower_delta": torch.zeros(count, 2),
            "damage_source": torch.zeros(count, 5),
            "opponent_elixir": torch.zeros(count, 5),
            "opponent_hand": torch.zeros(count, 8),
            "opponent_next_card": torch.zeros(count, 8),
            "future_value": torch.zeros(count),
            "entity_death": torch.full((count, entities), 8.0),
            "entity_position_delta": torch.zeros(count, entities, 2),
        }
        predictions["entity_death"][valid_rows, 0] = 0
        samples = SimpleNamespace(
            observations=observations, entity_targets=labels,
            mask=torch.tensor([1, 1, 1, 0, 1, 1, 1, 1]),
            episode_starts=torch.tensor([1, 0, 1, 0, 1, 0, 0, 0]),
            lstm_states=recurrent_module.RNNStates(
                (torch.zeros(1, sequences, 8), torch.zeros(1, sequences, 8)),
                (torch.zeros(1, sequences, 8), torch.zeros(1, sequences, 8)),
            ),
        )
        model = recurrent_module.NativeWorldModelPPO.__new__(recurrent_module.NativeWorldModelPPO)
        model.auxiliary_horizon = 2
        model.auxiliary_weights = model.DEFAULT_AUXILIARY_WEIGHTS
        _, losses = model._auxiliary_losses(samples, predictions)
        self.assertAlmostEqual(float(losses["entity_death"]), np.log(2), places=6)
        self.assertAlmostEqual(float(losses["entity_position"]), 0.02, places=6)

    def test_labels_follow_rollout_shuffle_padding_and_tbptt(self):
        count, steps = 3, 7
        buffer = recurrent_module.EntityTargetRecurrentDictRolloutBuffer(
            steps, vec_module.native_observation_space(),
            NativeBatchVecEnv(battles=count).action_space,
            (steps, 1, count, 8), n_envs=count, tbptt_steps=3, device="cpu",
        )
        zeros = torch.zeros(1, count, 8)
        states = recurrent_module.RNNStates((zeros, zeros), (zeros, zeros))
        for step in range(steps):
            observations = {
                key: np.zeros((count, *space.shape), dtype=space.dtype)
                for key, space in buffer.observation_space.spaces.items()
            }
            marker = (step * count + np.arange(count) + 1) / 100.0
            observations["scalar"][:, 0] = marker
            targets = np.zeros((count, vec_module.MAX_ENTITIES, 4), dtype=np.float32)
            targets[:, 0, 1] = marker
            targets[:, 0, 3] = 1
            buffer.set_entity_targets([
                {"entity_target_schema": vec_module.ENTITY_TARGET_SCHEMA,
                 "entity_targets": target}
                for target in targets
            ])
            starts = np.zeros(count)
            starts[:] = step == 0
            starts[0] = step in (0, 2)
            buffer.add(
                observations, np.zeros((count, 3)), np.zeros(count), starts,
                torch.zeros(count), torch.zeros(count), lstm_states=states,
            )
        buffer.advantages[:] = 0
        buffer.returns[:] = 0
        for _ in range(3):
            seen = []
            for samples in buffer.get(batch_size=8):
                mask = samples.mask > 0
                torch.testing.assert_close(
                    samples.entity_targets[mask, 0, 1],
                    samples.observations["scalar"][mask, 0],
                )
                self.assertFalse(bool(samples.entity_targets[~mask].any()))
                seen.extend(samples.entity_targets[mask, 0, 1].tolist())
            np.testing.assert_allclose(sorted(seen), np.arange(1, 22) / 100)
        self.assertLessEqual(buffer.max_sampled_sequence_length, 3)
        buffer.reset()
        self.assertFalse(buffer.entity_targets.any())
        with self.assertRaisesRegex(ValueError, "requires native_entity_id_v1"):
            buffer.set_entity_targets([{}] * count)
        with self.assertRaisesRegex(RuntimeError, "missing native entity targets"):
            buffer.add()


class ActionConditionedWorldModelTest(unittest.TestCase):
    def test_executed_action_uses_receipt_identity_and_rotated_coordinates(self):
        # Deck slots and policy hand slots are not card identities.
        play = vec_module.Play(owner=1, slot=7, x=15500, y=24500)
        receipt = SimpleNamespace(status=0, owner=1, card_id=vec_module.CARD_IDS[4])
        event = protocol.CardPlay(1, 91, 1, vec_module.CARD_IDS[4], 4)
        features = vec_module._executed_action_features(1, (play,), (receipt,), (event,))
        self.assertEqual(features[0], 0)
        self.assertEqual(features[5], 1)
        self.assertEqual(features[:9].sum(), 1)
        np.testing.assert_allclose(features[-2:], (2500 / 18000, 7500 / 32000))
        mirror = vec_module._executed_action_features(
            0, (replace(play, owner=0, x=2500, y=7500),),
            (receipt,), (replace(event, owner=0),),
        )
        np.testing.assert_array_equal(mirror, features)

    def test_wait_rejection_and_missing_execution_event_have_no_card_or_position(self):
        play = vec_module.Play(owner=0, slot=0, x=5500, y=6500)
        receipt = SimpleNamespace(status=0, owner=0, card_id=vec_module.CARD_IDS[0])
        event = protocol.CardPlay(1, 91, 0, vec_module.CARD_IDS[0], 3)
        wait = np.zeros(vec_module.EXECUTED_ACTION_FEATURES, dtype=np.float32)
        wait[0] = 1
        for owner, plays, receipts, events in (
            (0, (), (), ()),
            (0, (play,), (SimpleNamespace(status=1, owner=0, card_id=receipt.card_id),), (event,)),
            (0, (play,), (receipt,), ()),
            (1, (play,), (receipt,), (event,)),
        ):
            np.testing.assert_array_equal(
                vec_module._executed_action_features(owner, plays, receipts, events), wait
            )

    def test_environment_labels_precede_hand_cycle_and_terminal_reset(self):
        class CyclingBatch(FakeBatch):
            def step(self, actions=(), ticks=10):
                states = super().step(actions, ticks)
                self.states = tuple(replace(state, players=tuple(
                    replace(player, hand=tuple(replace(card, card=vec_module.CARD_IDS[7])
                        for card in player.hand)) for player in state.players
                )) for state in states)
                return self.states

        fake = CyclingBatch(2)
        fake.finish_slot_zero = True
        env = NativeBatchVecEnv(
            battles=2, learner_roles=(0, 1),
            opponent_policy=lambda *args: (0, 0, 0), batch_factory=lambda **kwargs: fake,
        )
        try:
            before = env.reset()
            after, _, dones, infos = env.step(np.asarray(((1, 7, 1), (1, 7, 1))))
            self.assertTrue(dones[0])
            self.assertEqual(after["hand"][1, 0], 8)
            for info in infos:
                self.assertEqual(info["executed_action_schema"], vec_module.EXECUTED_ACTION_SCHEMA)
                self.assertEqual(info["executed_action"][1], before["hand"][0, 0])
                np.testing.assert_allclose(info["executed_action"][-2:], (1500 / 18000, 7500 / 32000))
            self.assertNotIn("executed_action", after)
            _, _, _, rejected = env.step(np.asarray(((1, 99, 99), (0, 20, 10))))
            for info in rejected:
                self.assertEqual(info["executed_action"][0], 1)
                self.assertFalse(info["executed_action"][1:].any())
        finally:
            env.close()

    def test_executed_actions_follow_shuffle_padding_and_tbptt(self):
        count, steps = 2, 7
        buffer = recurrent_module.ActionTargetRecurrentDictRolloutBuffer(
            steps, vec_module.native_observation_space(),
            NativeBatchVecEnv(battles=count).action_space,
            (steps, 1, count, 8), n_envs=count, tbptt_steps=3, device="cpu",
        )
        zeros = torch.zeros(1, count, 8)
        states = recurrent_module.RNNStates((zeros, zeros), (zeros, zeros))
        for step in range(steps):
            observations = {
                key: np.zeros((count, *space.shape), dtype=space.dtype)
                for key, space in buffer.observation_space.spaces.items()
            }
            marker = (step * count + np.arange(count) + 1) / 100
            observations["scalar"][:, 0] = marker
            infos = []
            for value in marker:
                action = np.zeros(vec_module.EXECUTED_ACTION_FEATURES, dtype=np.float32)
                action[1] = 1
                action[-2:] = value
                infos.append({
                    "entity_target_schema": vec_module.ENTITY_TARGET_SCHEMA,
                    "entity_targets": np.zeros((vec_module.MAX_ENTITIES, 4)),
                    "executed_action_schema": vec_module.EXECUTED_ACTION_SCHEMA,
                    "executed_action": action,
                })
            buffer.set_entity_targets(infos)
            starts = np.full(count, step == 0, dtype=float)
            starts[0] = step in (0, 2)
            # PPO actions deliberately disagree with execution labels.
            buffer.add(observations, np.zeros((count, 3)), np.zeros(count), starts,
                torch.zeros(count), torch.zeros(count), lstm_states=states)
        for _ in range(3):
            seen = []
            for samples in buffer.get(batch_size=5):
                mask = samples.mask > 0
                torch.testing.assert_close(samples.executed_actions[mask, -1],
                    samples.observations["scalar"][mask, 0])
                self.assertFalse(bool(samples.executed_actions[~mask].any()))
                self.assertFalse(bool(samples.actions.any()))
                seen.extend(samples.executed_actions[mask, -1].tolist())
            np.testing.assert_allclose(sorted(seen), np.arange(1, 15) / 100)
        self.assertLessEqual(buffer.max_sampled_sequence_length, 3)
        buffer.reset()
        self.assertFalse(buffer.executed_actions.any())
        with self.assertRaisesRegex(ValueError, "requires executed_card_xy_v1"):
            buffer.set_entity_targets([{}] * count)
        with self.assertRaisesRegex(RuntimeError, "missing executed actions"):
            buffer.add()
        for bad_action in (np.full(11, np.nan), np.ones(11), np.zeros(10)):
            with self.assertRaises(ValueError):
                buffer.set_entity_targets([dict(infos[0], executed_action=bad_action)] * count)

    def test_action_changes_dynamics_but_not_current_belief_and_has_gradients(self):
        torch.manual_seed(41)
        head = recurrent_module.AuxiliaryWorldModelHead(16, 8, 16, action_conditioned=True)
        actor = torch.randn(1, 16, requires_grad=True)
        entities = torch.randn(1, 3, 8, requires_grad=True)
        actions = torch.zeros(4, vec_module.EXECUTED_ACTION_FEATURES)
        actions[0, 0] = 1  # wait
        actions[1:, 1] = 1
        actions[1:, -2:] = torch.tensor([[0.2, 0.3], [0.7, 0.8], [0.2, 0.3]])
        actions[3, 1] = 0
        actions[3, 5] = 1  # same placement, different card
        predictions = head(actor.expand(4, -1), entities.expand(4, -1, -1), actions)
        dynamics = ("tower_delta", "damage_source", "future_value", "entity_death", "entity_position_delta")
        for key in dynamics:
            for row in (0, 2, 3):
                self.assertFalse(torch.allclose(predictions[key][1], predictions[key][row]), key)
        for key in ("opponent_elixir", "opponent_hand", "opponent_next_card"):
            torch.testing.assert_close(predictions[key], predictions[key][:1].expand_as(predictions[key]))
        sum(predictions[key].square().mean() for key in dynamics).backward()
        for parameter in (head.action_encoder[0].weight, head.dynamics_encoder[0].weight, actor, entities):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "requires executed_actions"):
            head(actor, entities)

    def test_v6_joint_training_checkpoint_reload_and_legal_inference(self):
        fake = FakeBatch(2)
        env = NativeBatchVecEnv(battles=2, learner_roles=(0, 1),
            opponent_policy=lambda *args: (0, 0, 0), batch_factory=lambda **kwargs: fake)
        model = recurrent_module.NativeActionConditionedWorldModelPPO(
            recurrent_module.NativeActionConditionedWorldModelRecurrentPolicy, env,
            n_steps=4, batch_size=8, n_epochs=1, tbptt_steps=4, auxiliary_horizon=2,
            policy_kwargs={"lstm_hidden_size": 32, "world_model_hidden_dim": 32,
                "features_extractor_kwargs": {"features_dim": 64, "spatial_channels": 16,
                    "cross_dim": 16, "entity_layers": 1, "attention_heads": 4},
                "net_arch": {"pi": [32], "vf": [32]}}, device="cpu", seed=41)
        try:
            model.learn(total_timesteps=8)
            self.assertEqual(model.policy.features_extractor.architecture_version,
                "action_conditioned_world_model_stateful_v6")
            for name, value in model.logger.name_to_value.items():
                if name.startswith("train/aux"):
                    self.assertTrue(np.isfinite(value), name)
            parameter = model.policy.world_model.action_encoder[0].weight
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
            with TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "v6"
                model.save(checkpoint)
                runtime = importlib.import_module("training_runtime")
                config = {
                    "rounds": 1, "rollout_decisions": 8, "decision_ticks": 10,
                    "algorithm": "NativeActionConditionedWorldModelPPO",
                    "policy": "NativeActionConditionedWorldModelRecurrentPolicy",
                    "world_model_enabled": True, "world_model_hidden_dim": 32,
                    "auxiliary_coef": model.auxiliary_coef, "auxiliary_horizon": 2,
                    "auxiliary_weights": model.auxiliary_weights,
                }
                metadata = runtime._model_metadata(
                    model, checkpoint.with_suffix(".zip"), config, 0.1,
                    SimpleNamespace(episode_count=0), "completed",
                )
                self.assertTrue(metadata["world_model"]["action_conditioned"])
                self.assertEqual(metadata["world_model"]["executed_action_schema"],
                    vec_module.EXECUTED_ACTION_SCHEMA)
                summary = Path(directory) / "summary.md"
                runtime._write_summary_markdown(summary, metadata, Path(directory))
                self.assertIn(vec_module.EXECUTED_ACTION_SCHEMA, summary.read_text())
                loaded = recurrent_module.NativeActionConditionedWorldModelPPO.load(
                    checkpoint, env=env, device="cpu")
                torch.testing.assert_close(parameter, loaded.policy.world_model.action_encoder[0].weight)
                observations = env.reset()
                actions, _ = loaded.predict(observations, deterministic=True)
                for row, (card, y, x) in enumerate(actions):
                    self.assertTrue(observations["action_mask"][row, card, y, x])
                executed = torch.zeros(2, vec_module.EXECUTED_ACTION_FEATURES)
                executed[:, 0] = 1
                predictions = loaded.policy.auxiliary_predictions(executed)
                self.assertEqual(tuple(predictions["tower_delta"].shape), (2, 2))
                loaded.learn(total_timesteps=8)
                self.assertTrue(np.isfinite(loaded.logger.name_to_value["train/auxiliary_loss"]))
        finally:
            env.close()


class NativeDeploymentMaskTest(unittest.TestCase):
    def setUp(self):
        self.env = NativeBatchVecEnv(battles=1)
        state = make_state(0, 1)
        towers = []
        for owner in (0, 1):
            for x, y in (
                (9000, 3000 if owner == 0 else 29000),
                (3500, 6500 if owner == 0 else 25500),
                (14500, 6500 if owner == 0 else 25500),
            ):
                towers.append(replace(
                    state.entities[owner], id=len(towers) + 1,
                    owner=owner, x=x, y=y,
                ))
        players = tuple(replace(
            player, hand=tuple(
                protocol.HandCard(index, card, cost, 0)
                for index, (card, cost) in enumerate((
                    (26000000, 3), (26000003, 5),
                    (28000000, 4), (28000001, 3),
                ))
            ),
        ) for player in state.players)
        self.state = replace(state, entities=tuple(towers), players=players)
        self.slot = self.env.slots[0]
        self.env._begin_episode(self.slot, self.state, 0)

    def tearDown(self):
        self.env.close()

    def mask(self, state, owner):
        return self.env._action_mask(state, owner, self.slot.initial_towers)

    def remove_tower(self, owner, lane, enemy=True, zero_hp=False):
        physical_owner = 1 - owner if enemy else owner
        x = 3500 if (lane == "left") == (owner == 0) else 14500
        tower = next(
            entity for entity in self.state.entities
            if entity.owner == physical_owner and entity.x == x
        )
        entities = tuple(
            replace(entity, hp=0) if zero_hp and entity.id == tower.id else entity
            for entity in self.state.entities
            if zero_hp or entity.id != tower.id
        )
        return replace(self.state, entities=entities)

    def test_base_region_rear_edges_towers_and_river(self):
        for owner in (0, 1):
            with self.subTest(owner=owner):
                mask = self.mask(self.state, owner)
                self.assertEqual(int(mask[1].sum()), 222)
                self.assertEqual(mask[1, 0, 7], 1)  # behind king
                self.assertEqual(mask[1, 1, 0], 1)  # outer column
                self.assertEqual(mask[1, 13, 17], 1)
                for y, x in ((0, 0), (3, 8), (6, 3), (6, 14), (14, 0), (15, 3), (18, 3)):
                    self.assertEqual(mask[1, y, x], 0)
                self.assertTrue(np.all(mask[3:5] == 1))  # both spells, whole arena

    def test_each_destroyed_lane_unlocks_only_its_region_for_both_owners(self):
        for owner in (0, 1):
            for lane, bridge, opposite in (("left", 3, 14), ("right", 14, 3)):
                for zero_hp in (False, True):
                    with self.subTest(owner=owner, lane=lane, zero_hp=zero_hp):
                        state = self.remove_tower(owner, lane, zero_hp=zero_hp)
                        mask = self.mask(state, owner)
                        self.assertEqual(int(mask[1].sum()), 259)
                        self.assertEqual(mask[1, 15, bridge], 1)
                        self.assertEqual(mask[1, 16, bridge], 1)
                        self.assertEqual(mask[1, 18, bridge], 1)
                        self.assertEqual(mask[1, 20, bridge], 1)
                        self.assertEqual(mask[1, 18, opposite], 0)
                        self.assertEqual(mask[1, 15, opposite], 0)
                        self.assertEqual(mask[1, 21, bridge], 0)
                        self.assertEqual(mask[1, 15, 8], 0)  # river outside bridge

    def test_both_enemy_towers_union_and_own_destroyed_tower_footprint(self):
        for owner in (0, 1):
            enemy_king = next(
                entity for entity in self.state.entities
                if entity.owner != owner and entity.x == 9000
            )
            state = replace(self.state, entities=tuple(
                entity for entity in self.state.entities
                if entity.owner == owner or entity.id == enemy_king.id
            ))
            self.assertEqual(int(self.mask(state, owner)[1].sum()), 296)
            own_left_destroyed = self.remove_tower(owner, "left", enemy=False)
            mask = self.mask(own_left_destroyed, owner)
            self.assertEqual(mask[1, 6, 3], 1)
            self.assertEqual(mask[1, 6, 14], 0)
            self.assertEqual(mask[1, 18, 3], 0)

    def test_initial_identity_damage_and_crowns_do_not_falsely_unlock(self):
        missing = self.remove_tower(0, "left")
        self.env._begin_episode(self.slot, missing, 0)
        self.assertEqual(self.mask(missing, 0)[1, 18, 3], 0)
        self.env._begin_episode(self.slot, self.state, 0)
        damaged = replace(self.state, entities=tuple(
            replace(entity, hp=1) for entity in self.state.entities
        ), crowns=(1, 0))
        self.assertEqual(self.mask(damaged, 0)[1, 18, 3], 0)
        unknown_hp = replace(self.state, entities=tuple(
            replace(entity, hp=None) for entity in self.state.entities
        ))
        self.assertEqual(self.mask(unknown_hp, 0)[1, 18, 3], 0)
        self.assertEqual(self.mask(missing, 0)[1, 18, 3], 1)
        fresh = replace(self.state, generation=2)
        self.env._begin_episode(self.slot, fresh, 1)
        self.assertEqual(self.mask(fresh, 0)[1, 18, 3], 0)

    def test_expanded_actions_reach_correct_native_side_and_unaffordable_cards_stay_masked(self):
        for owner in (0, 1):
            self.slot.state = self.remove_tower(owner, "left")
            self.slot.learner_owner = owner
            self.assertEqual(self.env.env_method("action_masks")[0][1, 18, 3], 1)
            self.assertEqual(self.env._observation(self.slot)["action_mask"][1, 18, 3], 1)
            play, attempted = self.env._action_to_play(self.slot, owner, (1, 18, 3))
            self.assertTrue(attempted)
            expected = (3500, 18500) if owner == 0 else (14500, 13500)
            self.assertEqual((play.x, play.y), expected)
            blocked, _ = self.env._action_to_play(self.slot, owner, (1, 18, 14))
            self.assertIsNone(blocked)
        poor = replace(self.state, players=tuple(
            replace(player, elixir=0) for player in self.state.players
        ))
        mask = self.mask(poor, 0)
        self.assertEqual(int(mask.sum()), 1)
        finalized = replace(self.state, finalized=True)
        self.assertEqual(int(self.mask(finalized, 0).sum()), 1)


class NativeBatchVecEnvTest(unittest.TestCase):
    def make_env(self, battles=2, roles=(0, 1)):
        fake = FakeBatch(battles)
        env = NativeBatchVecEnv(
            battles=battles,
            decision_ticks=10,
            seed=100,
            learner_roles=roles,
            opponent_policy=lambda observation, mask, slot: (0, 0, 0),
            batch_factory=lambda **kwargs: fake,
        )
        return env, fake

    def test_release_batch_is_available_through_vec_env_method(self):
        env, fake = self.make_env()
        try:
            env.reset()
            result = env.env_method("release_batch")
            self.assertEqual(result, [None, None])
            self.assertTrue(fake.closed)
            self.assertFalse(env.closed)
        finally:
            env.close()

    def test_one_batch_step_covers_all_slots_and_transforms_roles(self):
        env, fake = self.make_env()
        try:
            observations = env.reset()
            self.assertEqual(observations["spatial"].shape, (2, 16, 32, 18))
            self.assertEqual(observations["entities"].shape, (2, 128, 24))
            self.assertEqual(observations["action_history"].shape, (2, 32, 8))
            self.assertEqual(observations["public_belief"].shape, (2, 20))
            self.assertEqual(observations["temporal"].shape, (2, 20))
            self.assertEqual(float(observations["temporal"].sum()), 0.0)
            for index in range(2):
                unbatched = {key: value[index] for key, value in observations.items()}
                self.assertTrue(env.observation_space.contains(unbatched))

            actions = np.asarray(((1, 7, 1), (1, 7, 1)), dtype=np.int64)
            next_observations, rewards, dones, infos = env.step(actions)
            self.assertEqual(len(fake.step_calls), 1)
            groups, ticks = fake.step_calls[0]
            self.assertEqual(ticks, 10)
            self.assertEqual(len(groups), 2)
            self.assertEqual((groups[0][0].x, groups[0][0].y), (1500, 7500))
            self.assertEqual((groups[1][0].x, groups[1][0].y), (16500, 24500))
            self.assertEqual(rewards.shape, (2,))
            self.assertFalse(dones.any())
            self.assertTrue(all(info["deployment_succeeded"] for info in infos))
            self.assertTrue(all(info["crown_diff"] == 0.0 for info in infos))
            self.assertTrue(all(info["tower_hp_diff"] == 0.0 for info in infos))
            self.assertEqual(next_observations["action_history"][:, 0, 0].tolist(), [1, 1])
            self.assertEqual(next_observations["temporal"][:, 0].tolist(), [1.0, 1.0])
        finally:
            env.close()

    def test_public_cycle_belief_becomes_exact_after_four_plays(self):
        cards = vec_module.CARD_IDS[:4]
        hand, next_card, confidence = vec_module._cycle_belief(tuple(cards))
        expected_hand = set(vec_module.CARD_IDS[4:])
        for card, probability in zip(vec_module.CARD_IDS, hand):
            self.assertEqual(probability, float(card in expected_hand))
        self.assertEqual(next_card[0], 1.0)
        self.assertEqual(sum(next_card), 1.0)
        self.assertEqual(confidence, 1.0)

    def test_temporal_features_capture_continuous_state_changes(self):
        env, _ = self.make_env()
        try:
            env.reset()
            slot = env.slots[0]
            own_troop = protocol.Entity(
                500, 0, vec_module.CARD_IDS[0], 9000, 9000, 600, 600,
                None, 0, "troop", 0, "melee", "ground", "ground", False,
            )
            enemy_troop = protocol.Entity(
                501, 1, vec_module.CARD_IDS[1], 9000, 23000, 500, 500,
                None, 0, "troop", 0, "projectile", "ground",
                "air_and_ground", False,
            )
            before = replace(
                slot.state,
                entities=(*slot.state.entities, own_troop),
            )
            players = list(before.players)
            players[0] = replace(players[0], elixir=80_000)
            own_tower = next(
                entity for entity in before.entities
                if entity.owner == 0 and entity.card == -1
            )
            after_entities = tuple(
                replace(entity, hp=900)
                if entity.id == own_tower.id else entity
                for entity in before.entities
                if entity.id != own_troop.id
            ) + (enemy_troop,)
            after = replace(
                before,
                tick=100,
                players=tuple(players),
                entities=after_entities,
                plays=(protocol.CardPlay(1, 91, 1, vec_module.CARD_IDS[1], 3),),
            )
            slot.previous_state = before
            slot.state = after
            temporal = env._temporal_features(slot, owner=0)
            self.assertEqual(temporal.shape, (20,))
            self.assertEqual(temporal[0], 1.0)
            self.assertLess(temporal[2], 0.0)  # own tower damage
            self.assertLess(temporal[4], 0.0)  # own elixir spent
            self.assertGreater(temporal[7], 0.0)  # enemy spawn
            self.assertGreater(temporal[8], 0.0)  # own death
            self.assertGreater(temporal[13], 0.0)  # enemy play
        finally:
            env.close()

    def test_model_opponent_is_inferred_once_for_the_whole_batch(self):
        class WaitingModel:
            def __init__(self):
                self.calls = 0

            def predict(self, observation, deterministic=False):
                self.calls += 1
                count = observation["scalar"].shape[0]
                return np.zeros((count, 3), dtype=np.int64), None

        fake = FakeBatch(4)
        opponent = WaitingModel()
        env = NativeBatchVecEnv(
            battles=4,
            seed=3,
            opponent_policy=opponent,
            batch_factory=lambda **kwargs: fake,
        )
        try:
            env.reset()
            env.step(np.zeros((4, 3), dtype=np.int64))
            self.assertEqual(opponent.calls, 1)
        finally:
            env.close()

    def test_terminal_slot_is_reset_without_rewinding_survivor(self):
        env, fake = self.make_env()
        try:
            env.reset()
            surviving_generation = env.slots[1].state.generation
            fake.finish_slot_zero = True
            observations, _, dones, infos = env.step(
                np.zeros((2, 3), dtype=np.int64)
            )
            self.assertEqual(dones.tolist(), [True, False])
            self.assertEqual(fake.reset_slot_calls, [((0,), (102,))])
            self.assertEqual(env.slots[0].state.tick, 90)
            self.assertEqual(env.slots[1].state.tick, 100)
            self.assertEqual(env.slots[1].state.generation, surviving_generation)
            self.assertEqual(infos[0]["terminal_observation"]["scalar"].shape, (12,))
            self.assertEqual(env.slots[0].learner_owner, 0)
            self.assertEqual(observations["action_history"][0].sum(), 0)
        finally:
            env.close()

    def test_opponent_lifecycle_is_fixed_per_episode_and_reported(self):
        class RecordingOpponent:
            def __init__(self):
                self.starts = []
                self.ends = []
                self.assignments = {}

            def on_episode_start(self, slot_index, episode, seed, learner_role):
                opponent_id = f"opponent-{slot_index}-{episode}"
                self.assignments[slot_index] = opponent_id
                self.starts.append((slot_index, episode, seed, learner_role))

            def opponent_metadata(self, slot_index):
                return {
                    "opponent_id": self.assignments[slot_index],
                    "opponent_category": "historical",
                }

            def on_episode_end(self, slot_index, winner, learner_role):
                self.ends.append((slot_index, winner, learner_role))
                return {"self_play_result": "win"}

            def predict(self, observation, deterministic=False):
                count = observation["scalar"].shape[0]
                return np.zeros((count, 3), dtype=np.int64), None

        fake = FakeBatch(2)
        opponent = RecordingOpponent()
        env = NativeBatchVecEnv(
            battles=2,
            seed=11,
            opponent_policy=opponent,
            batch_factory=lambda **kwargs: fake,
        )
        try:
            env.reset()
            self.assertEqual(len(opponent.starts), 2)
            original_id = opponent.assignments[0]
            fake.finish_slot_zero = True
            _, _, dones, infos = env.step(np.zeros((2, 3), dtype=np.int64))
            self.assertTrue(dones[0])
            self.assertEqual(infos[0]["opponent_id"], original_id)
            self.assertEqual(infos[0]["self_play_result"], "win")
            self.assertEqual(len(opponent.ends), 1)
            self.assertEqual(len(opponent.starts), 3)
            self.assertNotEqual(opponent.assignments[0], original_id)
        finally:
            env.close()

    def test_vec_monitor_accepts_the_custom_vec_env(self):
        env, fake = self.make_env()
        monitored = VecMonitor(env)
        try:
            monitored.reset()
            fake.finish_slot_zero = True
            _, _, dones, infos = monitored.step(np.zeros((2, 3), dtype=np.int64))
            self.assertTrue(dones[0])
            self.assertIn("episode", infos[0])
        finally:
            monitored.close()
        self.assertTrue(fake.closed)

    def test_hybrid_masked_policy_samples_and_backpropagates(self):
        fake = FakeBatch(4)
        env = NativeBatchVecEnv(
            battles=4,
            seed=7,
            learner_roles=(0, 1, 0, 1),
            opponent_policy=lambda observation, mask, slot: (0, 0, 0),
            batch_factory=lambda **kwargs: fake,
        )
        model = PPO(
            model_module.NativeHybridMaskedPolicy,
            env,
            n_steps=2,
            batch_size=8,
            n_epochs=1,
            device="cpu",
            seed=7,
            policy_kwargs={
                "features_extractor_kwargs": {
                    "features_dim": 64,
                    "spatial_channels": 16,
                    "entity_dim": 16,
                    "entity_layers": 1,
                    "entity_heads": 4,
                    "history_dim": 16,
                    "card_dim": 4,
                },
                "net_arch": {"pi": [32], "vf": [32]},
            },
            verbose=0,
        )
        try:
            observations = env.reset()
            actions, _ = model.predict(observations, deterministic=False)
            for index, (policy_slot, y, x) in enumerate(actions):
                self.assertEqual(
                    observations["action_mask"][index, policy_slot, y, x], 1.0
                )
            model.learn(total_timesteps=8)
            self.assertEqual(model.num_timesteps, 8)
            self.assertEqual(len(fake.step_calls), 2)
            self.assertEqual(
                model.policy.features_extractor.architecture_version,
                "cnn_entity_transformer_event_lstm_v2",
            )
        finally:
            env.close()

    def test_recurrent_state_persists_and_resets_one_slot(self):
        env, _ = self.make_env()
        try:
            observations = env.reset()
            policy = recurrent_module.NativeRecurrentMaskedPolicy(
                env.observation_space,
                env.action_space,
                lambda _: 3e-4,
                lstm_hidden_size=32,
            )
            _, first = policy.predict(
                observations,
                episode_start=np.asarray((True, True)),
                deterministic=True,
            )
            _, second = policy.predict(
                observations,
                state=first,
                episode_start=np.asarray((False, False)),
                deterministic=True,
            )
            self.assertFalse(np.allclose(first[0], second[0]))
            _, partially_reset = policy.predict(
                observations,
                state=second,
                episode_start=np.asarray((True, False)),
                deterministic=True,
            )
            single = {key: value[0:1] for key, value in observations.items()}
            _, fresh = policy.predict(
                single,
                episode_start=np.asarray((True,)),
                deterministic=True,
            )
            np.testing.assert_allclose(
                partially_reset[0][:, 0:1], fresh[0], rtol=1e-5, atol=1e-6
            )
            self.assertFalse(np.allclose(partially_reset[0][:, 1], fresh[0][:, 0]))
        finally:
            env.close()

    def test_recurrent_ppo_uses_truncated_sequence_minibatches(self):
        env, _ = self.make_env()
        model = recurrent_module.NativeRecurrentPPO(
            recurrent_module.NativeRecurrentMaskedPolicy,
            env,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            tbptt_steps=4,
            policy_kwargs={"lstm_hidden_size": 32},
            device="cpu",
            seed=0,
        )
        try:
            model.learn(total_timesteps=16)
            self.assertGreater(model.rollout_buffer.max_sampled_sequence_length, 0)
            self.assertLessEqual(
                model.rollout_buffer.max_sampled_sequence_length, 4
            )
            self.assertTrue(all(
                bool(np.isfinite(parameter.detach().numpy()).all())
                for parameter in model.policy.parameters()
            ))
        finally:
            env.close()

    def test_cross_modal_policy_uses_context_tokens_and_backpropagates(self):
        fake = FakeBatch(2)
        env = NativeBatchVecEnv(
            battles=2,
            seed=19,
            learner_roles=(0, 1),
            opponent_policy=lambda observation, mask, slot: (0, 0, 0),
            batch_factory=lambda **kwargs: fake,
        )
        model = recurrent_module.NativeRecurrentPPO(
            recurrent_module.NativeCrossModalRecurrentPolicy,
            env,
            n_steps=2,
            batch_size=4,
            n_epochs=1,
            tbptt_steps=2,
            policy_kwargs={
                "lstm_hidden_size": 32,
                "features_extractor_kwargs": {
                    "features_dim": 64,
                    "spatial_channels": 16,
                    "cross_dim": 16,
                    "entity_layers": 1,
                    "attention_heads": 4,
                },
                "net_arch": {"pi": [32], "vf": [32]},
            },
            device="cpu",
            seed=19,
        )
        try:
            observations = env.reset()
            actions, _ = model.predict(observations, deterministic=False)
            for index, (policy_slot, y, x) in enumerate(actions):
                self.assertEqual(
                    observations["action_mask"][index, policy_slot, y, x], 1.0
                )

            extractor = model.policy.features_extractor
            tensors, _ = model.policy.obs_to_tensor(observations)
            with torch.no_grad():
                extractor(tensors)
                card_tokens, placement_tokens = extractor.action_tokens()
                original_cards = card_tokens.clone()
                changed = dict(tensors)
                changed["public_belief"] = tensors["public_belief"].clone()
                changed["public_belief"][:, 0] = 1.0
                extractor(changed)
                changed_cards, _ = extractor.action_tokens()
            self.assertEqual(tuple(original_cards.shape), (2, 4, 16))
            self.assertEqual(tuple(placement_tokens.shape), (2, 32 * 18, 16))
            self.assertFalse(torch.allclose(original_cards, changed_cards))
            self.assertFalse(torch.allclose(
                placement_tokens[:, 0], placement_tokens[:, -1]
            ))

            model.learn(total_timesteps=4)
            attention_parameters = (
                extractor.card_context_attention.in_proj_weight,
                extractor.history_entity_attention.in_proj_weight,
                extractor.placement_attention.in_proj_weight,
            )
            self.assertTrue(all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in attention_parameters
            ))
            self.assertEqual(
                extractor.architecture_version, "cross_modal_stateful_v3"
            )
        finally:
            env.close()

    def test_spatial_heatmap_policy_preserves_map_and_conditions_on_card(self):
        fake = FakeBatch(2)
        env = NativeBatchVecEnv(
            battles=2,
            seed=23,
            learner_roles=(0, 1),
            opponent_policy=lambda observation, mask, slot: (0, 0, 0),
            batch_factory=lambda **kwargs: fake,
        )
        model = recurrent_module.NativeRecurrentPPO(
            recurrent_module.NativeSpatialHeatmapRecurrentPolicy,
            env,
            n_steps=2,
            batch_size=4,
            n_epochs=1,
            tbptt_steps=2,
            policy_kwargs={
                "lstm_hidden_size": 32,
                "features_extractor_kwargs": {
                    "features_dim": 64,
                    "spatial_channels": 16,
                    "cross_dim": 16,
                    "entity_layers": 1,
                    "attention_heads": 4,
                },
                "net_arch": {"pi": [32], "vf": [32]},
            },
            device="cpu",
            seed=23,
        )
        try:
            observations = env.reset()
            actions, _ = model.predict(observations, deterministic=False)
            for index, (policy_slot, y, x) in enumerate(actions):
                self.assertEqual(
                    observations["action_mask"][index, policy_slot, y, x], 1.0
                )

            policy = model.policy
            extractor = policy.features_extractor
            tensors, _ = policy.obs_to_tensor(observations)
            actor = torch.zeros(
                (2, policy.mlp_extractor.latent_dim_pi),
                device=policy.device,
            )
            with torch.no_grad():
                extractor(tensors)
                cards, spatial_tokens = extractor.action_tokens()
                policy.action_net(actor, cards, spatial_tokens)
                original = policy.action_net.heatmaps().clone()

                changed = dict(tensors)
                changed["spatial"] = tensors["spatial"].clone()
                changed["spatial"][:, 0, 16, 9] = 1.0
                extractor(changed)
                changed_cards, changed_spatial = extractor.action_tokens()
                policy.action_net(actor, changed_cards, changed_spatial)
                perturbed = policy.action_net.heatmaps().clone()

            self.assertEqual(
                extractor.spatial_feature_shape,
                (vec_module.ARENA_HEIGHT, vec_module.ARENA_WIDTH),
            )
            self.assertEqual(tuple(spatial_tokens.shape), (2, 32 * 18, 16))
            self.assertEqual(tuple(original.shape), (2, 4, 32, 18))
            self.assertFalse(torch.allclose(original[:, 0], original[:, 1]))
            self.assertFalse(torch.allclose(original, perturbed))

            model.learn(total_timesteps=4)
            spatial_parameters = (
                extractor.spatial_backbone[0].weight,
                policy.action_net.card_projection.weight,
                policy.action_net.spatial_key_projection.weight,
            )
            self.assertTrue(all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in spatial_parameters
            ))
            self.assertEqual(
                extractor.architecture_version,
                "spatial_heatmap_stateful_v4",
            )
        finally:
            env.close()

    def test_world_model_auxiliary_tasks_train_with_recurrent_ppo(self):
        class CombatFakeBatch(FakeBatch):
            def _reset_state(self, slot):
                state = super()._reset_state(slot)
                troop = protocol.Entity(
                    slot * 100 + 50,
                    slot % 2,
                    vec_module.CARD_IDS[0],
                    9000,
                    9000 if slot % 2 == 0 else 23000,
                    600,
                    600,
                    None,
                    0,
                    "troop",
                    0,
                    "melee",
                    "ground",
                    "ground",
                    False,
                )
                return replace(state, entities=(*state.entities, troop))

        fake = CombatFakeBatch(2)
        env = NativeBatchVecEnv(
            battles=2,
            seed=29,
            learner_roles=(0, 1),
            opponent_policy=lambda observation, mask, slot: (0, 0, 0),
            batch_factory=lambda **kwargs: fake,
        )
        model = recurrent_module.NativeWorldModelPPO(
            recurrent_module.NativeWorldModelRecurrentPolicy,
            env,
            n_steps=4,
            batch_size=8,
            n_epochs=1,
            tbptt_steps=4,
            auxiliary_coef=0.5,
            auxiliary_horizon=2,
            policy_kwargs={
                "lstm_hidden_size": 32,
                "world_model_hidden_dim": 32,
                "features_extractor_kwargs": {
                    "features_dim": 64,
                    "spatial_channels": 16,
                    "cross_dim": 16,
                    "entity_layers": 1,
                    "attention_heads": 4,
                },
                "net_arch": {"pi": [32], "vf": [32]},
            },
            device="cpu",
            seed=29,
        )
        try:
            observations = env.reset()
            model.predict(
                observations,
                episode_start=np.ones(2, dtype=bool),
                deterministic=False,
            )
            predictions = model.policy.auxiliary_predictions()
            self.assertEqual(tuple(predictions["tower_delta"].shape), (2, 2))
            self.assertEqual(
                tuple(predictions["entity_death"].shape), (2, 128)
            )
            self.assertEqual(
                tuple(predictions["entity_position_delta"].shape),
                (2, 128, 2),
            )
            self.assertEqual(
                tuple(predictions["opponent_hand"].shape), (2, 8)
            )

            model.learn(total_timesteps=8)
            self.assertIsInstance(
                model.rollout_buffer,
                recurrent_module.EntityTargetRecurrentDictRolloutBuffer,
            )
            self.assertNotIn("entity_targets", observations)
            self.assertNotIn("entity_ids", observations)
            self.assertEqual(
                model.policy.features_extractor.architecture_version,
                "auxiliary_world_model_stateful_v5",
            )
            self.assertFalse(model.policy.action_conditioned)
            self.assertFalse(any("action_encoder" in key or "dynamics_encoder" in key
                for key in model.policy.world_model.state_dict()))
            self.assertIn("train/auxiliary_loss", model.logger.name_to_value)
            self.assertTrue(np.isfinite(
                model.logger.name_to_value["train/auxiliary_loss"]
            ))
            auxiliary_parameters = (
                model.policy.world_model.tower_delta.weight,
                model.policy.world_model.entity_death.weight,
                model.policy.world_model.opponent_hand.weight,
            )
            self.assertTrue(all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                and float(parameter.grad.abs().sum()) > 0.0
                for parameter in auxiliary_parameters
            ))
            with TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "world_model"
                model.save(checkpoint)
                loaded = recurrent_module.NativeWorldModelPPO.load(
                    checkpoint, env=env, device="cpu"
                )
                loaded.predict(observations, deterministic=True)
                loaded.learn(total_timesteps=8)
                self.assertTrue(np.isfinite(
                    loaded.logger.name_to_value["train/auxiliary_loss"]
                ))
        finally:
            env.close()

    def test_training_device_requires_real_mps(self):
        self.assertEqual(train_module.require_training_device("cpu"), "cpu")
        with self.assertRaisesRegex(ValueError, "mps.*cpu"):
            train_module.require_training_device("auto")
        with (
            patch.object(train_module.torch.backends.mps, "is_built", return_value=True),
            patch.object(train_module.torch.backends.mps, "is_available", return_value=True),
        ):
            self.assertEqual(train_module.require_training_device("mps"), "mps")
        with (
            patch.object(train_module.torch.backends.mps, "is_built", return_value=True),
            patch.object(train_module.torch.backends.mps, "is_available", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                train_module.require_training_device("mps")


if __name__ == "__main__":
    unittest.main()
