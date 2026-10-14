"""Regression cases for lost card roles and same-cell combat information."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from native_engine.native_vec_env import NativeBatchVecEnv, MAX_ENTITIES, native_observation_space
from native_engine.native_recurrent_model import (
    NativeActionConditionedWorldModelPPO,
    NativeActionConditionedWorldModelRecurrentPolicy,
    NativeRoleAwareActionConditionedWorldModelRecurrentPolicy,
)
from policy_contracts import observations_for_policy
from test_native_vec_env import FakeBatch


class ModelInformationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def env(self, schema="native_model_v4"):
        env = NativeBatchVecEnv(battles=1, observation_schema=schema,
            batch_factory=lambda **kwargs: FakeBatch(kwargs["battles"]))
        self.addCleanup(env.close)
        env.reset()
        return env

    def model(self, env, policy=NativeRoleAwareActionConditionedWorldModelRecurrentPolicy):
        return NativeActionConditionedWorldModelPPO(policy, env,
            n_steps=4, batch_size=4, n_epochs=1, tbptt_steps=4, auxiliary_horizon=2,
            seed=7, device="cpu", policy_kwargs={
                "lstm_hidden_size": 16, "world_model_hidden_dim": 16,
                "features_extractor_kwargs": {"features_dim": 32, "spatial_channels": 8,
                    "cross_dim": 8, "entity_layers": 1, "attention_heads": 4},
                "net_arch": {"pi": [16], "vf": [16]}})

    def test_next_card_swap_changes_features_and_value_but_hand_permutation_does_not(self):
        env = self.env()
        model = self.model(env)
        obs = env.reset()
        obs["hand"][:] = (1, 2, 3, 4, 5)
        tensors, _ = model.policy.obs_to_tensor(obs)
        extractor = model.policy.features_extractor
        model.policy.set_training_mode(False)
        with torch.no_grad():
            features = extractor(tensors).clone()
            values, _, _ = model.policy.evaluate_actions(tensors, torch.zeros((1, 3)),
                model._last_lstm_states, torch.ones(1))
            swapped = dict(tensors, hand=torch.tensor([[1, 2, 3, 5, 4]]))
            self.assertFalse(torch.allclose(features, extractor(swapped)))
            new_values, _, _ = model.policy.evaluate_actions(swapped, torch.zeros((1, 3)),
                model._last_lstm_states, torch.ones(1))
            self.assertGreater(float((values - new_values).abs().max()), 1e-8)
            permuted = dict(tensors, hand=torch.tensor([[4, 3, 2, 1, 5]]))
            self.assertTrue(torch.allclose(features, extractor(permuted), atol=1e-6))
            missing = dict(tensors, hand=torch.zeros((1, 5), dtype=torch.long))
            self.assertTrue(torch.isfinite(extractor(missing)).all())
            self.assertEqual(int(torch.count_nonzero(extractor.action_tokens()[0])), 0)

    def test_same_cell_counts_and_hp_preserve_legacy_prefix_and_player_perspective(self):
        env = self.env()
        slot = env.slots[0]
        base = slot.state
        troop = replace(base.entities[0], id=99, owner=1, card=26000000,
            entity_type="troop", x=8500, y=17500, hp=100, max_hp=100)
        slot.state = replace(base, entities=base.entities + (troop,))
        one = env._observation(slot)
        slot.state = replace(base, entities=base.entities + (troop, replace(troop, id=100)))
        two = env._observation(slot)
        np.testing.assert_array_equal(one["spatial"][:16], two["spatial"][:16])
        self.assertAlmostEqual(two["spatial"][17, 17, 8], 2 / MAX_ENTITIES)
        self.assertAlmostEqual(two["spatial"][19, 17, 8], .02)
        self.assertGreater(two["spatial"][17, 17, 8], one["spatial"][17, 17, 8])
        own = env._observation(slot, owner=1)
        self.assertAlmostEqual(own["spatial"][16, 14, 9], 2 / MAX_ENTITIES)
        self.assertAlmostEqual(own["spatial"][18, 14, 9], .02)
        self.assertTrue(env.observation_space.contains(two))
        legacy = self.env("native_model_v3")
        legacy.slots[0].state = slot.state
        np.testing.assert_array_equal(legacy._observation(legacy.slots[0])["spatial"], two["spatial"][:16])

    def test_aggregation_includes_entities_beyond_token_cap_and_clips(self):
        env = self.env()
        slot = env.slots[0]
        base = slot.state
        troop = replace(base.entities[0], owner=1, card=26000000,
            entity_type="troop", x=8500, y=17500, hp=100, max_hp=100)
        slot.state = replace(base, entities=base.entities + tuple(
            replace(troop, id=100 + row) for row in range(MAX_ENTITIES + 2)))
        obs = env._observation(slot)
        self.assertEqual(float(obs["spatial"][17, 17, 8]), 1.)
        self.assertEqual(float(obs["spatial"][19, 17, 8]), 1.)
        self.assertEqual(float(obs["entity_mask"].sum()), MAX_ENTITIES)
        self.assertTrue(env.observation_space.contains(obs))

    def test_v7_train_save_load_and_legacy_opponent(self):
        env = self.env()
        model = self.model(env)
        model.learn(total_timesteps=4)
        self.assertTrue(np.isfinite(model.logger.name_to_value["train/loss"]))
        gradient = model.policy.features_extractor.card_role_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)
        observations = env.reset()
        expected, state = model.predict(observations, deterministic=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "v7.zip"
            model.save(path)
            loaded = NativeActionConditionedWorldModelPPO.load(path, env=env, device="cpu")
            actual, restored = loaded.predict(observations, deterministic=True)
            np.testing.assert_array_equal(actual, expected)
            for left, right in zip(state, restored):
                np.testing.assert_array_equal(left, right)
            loaded.learn(total_timesteps=4)
            legacy_env = self.env("native_model_v3")
            legacy = self.model(legacy_env, NativeActionConditionedWorldModelRecurrentPolicy)
            legacy.save(Path(directory) / "v6.zip")
            legacy = NativeActionConditionedWorldModelPPO.load(Path(directory) / "v6.zip", device="cpu")
            self.assertFalse(hasattr(legacy.policy.features_extractor, "card_role_embedding"))
            adapted = observations_for_policy(legacy, observations)
            self.assertEqual(adapted["spatial"].shape, (1, 16, 32, 18))
            np.testing.assert_array_equal(adapted["spatial"], observations["spatial"][:, :16])
            env.opponent_policy = legacy
            env.reset()
            env.step(np.zeros((1, 3), dtype=np.int64))

    def test_unknown_observation_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported observation schema"):
            native_observation_space("typo")


if __name__ == "__main__":
    unittest.main()
