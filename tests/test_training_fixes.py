import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_native_vec_env import FakeBatch, make_state
from test_training_experiments import TerminalBatch
import test_training_experiments as experiment_tests
from native_engine.native_vec_env import NativeBatchVecEnv, _public_transition_targets, MAX_ENTITIES
from native_engine.native_recurrent_model import (
    NativeActionConditionedWorldModelPPO, EntityTargetRecurrentDictRolloutBuffer, RNNStates,
)
from native_engine.native_vec_env import native_observation_space, ENTITY_TARGET_SCHEMA, TRANSITION_TARGET_SCHEMA
from types import SimpleNamespace
from training_runtime import run_sb3_training, resume_directory
from self_play import SelfPlayOpponentPool
from auxiliary_validation import collect_dataset, score_dataset
from native_visualizer import record_replay, write_replay


class TrainingFixesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        result = experiment_tests.TrainingExperimentsTest().config()
        result.update(rounds=2, checkpoint_every_rounds=1, fixed_evaluation_enabled=False,
                      self_play_snapshot_every_rounds=1)
        return result

    def train(self, root, config=None):
        config = config or self.config()
        with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
            result = run_sb3_training(config, root)
        return result

    def test_historical_checkpoint_loading_preserves_real_rng_streams(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            self.train(root / 'train', config)
            pool = SelfPlayOpponentPool(root / 'league', NativeActionConditionedWorldModelPPO,
                                        'cpu', 2, random_ratio=0, current_ratio=0, historical_ratio=1)
            pool.register_snapshot('snapshot', root / 'train/final_model.zip', 2)
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            expected = (random.random(), np.random.rand(), torch.rand(()))
            random.seed(123); np.random.seed(123); torch.manual_seed(123)
            pool._load_snapshot('snapshot')
            actual = (random.random(), np.random.rand(), torch.rand(()))
            self.assertEqual(actual[:2], expected[:2])
            torch.testing.assert_close(actual[2], expected[2])
            pool.close()

    def test_projectile_expiry_and_spawn_hp_do_not_invent_damage(self):
        before = make_state(0, 1)
        unit = replace(before.entities[0], id=9, card=26000000, entity_type='troop', hp=100, max_hp=100)
        projectile = replace(unit, id=10, entity_type='projectile', hp=None, max_hp=None)
        before = replace(before, entities=(*before.entities, unit, projectile))
        after = replace(before, tick=100, entities=(*before.entities[:2], unit))
        target = _public_transition_targets(before, after, 0, (1000, 1000))
        self.assertEqual(np.argmax(target[2:7]), 0)
        damaged = replace(unit, hp=70)
        spawned = replace(unit, id=11, hp=1000, max_hp=1000)
        after = replace(after, entities=(*after.entities[:2], damaged, spawned))
        target = _public_transition_targets(before, after, 0, (1000, 1000))
        self.assertEqual(np.argmax(target[2:7]), 3)
        self.assertAlmostEqual(target[5], .3)

    def test_spatial_keeps_overflow_entities_and_same_cell_target(self):
        fake = FakeBatch(1)
        env = NativeBatchVecEnv(battles=1, learner_roles='p0', batch_factory=lambda **kw: fake)
        try:
            env.reset()
            base = env.slots[0].state
            unit = replace(base.entities[0], id=3, card=26000000, entity_type='troop', x=5000, y=10000,
                           target_id=base.entities[1].id)
            idle = replace(unit, id=4, target_id=None)
            padding = [replace(idle, id=10+i, x=9000, y=10000) for i in range(MAX_ENTITIES)]
            overflow = replace(unit, id=10000, x=15000, y=20000)
            env.slots[0].state = replace(base, entities=(*base.entities, unit, idle, *padding, overflow))
            observation = env._observation(env.slots[0])
            self.assertEqual(observation['spatial'][14, 10, 5], 1)
            self.assertEqual(observation['spatial'][0, 20, 15], 1)
            self.assertEqual(observation['entity_mask'].sum(), MAX_ENTITIES)
            self.assertFalse(np.any(np.isclose(observation['entities'][:, 14], 15/18)))
        finally:
            env.close()

    def test_independent_labels_include_terminal_and_last_sequence_step(self):
        observation = {key: torch.zeros(2, *space.shape)
                       for key, space in native_observation_space().spaces.items()}
        observation['entity_mask'][1, 0] = 1
        entities = torch.zeros(2, MAX_ENTITIES, 4)
        entities[1, 0, 0] = entities[1, 0, 3] = 1
        transition = torch.zeros(2, 8)
        transition[1, :2] = torch.tensor([1., -1.])
        transition[1, 7] = 1
        predictions = {name: torch.zeros(2, size) for name, size in (
            ('tower_delta', 2), ('damage_source', 5), ('opponent_elixir', 5),
            ('opponent_hand', 8), ('opponent_next_card', 8), ('entity_death', MAX_ENTITIES))}
        predictions.update(entity_position_delta=torch.zeros(2, MAX_ENTITIES, 2), future_value=torch.zeros(2))
        predictions['tower_delta'].requires_grad_()
        samples = SimpleNamespace(observations=observation, entity_targets=entities,
            transition_targets=transition, future_targets=torch.zeros(2, 2), mask=torch.ones(2),
            lstm_states=RNNStates((torch.zeros(1, 1, 8),)*2, (torch.zeros(1, 1, 8),)*2))
        model = NativeActionConditionedWorldModelPPO.__new__(NativeActionConditionedWorldModelPPO)
        model.auxiliary_weights = model.DEFAULT_AUXILIARY_WEIGHTS
        total, losses = model._auxiliary_losses(samples, predictions)
        self.assertAlmostEqual(float(losses['tower_delta'].detach()), .5)
        self.assertAlmostEqual(float(losses['entity_death']), np.log(2), places=6)
        total.backward()
        self.assertGreater(float(predictions['tower_delta'].grad[1].abs().sum()), 0)

    def test_future_targets_cross_tbptt_but_never_episodes(self):
        env = NativeBatchVecEnv(battles=1)
        buffer = EntityTargetRecurrentDictRolloutBuffer(4, native_observation_space(), env.action_space,
            (4, 1, 1, 8), n_envs=1, device='cpu', tbptt_steps=2, auxiliary_horizon=2)
        states = RNNStates((torch.zeros(1, 1, 8),)*2, (torch.zeros(1, 1, 8),)*2)
        for step in range(4):
            observation = {key: np.zeros((1, *space.shape), dtype=space.dtype)
                           for key, space in native_observation_space().spaces.items()}
            observation['scalar'][0, 0] = (step + 1) / 10
            observation['scalar'][0, 9] = .2 if step == 2 else .9 if step == 3 else 0
            info = {'entity_target_schema': ENTITY_TARGET_SCHEMA, 'entity_targets': np.zeros((MAX_ENTITIES, 4)),
                    'transition_target_schema': TRANSITION_TARGET_SCHEMA, 'transition_targets': np.zeros(8)}
            if step == 2:
                terminal = np.zeros(12); terminal[9] = .3
                info['terminal_observation'] = {'scalar': terminal}
            buffer.set_entity_targets([info])
            buffer.add(observation, np.zeros((1, 3)), np.zeros(1), np.asarray([step in (0, 3)]),
                       torch.zeros(1), torch.zeros(1), lstm_states=states)
        for sample in buffer.get(batch_size=4):
            for marker, target in zip(sample.observations['scalar'][:, 0][sample.mask > 0],
                                      sample.future_targets[sample.mask > 0]):
                step = round(float(marker) * 10) - 1
                if step < 2:
                    self.assertEqual(target[1], 1)
                    self.assertAlmostEqual(float(target[0]), (.2, .3)[step], places=6)
                else:
                    self.assertEqual(target[1], 0)
        env.close()

    def test_checkpoints_and_logs_include_the_named_round_update(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            self.train(root, config)
            for completed in (1, 2):
                model = NativeActionConditionedWorldModelPPO.load(
                    root / f'checkpoints/model_{completed * 8}_steps.zip', device='cpu')
                self.assertEqual(model._n_updates, completed)
                self.assertTrue(resume_directory(root / f'checkpoints/model_{completed * 8}_steps.zip').is_dir())
            records = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['round'] for row in records], [1, 2])
            self.assertTrue(all(row['train/n_updates'] == row['round'] for row in records))
            self.assertTrue(all('exploration/wait_probability' in row for row in records))
            for row in records:
                total = row['exploration/wait_rate'] + sum(row[f'exploration/card_{card}_rate'] for card in range(1, 9))
                self.assertAlmostEqual(total, 1, places=6)

    def test_deferred_and_independent_evaluations_never_reset_training(self):
        for separate in (False, True):
            with self.subTest(independent_port=separate), TemporaryDirectory() as directory:
                config = self.config()
                config.update(fixed_evaluation_enabled=True, port=101)
                if separate:
                    config['evaluation_port'] = 102
                batches = []
                class CountBatch(TerminalBatch):
                    resets = 0
                    def reset(self, seed=1):
                        self.resets += 1
                        return super().reset(seed)
                def factory(**kw):
                    batch = CountBatch(kw['battles'])
                    batches.append(batch)
                    return batch
                with patch('native_engine.native_vec_env.Batch', side_effect=factory):
                    metadata = run_sb3_training(config, directory)
                self.assertEqual(batches[0].resets, 1)
                self.assertEqual(metadata['fixed_evaluation']['evaluated_rounds'], [0, 1, 2])
                schedule = json.loads((Path(directory) / 'evaluation/schedule.json').read_text())
                self.assertEqual(schedule['execution'], 'immediate' if separate else 'after_training')

    def test_resume_restores_optimizer_league_budget_and_seed_cursor(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            config.update(self_play_enabled=True, self_play_current_ratio=0, self_play_historical_ratio=1)
            original_config = copy.deepcopy(config)
            first = self.train(root / 'first', config)
            self.assertEqual(config, original_config)
            checkpoint = root / 'first/final_model.zip'
            state = json.loads((resume_directory(checkpoint) / 'runtime.json').read_text())
            resumed = json.loads((resume_directory(checkpoint) / 'config.json').read_text())
            resumed.update(rounds=3, resume_from=str(checkpoint))
            second = self.train(root / 'second', resumed)
            self.assertEqual(second['timesteps'], 24)
            self.assertEqual(second['session_timesteps'], 8)
            records = [json.loads(line) for line in (root / 'second/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['session_timesteps'], 8)
            model = NativeActionConditionedWorldModelPPO.load(root / 'second/final_model.zip', device='cpu')
            self.assertEqual(model._n_updates, 3)
            self.assertGreater(second['self_play']['self_play/games'], first['self_play']['self_play/games'])
            new_state = json.loads((resume_directory(root / 'second/final_model.zip') / 'runtime.json').read_text())
            self.assertGreater(new_state['next_native_seed'], state['next_native_seed'])
            self.assertTrue((resume_directory(root / 'second/final_model.zip') / 'league/state.json').is_file())

    def test_final_checkpoint_survives_deferred_evaluation_failure(self):
        with TemporaryDirectory() as directory:
            config = self.config()
            config.update(checkpoint_every_rounds=5, fixed_evaluation_enabled=True)
            with patch('fixed_evaluation.FixedEvaluationCallback.evaluate', side_effect=RuntimeError('evaluation failed')):
                with self.assertRaisesRegex(RuntimeError, 'evaluation failed'):
                    self.train(directory, config)
            checkpoint = Path(directory) / 'checkpoints/model_16_steps.zip'
            model = NativeActionConditionedWorldModelPPO.load(checkpoint, device='cpu')
            self.assertEqual(model._n_updates, 2)
            self.assertTrue((resume_directory(checkpoint) / 'runtime.json').is_file())

    def test_independent_auxiliary_dataset_and_actual_heatmap_replay(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            self.train(root / 'train', config)
            model = NativeActionConditionedWorldModelPPO.load(root / 'train/final_model.zip', device='cpu')
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                dataset = collect_dataset(config, model, root / 'validation.npz', seed_count=2,
                                          decisions=4, battles=2)
                frames = record_replay(config, model, steps=3)
            before = {key: value.clone() for key, value in model.policy.state_dict().items()}
            result = score_dataset(config, model, dataset, batch_size=2)
            microbatched = score_dataset(config, model, dataset, batch_size=1)
            for key, value in result['metrics'].items():
                other = microbatched['metrics'][key]
                if value is None:
                    self.assertIsNone(other)
                else:
                    self.assertAlmostEqual(value, other, places=5)
            self.assertEqual(result['training_updates'], 0)
            self.assertEqual(result['dataset']['samples'], 8)
            self.assertEqual(result['metrics']['future_labels'], 4)
            self.assertTrue(np.isfinite(result['metrics']['tower_delta_mae']))
            for key, value in model.policy.state_dict().items():
                torch.testing.assert_close(value, before[key])
            write_replay(root / 'replay.html', frames)
            self.assertIn('P(tile | card)', (root / 'replay.html').read_text())
            for frame in frames:
                probabilities = np.asarray(frame['probabilities'])
                legal = np.asarray(frame['legal'])
                self.assertTrue(np.all(probabilities[~legal] == 0))


if __name__ == '__main__':
    unittest.main()
