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
from native_engine.native_vec_env import CARD_IDS, _cycle_belief
from types import SimpleNamespace
from training_runtime import run_sb3_training, resume_directory, load_inference_model
from one_click_train import validate_config
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

    def test_inference_loading_preserves_recurrent_predictions_without_training_buffers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self.train(root)
            path = root / 'final_model.zip'
            original = NativeActionConditionedWorldModelPPO.load(path, device='cpu')
            light = load_inference_model(NativeActionConditionedWorldModelPPO, path, 'cpu')
            self.assertEqual(light.n_steps, 1)
            self.assertIsNone(light.rollout_buffer)
            self.assertFalse(light.policy.optimizer.state)
            self.assertTrue(original.policy.optimizer.state)
            env = NativeBatchVecEnv(battles=2, batch_factory=lambda **kw: FakeBatch(2))
            try:
                observation = env.reset()
                original_state = light_state = None
                for starts in ([True, True], [False, False], [False, True]):
                    expected, original_state = original.predict(observation, state=original_state,
                        episode_start=np.asarray(starts), deterministic=True)
                    actual, light_state = light.predict(observation, state=light_state,
                        episode_start=np.asarray(starts), deterministic=True)
                    np.testing.assert_array_equal(actual, expected)
                    for left, right in zip(original_state, light_state):
                        np.testing.assert_allclose(left, right)
                env.opponent_policy = light
                env.reset()
                recorded = []
                predict = light.predict
                def capture(*args, **kwargs):
                    recorded.append((kwargs['state'], kwargs['episode_start'].copy()))
                    return predict(*args, **kwargs)
                with patch.object(light, 'predict', side_effect=capture):
                    env.step(np.zeros((2, 3), dtype=np.int64))
                    env.batch.finish_slot_zero = True
                    env.step(np.zeros((2, 3), dtype=np.int64))
                    env.step(np.zeros((2, 3), dtype=np.int64))
                self.assertIsNone(recorded[0][0])
                self.assertIsNotNone(recorded[1][0])
                np.testing.assert_array_equal(recorded[0][1], [True, True])
                np.testing.assert_array_equal(recorded[1][1], [False, False])
                np.testing.assert_array_equal(recorded[2][1], [True, False])
            finally:
                env.close()

    def test_incremental_cycle_beliefs_equal_complete_history_reconstruction(self):
        env = NativeBatchVecEnv(battles=1, batch_factory=lambda **kw: FakeBatch(1))
        try:
            env.reset()
            slot = env.slots[0]
            sequences = [[], []]
            # Include both viewpoints, long histories and an impossible cycle.
            for index, card in enumerate((*CARD_IDS, *CARD_IDS, CARD_IDS[-1])):
                for owner in (0, 1):
                    sequences[owner].append(card)
                    event = SimpleNamespace(tick=91 + index, owner=owner, card=card, cost=1)
                    env._record_plays(slot, SimpleNamespace(plays=(event,)), (), ())
                    expected = _cycle_belief(tuple(sequences[owner]))
                    for actual, reference in zip(slot.cycle_beliefs[owner], expected):
                        np.testing.assert_allclose(actual, reference)
                    observation = env._observation(slot, owner=1-owner)
                    np.testing.assert_allclose(observation['public_belief'][:16], (*expected[0], *expected[1]))
            env.reset()
            self.assertEqual(slot.cycle_beliefs, [_cycle_belief(())] * 2)
        finally:
            env.close()

    def test_environment_rejects_truncated_receipts_and_partial_reset_results(self):
        fake = FakeBatch(2)
        env = NativeBatchVecEnv(battles=2, batch_factory=lambda **kw: fake)
        try:
            env.reset()
            step = fake.step
            def truncated(*args, **kwargs):
                result = step(*args, **kwargs)
                fake.last_receipts = fake.last_receipts[:1]
                return result
            with patch.object(fake, 'step', side_effect=truncated):
                with self.assertRaisesRegex(RuntimeError, 'receipt group'):
                    env.step(np.zeros((2, 3), dtype=np.int64))
            env.reset()
            fake.finish_slot_zero = True
            with patch.object(fake, 'reset_slots', return_value=()):
                with self.assertRaisesRegex(RuntimeError, 'partial reset'):
                    env.step(np.zeros((2, 3), dtype=np.int64))
        finally:
            env.close()

    def test_configuration_rejects_invalid_values_before_training(self):
        for key, value in (
            ('rollout_decisions', 0), ('epochs_per_round', 0), ('decision_ticks', 0),
            ('decision_ticks', 1_000_001), ('port', 0), ('seed', -1),
            ('learning_rate', float('nan')), ('gamma', 1.1), ('gae_lambda', -1),
            ('entropy_coef', float('inf')), ('tbptt_steps', 0),
            ('cross_modal_heads', 0), ('cross_modal_entity_layers', 0),
            ('spatial_feature_channels', 0), ('auxiliary_horizon', 0),
            ('world_model_hidden_dim', 0), ('auxiliary_coef', float('nan')),
            ('auxiliary_weights', {'entity_death': float('inf')}),
            ('auxiliary_weights', {'typo': 1}),
        ):
            config = self.config()
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)
        config = self.config()
        config.update(self_play_enabled=True, self_play_random_ratio=-.1,
                      self_play_current_ratio=.4, self_play_historical_ratio=.7)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_model_construction_failure_closes_environment(self):
        env = NativeBatchVecEnv(battles=2, batch_factory=lambda **kw: FakeBatch(2))
        with TemporaryDirectory() as directory:
            import training_runtime
            actual_load = training_runtime.load_object
            def load(reference):
                if reference == self.config()['environment']:
                    return lambda **kwargs: env
                if reference == self.config()['algorithm']:
                    def fail(*args, **kwargs):
                        raise RuntimeError('model construction failed')
                    return fail
                return actual_load(reference)
            with patch('training_runtime.load_object', side_effect=load):
                with self.assertRaisesRegex(RuntimeError, 'model construction failed'):
                    run_sb3_training(self.config(), directory)
        self.assertTrue(env.closed)

    def test_source_changes_during_training_invalidate_the_run(self):
        from training_experiments import source_hashes

        sources = source_hashes()
        self.assertIn('requirements-training.txt', sources)
        self.assertIn('vendor/firstlight/probe/minimal_capture.inc', sources)
        with TemporaryDirectory() as directory:
            with patch('training_experiments.source_hashes', side_effect=[sources, {}]):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    self.train(directory)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'interrupted')
            self.assertEqual(summary['source_sha256'], sources)

    def test_singleton_final_minibatch_keeps_parameters_finite(self):
        with TemporaryDirectory() as directory:
            config = self.config()
            config.update(rounds=1, batch_size=7)
            self.train(directory, config)
            model = NativeActionConditionedWorldModelPPO.load(Path(directory) / 'final_model.zip', device='cpu')
            self.assertTrue(all(torch.isfinite(value).all() for value in model.policy.parameters()))

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
        self.assertEqual(float(losses['opponent_hand']), 0)
        self.assertEqual(float(losses['opponent_next_card']), 0)
        # One real posterior and one empty posterior: invalid rows neither
        # contribute negative-card labels nor dilute the valid loss.
        observation['public_belief'][1, :8] = .5
        observation['public_belief'][1, 8:16] = 1 / 8
        predictions['opponent_hand'].requires_grad_()
        predictions['opponent_next_card'].requires_grad_()
        total, losses = model._auxiliary_losses(samples, predictions)
        self.assertAlmostEqual(float(losses['opponent_hand'].detach()), np.log(2), places=6)
        self.assertAlmostEqual(float(losses['opponent_next_card'].detach()), np.log(8), places=6)
        total.backward()
        self.assertGreater(float(predictions['tower_delta'].grad[1].abs().sum()), 0)
        self.assertEqual(float(predictions['opponent_hand'].grad[0].abs().sum()), 0)
        self.assertEqual(float(predictions['opponent_next_card'].grad[0].abs().sum()), 0)

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

    def test_resume_rejects_missing_or_changed_fixed_baseline(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            config['fixed_evaluation_enabled'] = True
            self.train(root / 'first', config)
            checkpoint = root / 'first/final_model.zip'
            saved = resume_directory(checkpoint)
            baseline = saved / 'initial_policy.zip'
            original = baseline.read_bytes()
            resumed = json.loads((saved / 'config.json').read_text())
            resumed.update(rounds=3, resume_from=str(checkpoint))
            baseline.unlink()
            with self.assertRaisesRegex(ValueError, 'missing.*baseline'):
                self.train(root / 'missing', resumed)
            baseline.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'baseline no longer matches'):
                self.train(root / 'changed', resumed)
            baseline.write_bytes(original)
            self.train(root / 'valid', resumed)
            self.assertEqual((resume_directory(root / 'valid/final_model.zip') / 'initial_policy.zip').read_bytes(), original)

    def test_auxiliary_dataset_rejects_misaligned_and_corrupt_labels_before_inference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            self.train(root / 'train', config)
            model = load_inference_model(NativeActionConditionedWorldModelPPO, root / 'train/final_model.zip', 'cpu')
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                dataset = collect_dataset(config, model, root / 'validation.npz', seed_count=3, decisions=4, battles=2)
            with np.load(dataset, allow_pickle=False) as archive:
                original = {key: archive[key].copy() for key in archive.files}
            def change_schema(data):
                metadata = json.loads(str(data['metadata']))
                metadata['schema'] = 'native_auxiliary_validation_v1'
                data['metadata'] = np.asarray(json.dumps(metadata))
            def reorder(data):
                order = np.arange(len(data['slot']))
                order[:4] = [0, 2, 1, 3]
                for key in data:
                    if key != 'metadata':
                        data[key] = data[key][order]
            mutations = [change_schema, reorder,
                         lambda d: d.update(obs_scalar=d['obs_scalar'][:, :-1]),
                         lambda d: d['obs_scalar'].__setitem__((0, 0), np.nan),
                         lambda d: d['episode_start'].__setitem__(2, True),
                         lambda d: d['seed'].__setitem__(2, 1),
                         lambda d: d['slot'].__setitem__(1, 0),
                         lambda d: d['future_targets'].__setitem__((0, 0), 999),
                         lambda d: d['future_targets'].__setitem__((4, 1), 1),
                         lambda d: d['executed_action'].__setitem__((0, slice(None)), 0)]
            for index, mutate in enumerate(mutations):
                data = {key: value.copy() for key, value in original.items()}
                mutate(data)
                path = root / f'corrupt_{index}.npz'
                np.savez_compressed(path, **data)
                with self.subTest(mutation=index), patch.object(model.policy, 'get_distribution') as infer:
                    with self.assertRaises(ValueError):
                        score_dataset(config, model, path)
                    infer.assert_not_called()
            result = score_dataset(config, model, dataset, batch_size=1)
            self.assertEqual(result['dataset']['samples'], 12)

    def test_replay_records_real_tick_when_terminal_step_is_short(self):
        class ShortTerminalBatch(FakeBatch):
            def step(self, actions=(), ticks=5):
                states = super().step(actions, ticks=3)
                self.states = tuple(replace(state, finalized=True, result=0, crowns=(1, 0)) for state in states)
                return tuple(self.states)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            self.train(root / 'train', config)
            model = load_inference_model(NativeActionConditionedWorldModelPPO, root / 'train/final_model.zip', 'cpu')
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: ShortTerminalBatch(kw['battles'])):
                frame = record_replay(config, model, steps=1)[0]
            self.assertEqual(frame['tick'], 90)
            self.assertEqual(frame['next_tick'], 93)
            self.assertTrue(frame['terminal'])

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
