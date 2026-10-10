import copy
import json
import random
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock

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
from training_runtime import (
    run_sb3_training, resume_directory, load_inference_model, save_resume_state,
    read_resume_state, restore_resume_state, TrainingTelemetryCallback,
)
from native_engine.native_vec_env import NEXT_SCENE_TARGET_SCHEMA, scene_value
from one_click_train import validate_config
from self_play import SelfPlayOpponentPool
from auxiliary_validation import collect_dataset, score_dataset
from native_visualizer import record_replay, write_replay
from fixed_evaluation import evaluate_checkpoint, initial_policy_path, FixedEvaluationCallback


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

    def test_incremental_public_elixir_matches_history_at_all_query_times(self):
        from native_engine.native_vec_env import FIRST_PLAYABLE_TICK, TICK_SECONDS, _regenerated_elixir
        env = NativeBatchVecEnv(battles=1, batch_factory=lambda **kw: FakeBatch(1))
        events = []
        def reference(owner, tick):
            value, previous = 5., 0.
            for event in events:
                if event.owner == owner and event.tick <= tick:
                    seconds = max(0., (event.tick - FIRST_PLAYABLE_TICK) * TICK_SECONDS)
                    value = max(0., min(10., value + _regenerated_elixir(previous, seconds)) - event.cost)
                    previous = seconds
            return min(10., value + _regenerated_elixir(previous, max(0., (tick - FIRST_PLAYABLE_TICK) * TICK_SECONDS)))
        try:
            env.reset()
            slot = env.slots[0]
            ticks = (91, 91, 120, 400, 2490, 2491, 4890, 4891, 6090)
            for index, tick in enumerate(ticks):
                for owner in (0, 1):
                    event = SimpleNamespace(tick=tick, owner=owner, card=CARD_IDS[index % 8], cost=(index % 5) + 1)
                    events.append(event)
                    env._record_plays(slot, SimpleNamespace(plays=(event,)), (), ())
                for query in (90, 91, tick - 1, tick, tick + 10, tick + 1000):
                    for owner in (0, 1):
                        self.assertAlmostEqual(env._public_elixir(slot, owner, query), reference(owner, query), places=12)
            slot.state = replace(slot.state, tick=ticks[-1] + 20)
            before = [env._observation(slot, owner=owner) for owner in (0, 1)]
            class UnreadableLedger(list):
                def __iter__(self):
                    raise AssertionError('observation replayed the complete ledger')
                def __reversed__(self):
                    raise AssertionError('observation replayed the complete ledger')
            slot.ledger = UnreadableLedger(slot.ledger)
            for owner in (0, 1):
                for key, value in env._observation(slot, owner=owner).items():
                    np.testing.assert_array_equal(value, before[owner][key])
            with self.assertRaisesRegex(ValueError, 'nondecreasing'):
                env._record_plays(slot, SimpleNamespace(plays=(events[0],)), (), ())
            env.reset()
            for owner in (0, 1):
                self.assertEqual(env._public_elixir(slot, owner, 90), 5.)
        finally:
            env.close()

    def test_actions_reject_lossy_conversion_before_native_submission(self):
        env = NativeBatchVecEnv(battles=2, batch_factory=lambda **kw: FakeBatch(2))
        bad = [np.full((2, 3), value) for value in (.5, np.nan, np.inf, True, '1', 2.**63, -2.**64)]
        bad.append(np.full((2, 3), 2**64 - 1, dtype=np.uint64))
        try:
            env.reset()
            for actions in bad:
                with self.subTest(dtype=actions.dtype, value=actions[0, 0]), self.assertRaises(ValueError):
                    env.step_async(actions)
                self.assertFalse(env.waiting)
            self.assertFalse(env.batch.step_calls)
            env.step(np.zeros((2, 3), dtype=float))
            for policy in (lambda *args: (1.9, 7, 1), SimpleNamespace(predict=lambda *args, **kw: (np.full((2, 3), .5), None))):
                env.opponent_policy = policy
                count = len(env.batch.step_calls)
                with self.assertRaises(ValueError):
                    env.step(np.zeros((2, 3), dtype=np.int64))
                self.assertEqual(len(env.batch.step_calls), count)
            from stable_baselines3.common.base_class import BaseAlgorithm
            direct = MagicMock(spec=BaseAlgorithm)
            direct.predict.return_value = (np.full((2, 3), .5), ('new memory',))
            env.opponent_policy = direct
            prior_state = env._opponent_state
            prior_starts = env._opponent_episode_starts.copy()
            with self.assertRaises(ValueError):
                env.step(np.zeros((2, 3), dtype=np.int64))
            self.assertIs(env._opponent_state, prior_state)
            np.testing.assert_array_equal(env._opponent_episode_starts, prior_starts)
            env.opponent_policy = None
            _, _, _, infos = env.step(np.full((2, 3), -1, dtype=np.int64))
            self.assertTrue(all(info['reward_components']['invalid'] < 0 for info in infos))
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
            ('rounds', 2.9), ('battles', 2.5), ('seed', True), ('port', 26789.5),
            ('evaluation_battles', float('inf')),
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

    def test_training_rejects_existing_artifacts_before_mutating_them(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('metrics.jsonl', 'source_sha256.json', 'final_model.zip'):
                output = root / name.replace('.', '_')
                output.mkdir()
                artifact = output / name
                artifact.write_bytes(b'previous run')
                with self.subTest(artifact=name), patch('training_runtime.load_object') as load:
                    with self.assertRaisesRegex(ValueError, 'fresh run directory'):
                        run_sb3_training(self.config(), output)
                    self.assertEqual(artifact.read_bytes(), b'previous run')
                    self.assertEqual(list(output.iterdir()), [artifact])
                    load.assert_not_called()
            fresh = root / 'prepared'
            fresh.mkdir()
            (fresh / 'config.json').write_text(json.dumps(self.config()))
            self.train(fresh)

    def test_offline_evaluation_uses_original_baseline_and_rejects_missing_one(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            config['fixed_evaluation_enabled'] = True
            self.train(root / 'train', config)
            checkpoint = root / 'train/final_model.zip'
            sidecar = resume_directory(checkpoint)
            expected = json.loads((sidecar / 'runtime.json').read_text())['initial_policy_sha256']
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                evaluate_checkpoint(config, checkpoint, root / 'offline')
            schedule = json.loads((root / 'offline/evaluation/schedule.json').read_text())
            self.assertEqual(schedule['initial_policy']['sha256'], expected)
            self.assertNotEqual(expected, json.loads((sidecar / 'runtime.json').read_text())['model_sha256'])
            (sidecar / 'initial_policy.zip').unlink()
            original_baseline = root / 'train/evaluation/baselines/initial_policy.zip'
            original = original_baseline.read_bytes()
            original_baseline.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                initial_policy_path(config, checkpoint)
            original_baseline.unlink()
            with self.assertRaisesRegex(ValueError, 'original initial-policy baseline'):
                evaluate_checkpoint(config, checkpoint, root / 'missing')
            self.assertFalse((root / 'missing').exists())
            explicit = root / 'explicit.zip'
            explicit.write_bytes(original)
            config['evaluation_initial_policy_path'] = str(explicit)
            self.assertEqual(initial_policy_path(config, checkpoint), explicit)

    def test_direct_evaluation_rejects_invalid_schedule_counts(self):
        with TemporaryDirectory() as directory:
            kwargs = dict(run_dir=directory, env_class=NativeBatchVecEnv,
                          algorithm_class=NativeActionConditionedWorldModelPPO,
                          env_kwargs={}, rollout_decisions=8, every_rounds=1, seed_base=900001,
                          seed_count=1, battles=1, max_decisions=4, opponents=['random'],
                          initial_policy_path=Path(directory) / 'unused.zip', device='cpu')
            for key, value in (('rollout_decisions', 0), ('every_rounds', 0), ('seed_base', -1),
                               ('seed_count', 0), ('battles', 0), ('battles', 17), ('max_decisions', 0)):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    FixedEvaluationCallback(**dict(kwargs, **{key: value}))

    def test_offline_evaluation_publishes_completion_after_required_artifacts(self):
        import fixed_evaluation
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(self.config(), evaluation_opponents=['random'])
            self.train(root / 'train', config)
            checkpoint = root / 'train/final_model.zip'
            output = root / 'offline'
            write = fixed_evaluation._atomic_json
            def check(path, payload):
                if Path(path).name == 'status.json' and payload.get('status') == 'completed':
                    self.assertTrue((output / 'evaluation/checkpoint.json').is_file())
                    self.assertTrue((output / 'charts/fixed_evaluation.svg').is_file())
                return write(path, payload)
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                with patch('fixed_evaluation._atomic_json', side_effect=check):
                    result = evaluate_checkpoint(config, checkpoint, output)
            self.assertTrue(result['latest']['valid'])

    def test_failed_evaluation_finalization_remains_retryable(self):
        import fixed_evaluation
        from training_experiments import _prepare_held_out
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(self.config(), evaluation_opponents=['random'])
            self.train(root / 'train', config)
            checkpoint = root / 'train/final_model.zip'
            for fault in ('chart', 'identity'):
                output = root / fault / 'post_training'
                write = fixed_evaluation._atomic_json
                def fail_identity(path, payload):
                    if Path(path).name == 'checkpoint.json':
                        raise OSError('identity write failed')
                    return write(path, payload)
                failure = (patch('fixed_evaluation._svg_chart', side_effect=OSError('chart write failed'))
                           if fault == 'chart' else patch('fixed_evaluation._atomic_json', side_effect=fail_identity))
                with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                    with failure, self.assertRaises(OSError):
                        evaluate_checkpoint(config, checkpoint, output)
                status = json.loads((output / 'evaluation/status.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertTrue(_prepare_held_out(root / fault))
                self.assertTrue((output / 'evaluation_failed_001').is_dir())

    def test_offline_evaluation_rejects_checkpoint_replaced_during_inference(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(self.config(), evaluation_opponents=['random'])
            self.train(root / 'train', config)
            checkpoint = root / 'train/final_model.zip'
            original = checkpoint.read_bytes()
            run_wave = FixedEvaluationCallback._run_wave
            def replace_after_wave(callback, *args, **kwargs):
                result = run_wave(callback, *args, **kwargs)
                checkpoint.write_bytes(original + b'changed during inference')
                return result
            output = root / 'offline'
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                with patch.object(FixedEvaluationCallback, '_run_wave', replace_after_wave):
                    with self.assertRaisesRegex(ValueError, 'hash changed'):
                        evaluate_checkpoint(config, checkpoint, output)
            self.assertEqual(json.loads((output / 'evaluation/status.json').read_text())['status'], 'failed')

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

    def test_v4_singleton_minibatches_keep_parameters_finite(self):
        from training_experiments import variant_config
        from native_engine.native_recurrent_model import NativeRecurrentPPO
        for batch_size in (7, 1):
            with self.subTest(batch_size=batch_size), TemporaryDirectory() as directory:
                config = variant_config(self.config(), 'v4')
                config.update(rounds=1, batch_size=batch_size)
                self.train(directory, config)
                model = NativeRecurrentPPO.load(Path(directory) / 'final_model.zip', device='cpu')
                self.assertTrue(all(torch.isfinite(value).all() for value in model.policy.parameters()))

    def test_shared_ppo_update_matches_upstream_for_regular_v4_batches(self):
        from sb3_contrib import RecurrentPPO
        from training_experiments import variant_config
        from native_engine.native_recurrent_model import NativeRecurrentPPO
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = variant_config(self.config(), 'v4')
            config.update(rounds=1, epochs_per_round=2)
            config['algorithm_kwargs'].update(clip_range_vf=.2, target_kl=.03)
            self.train(root / 'shared', config)
            with patch.object(NativeRecurrentPPO, 'train', RecurrentPPO.train):
                self.train(root / 'upstream', config)
            shared = NativeRecurrentPPO.load(root / 'shared/final_model.zip', device='cpu')
            upstream = NativeRecurrentPPO.load(root / 'upstream/final_model.zip', device='cpu')
            self.assertEqual(shared._n_updates, upstream._n_updates)
            for name, value in shared.policy.state_dict().items():
                torch.testing.assert_close(value, upstream.policy.state_dict()[name], rtol=0, atol=0)

    def test_no_aux_skips_predictions_and_matches_ppo_updates_exactly(self):
        from sb3_contrib import RecurrentPPO
        from training_experiments import variant_config
        from native_engine.native_recurrent_model import (
            NativeRecurrentPPO, NativeActionConditionedWorldModelRecurrentPolicy,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = variant_config(self.config(), 'no_aux')
            config.update(rounds=1, epochs_per_round=2)
            with patch.object(NativeActionConditionedWorldModelRecurrentPolicy, 'auxiliary_predictions',
                              side_effect=AssertionError('disabled auxiliary heads must not run')):
                self.train(root / 'skipped', config)
            with patch.object(NativeRecurrentPPO, 'train', RecurrentPPO.train):
                self.train(root / 'ppo', config)
            skipped = NativeActionConditionedWorldModelPPO.load(root / 'skipped/final_model.zip', device='cpu')
            ppo = NativeActionConditionedWorldModelPPO.load(root / 'ppo/final_model.zip', device='cpu')
            self.assertEqual(skipped.n_epochs, 2)
            self.assertEqual(skipped._n_updates, ppo._n_updates)
            for name, value in skipped.policy.state_dict().items():
                torch.testing.assert_close(value, ppo.policy.state_dict()[name], rtol=0, atol=0)
            for parameter in skipped.policy.world_model.parameters():
                self.assertNotIn(parameter, skipped.policy.optimizer.state)

    def test_all_zero_auxiliary_weights_skip_predictions(self):
        from native_engine.native_recurrent_model import NativeActionConditionedWorldModelRecurrentPolicy
        with TemporaryDirectory() as directory:
            config = self.config()
            config.update(rounds=1, auxiliary_weights={
                name: 0. for name in NativeActionConditionedWorldModelPPO.DEFAULT_AUXILIARY_WEIGHTS
            })
            with patch.object(NativeActionConditionedWorldModelRecurrentPolicy, 'auxiliary_predictions',
                              side_effect=AssertionError('zero-weight auxiliary heads must not run')):
                self.train(directory, config)
            model = NativeActionConditionedWorldModelPPO.load(Path(directory) / 'final_model.zip', device='cpu')
            self.assertTrue(all(torch.isfinite(value).all() for value in model.policy.parameters()))

    def test_zero_weight_tasks_ignore_nonfinite_or_missing_predictions(self):
        observation = {key: torch.zeros(2, *space.shape)
                       for key, space in native_observation_space().spaces.items()}
        observation['entity_mask'][:, 0] = 1
        observation['public_belief'][:, :8] = .5
        observation['public_belief'][:, 8:16] = 1 / 8
        entities = torch.zeros(2, MAX_ENTITIES, 4)
        entities[:, 0, 3] = 1
        transition = torch.zeros(2, 8)
        transition[:, 2] = transition[:, 7] = 1
        samples = SimpleNamespace(observations=observation, entity_targets=entities,
            transition_targets=transition, future_targets=torch.ones(2, 2), mask=torch.ones(2),
            lstm_states=RNNStates((torch.zeros(1, 1, 8),)*2, (torch.zeros(1, 1, 8),)*2))
        shapes = {'tower_delta': (2, 2), 'damage_source': (2, 5), 'opponent_elixir': (2, 5),
                  'opponent_hand': (2, 8), 'opponent_next_card': (2, 8), 'future_value': (2,),
                  'entity_death': (2, MAX_ENTITIES), 'entity_position_delta': (2, MAX_ENTITIES, 2)}
        model = NativeActionConditionedWorldModelPPO.__new__(NativeActionConditionedWorldModelPPO)
        for task in model.DEFAULT_AUXILIARY_WEIGHTS:
            with self.subTest(task=task):
                output = 'entity_position_delta' if task == 'entity_position' else task
                model.auxiliary_weights = dict(model.DEFAULT_AUXILIARY_WEIGHTS, **{task: 0.})
                predictions = {name: torch.full(shape, float('nan') if name == output else 0.,
                                               requires_grad=True) for name, shape in shapes.items()}
                total, losses = model._auxiliary_losses(samples, predictions)
                self.assertNotIn(task, losses)
                self.assertTrue(torch.isfinite(total))
                total.backward()
                self.assertIsNone(predictions[output].grad)
                self.assertTrue(all(torch.isfinite(value.grad).all()
                                    for name, value in predictions.items() if name != output))
                without, _ = model._auxiliary_losses(samples, {
                    name: value for name, value in predictions.items() if name != output
                })
                torch.testing.assert_close(total, without)

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

    def test_future_targets_keep_last_observed_nonterminal_successor(self):
        env = NativeBatchVecEnv(battles=2, batch_factory=lambda **kw: TerminalBatch(2))
        try:
            for horizon in (0, -1):
                with self.assertRaisesRegex(ValueError, 'horizon'):
                    EntityTargetRecurrentDictRolloutBuffer(4, native_observation_space(), env.action_space,
                        (4, 1, 2, 8), n_envs=2, device='cpu', tbptt_steps=2, auxiliary_horizon=horizon)
            buffer = EntityTargetRecurrentDictRolloutBuffer(4, native_observation_space(), env.action_space,
                (4, 1, 2, 8), n_envs=2, device='cpu', tbptt_steps=2, auxiliary_horizon=2)
            states = RNNStates((torch.zeros(1, 2, 8),)*2, (torch.zeros(1, 2, 8),)*2)
            for step in range(4):
                observation = {key: np.zeros((2, *space.shape), dtype=space.dtype)
                               for key, space in native_observation_space().spaces.items()}
                observation['scalar'][:, 0] = (step + 1) / 10
                observation['scalar'][:, 8] = (0, 1)
                infos = [{'entity_target_schema': ENTITY_TARGET_SCHEMA, 'entity_targets': np.zeros((MAX_ENTITIES, 4)),
                          'transition_target_schema': TRANSITION_TARGET_SCHEMA, 'transition_targets': np.zeros(8),
                          'next_scene_target_schema': NEXT_SCENE_TARGET_SCHEMA, 'next_scene_value': (step + 1) / 10}
                         for _ in range(2)]
                buffer.set_entity_targets(infos)
                buffer.add(observation, np.zeros((2, 3)), np.zeros(2), np.asarray([step in (0, 3), step == 0]),
                           torch.zeros(2), torch.zeros(2), lstm_states=states)
            for sample in buffer.get(batch_size=8):
                selected = sample.mask > 0
                for scalar, target in zip(sample.observations['scalar'][selected], sample.future_targets[selected]):
                    step, slot = round(float(scalar[0]) * 10) - 1, int(scalar[8])
                    valid = step < 3 and not (slot == 0 and step == 2)
                    self.assertEqual(bool(target[1]), valid)
                    if valid:
                        self.assertAlmostEqual(float(target[0]), (step + 2) / 10, places=6)
            observation = env.reset()
            for _ in range(3):
                observation, _, dones, infos = env.step(np.zeros((2, 3), dtype=np.int64))
                for slot, info in enumerate(infos):
                    scalar = info['terminal_observation']['scalar'] if dones[slot] else observation['scalar'][slot]
                    self.assertEqual(info['next_scene_target_schema'], NEXT_SCENE_TARGET_SCHEMA)
                    self.assertAlmostEqual(info['next_scene_value'], float(scene_value(scalar)), places=7)
            self.assertTrue(dones.all())
            self.assertNotEqual(infos[0]['next_scene_value'], float(scene_value(observation['scalar'][0])))
        finally:
            env.close()

    def test_buffer_rejects_bad_labels_without_reusing_pending_values(self):
        env = NativeBatchVecEnv(battles=1)
        try:
            buffer = EntityTargetRecurrentDictRolloutBuffer(4, native_observation_space(), env.action_space,
                (4, 1, 1, 8), n_envs=1, device='cpu', tbptt_steps=2, auxiliary_horizon=1)
            good = {'entity_target_schema': ENTITY_TARGET_SCHEMA, 'entity_targets': np.zeros((MAX_ENTITIES, 4)),
                    'transition_target_schema': TRANSITION_TARGET_SCHEMA, 'transition_targets': np.zeros(8),
                    'next_scene_target_schema': NEXT_SCENE_TARGET_SCHEMA, 'next_scene_value': .2}
            bad_infos = []
            bad = copy.deepcopy(good); bad['entity_targets'][0, 0] = -.1; bad_infos.append(bad)
            bad = copy.deepcopy(good); bad['entity_targets'][0, 1] = 2; bad_infos.append(bad)
            bad = copy.deepcopy(good); bad['transition_targets'][7] = .5; bad_infos.append(bad)
            bad = copy.deepcopy(good); bad['next_scene_value'] = float('nan'); bad_infos.append(bad)
            bad = copy.deepcopy(good); bad['next_scene_target_schema'] = 'wrong'; bad_infos.append(bad)
            for bad in bad_infos:
                buffer.set_entity_targets([good])
                with self.assertRaises(ValueError):
                    buffer.set_entity_targets([bad])
                with self.assertRaisesRegex(RuntimeError, 'missing native entity targets'):
                    buffer.add()
            terminal = dict(good)
            terminal.pop('next_scene_value')
            terminal['terminal_observation'] = {'scalar': np.full(12, np.nan)}
            with self.assertRaisesRegex(ValueError, 'terminal scalar'):
                buffer.set_entity_targets([terminal])
        finally:
            env.close()

    def test_interrupt_at_last_sampling_step_is_not_a_completed_update(self):
        for stop in ('interrupt', 'callback_false'):
            with self.subTest(stop=stop), TemporaryDirectory() as directory:
                root = Path(directory)
                original = TrainingTelemetryCallback._on_step
                def interrupt(callback):
                    result = original(callback)
                    if callback.num_timesteps == 16:
                        if stop == 'interrupt':
                            raise KeyboardInterrupt('before final rollout sample is added')
                        return False
                    return result
                with patch.object(TrainingTelemetryCallback, '_on_step', interrupt):
                    with self.assertRaises(KeyboardInterrupt if stop == 'interrupt' else ValueError):
                        self.train(root)
                interrupted = root / 'interrupted_model.zip'
                model = NativeActionConditionedWorldModelPPO.load(interrupted, device='cpu')
                self.assertEqual(model.num_timesteps, 16)
                self.assertEqual(model._n_updates, 1)
                self.assertFalse(resume_directory(interrupted).exists())
                self.assertFalse((root / 'final_model.zip').exists())
                completed = root / 'checkpoints/model_8_steps.zip'
                self.assertEqual(read_resume_state(completed)['timesteps'], 8)

    def test_cancelled_rollouts_do_not_publish_unupdated_round_artifacts(self):
        for stop_at in (2, 16):
            for separate in (False, True):
                with self.subTest(stop_at=stop_at, separate=separate), TemporaryDirectory() as directory:
                    root = Path(directory)
                    config = dict(self.config(), self_play_enabled=True, self_play_current_ratio=0,
                                  self_play_historical_ratio=1, fixed_evaluation_enabled=True, port=101)
                    if separate:
                        config['evaluation_port'] = 102
                    original = TrainingTelemetryCallback._on_step
                    def stop(callback):
                        original(callback)
                        return callback.num_timesteps != stop_at
                    with patch.object(TrainingTelemetryCallback, '_on_step', stop):
                        with self.assertRaisesRegex(ValueError, 'before completing'):
                            self.train(root, config)
                    completed = 0 if stop_at == 2 else 1
                    summary = json.loads((root / 'summary.json').read_text())
                    self.assertEqual(summary['completed_rounds'], completed)
                    snapshots = list((root / 'self_play/snapshots').glob('*.zip'))
                    self.assertEqual({path.stem for path in snapshots},
                                     {f'snapshot_round_{i:06d}' for i in range(completed + 1)})
                    for path in snapshots:
                        model = NativeActionConditionedWorldModelPPO.load(path, device='cpu')
                        self.assertEqual(model._n_updates, int(path.stem.rsplit('_', 1)[1]))
                    self.assertFalse((root / f'evaluation/checkpoints/round_{completed + 1:06d}.zip').exists())
                    metrics = root / 'metrics.jsonl'
                    records = [json.loads(line) for line in metrics.read_text().splitlines()] if metrics.exists() else []
                    self.assertTrue(all(row['round'] <= completed for row in records))
                    matches = root / 'evaluation/matches.jsonl'
                    if matches.exists():
                        self.assertTrue(all(json.loads(line)['round'] <= completed
                                            for line in matches.read_text().splitlines()))

    def test_resume_manifest_is_last_and_checks_rng_config_and_league(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'partial.zip'
            checkpoint.write_bytes(b'weights')
            model = SimpleNamespace(num_timesteps=8, get_env=lambda: SimpleNamespace(get_attr=lambda _: [99]))
            pool = SimpleNamespace(save_state=MagicMock(side_effect=RuntimeError('league copy failed')))
            with self.assertRaisesRegex(RuntimeError, 'league copy failed'):
                save_resume_state(checkpoint, model, dict(self.config(), self_play_enabled=True), pool)
            self.assertFalse((resume_directory(checkpoint) / 'runtime.json').exists())
            self.train(root / 'train', dict(self.config(), self_play_enabled=True,
                                           self_play_current_ratio=0, self_play_historical_ratio=1))
            checkpoint = root / 'train/final_model.zip'
            directory = resume_directory(checkpoint)
            self.assertEqual(read_resume_state(checkpoint)['schema'], 'native_training_resume_v2')
            for name in ('config.json', 'rng.pt', 'league/state.json'):
                path = directory / name
                original = path.read_bytes()
                path.write_bytes(original + b' ')
                env, pool = MagicMock(), MagicMock()
                with self.subTest(file=name), self.assertRaisesRegex(ValueError, 'missing or changed'):
                    restore_resume_state(checkpoint, env, pool)
                env.seed.assert_not_called()
                pool.restore_state.assert_not_called()
                path.write_bytes(original)
            self.assertEqual(read_resume_state(checkpoint)['timesteps'], 16)

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

    def test_auxiliary_validation_rejects_invalid_predictions_and_collection_schemas(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config()
            self.train(root / 'train', config)
            model = load_inference_model(NativeActionConditionedWorldModelPPO, root / 'train/final_model.zip', 'cpu')
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: TerminalBatch(kw['battles'])):
                dataset = collect_dataset(config, model, root / 'validation.npz', seed_count=2, decisions=4, battles=2)
            predict = model.policy.auxiliary_predictions
            for kind in ('nan', 'inf', 'shape', 'missing'):
                def corrupt(*args, **kwargs):
                    values = predict(*args, **kwargs)
                    if kind in ('nan', 'inf'):
                        values['tower_delta'] = torch.full_like(values['tower_delta'], float(kind))
                    elif kind == 'shape':
                        values['future_value'] = values['future_value'].unsqueeze(-1)
                    else:
                        values.pop('entity_death')
                    return values
                with self.subTest(kind=kind), patch.object(model.policy, 'auxiliary_predictions', side_effect=corrupt):
                    with self.assertRaisesRegex(ValueError, 'invalid auxiliary prediction'):
                        score_dataset(config, model, dataset)
            with np.load(dataset, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            data['obs_public_belief'][:, 8:16] *= .5
            scaled = root / 'scaled_belief.npz'
            np.savez_compressed(scaled, **data)
            def uniform_next_card(*args, **kwargs):
                values = predict(*args, **kwargs)
                values['opponent_next_card'] = torch.zeros_like(values['opponent_next_card'])
                return values
            with patch.object(model.policy, 'auxiliary_predictions', side_effect=uniform_next_card):
                metrics = score_dataset(config, model, scaled)['metrics']
                self.assertAlmostEqual(metrics['public_next_card_cross_entropy'], np.log(8), places=6)
            class WrongSchemaEnv(NativeBatchVecEnv):
                def step_wait(self):
                    observation, rewards, dones, infos = super().step_wait()
                    infos[0]['entity_target_schema'] = 'wrong'
                    return observation, rewards, dones, infos
            env = WrongSchemaEnv(battles=2, batch_factory=lambda **kw: TerminalBatch(2))
            with patch('auxiliary_validation.load_object', return_value=lambda **kw: env):
                with self.assertRaisesRegex(ValueError, 'incompatible entity_target_schema'):
                    collect_dataset(config, model, root / 'bad_collection.npz', seed_count=2, decisions=4, battles=2)
            self.assertTrue(env.closed)
            self.assertFalse((root / 'bad_collection.npz').exists())

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
