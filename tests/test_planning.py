import copy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_native_vec_env import FakeBatch
from native_engine.native_vec_env import NativeBatchVecEnv, _regenerated_elixir
from native_engine.planning_rules import (ACTION_COUNT, TILES, canonical_actions, public_actions,
                                          canonical_mask, executed_identity, action_features, candidate_set)
from native_engine.planning_rules import proposal_log_probabilities
from native_engine.planning_model import PlanningModel, PlanningState
from native_engine.native_search import SampledSearch, SearchSettings
from native_engine.planning_policy import NativePlanningAgent
from native_engine.planning_replay import PlanningStep, SequenceReplay
from native_engine.planning_training import PlanningLearner, run_planning_training
from native_engine.planning_validation import validate_replay, collect_replay
from one_click_train import load_profiles, prepare_config, validate_config


def environment(**kwargs):
    return NativeBatchVecEnv(planning_labels=True, batch_factory=lambda **kw: FakeBatch(kw['battles']), **kwargs)


def agent():
    return NativePlanningAgent(dict(hidden_dim=32, spatial_dim=16, chance_codes=4, entity_layers=1),
                               dict(budget=8, depth=3, candidates=3, internal_candidates=3, outcome_samples=1))


def collect(model, steps=5, finish=False):
    env = environment(battles=2)
    observation = env.reset()
    state = None
    replay = SequenceReplay(128)
    for tick in range(steps):
        actions, state = model.predict(observation, state, [tick == 0] * 2)
        records = copy.deepcopy(model.last_search)
        identities = canonical_actions(observation['hand'], actions)
        if finish and tick == 1:
            env.batch.finish_slot_zero = True
        after, rewards, dones, infos = env.step(actions)
        for row, info in enumerate(infos):
            label = info['planning']
            replay.append(row, PlanningStep({key: value[row].copy() for key, value in observation.items()},
                          label['next_observation'], int(identities[row]), label['opponent_action'], label['accepted'],
                          float(rewards[row]), bool(dones[row]), label['ticks'], records[row]))
        observation = after
    env.close()
    return replay, observation


class TacticalModel:
    """A costly investment pays off one step later; greedy play misses it."""
    def evaluate(self, state):
        count = len(state.memory)
        own = torch.full((count, ACTION_COUNT), -100.)
        own[:, :2] = np.log(.5)
        opponent = torch.full_like(own, -100.)
        opponent[:, 0] = 0
        return dict(policy=own, opponent=opponent, value=torch.zeros(count))

    def predicted_mask(self, state):
        masks = torch.zeros((len(state.memory), ACTION_COUNT), dtype=torch.bool)
        masks[:, :2] = True
        return masks

    def chance_distribution(self, state, own, enemy):
        return torch.zeros((len(own), 2))

    def dynamics(self, state, own, enemy, codes):
        own = torch.tensor(own)
        step, invested = state.memory.T
        reward = torch.where(step == 0, -(own == 1).float(), 5 * invested)
        memory = torch.stack((step + 1, torch.where(step == 0, (own == 1).float(), invested)), -1)
        return replace(state, memory=memory), dict(reward=reward, terminal=torch.where(step + 1 >= 2, 20., -20.))


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_action_identity_roundtrip_and_mirrored_native_coordinates(self):
        hand = np.asarray([[1, 2, 3, 4, 5], [8, 7, 6, 5, 4]])
        actions = np.asarray([[3, 7, 4], [1, 31, 17]])
        identities = canonical_actions(hand, actions)
        np.testing.assert_equal(public_actions(hand, identities), actions)
        features = action_features(identities).numpy()
        self.assertEqual(executed_identity(features[0]), identities[0])
        self.assertEqual(executed_identity(features[1], mirror=True), 1 + 7 * TILES)
        self.assertEqual(executed_identity(action_features([0]).numpy()[0]), 0)
        with self.assertRaises(ValueError):
            canonical_actions(hand, [[0, 1, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            canonical_actions(hand, [[1, .5, 0], [0, 0, 0]])

    def test_candidates_preserve_card_diversity_and_legality(self):
        legal = np.zeros(ACTION_COUNT, bool); legal[0] = True
        for card in range(4):
            legal[1 + card * TILES:1 + (card + 1) * TILES] = True
        probabilities = np.ones(ACTION_COUNT); probabilities[1:577] = 100
        selected = candidate_set(probabilities, legal, 8, np.random.default_rng(1))
        self.assertIn(0, selected)
        self.assertTrue(legal[selected].all())
        self.assertTrue({0, 1, 2, 3} <= set((selected[selected > 0] - 1) // TILES))

    def test_restricted_prior_preserves_card_mass_instead_of_amplifying_wait(self):
        logits = torch.full((ACTION_COUNT,), -100.)
        logits[0] = np.log(.2)
        logits[1:577] = np.log(.8 / 576)
        candidates = torch.tensor([0, 1, 12, 400])
        probabilities = proposal_log_probabilities(logits, candidates).exp()
        self.assertAlmostEqual(float(probabilities[0]), .2, places=6)
        self.assertAlmostEqual(float(probabilities[1:].sum()), .8, places=6)

    def test_search_reaches_future_and_avoids_greedy_decision(self):
        state = PlanningState(torch.zeros(1, 2), *(torch.zeros(1, 1) for _ in range(5)))
        model = TacticalModel(); mask = model.predicted_mask(state)
        shallow = SampledSearch(model, SearchSettings(budget=128, depth=1, candidates=2, internal_candidates=2,
                                                     outcome_samples=1, discount=1.))
        deep = SampledSearch(model, SearchSettings(budget=128, depth=2, candidates=2, internal_candidates=2,
                                                  outcome_samples=1, discount=1.))
        first, shallow_record = shallow.run(state, mask, [np.random.default_rng(1)], True)
        second, deep_record = deep.run(state, mask, [np.random.default_rng(1)], True)
        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], 1)
        self.assertEqual(deep_record[0]['depth'], 2)
        self.assertEqual(deep_record[0]['transitions'], 128)
        self.assertLessEqual(int(deep_record[0]['visits'].sum()), 128)

    def test_terminal_stops_search_and_zero_budget_uses_prior(self):
        class Terminal(TacticalModel):
            def dynamics(self, *args):
                state, result = super().dynamics(*args)
                result['terminal'][:] = 20.
                return state, result
        state = PlanningState(torch.zeros(1, 2), *(torch.zeros(1, 1) for _ in range(5)))
        model = Terminal(); mask = model.predicted_mask(state)
        _, records = SampledSearch(model, SearchSettings(budget=12, depth=8, candidates=2,
                                                       internal_candidates=2, outcome_samples=1)).run(
                                                           state, mask, [np.random.default_rng(2)])
        self.assertEqual(records[0]['depth'], 1)
        _, records = SampledSearch(model, SearchSettings(budget=0)).run(state, mask, [np.random.default_rng(2)])
        self.assertEqual(records[0]['transitions'], 0)
        self.assertAlmostEqual(float(records[0]['probabilities'].sum()), 1.)

    def test_outcome_codes_change_future_not_current_opponent_distribution(self):
        model = agent().policy.eval()
        env = environment(battles=1); obs = env.reset()
        with torch.no_grad():
            root = model.encode(obs)
            enemy_before = model.evaluate(root)['opponent'].clone()
            first, _ = model.dynamics(root, [1], [0], [0])
            second, _ = model.dynamics(root, [1], [0], [1])
            self.assertFalse(torch.equal(first.memory, second.memory))
            torch.testing.assert_close(model.evaluate(root)['opponent'], enemy_before)
        env.close()

    def test_known_card_and_elixir_rules_include_phase_boundary(self):
        model = agent().policy.eval()
        env = environment(battles=1); obs = env.reset()
        obs['scalar'][:, 4] = 119.8 / 300.
        with torch.no_grad():
            model.success.weight.zero_(); model.success.bias.fill_(20)
            root = model.encode(obs)
            successor, _ = model.dynamics(root, [1], [0], [0])
        self.assertEqual(successor.hand[0, 0], 0)
        self.assertEqual(successor.hand[0, 4], 1)
        expected = .7 + _regenerated_elixir(119.8, 120.3) / 10
        self.assertAlmostEqual(float(successor.scalar[0, 0]), expected, places=5)
        self.assertEqual(successor.scalar[0, 6], 1)
        env.close()

    def test_predict_checkpoint_batching_and_episode_memory(self):
        model = agent(); env = environment(battles=2); obs = env.reset()
        actions, state = model.predict(obs, deterministic=True)
        np.testing.assert_equal(canonical_mask(obs)[np.arange(2), canonical_actions(obs['hand'], actions)], True)
        again, reset_state = model.predict(obs, state=state, episode_start=[True, True], deterministic=True)
        np.testing.assert_array_equal(actions, again)
        np.testing.assert_array_equal(state[0], reset_state[0])
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.zip'; model.save(path)
            loaded = NativePlanningAgent.load(path)
            np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], actions)
            for row in range(2):
                single = {key: value[row:row + 1] for key, value in obs.items()}
                np.testing.assert_array_equal(loaded.predict(single, deterministic=True)[0][0], actions[row])
        env.close()

    def test_terminal_replay_labels_do_not_use_autoreset(self):
        replay, _ = collect(agent(), finish=True)
        completed = [episode for episode in replay.episodes if episode[-1].done]
        self.assertTrue(completed)
        terminal = completed[0][-1]
        self.assertNotEqual(float(terminal.next_observation['scalar'][2]), float(terminal.observation['scalar'][2]))
        for prefix, sequence in replay.sample(np.random.default_rng(2), 20, 8, 4):
            self.assertFalse(any(step.done for step in sequence[:-1]))

    def test_multistep_gradients_reanalysis_and_finite_optimizer(self):
        model = agent(); replay, _ = collect(model)
        learner = PlanningLearner(model, n_steps=2)
        samples = [(episode[:1], episode[1:]) for episode in replay.episodes]
        learner.reanalyze(samples[:1])
        before = model.policy.transition.weight_ih.detach().clone()
        metrics = learner.train_batch(samples, 3)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertFalse(torch.equal(before, model.policy.transition.weight_ih))
        for module in ('encoder', 'filter', 'transition', 'prior', 'posterior', 'own_card', 'enemy_card', 'value',
                       'board_transition', 'spatial_decoder', 'success', 'next_card_head'):
            gradients = [p.grad for p in getattr(model.policy, module).parameters() if p.grad is not None]
            self.assertTrue(gradients, module)
            self.assertTrue(any(bool(g.abs().sum() > 0) for g in gradients), module)

    def test_replay_capacity_and_contiguity(self):
        replay, _ = collect(agent())
        step = replay.episodes[0][0]
        small = SequenceReplay(2)
        for _ in range(5):
            small.append(0, replace(step, observation=step.observation, next_observation=step.observation, done=False))
        self.assertEqual(len(small), 2)
        bad = copy.deepcopy(step); bad.observation['scalar'][0] += .123
        with self.assertRaisesRegex(ValueError, 'contiguous'):
            small.append(0, bad)
        restored = SequenceReplay.from_state_dict(small.state_dict())
        self.assertFalse(restored.active)

    def test_frozen_collection_and_multistep_validation(self):
        model = agent()
        profile = next(p for p in load_profiles() if p['profile_id'] == 'native_planning')
        config = prepare_config(profile, defaults_only=True)
        before = {key: value.clone() for key, value in model.policy.state_dict().items()}
        with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: FakeBatch(kw['battles'])):
            replay = collect_replay(model, config, decisions=12, battles=2)
        result = validate_replay(model, replay, horizons=(1, 3, 8), samples=8, outcome_samples=2)
        self.assertGreater(result['horizons']['1']['samples'], 0)
        self.assertGreater(result['horizons']['3']['samples'], 0)
        self.assertEqual(result['horizons']['8']['samples'], 0)
        self.assertTrue(all(step.seed >= 2900001 for episode in replay.episodes for step in episode))
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, before[key])
        with self.assertRaisesRegex(ValueError, 'training seed'):
            collect_replay(model, config, decisions=4, battles=2, seed=0)

    def test_historical_planner_generator_survives_league_restore(self):
        from self_play import SelfPlayOpponentPool
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = agent(); model.save(root / 'snapshot.zip')
            pool = SelfPlayOpponentPool(root / 'first', NativePlanningAgent, 'cpu', 1,
                                        random_ratio=0, current_ratio=0, historical_ratio=1)
            pool.register_snapshot('old', root / 'snapshot.zip', 0)
            historical = pool._load_snapshot('old')
            historical.rng.random(7)
            pool.save_state(root / 'state')
            expected = historical.rng.random(5)
            restored = SelfPlayOpponentPool(root / 'second', NativePlanningAgent, 'cpu', 1,
                                            random_ratio=0, current_ratio=0, historical_ratio=1)
            restored.restore_state(root / 'state')
            np.testing.assert_equal(restored._load_snapshot('old').rng.random(5), expected)

    def test_profile_training_snapshot_evaluation_and_resume(self):
        profile = next(p for p in load_profiles() if p['profile_id'] == 'native_planning')
        config = prepare_config(profile, defaults_only=True)
        config.update(rounds=2, battles=2, rollout_decisions=8, batch_size=4, device='cpu',
                      planning_hidden_dim=32, planning_spatial_dim=16, planning_chance_codes=4,
                      cross_modal_entity_layers=1, planning_budget=4, planning_depth=3,
                      planning_candidates=3, planning_internal_candidates=3, planning_outcome_samples=1,
                      planning_batch_size=2, planning_updates=1, planning_unroll=2, planning_n_steps=2,
                      planning_burn_in=2, planning_reanalyze_every=1,
                      checkpoint_every_rounds=1, self_play_snapshot_every_rounds=1,
                      evaluation_seed_count=1, evaluation_battles=1, evaluation_max_decisions=4)
        validate_config(config)
        class FinishingBatch(FakeBatch):
            def step(self, *args, **kwargs):
                self.finish_slot_zero = True
                return super().step(*args, **kwargs)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            # Actual native VecEnv, league, checkpoint and fixed-evaluation code;
            # only the native transport is substituted.
            with patch('native_engine.native_vec_env.Batch', side_effect=lambda **kw: FinishingBatch(kw['battles'])):
                result = run_planning_training(config, root / 'first')
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(result['timesteps'], 16)
                manifest = json.loads((root / 'first/final_model.zip.state/runtime.json').read_text())
                self.assertEqual(manifest['schema'], 'planning_resume_v1')
                resumed = dict(config, rounds=3, resume_from=str(root / 'first/final_model.zip'), fixed_evaluation_enabled=True)
                result = run_planning_training(resumed, root / 'second')
                self.assertEqual(result['timesteps'], 24)
                self.assertTrue((root / 'second/fixed_evaluations/round_000003/evaluation/summary.csv').is_file())


if __name__ == '__main__':
    unittest.main()
