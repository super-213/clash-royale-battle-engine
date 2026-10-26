import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from self_play import SelfPlayOpponentPool
from fixed_evaluation import FixedPolicyOpponent


class FakePolicy:
    def __init__(self, action=(1, 7, 1)):
        self.action = np.asarray(action, dtype=np.int64)
        self.calls = []

    def predict(
        self,
        observations,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        count = observations["action_mask"].shape[0]
        self.calls.append((count, state, np.asarray(episode_start).copy()))
        if state is None:
            state = (
                np.zeros((1, count, 2), dtype=np.float32),
                np.zeros((1, count, 2), dtype=np.float32),
            )
        next_state = tuple(component + 1.0 for component in state)
        return np.repeat(self.action[None, :], count, axis=0), next_state


class FakeAlgorithm:
    loads = []

    @classmethod
    def load(cls, path, device):
        cls.loads.append((str(path), str(device)))
        return FakePolicy()


def observations(count):
    mask = np.zeros((count, 5, 32, 18), dtype=np.float32)
    mask[:, 0, 0, 0] = 1.0
    mask[:, 1, 7:15, 1:17] = 1.0
    return {
        "action_mask": mask,
        "scalar": np.zeros((count, 12), dtype=np.float32),
    }


class SelfPlayOpponentPoolTest(unittest.TestCase):
    def setUp(self):
        FakeAlgorithm.loads.clear()

    def test_staged_opponent_is_frozen_and_stage_results_exclude_old_games(self):
        with TemporaryDirectory() as directory:
            pool = SelfPlayOpponentPool(directory, FakeAlgorithm, "cpu", 2,
                                        mode="staged", snapshot_pool_size=1)
            live = FakePolicy()
            pool.set_current_model(live)
            initial = pool.snapshots_dir / "initial.zip"
            initial.write_bytes(b"initial frozen weights")
            pool.register_snapshot("initial", initial, 0)
            for slot in (0, 1):
                pool.on_episode_start(slot, 0, slot, slot)
            live.action[:] = (0, 0, 0)
            actions, _ = pool.predict(observations(2))
            np.testing.assert_array_equal(actions, [[1, 7, 1], [1, 7, 1]])
            self.assertFalse(live.calls)
            pool.on_episode_end(0, winner=0, learner_role=0)
            self.assertEqual(pool.metrics()["self_play/stage_win_rate_100"], 1.0)

            newer = pool.snapshots_dir / "newer.zip"
            newer.write_bytes(b"next frozen weights")
            pool.register_snapshot("newer", newer, 200)
            pool.ratings["initial"] = 5000
            pool.begin_stage()
            self.assertEqual(pool.metrics()["self_play/stage_games"], 0)
            self.assertIsNone(pool.metrics()["self_play/stage_win_rate_100"])
            # Finish an old game after switching: it must not inflate the new stage.
            self.assertEqual(pool.assignments[1]["id"], "initial")
            pool.on_episode_end(1, winner=1, learner_role=1)
            self.assertEqual(pool.metrics()["self_play/stage_games"], 0)
            for slot in (0, 1):
                pool.on_episode_start(slot, 1, slot + 2, slot)
                self.assertEqual(pool.assignments[slot]["id"], "newer")
            pool.on_episode_end(0, winner=1, learner_role=0)
            pool.on_episode_end(1, winner=None, learner_role=1)
            self.assertEqual(pool.metrics()["self_play/stage_games"], 2)
            self.assertEqual(pool.metrics()["self_play/stage_win_rate_100"], 0.0)
            self.assertEqual(pool.metrics()["self_play/stage_score_rate_100"], 0.25)
            self.assertEqual(pool.metrics()["self_play/games"], 4)
            self.assertEqual([entry["id"] for entry in pool.snapshots], ["initial", "newer"])
            self.assertTrue(initial.is_file())
            self.assertEqual([entry["id"] for entry in pool._active_snapshots()], ["newer"])

            saved = Path(directory) / "saved"
            pool.save_state(saved)
            restored = SelfPlayOpponentPool(Path(directory) / "resumed", FakeAlgorithm, "cpu", 2,
                                            mode="staged")
            restored.restore_state(saved)
            restored.begin_stage()
            self.assertEqual(restored.stage, pool.stage)
            self.assertEqual(list(restored.stage_recent_results), [0.0, 0.5])
            self.assertEqual(restored._choose_opponent()["id"], "newer")
            pool.close()
            restored.close()

    def test_historical_snapshot_identity_is_checked_on_load_save_and_resume(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'snapshot.zip'
            snapshot.write_bytes(b'original snapshot')
            pool = SelfPlayOpponentPool(root, FakeAlgorithm, 'cpu', 1)
            pool.register_snapshot('first', snapshot, 0)
            pool.save_state(root / 'saved')
            snapshot.write_bytes(b'changed snapshot')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                pool._load_snapshot('first')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                pool.save_state(root / 'changed')
            FakeAlgorithm.loads.clear()
            restored = SelfPlayOpponentPool(root / 'restore', FakeAlgorithm, 'cpu', 1)
            restored.restore_state(root / 'saved')
            self.assertEqual(Path(restored.snapshots[0]['path']).read_bytes(), b'original snapshot')
            saved_snapshot = root / 'saved/first.zip'
            saved_snapshot.write_bytes(b'changed snapshot')
            with self.assertRaisesRegex(ValueError, 'resume historical snapshot changed'):
                restored.restore_state(root / 'saved')
            saved_snapshot.unlink()
            with self.assertRaisesRegex(ValueError, 'resume historical snapshot is missing'):
                restored.restore_state(root / 'saved')
            pool.close()
            restored.close()

    def test_snapshot_ids_cannot_alias_categories_or_change_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'first.zip', root / 'second.zip'
            first.write_bytes(b'first')
            second.write_bytes(b'second')
            pool = SelfPlayOpponentPool(root, FakeAlgorithm, 'cpu', 1,
                                       current_ratio=0, historical_ratio=1, snapshot_pool_size=1)
            for name in ('random', 'current', '', '..', '../first', 'nested/first', 'nested\\first'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    pool.register_snapshot(name, first, 0)
            pool.register_snapshot('first', first, 0)
            pool.register_snapshot('first', first, 0)
            self.assertEqual(len(pool.snapshots), 1)
            for path, round_number in ((second, 0), (first, 1)):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    pool.register_snapshot('first', path, round_number)
            pool.on_episode_start(0, 0, 0, 0)
            pool.register_snapshot('second', second, 1)
            with self.assertRaisesRegex(ValueError, 'retired'):
                pool.register_snapshot('first', second, 2)
            self.assertEqual(pool._snapshot_entry('first')['path'], str(first.resolve()))
            pool.predict(observations(1))
            self.assertEqual(FakeAlgorithm.loads[-1][0], str(first.resolve()))
            pool.save_state(root / 'saved')
            state_path = root / 'saved/state.json'
            saved = json.loads(state_path.read_text())
            for entries in ([dict(saved['snapshots'][0], id='current')], saved['snapshots'] * 2):
                state_path.write_text(json.dumps(dict(saved, snapshots=entries)))
                before = list(pool.snapshots)
                with self.assertRaises(ValueError):
                    pool.restore_state(root / 'saved')
                self.assertEqual(pool.snapshots, before)
            pool.close()

    def test_bad_predictions_do_not_advance_opponent_memory_or_truncate_actions(self):
        with TemporaryDirectory() as directory:
            model = FakePolicy()
            pool = SelfPlayOpponentPool(directory, FakeAlgorithm, 'cpu', 2,
                                       current_ratio=1, historical_ratio=0)
            pool.set_current_model(model)
            fixed = FixedPolicyOpponent(model, 2)
            for wrapper in (pool, fixed):
                for slot in (0, 1):
                    wrapper.on_episode_start(slot, 0, 0, 0)
                for actions in (np.full((2, 3), .5), np.zeros((1, 3)), np.full((2, 3), np.nan)):
                    model.action = actions
                    original = model.predict
                    def invalid(observations, **kwargs):
                        _, state = original(observations, **kwargs)
                        return model.action, state
                    with patch.object(model, 'predict', side_effect=invalid):
                        with self.assertRaises(ValueError):
                            wrapper.predict(observations(2))
                    states = wrapper.assignments if wrapper is pool else wrapper.states
                    self.assertTrue(all(slot['episode_start'] for slot in states.values()))
                    self.assertTrue(all(slot.get('recurrent_state', slot.get('state')) is None for slot in states.values()))
                model.action = np.asarray((1, 7, 1))
                actions, _ = wrapper.predict(observations(2))
                np.testing.assert_array_equal(actions, [[1, 7, 1], [1, 7, 1]])
                self.assertTrue(all(not slot['episode_start'] for slot in states.values()))
            pool.close()

    def test_historical_opponent_is_batched_and_results_update_elo(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot.zip"
            snapshot.write_bytes(b"fake snapshot")
            pool = SelfPlayOpponentPool(
                run_dir=root,
                algorithm_class=FakeAlgorithm,
                device="cpu",
                battles=2,
                seed=3,
                random_ratio=0.0,
                current_ratio=0.0,
                historical_ratio=1.0,
            )
            pool.register_snapshot("snapshot_0", snapshot, round_number=0)
            pool.on_episode_start(0, episode=0, seed=10, learner_role=0)
            pool.on_episode_start(1, episode=0, seed=11, learner_role=1)
            actions, _ = pool.predict(observations(2))
            self.assertEqual(actions.tolist(), [[1, 7, 1], [1, 7, 1]])
            self.assertEqual(len(FakeAlgorithm.loads), 1)
            loaded_policy = next(iter(pool._loaded_models.values()))
            pool.predict(observations(2))
            self.assertIsNotNone(loaded_policy.calls[1][1])
            self.assertEqual(loaded_policy.calls[1][2].tolist(), [False, False])

            win = pool.on_episode_end(0, winner=0, learner_role=0)
            loss = pool.on_episode_end(1, winner=0, learner_role=1)
            self.assertEqual(win["self_play_result"], "win")
            self.assertEqual(loss["self_play_result"], "loss")
            self.assertEqual(pool.metrics()["self_play/games"], 2)
            self.assertTrue((root / "self_play" / "matches.jsonl").is_file())
            manifest = json.loads(
                (root / "self_play" / "pool.json").read_text()
            )
            self.assertEqual(manifest["aggregate"]["games"], 2)
            self.assertEqual(manifest["snapshots"][0]["id"], "snapshot_0")
            self.assertEqual(len(manifest["matchups"]), 1)
            pool.close()

    def test_later_group_failure_does_not_advance_any_opponent_memory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / 'snapshot.zip'
            snapshot.write_bytes(b'historical model')
            current, historical = FakePolicy(), FakePolicy()
            pool = SelfPlayOpponentPool(root, FakeAlgorithm, 'cpu', 2,
                                       current_ratio=1, historical_ratio=0)
            pool.set_current_model(current)
            pool.register_snapshot('historical', snapshot, 0)
            for slot in (0, 1):
                pool.on_episode_start(slot, 0, slot, 0)
            pool.assignments[1].update(id='historical', category='historical')
            with patch.object(pool, '_load_snapshot', return_value=historical):
                pool.predict(observations(2))
                # Keep one ongoing game and one fresh game in different groups.
                pool.on_episode_start(1, 1, 2, 0)
                pool.assignments[1].update(id='historical', category='historical')
                original = historical.predict
                for failure in ('fractional_action', 'exception'):
                    def fail(observation, **kwargs):
                        if failure == 'exception':
                            raise RuntimeError('later opponent failed')
                        _, state = original(observation, **kwargs)
                        return np.full((1, 3), .5), state
                    with self.subTest(failure=failure), patch.object(historical, 'predict', side_effect=fail):
                        with self.assertRaises((ValueError, RuntimeError)):
                            pool.predict(observations(2))
                    self.assertFalse(pool.assignments[0]['episode_start'])
                    for component in pool.assignments[0]['recurrent_state']:
                        np.testing.assert_array_equal(component, np.ones((1, 1, 2)))
                    self.assertTrue(pool.assignments[1]['episode_start'])
                    self.assertIsNone(pool.assignments[1]['recurrent_state'])
                pool.predict(observations(2))
                for slot, expected in ((0, 2), (1, 1)):
                    self.assertFalse(pool.assignments[slot]['episode_start'])
                    for component in pool.assignments[slot]['recurrent_state']:
                        np.testing.assert_array_equal(component, np.full((1, 1, 2), expected))
            pool.close()

    def test_random_current_and_history_ratios_must_sum_to_one(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                SelfPlayOpponentPool(
                    run_dir=directory,
                    algorithm_class=FakeAlgorithm,
                    device="cpu",
                    battles=1,
                    random_ratio=0.5,
                    current_ratio=0.5,
                    historical_ratio=0.5,
                )

    def test_default_training_opponents_exclude_random(self):
        with TemporaryDirectory() as directory:
            pool = SelfPlayOpponentPool(
                run_dir=directory,
                algorithm_class=FakeAlgorithm,
                device="cpu",
                battles=1,
            )
            pool.set_current_model(FakePolicy())
            self.assertEqual(pool.ratios.tolist(), [0.0, 0.3, 0.7])
            self.assertEqual(
                {pool._choose_opponent()["id"] for _ in range(100)}, {"current"}
            )
            pool.close()

    def test_pool_retains_stronger_models_and_deletes_weak_owned_snapshots(self):
        with TemporaryDirectory() as directory:
            pool = SelfPlayOpponentPool(
                run_dir=directory,
                algorithm_class=FakeAlgorithm,
                device="cpu",
                battles=1,
                random_ratio=0.0,
                current_ratio=0.0,
                historical_ratio=1.0,
                snapshot_pool_size=3,
                active_historical=2,
            )
            for round_number in range(4):
                path = pool.snapshots_dir / f"round_{round_number}.zip"
                path.write_bytes(b"fake snapshot")
                pool.register_snapshot(f"round_{round_number}", path, round_number)
                if round_number == 0:
                    pool.ratings["round_0"] = 1200.0
                elif round_number == 1:
                    pool.ratings["round_1"] = 700.0
                elif round_number == 2:
                    pool.ratings["round_2"] = 1100.0

            self.assertEqual(
                [item["id"] for item in pool.snapshots],
                ["round_0", "round_2", "round_3"],
            )
            self.assertEqual(
                {item["id"] for item in pool._active_snapshots()},
                {"round_0", "round_3"},
            )
            self.assertFalse((pool.snapshots_dir / "round_1.zip").exists())
            self.assertEqual(len(list(pool.snapshots_dir.glob("*.zip"))), 3)
            manifest = json.loads((pool.root / "pool.json").read_text())
            self.assertEqual(len(manifest["snapshots"]), 3)
            self.assertNotIn("round_1", manifest["ratings"])
            pool.close()

    def test_evicted_opponent_remains_available_until_its_episode_ends(self):
        with TemporaryDirectory() as directory:
            pool = SelfPlayOpponentPool(
                run_dir=directory,
                algorithm_class=FakeAlgorithm,
                device="cpu",
                battles=1,
                random_ratio=0.0,
                current_ratio=0.0,
                historical_ratio=1.0,
                snapshot_pool_size=1,
            )
            first = pool.snapshots_dir / "first.zip"
            first.write_bytes(b"first")
            pool.register_snapshot("first", first, 0)
            pool.on_episode_start(0, episode=0, seed=0, learner_role=0)
            second = pool.snapshots_dir / "second.zip"
            second.write_bytes(b"second")
            pool.register_snapshot("second", second, 1)
            self.assertEqual([item["id"] for item in pool.snapshots], ["second"])
            self.assertTrue(first.exists())
            pool.predict(observations(1))
            pool.on_episode_end(0, winner=0, learner_role=0)
            pool.on_episode_start(0, episode=1, seed=1, learner_role=0)
            self.assertFalse(first.exists())
            self.assertNotIn("first", pool.ratings)
            pool.close()


if __name__ == "__main__":
    unittest.main()
