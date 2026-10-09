import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from self_play import SelfPlayOpponentPool


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
