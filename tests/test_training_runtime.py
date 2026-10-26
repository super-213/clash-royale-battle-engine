import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

import one_click_train
from training_runtime import generate_charts, load_object, run_sb3_training


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps == 4
        return np.zeros(4, dtype=np.float32), 1.0, terminated, False, {}


class ToyVecEnv(DummyVecEnv):
    def __init__(
        self,
        battles,
        port,
        decision_ticks,
        seed,
        learner_roles=None,
        opponent_policy=None,
    ):
        if battles != 1:
            raise ValueError("toy environment only supports one slot")
        super().__init__([ToyEnv])


class ToySelfPlayVecEnv(VecEnv):
    def __init__(self, battles, port, decision_ticks, seed, opponent_policy=None):
        self.battles = int(battles)
        self.opponent_policy = opponent_policy
        self.steps = np.zeros(self.battles, dtype=np.int64)
        observation_space = spaces.Dict({
            "action_mask": spaces.Box(
                0.0, 1.0, shape=(5, 32, 18), dtype=np.float32
            ),
            "scalar": spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
        })
        super().__init__(
            self.battles,
            observation_space,
            spaces.MultiDiscrete([5, 32, 18]),
        )
        self._actions = None

    def _observations(self):
        mask = np.zeros((self.battles, 5, 32, 18), dtype=np.float32)
        mask[:, 0, 0, 0] = 1.0
        mask[:, 1, 7:15, 1:17] = 1.0
        return {
            "action_mask": mask,
            "scalar": np.zeros((self.battles, 2), dtype=np.float32),
        }

    def reset(self):
        self.steps[:] = 0
        for slot in range(self.battles):
            self.opponent_policy.on_episode_start(slot, 0, slot, slot % 2)
        return self._observations()

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        observations = self._observations()
        self.opponent_policy.predict(observations)
        self.steps += 1
        dones = self.steps >= 2
        rewards = np.ones(self.battles, dtype=np.float32)
        infos = []
        for slot, done in enumerate(dones):
            info = self.opponent_policy.opponent_metadata(slot)
            if done:
                info.update(self.opponent_policy.on_episode_end(slot, slot % 2, slot % 2))
                info["episode"] = {"r": 2.0, "l": 2, "t": 0.0}
                info["terminal_observation"] = {
                    key: value[slot] for key, value in observations.items()
                }
                self.steps[slot] = 0
                self.opponent_policy.on_episode_start(slot, 1, slot + 10, slot % 2)
            infos.append(info)
        return self._observations(), rewards, dones, infos

    def close(self):
        if self.opponent_policy is not None:
            self.opponent_policy.close()

    def get_attr(self, attr_name, indices=None):
        selected = range(self.battles) if indices is None else indices
        return [getattr(self, attr_name) for _ in selected]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return []

    def env_is_wrapped(self, wrapper_class, indices=None):
        selected = range(self.battles) if indices is None else indices
        return [False for _ in selected]


class TrainingRuntimeTest(unittest.TestCase):
    def test_default_profile_resolves_replaceable_components(self):
        profile = one_click_train.load_profiles()[0]
        self.assertTrue(callable(load_object(profile["trainer"])))
        self.assertTrue(callable(load_object(profile["environment"])))
        self.assertTrue(callable(load_object(profile["algorithm"])))
        self.assertTrue(callable(load_object(profile["policy"])))

    def test_defaults_dry_run_writes_agent_readable_config_and_status(self):
        with TemporaryDirectory() as directory:
            result = one_click_train.main([
                "--profile",
                "native_hybrid_ppo",
                "--defaults",
                "--dry-run",
                "--runs-dir",
                directory,
            ])
            self.assertEqual(result, 0)
            runs = [path for path in Path(directory).iterdir() if path.is_dir()]
            self.assertEqual(len(runs), 1)
            config = json.loads((runs[0] / "config.json").read_text())
            status = json.loads((runs[0] / "status.json").read_text())
            latest = json.loads((Path(directory) / "latest.json").read_text())
            self.assertEqual(
                config["policy"],
                "native_engine.native_recurrent_model:NativeRoleAwareActionConditionedWorldModelRecurrentPolicy",
            )
            self.assertEqual(
                config["algorithm"],
                "native_engine.native_recurrent_model:NativeActionConditionedWorldModelPPO",
            )
            self.assertEqual(config["device"], "mps")
            self.assertEqual(config["self_play_mode"], "staged")
            self.assertEqual(config["self_play_snapshot_every_rounds"], 100)
            self.assertTrue(config["recurrent_enabled"])
            self.assertEqual(config["tbptt_steps"], 32)
            self.assertTrue(config["cross_modal_enabled"])
            self.assertEqual(config["cross_modal_dim"], 64)
            self.assertTrue(config["spatial_heatmap_enabled"])
            self.assertEqual(config["spatial_feature_channels"], 64)
            self.assertTrue(config["world_model_enabled"])
            self.assertEqual(config["auxiliary_horizon"], 4)
            self.assertTrue(config["fixed_evaluation_enabled"])
            self.assertEqual(config["evaluation_seed_count"], 16)
            self.assertEqual(
                config["evaluation_opponents"], ["random", "initial_policy"]
            )
            self.assertEqual(status["status"], "dry_run_completed")
            self.assertEqual(Path(latest["run_dir"]), runs[0].resolve())

    def test_stage_interval_is_configurable_in_both_launchers(self):
        import train_native

        with TemporaryDirectory() as directory:
            root = Path(directory)
            one_click_train.main([
                "--profile", "native_hybrid_ppo", "--defaults", "--dry-run",
                "--runs-dir", str(root / "interactive"),
                "--self-play-snapshot-every-rounds", "200",
            ])
            config_path = next((root / "interactive").glob("*/config.json"))
            config = json.loads(config_path.read_text())
            self.assertEqual(config["self_play_snapshot_every_rounds"], 200)
            self.assertEqual(config["self_play_mode"], "staged")
            train_native.main([
                "--dry-run", "--output", str(root / "direct"),
                "--self-play-snapshot-every-rounds", "1000",
            ])
            direct = json.loads((root / "direct/config.json").read_text())
            self.assertEqual(direct["self_play_snapshot_every_rounds"], 1000)

    def test_cross_modal_dimension_must_be_divisible_by_heads(self):
        profile = one_click_train.load_profiles()[0]
        config = one_click_train.prepare_config(profile, defaults_only=True)
        config["cross_modal_dim"] = 62
        with self.assertRaisesRegex(ValueError, "token.*头数"):
            one_click_train.validate_config(config)

    def test_auxiliary_horizon_must_fit_inside_tbptt(self):
        profile = one_click_train.load_profiles()[0]
        config = one_click_train.prepare_config(profile, defaults_only=True)
        config["auxiliary_horizon"] = config["tbptt_steps"]
        with self.assertRaisesRegex(ValueError, "预测步数.*TBPTT"):
            one_click_train.validate_config(config)

    def test_fixed_evaluation_rejects_unknown_opponent(self):
        profile = one_click_train.load_profiles()[0]
        config = one_click_train.prepare_config(profile, defaults_only=True)
        config["evaluation_opponents"] = ["moving_league"]
        with self.assertRaisesRegex(ValueError, "random 和 initial_policy"):
            one_click_train.validate_config(config)

    def test_svg_charts_are_generated_without_matplotlib(self):
        records = [
            {
                "timesteps": 8192,
                "fps": 300.0,
                "episode_reward_mean_100": 1.0,
                "episode_length_mean_100": 300.0,
                "train/loss": 2.5,
                "train/policy_gradient_loss": -0.1,
                "train/value_loss": 5.0,
                "train/entropy_loss": -1.2,
                "train/explained_variance": 0.2,
                "train/approx_kl": 0.01,
                "train/clip_fraction": 0.1,
                "train/learning_rate": 0.001,
                "self_play/win_rate_100": 0.4,
                "self_play/score_rate_100": 0.5,
                "self_play/learner_elo": 1000.0,
                "self_play/snapshot_count": 1,
            },
            {
                "timesteps": 16384,
                "fps": 320.0,
                "episode_reward_mean_100": 1.5,
                "episode_length_mean_100": 290.0,
                "train/loss": 2.0,
                "train/policy_gradient_loss": -0.05,
                "train/value_loss": 4.0,
                "train/entropy_loss": -1.1,
                "train/explained_variance": 0.3,
                "train/approx_kl": 0.008,
                "train/clip_fraction": 0.08,
                "train/learning_rate": 0.0009,
                "self_play/win_rate_100": 0.5,
                "self_play/score_rate_100": 0.6,
                "self_play/learner_elo": 1010.0,
                "self_play/snapshot_count": 2,
            },
        ]
        with TemporaryDirectory() as directory:
            paths = generate_charts(directory, records)
            self.assertEqual(len(paths), 7)
            for path in paths:
                content = path.read_text()
                self.assertIn("<svg", content)
                self.assertTrue(
                    "timesteps" in content or "no samples yet" in content
                )

    def test_tiny_training_writes_logs_model_summary_and_charts(self):
        config = {
            "rounds": 2,
            "rollout_decisions": 8,
            "battles": 1,
            "decision_ticks": 1,
            "port": 1,
            "seed": 0,
            "device": "cpu",
            "checkpoint_every_rounds": 1,
            "batch_size": 4,
            "epochs_per_round": 1,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "entropy_coef": 0.0,
            "clip_range": 0.2,
            "environment": f"{__name__}:ToyVecEnv",
            "algorithm": "stable_baselines3:PPO",
            "policy": "stable_baselines3.common.policies:ActorCriticPolicy",
            "environment_kwargs": {},
            "algorithm_kwargs": {},
            "fixed_evaluation_enabled": True,
            "evaluation_every_rounds": 1,
            "evaluation_seed_base": 1234,
            "evaluation_seed_count": 1,
            "evaluation_battles": 1,
            "evaluation_max_decisions": 8,
            "evaluation_opponents": ["random", "initial_policy"],
        }
        with TemporaryDirectory() as directory:
            metadata = run_sb3_training(config, directory)
            root = Path(directory)
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["completed_rounds"], 2)
            self.assertTrue((root / "final_model.zip").is_file())
            self.assertTrue((root / "summary.json").is_file())
            self.assertTrue((root / "summary.md").is_file())
            self.assertTrue((root / "metrics.jsonl").is_file())
            self.assertTrue((root / "metrics.csv").is_file())
            self.assertTrue((root / "charts" / "loss_curves.svg").is_file())
            self.assertTrue((root / "charts" / "optimization_diagnostics.svg").is_file())
            self.assertTrue(any((root / "checkpoints").glob("*.zip")))
            self.assertTrue(any((root / "sb3").glob("events.out.tfevents.*")))
            self.assertTrue((root / "episodes.monitor.csv").is_file())
            self.assertTrue((root / "evaluation" / "schedule.json").is_file())
            self.assertTrue((root / "evaluation" / "matches.jsonl").is_file())
            self.assertTrue((root / "evaluation" / "summary.csv").is_file())
            self.assertTrue((root / "evaluation" / "latest.json").is_file())
            self.assertTrue(
                (root / "evaluation" / "baselines" / "initial_policy.zip").is_file()
            )
            self.assertTrue((root / "charts" / "fixed_evaluation.svg").is_file())
            schedule = json.loads(
                (root / "evaluation" / "schedule.json").read_text()
            )
            self.assertEqual(schedule["seeds"], [1234])
            self.assertEqual(schedule["roles_per_seed"], [0, 1])
            matches = [
                json.loads(line)
                for line in (root / "evaluation" / "matches.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                sorted({match["round"] for match in matches}), [0, 1, 2]
            )
            for round_number in (0, 1, 2):
                selected = [match for match in matches
                            if match["round"] == round_number]
                self.assertEqual(len(selected), 4)
                self.assertEqual(
                    sorted((match["opponent_id"], match["learner_role"])
                           for match in selected),
                    [
                        ("initial_policy", 0),
                        ("initial_policy", 1),
                        ("random", 0),
                        ("random", 1),
                    ],
                )
            self.assertIn("fixed_evaluation", metadata)
            self.assertEqual(
                metadata["fixed_evaluation"]["evaluated_rounds"], [0, 1, 2]
            )
            self.assertIn("## Fixed evaluation", (root / "summary.md").read_text())
            status = json.loads((root / "status.json").read_text())
            self.assertEqual(status["status"], "completed")

    def self_play_config(self):
        return {
            "rounds": 2,
            "rollout_decisions": 8,
            "battles": 2,
            "decision_ticks": 1,
            "port": 1,
            "seed": 0,
            "device": "cpu",
            "checkpoint_every_rounds": 1,
            "batch_size": 4,
            "epochs_per_round": 1,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "entropy_coef": 0.0,
            "clip_range": 0.2,
            "environment": f"{__name__}:ToySelfPlayVecEnv",
            "algorithm": "native_engine.native_recurrent_model:NativeRecurrentPPO",
            "policy": "sb3_contrib.common.recurrent.policies:RecurrentMultiInputActorCriticPolicy",
            "environment_kwargs": {},
            "algorithm_kwargs": {},
            "recurrent_enabled": True,
            "tbptt_steps": 2,
            "recurrent_hidden_size": 16,
            "recurrent_layers": 1,
            "self_play_enabled": True,
            "self_play_random_ratio": 0.0,
            "self_play_current_ratio": 0.0,
            "self_play_historical_ratio": 1.0,
            "self_play_snapshot_every_rounds": 1,
            "self_play_pool_size": 5,
            "self_play_active_historical": 2,
            "self_play_loaded_model_limit": 2,
            "self_play_elo_k": 16.0,
        }

    def test_tiny_self_play_training_publishes_league_artifacts(self):
        config = self.self_play_config()
        with TemporaryDirectory() as directory:
            metadata = run_sb3_training(config, directory)
            root = Path(directory)
            pool = json.loads((root / "self_play" / "pool.json").read_text())
            self.assertGreater(pool["aggregate"]["games"], 0)
            self.assertEqual(
                [item["round"] for item in pool["snapshots"]], [0, 1, 2]
            )
            self.assertIn("self_play", metadata)
            summary = (root / "summary.md").read_text()
            self.assertIn("## Self-play", summary)
            self.assertIn("## Recurrent memory", summary)
            self.assertTrue((root / "self_play" / "matches.jsonl").is_file())
            self.assertTrue((root / "charts" / "self_play_progress.svg").is_file())

    def test_staged_training_switches_on_interval_and_resumes_an_unfinished_stage(self):
        config = self.self_play_config()
        config.update(rounds=5, self_play_mode="staged", self_play_snapshot_every_rounds=2)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "first"
            metadata = run_sb3_training(config, root)
            pool = json.loads((root / "self_play/pool.json").read_text())
            self.assertEqual([entry["round"] for entry in pool["snapshots"]], [0, 2, 4])
            records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["self_play/stage_opponent_round"] for r in records], [0, 0, 2, 2, 4])
            self.assertGreater(records[1]["self_play/stage_games"], records[0]["self_play/stage_games"])
            self.assertLess(records[2]["self_play/stage_games"], records[1]["self_play/stage_games"])
            self.assertGreater(records[3]["self_play/stage_games"], records[2]["self_play/stage_games"])
            self.assertEqual(metadata["self_play"]["stage"]["opponent_round"], 4)
            self.assertTrue((root / "charts/self_play_stages.svg").is_file())
            matches = [json.loads(line) for line in (root / "self_play/matches.jsonl").read_text().splitlines()]
            for generation, opponent in [(1, 0), (3, 2)]:
                opponents = {m["opponent_id"] for m in matches
                             if m["learner_id"] == f"learner_round_{generation:06d}"}
                self.assertEqual(opponents, {f"snapshot_round_{opponent:06d}"})

            resumed = json.loads((Path(str(root / "final_model.zip") + ".state") / "config.json").read_text())
            resumed.update(rounds=7, resume_from=str(root / "final_model.zip"))
            second = Path(directory) / "resumed"
            result = run_sb3_training(resumed, second)
            self.assertEqual(result["self_play"]["stage"]["opponent_round"], 6)
            second_pool = json.loads((second / "self_play/pool.json").read_text())
            self.assertEqual([entry["round"] for entry in second_pool["snapshots"]], [0, 2, 4, 6])
            resumed_records = [json.loads(line) for line in (second / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r["self_play/stage_opponent_round"] for r in resumed_records], [4, 6])
            self.assertGreater(resumed_records[0]["self_play/stage_games"], records[-1]["self_play/stage_games"])


if __name__ == "__main__":
    unittest.main()
