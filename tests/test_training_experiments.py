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

import one_click_train
import training_experiments as experiments
from fixed_evaluation import (
    _summarize, checkpoint_spec, evaluate_checkpoint, preserve_random_state,
)
from native_engine.native_vec_env import NativeBatchVecEnv
from native_engine.native_recurrent_model import (
    NativeRecurrentPPO, NativeSpatialHeatmapRecurrentPolicy,
    NativeActionConditionedWorldModelPPO,
)
from test_native_vec_env import FakeBatch


class TerminalBatch(FakeBatch):
    """Small native-state fixture; every slot terminates after three decisions."""

    def step(self, actions=(), ticks=10):
        states = super().step(actions, ticks)
        self.states = tuple(replace(
            state, finalized=state.tick >= 120,
            result=0 if state.tick >= 120 else None,
            crowns=(1, 0) if state.tick >= 120 else (0, 0),
        ) for state in states)
        return self.states


class TrainingExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        base = one_click_train.prepare_config(one_click_train.load_profiles()[0], defaults_only=True)
        base.update(
            battles=2, device="cpu", rollout_decisions=8, batch_size=8,
            epochs_per_round=1, tbptt_steps=4, auxiliary_horizon=2,
            recurrent_hidden_size=16, cross_modal_dim=8, cross_modal_entity_layers=1,
            spatial_feature_channels=8, world_model_hidden_dim=16,
            self_play_enabled=False, evaluation_seed_count=1, evaluation_battles=2,
            evaluation_every_rounds=1, evaluation_max_decisions=8,
            algorithm_kwargs={"policy_kwargs": {
                "net_arch": {"pi": [16], "vf": [16]},
                "features_extractor_kwargs": {"features_dim": 32},
            }},
        )
        return base

    def frozen_checkpoint(self, root):
        with patch("native_engine.native_vec_env.Batch", side_effect=lambda **kwargs: TerminalBatch(kwargs["battles"])):
            env = NativeBatchVecEnv(battles=2)
            model = NativeRecurrentPPO(
                NativeSpatialHeatmapRecurrentPolicy, env, n_steps=4, batch_size=8,
                tbptt_steps=4, seed=17, device="cpu",
                policy_kwargs={"lstm_hidden_size": 16,
                    "features_extractor_kwargs": {"features_dim": 32, "spatial_channels": 8,
                        "cross_dim": 8, "entity_layers": 1, "attention_heads": 4},
                    "net_arch": {"pi": [16], "vf": [16]}},
            )
            path = Path(root) / "frozen.zip"
            model.save(path)
            env.close()
        return {"frozen_v4": checkpoint_spec(path, "native_engine.native_recurrent_model:NativeRecurrentPPO")}

    def test_variants_change_only_the_intended_training_controls(self):
        base = self.config()
        original = copy.deepcopy(base)
        no_aux = experiments.variant_config(base, "no_aux")
        self.assertEqual(no_aux["auxiliary_coef"], 0)
        self.assertEqual(no_aux["algorithm"], base["algorithm"])
        self.assertEqual(no_aux["policy"], base["policy"])
        self.assertTrue(experiments.variant_config(base, "state_only")["algorithm"].endswith(":NativeWorldModelPPO"))
        self.assertFalse(experiments.variant_config(base, "v4")["world_model_enabled"])
        for task in experiments.TASKS:
            config = experiments.variant_config(base, f"drop_{task}")
            for name, weight in original["auxiliary_weights"].items():
                self.assertEqual(config["auxiliary_weights"][name], 0 if name == task else weight)
        self.assertEqual(base, original)

    def test_plan_rejects_unequal_budget_insufficient_seeds_and_test_overlap(self):
        with TemporaryDirectory() as directory:
            opponents = self.frozen_checkpoint(directory)
            for budget, seeds, test_base, message in (
                (9, [0, 1, 2], 1900001, "multiple"),
                (16, [0, 1], 1900001, "three"),
                (16, [0, 1, 2], 900001, "disjoint"),
                (16, [0, 1, 2], 1, "disjoint"),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    experiments.create_plan(self.config(), Path(directory) / "suite", opponents,
                                            budget, seeds, test_seed_base=test_base, test_seed_count=1)
            self.assertFalse((Path(directory) / "suite").exists())

    def test_checkpoint_identity_and_relative_manifest_paths(self):
        with TemporaryDirectory() as directory:
            opponents = self.frozen_checkpoint(directory)
            manifest = Path(directory) / "opponents.json"
            spec = dict(opponents["frozen_v4"], path="frozen.zip")
            manifest.write_text(json.dumps({"opponents": {"old_model": spec}}))
            self.assertEqual(experiments.load_opponents(manifest)["old_model"], opponents["frozen_v4"])
            Path(opponents["frozen_v4"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                experiments.load_opponents(manifest)

    def test_timeouts_are_excluded_from_scores_and_invalidate_schedule(self):
        matches = [
            {"result": "win", "timed_out": False, "decisions": 4, "invalid_actions": 1},
            {"result": "draw", "timed_out": False, "decisions": 4},
            {"result": "incomplete", "timed_out": True, "decisions": 8},
        ]
        row = _summarize(matches, 1, 8, "opponent", 0)
        self.assertEqual(row["completed_games"], 2)
        self.assertEqual(row["draws"], 1)
        self.assertEqual(row["score_rate"], 0.75)
        self.assertEqual(row["win_rate"], 0.5)
        self.assertEqual(row["invalid_action_rate"], 1 / 16)
        self.assertFalse(row["valid"])
        row = _summarize(matches[-1:], 1, 8, "opponent", 0)
        self.assertIsNone(row["score_rate"])
        self.assertIsNone(row["win_rate"])

    def test_evaluation_restores_python_numpy_and_torch_rng_even_on_failure(self):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "failure"):
            with preserve_random_state():
                random.seed(456)
                np.random.seed(456)
                torch.manual_seed(456)
                raise RuntimeError("failure")
        self.assertEqual(random.getstate(), python_state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_state)

    def test_real_cpu_suite_trains_four_variants_and_reloads_on_held_out_schedule(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite"
            opponents = self.frozen_checkpoint(root)
            plan = experiments.create_plan(self.config(), suite, opponents, 16, [0, 1, 2], test_seed_count=1)
            self.assertEqual(len(plan["runs"]), 12)
            self.assertEqual(plan["total_training_decisions"], 192)
            self.assertTrue(Path(plan["frozen_opponents"]["frozen_v4"]["path"]).is_relative_to(suite.resolve()))
            with patch("native_engine.native_vec_env.Batch", side_effect=lambda **kwargs: TerminalBatch(kwargs["battles"])):
                report = experiments.run_suite(suite)
                self.assertTrue(report["valid"])
                self.assertEqual(len(report["groups"]), 4 * 7)  # aggregate + two opponents × (combined/p0/p1)
                self.assertEqual(len(report["paired_differences"]), 3 * 7)
                aggregate = {row["variant"]: row for row in report["groups"] if row["opponent_id"] == "aggregate"}
                self.assertEqual(aggregate["full"]["parameters"], aggregate["no_aux"]["parameters"])
                self.assertEqual(aggregate["full"]["score_rate"]["mean"], 0.5)
                self.assertEqual(aggregate["full"]["score_rate"]["std"], 0)
                for run in plan["runs"][:6]:
                    run_dir = Path(run["run_dir"])
                    initial = NativeActionConditionedWorldModelPPO.load(
                        run_dir / "evaluation" / "baselines" / "initial_policy.zip", device="cpu")
                    final = NativeActionConditionedWorldModelPPO.load(run_dir / "final_model.zip", device="cpu")
                    changes = [not torch.equal(value, final.policy.world_model.state_dict()[key])
                               for key, value in initial.policy.world_model.state_dict().items()]
                    self.assertEqual(any(changes), run["variant"] == "full")
                # Re-running skips completed runs and preserves the same comparison.
                with patch("training_runtime.run_sb3_training", side_effect=AssertionError("must skip completed training")):
                    self.assertTrue(experiments.run_suite(suite)["valid"])
                config = plan["runs"][0]["config"]
                model = Path(plan["runs"][0]["run_dir"]) / "final_model.zip"
                timeout_config = experiments.test_config(plan, config)
                timeout_config["evaluation_max_decisions"] = 1
                evaluation = evaluate_checkpoint(timeout_config, model, root / "timeout")
                self.assertFalse(evaluation["latest"]["valid"])
                self.assertEqual(evaluation["latest"]["timed_out"], 4)
                self.assertIsNone(evaluation["latest"]["score_rate"])
            # Duplicate records, edited schedules and unequal budgets cannot enter the comparison.
            run_dir = Path(plan["runs"][0]["run_dir"])
            matches = run_dir / "post_training" / "evaluation" / "matches.jsonl"
            original = matches.read_text()
            matches.write_text(original + original.splitlines()[0] + "\n")
            self.assertFalse(experiments.build_report(suite)["valid"])
            matches.write_text(original)
            schedule_path = run_dir / "post_training" / "evaluation" / "schedule.json"
            schedule = json.loads(schedule_path.read_text())
            schedule["seeds"] = [1]
            schedule_path.write_text(json.dumps(schedule))
            self.assertFalse(experiments.build_report(suite)["valid"])
            metadata_path = run_dir / "summary.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["timesteps"] += 8
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "unequal"):
                experiments.read_run_result(plan, plan["runs"][0])


if __name__ == "__main__":
    unittest.main()
