"""League-style self-play opponent pool for native vector training."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from training_runtime import load_inference_model, _sha256, completed_update_at_training_end


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_snapshot_id(snapshot_id):
    if (not snapshot_id or snapshot_id in {"random", "current", ".", ".."}
            or "/" in snapshot_id or "\\" in snapshot_id):
        raise ValueError(f"invalid historical snapshot id: {snapshot_id!r}")


class SelfPlayOpponentPool:
    """Assign one stable opponent per slot episode and track league results.

    The pool mixes three opponent categories:

    * an optional legal random policy;
    * the live learner, used as a mirror opponent;
    * immutable historical snapshots, used to prevent forgetting.

    Historical models are inferred in groups and loaded through a small LRU
    cache so a 16-slot step never becomes 16 separate model forwards.
    """

    def __init__(
        self,
        run_dir,
        algorithm_class,
        device,
        battles,
        seed=0,
        random_ratio=0.0,
        current_ratio=0.3,
        historical_ratio=0.7,
        snapshot_pool_size=20,
        active_historical=4,
        loaded_model_limit=4,
        elo_k=16.0,
        random_wait_probability=0.2,
    ):
        ratios = np.asarray(
            (random_ratio, current_ratio, historical_ratio), dtype=np.float64
        )
        if np.any(ratios < 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-6):
            raise ValueError("self-play random/current/historical ratios must sum to 1")
        if int(battles) <= 0:
            raise ValueError("battles must be positive")
        self.root = Path(run_dir).resolve() / "self_play"
        self.snapshots_dir = self.root / "snapshots"
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.algorithm_class = algorithm_class
        self.device = str(device)
        self.battles = int(battles)
        self.rng = np.random.default_rng(int(seed))
        self.ratios = ratios / ratios.sum()
        self.snapshot_pool_size = max(1, int(snapshot_pool_size))
        self.active_historical_limit = max(1, int(active_historical))
        self.loaded_model_limit = max(1, int(loaded_model_limit))
        self.elo_k = float(elo_k)
        self.random_wait_probability = float(random_wait_probability)
        self.current_model = None
        self.learner_generation = 0
        self.learner_elo = 1000.0
        self.snapshots = []
        self._retired_snapshots = {}
        self.assignments = {}
        self._loaded_models = OrderedDict()
        self.ratings = {"random": 800.0, "current": 1000.0}
        self.aggregate = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        self.recent_results = deque(maxlen=100)
        self.matchups = defaultdict(
            lambda: {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        )
        self.matches_path = self.root / "matches.jsonl"
        self.ratings_path = self.root / "ratings.csv"
        with self.ratings_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                ("timestamp", "games", "learner_generation", "learner_elo", "opponent_id", "opponent_elo")
            )
        self._persist()

    def set_current_model(self, model):
        self.current_model = model

    def set_learner_generation(self, generation):
        self.learner_generation = max(0, int(generation))
        self._persist()

    def _active_snapshots(self):
        if not self.snapshots:
            return []
        newest = self.snapshots[-1]
        strongest = sorted(
            self.snapshots[:-1],
            key=lambda item: (self.ratings[item["id"]], item["round"]),
            reverse=True,
        )[: self.active_historical_limit - 1]
        return strongest + [newest]

    def _cleanup_retired(self):
        changed = False
        for snapshot_id, entry in list(self._retired_snapshots.items()):
            if self._assigned(snapshot_id):
                continue
            self._loaded_models.pop(snapshot_id, None)
            self.ratings.pop(snapshot_id, None)
            path = Path(entry["path"])
            if path.is_relative_to(self.snapshots_dir):
                path.unlink(missing_ok=True)
            del self._retired_snapshots[snapshot_id]
            changed = True
        if changed:
            self._empty_mps_cache()
        return changed

    def _prune_snapshots(self):
        while len(self.snapshots) > self.snapshot_pool_size:
            # Keep the just-published policy as a fresh challenger. Among older
            # policies, prefer measured strength and break ties by recency.
            weakest = min(
                self.snapshots[:-1],
                key=lambda item: (self.ratings[item["id"]], item["round"]),
            )
            self.snapshots.remove(weakest)
            self._retired_snapshots[weakest["id"]] = weakest
        self._cleanup_retired()

    def register_snapshot(self, snapshot_id, path, round_number):
        snapshot_id = str(snapshot_id)
        _validate_snapshot_id(snapshot_id)
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if snapshot_id in self._retired_snapshots:
            raise ValueError(f"historical snapshot id is still retired: {snapshot_id}")
        digest = _sha256(path)
        previous = self._snapshot_entry(snapshot_id)
        if previous is not None:
            if (previous["path"] != str(path) or previous["round"] != int(round_number)
                    or previous.get("sha256") != digest):
                raise ValueError(f"historical snapshot identity changed: {snapshot_id}")
            return
        entry = {
            "id": snapshot_id,
            "path": str(path),
            "round": int(round_number),
            "created_at": _timestamp(),
            "sha256": digest,
        }
        self.snapshots.append(entry)
        self.ratings.setdefault(snapshot_id, self.learner_elo)
        self._prune_snapshots()
        active_ids = {item["id"] for item in self._active_snapshots()}
        for loaded_id in list(self._loaded_models):
            if loaded_id not in active_ids and not self._assigned(loaded_id):
                del self._loaded_models[loaded_id]
        self._empty_mps_cache()
        self._persist()

    def _assigned(self, opponent_id):
        return any(item["id"] == opponent_id for item in self.assignments.values())

    def _choose_opponent(self):
        available_history = self._active_snapshots()
        category_probabilities = self.ratios.copy()
        if self.current_model is None:
            fallback = 2 if available_history else 0
            category_probabilities[fallback] += category_probabilities[1]
            category_probabilities[1] = 0.0
        if not available_history:
            fallback = 1 if self.current_model is not None else 0
            category_probabilities[fallback] += category_probabilities[2]
            category_probabilities[2] = 0.0
        category_probabilities /= category_probabilities.sum()
        category = int(self.rng.choice(3, p=category_probabilities))
        if category == 0:
            return {"id": "random", "category": "random"}
        if category == 1:
            return {"id": "current", "category": "current"}
        snapshot = available_history[int(self.rng.integers(len(available_history)))]
        return {"id": snapshot["id"], "category": "historical"}

    def on_episode_start(self, slot_index, episode, seed, learner_role):
        assignment = self._choose_opponent()
        assignment.update(
            episode=int(episode),
            seed=int(seed),
            learner_role=int(learner_role),
            recurrent_state=None,
            episode_start=True,
        )
        self.assignments[int(slot_index)] = assignment
        if self._cleanup_retired():
            self._persist()
        return dict(assignment)

    def opponent_metadata(self, slot_index):
        assignment = self.assignments.get(int(slot_index))
        if assignment is None:
            return {"opponent_id": "unassigned", "opponent_category": "unknown"}
        return {
            "opponent_id": assignment["id"],
            "opponent_category": assignment["category"],
        }

    @staticmethod
    def _slice_observations(observations, indices):
        return {key: value[indices] for key, value in observations.items()}

    def _random_action(self, observation):
        mask = np.asarray(observation["action_mask"])
        legal = np.argwhere(mask[1:] > 0)
        if len(legal) == 0 or self.rng.random() < self.random_wait_probability:
            return np.asarray((0, 0, 0), dtype=np.int64)
        card_offset, y, x = legal[int(self.rng.integers(len(legal)))]
        return np.asarray((int(card_offset) + 1, int(y), int(x)), dtype=np.int64)

    def _snapshot_entry(self, snapshot_id):
        return next(
            (item for item in self.snapshots if item["id"] == snapshot_id),
            self._retired_snapshots.get(snapshot_id),
        )

    def _load_snapshot(self, snapshot_id):
        if snapshot_id in self._loaded_models:
            model = self._loaded_models.pop(snapshot_id)
            self._loaded_models[snapshot_id] = model
            return model
        entry = self._snapshot_entry(snapshot_id)
        if entry.get("sha256") is not None and _sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"historical snapshot changed: {snapshot_id}")
        model = load_inference_model(self.algorithm_class, entry["path"], self.device)
        self._loaded_models[snapshot_id] = model
        while len(self._loaded_models) > self.loaded_model_limit:
            removable = next(
                (
                    item_id for item_id in self._loaded_models
                    if item_id != snapshot_id and not self._assigned(item_id)
                ),
                None,
            )
            if removable is None:
                break
            del self._loaded_models[removable]
        self._empty_mps_cache()
        return model

    def predict(self, observations, deterministic=False):
        count = int(next(iter(observations.values())).shape[0])
        if count != self.battles:
            raise ValueError(f"self-play expected {self.battles} observations, got {count}")
        actions = np.zeros((count, 3), dtype=np.int64)
        grouped = defaultdict(list)
        for slot_index in range(count):
            assignment = self.assignments.get(slot_index)
            if assignment is None:
                assignment = self.on_episode_start(slot_index, 0, 0, 0)
            grouped[assignment["id"]].append(slot_index)

        for opponent_id, indices in grouped.items():
            if opponent_id == "random":
                for index in indices:
                    single = {key: value[index] for key, value in observations.items()}
                    actions[index] = self._random_action(single)
                continue
            model = (
                self.current_model
                if opponent_id == "current"
                else self._load_snapshot(opponent_id)
            )
            if model is None:
                for index in indices:
                    single = {key: value[index] for key, value in observations.items()}
                    actions[index] = self._random_action(single)
                continue
            slot_states = [
                self.assignments[index].get("recurrent_state") for index in indices
            ]
            state = None
            template = next((value for value in slot_states if value is not None), None)
            if template is not None:
                state = tuple(
                    np.concatenate(
                        [
                            value[component]
                            if value is not None
                            else np.zeros_like(template[component][:, :1, :])
                            for value in slot_states
                        ],
                        axis=1,
                    )
                    for component in range(len(template))
                )
            episode_starts = np.asarray(
                [self.assignments[index]["episode_start"] for index in indices],
                dtype=bool,
            )
            predicted, next_state = model.predict(
                    self._slice_observations(observations, indices),
                    state=state,
                    episode_start=episode_starts,
                    deterministic=bool(deterministic),
                )
            predicted = np.asarray(predicted)
            if predicted.shape != (len(indices), 3):
                raise ValueError(
                    f"opponent {opponent_id} returned {predicted.shape}, "
                    f"expected {(len(indices), 3)}"
                )
            actions[indices] = predicted
            for offset, index in enumerate(indices):
                self.assignments[index]["episode_start"] = False
                self.assignments[index]["recurrent_state"] = (
                    None
                    if next_state is None
                    else tuple(
                        component[:, offset : offset + 1, :].copy()
                        for component in next_state
                    )
                )
        return actions, None

    def on_episode_end(self, slot_index, winner, learner_role):
        assignment = self.assignments.get(int(slot_index))
        if assignment is None:
            return {}
        learner_role = int(learner_role)
        if winner is None or int(winner) not in (0, 1):
            result, score = "draw", 0.5
        elif int(winner) == learner_role:
            result, score = "win", 1.0
        else:
            result, score = "loss", 0.0
        opponent_id = assignment["id"]
        opponent_elo = float(self.ratings.get(opponent_id, 1000.0))
        expected = 1.0 / (1.0 + 10.0 ** ((opponent_elo - self.learner_elo) / 400.0))
        delta = self.elo_k * (score - expected)
        self.learner_elo += delta
        self.ratings[opponent_id] = opponent_elo - delta

        self.aggregate["games"] += 1
        self.aggregate[{"win": "wins", "loss": "losses", "draw": "draws"}[result]] += 1
        self.recent_results.append(score)
        learner_id = f"learner_round_{self.learner_generation:06d}"
        matchup = self.matchups[(learner_id, opponent_id)]
        matchup["games"] += 1
        matchup[{"win": "wins", "loss": "losses", "draw": "draws"}[result]] += 1
        match = {
            "timestamp": _timestamp(),
            "slot": int(slot_index),
            "episode": assignment["episode"],
            "seed": assignment["seed"],
            "learner_id": learner_id,
            "learner_role": learner_role,
            "opponent_id": opponent_id,
            "opponent_category": assignment["category"],
            "winner": None if winner is None else int(winner),
            "result": result,
            "learner_elo": round(self.learner_elo, 3),
            "opponent_elo": round(self.ratings[opponent_id], 3),
        }
        with self.matches_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(match, ensure_ascii=False, sort_keys=True) + "\n")
        with self.ratings_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                (
                    match["timestamp"],
                    self.aggregate["games"],
                    self.learner_generation,
                    match["learner_elo"],
                    opponent_id,
                    match["opponent_elo"],
                )
            )
        self._persist()
        return {
            "self_play_result": result,
            "learner_elo": match["learner_elo"],
            "opponent_elo": match["opponent_elo"],
        }

    def metrics(self):
        games = self.aggregate["games"]
        recent = list(self.recent_results)
        return {
            "self_play/games": games,
            "self_play/wins": self.aggregate["wins"],
            "self_play/losses": self.aggregate["losses"],
            "self_play/draws": self.aggregate["draws"],
            "self_play/win_rate": self.aggregate["wins"] / games if games else None,
            "self_play/win_rate_100": (
                sum(value == 1.0 for value in recent) / len(recent) if recent else None
            ),
            "self_play/score_rate_100": float(np.mean(recent)) if recent else None,
            "self_play/learner_elo": self.learner_elo,
            "self_play/snapshot_count": len(self.snapshots),
            "self_play/active_historical": len(self._active_snapshots()),
        }

    def summary(self):
        return {
            **self.metrics(),
            "ratios": {
                "random": float(self.ratios[0]),
                "current": float(self.ratios[1]),
                "historical": float(self.ratios[2]),
            },
            "pool_manifest": str(self.root / "pool.json"),
            "matches": str(self.matches_path),
            "ratings": str(self.ratings_path),
        }

    def save_state(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        snapshots = []
        for entry in self.snapshots:
            filename = f"{entry['id']}.zip"
            digest = _sha256(entry["path"])
            if entry.get("sha256") is not None and entry["sha256"] != digest:
                raise ValueError(f"historical snapshot changed: {entry['id']}")
            shutil.copyfile(entry["path"], directory / filename)
            snapshots.append({**entry, "path": filename, "sha256": digest})
        _atomic_json(directory / "state.json", {
            "snapshots": snapshots, "ratings": self.ratings,
            "learner_elo": self.learner_elo, "learner_generation": self.learner_generation,
            "aggregate": self.aggregate, "recent_results": list(self.recent_results),
            "matchups": [[*key, value] for key, value in self.matchups.items()],
            "rng": self.rng.bit_generator.state,
        })

    def restore_state(self, directory):
        directory = Path(directory)
        saved = json.loads((directory / "state.json").read_text())
        snapshot_ids = set()
        for entry in saved["snapshots"]:
            _validate_snapshot_id(entry["id"])
            if entry["id"] in snapshot_ids:
                raise ValueError(f"duplicate resume historical snapshot id: {entry['id']}")
            snapshot_ids.add(entry["id"])
            if entry["path"] != f"{entry['id']}.zip":
                raise ValueError(f"invalid resume historical snapshot path: {entry['id']}")
            source = directory / entry["path"]
            if not source.is_file():
                raise ValueError(f"resume historical snapshot is missing: {entry['id']}")
            if entry.get("sha256") is not None and _sha256(source) != entry["sha256"]:
                raise ValueError(f"resume historical snapshot changed: {entry['id']}")
        self._loaded_models.clear()
        self._retired_snapshots.clear()
        self.snapshots = []
        for entry in saved["snapshots"]:
            source = directory / entry["path"]
            destination = self.snapshots_dir / source.name
            shutil.copyfile(source, destination)
            self.snapshots.append({**entry, "path": str(destination.resolve())})
        self.ratings = saved["ratings"]
        self.learner_elo = saved["learner_elo"]
        self.learner_generation = saved["learner_generation"]
        self.aggregate = saved["aggregate"]
        self.recent_results = deque(saved["recent_results"], maxlen=100)
        self.matchups.clear()
        self.matchups.update({(learner, opponent): values
                             for learner, opponent, values in saved["matchups"]})
        self.rng.bit_generator.state = saved["rng"]
        # Native episodes cannot be restored: resumed slots start new games.
        self.assignments.clear()
        self._persist()

    def _persist(self):
        active_ids = {item["id"] for item in self._active_snapshots()}
        snapshots = [
            {
                **item,
                "active": item["id"] in active_ids,
                "elo": round(float(self.ratings.get(item["id"], 1000.0)), 3),
            }
            for item in self.snapshots
        ]
        matchups = [
            {"learner_id": learner, "opponent_id": opponent, **values}
            for (learner, opponent), values in sorted(self.matchups.items())
        ]
        _atomic_json(
            self.root / "pool.json",
            {
                "version": 1,
                "updated_at": _timestamp(),
                "learner_generation": self.learner_generation,
                "learner_elo": round(self.learner_elo, 3),
                "snapshot_pool_size": self.snapshot_pool_size,
                "active_historical_limit": self.active_historical_limit,
                "ratios": {
                    "random": float(self.ratios[0]),
                    "current": float(self.ratios[1]),
                    "historical": float(self.ratios[2]),
                },
                "aggregate": dict(self.aggregate),
                "recent_score_rate_100": (
                    float(np.mean(self.recent_results)) if self.recent_results else None
                ),
                "ratings": {
                    key: round(float(value), 3) for key, value in self.ratings.items()
                },
                "snapshots": snapshots,
                "matchups": matchups,
            },
        )

    @staticmethod
    def _empty_mps_cache():
        if torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def close(self):
        self.current_model = None
        self.assignments.clear()
        self._loaded_models.clear()
        if self._cleanup_retired():
            self._persist()
        self._empty_mps_cache()


class SelfPlaySnapshotCallback(BaseCallback):
    """Publish immutable learner snapshots after completed PPO rounds."""

    def __init__(self, pool, rollout_decisions, every_rounds, final_round):
        super().__init__(verbose=0)
        self.pool = pool
        self.rollout_decisions = int(rollout_decisions)
        self.every_rounds = max(1, int(every_rounds))
        self.final_round = int(final_round)
        self.saved_rounds = set()

    def save_snapshot(self, round_number):
        round_number = int(round_number)
        if round_number in self.saved_rounds:
            return
        snapshot_id = f"snapshot_round_{round_number:06d}"
        path = self.pool.snapshots_dir / f"{snapshot_id}.zip"
        self.model.save(path.with_suffix(""))
        self.pool.register_snapshot(snapshot_id, path, round_number)
        self.saved_rounds.add(round_number)
        print(
            f"[self-play-snapshot] round={round_number} id={snapshot_id} path={path}",
            flush=True,
        )

    def _on_rollout_start(self):
        completed = int(self.num_timesteps // self.rollout_decisions)
        self.pool.set_learner_generation(completed)
        if completed > 0 and completed % self.every_rounds == 0:
            self.save_snapshot(completed)

    def _on_step(self):
        return True

    def _on_training_end(self):
        if not completed_update_at_training_end(self.model, self.rollout_decisions):
            return
        completed = int(self.num_timesteps // self.rollout_decisions)
        self.pool.set_learner_generation(completed)
        self.save_snapshot(completed)


__all__ = ["SelfPlayOpponentPool", "SelfPlaySnapshotCallback"]
