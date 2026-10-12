"""Deterministic, training-independent evaluation schedule for native policies."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from policy_contracts import validated_actions


EVALUATION_COLUMNS = (
    "timestamp",
    "round",
    "timesteps",
    "opponent_id",
    "games",
    "completed_games",
    "wins",
    "losses",
    "draws",
    "win_rate",
    "score_rate",
    "mean_crown_diff",
    "mean_tower_hp_diff",
    "mean_episode_reward",
    "timed_out",
    "elapsed_seconds",
    "valid",
    "invalid_actions",
    "action_decisions",
    "invalid_action_rate",
    "mean_prediction_seconds", "mean_search_transitions", "max_search_depth",
)


def checkpoint_spec(path, algorithm, expected_sha256=None):
    path = Path(path).resolve()
    from training_runtime import _sha256

    digest = _sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"checkpoint hash changed: {path}")
    return {"path": str(path), "algorithm": str(algorithm), "sha256": digest}


def initial_policy_path(config, checkpoint):
    """Resolve the original baseline, never substitute the evaluated learner."""
    from training_runtime import resume_directory

    if "initial_policy" not in config["evaluation_opponents"]:
        return Path(checkpoint)
    explicit = config.get("evaluation_initial_policy_path")
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise ValueError("explicit initial-policy baseline is missing")
        return path
    sidecar = resume_directory(checkpoint)
    runtime = sidecar / "runtime.json"
    expected = json.loads(runtime.read_text()).get("initial_policy_sha256") if runtime.is_file() else None
    baseline = sidecar / "initial_policy.zip"
    if baseline.is_file():
        checkpoint_spec(baseline, config["algorithm"], expected)
        return baseline
    checkpoint = Path(checkpoint).resolve()
    candidates = [config.get("_initial_policy_path"),
                  checkpoint.parent / "evaluation/baselines/initial_policy.zip",
                  checkpoint.parent.parent / "evaluation/baselines/initial_policy.zip"]
    for candidate in candidates:
        if candidate is not None and Path(candidate).is_file():
            checkpoint_spec(candidate, config["algorithm"], expected)
            return Path(candidate)
    raise ValueError("original initial-policy baseline not found; provide evaluation_initial_policy_path")


@contextmanager
def preserve_random_state():
    """Loading opponents must not change PPO's sampling/minibatch RNG stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _record_evaluation_failure(root, round_number, error):
    try:
        _atomic_json(root / "status.json", {
            "status": "failed", "updated_at": _timestamp(), "round": round_number,
            "error_type": type(error).__name__, "error": str(error),
        })
    except OSError as status_error:
        # A full/read-only filesystem may prevent even the failure marker.
        # Keep the original exception, rather than hiding it with this write.
        error.add_note(f"could not write evaluation failure status: {status_error}")


class FixedPolicyOpponent:
    """Expose one frozen recurrent policy as a NativeBatchVecEnv opponent."""

    def __init__(self, model, battles, opponent_id="initial_policy"):
        self.model = model
        self.battles = int(battles)
        self.opponent_id = str(opponent_id)
        self.states = {}

    def on_episode_start(self, slot_index, episode, seed, learner_role):
        self.states[int(slot_index)] = {
            "state": None,
            "episode_start": True,
        }

    def opponent_metadata(self, slot_index):
        return {
            "opponent_id": self.opponent_id,
            "opponent_category": "fixed_evaluation",
        }

    def predict(self, observations, deterministic=True):
        count = int(next(iter(observations.values())).shape[0])
        if count != self.battles:
            raise ValueError(
                f"fixed opponent expected {self.battles} observations, got {count}"
            )
        slots = [self.states.setdefault(index, {
            "state": None, "episode_start": True,
        }) for index in range(count)]
        state = self._stack_states([slot["state"] for slot in slots])
        episode_start = np.asarray(
            [slot["episode_start"] for slot in slots], dtype=bool
        )
        actions, next_state = self.model.predict(
            observations,
            state=state,
            episode_start=episode_start,
            deterministic=True,
        )
        actions = validated_actions(actions, count, "fixed opponent")
        for index, slot in enumerate(slots):
            slot["episode_start"] = False
            slot["state"] = self._slice_state(next_state, index)
        return actions, None

    @staticmethod
    def _stack_states(states):
        template = next((state for state in states if state is not None), None)
        if template is None:
            return None
        return tuple(
            np.concatenate([
                state[component]
                if state is not None
                else np.zeros_like(template[component][:, :1, :])
                for state in states
            ], axis=1)
            for component in range(len(template))
        )

    @staticmethod
    def _slice_state(state, index):
        if state is None:
            return None
        return tuple(
            component[:, index : index + 1, :].copy()
            for component in state
        )

    def close(self):
        # The runner owns and reuses the frozen model across schedule waves.
        self.states.clear()


def _terminal_metrics(info):
    crown_diff = info.get("crown_diff")
    tower_diff = info.get("tower_hp_diff")
    terminal = info.get("terminal_observation")
    if isinstance(terminal, dict) and "scalar" in terminal:
        scalar = np.asarray(terminal["scalar"], dtype=np.float32)
        if scalar.size >= 11:
            if crown_diff is None:
                crown_diff = float((scalar[2] - scalar[3]) * 3.0)
            if tower_diff is None:
                tower_diff = float(scalar[9] - scalar[10])
    return crown_diff, tower_diff


def _summarize(matches, round_number, timesteps, opponent_id, elapsed):
    games = len(matches)
    completed = [match for match in matches if not match.get("timed_out")]
    wins = sum(match["result"] == "win" for match in completed)
    losses = sum(match["result"] == "loss" for match in completed)
    draws = len(completed) - wins - losses
    scores = [1.0 if match["result"] == "win" else 0.0
              if match["result"] == "loss" else 0.5 for match in completed]

    def mean(name):
        values = [float(match[name]) for match in completed
                  if isinstance(match.get(name), (int, float))
                  and math.isfinite(float(match[name]))]
        return float(np.mean(values)) if values else None

    return {
        "timestamp": _timestamp(),
        "round": int(round_number),
        "timesteps": int(timesteps),
        "opponent_id": str(opponent_id),
        "games": games,
        "completed_games": len(completed),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / len(completed) if completed else None,
        "score_rate": float(np.mean(scores)) if scores else None,
        "mean_crown_diff": mean("crown_diff"),
        "mean_tower_hp_diff": mean("tower_hp_diff"),
        "mean_episode_reward": mean("episode_reward"),
        "timed_out": sum(bool(match.get("timed_out")) for match in matches),
        "elapsed_seconds": round(float(elapsed), 3),
        "valid": len(completed) == games and games > 0,
        "invalid_actions": sum(match.get("invalid_actions", 0) for match in matches),
        "action_decisions": sum(match["decisions"] for match in matches),
        "invalid_action_rate": (
            sum(match.get("invalid_actions", 0) for match in matches)
            / max(1, sum(match["decisions"] for match in matches))
        ),
        "mean_prediction_seconds": mean('prediction_seconds_per_decision'),
        "mean_search_transitions": mean('search_transitions_per_decision'),
        "max_search_depth": max((match.get('search_depth', 0) for match in matches), default=0),
    }


def _svg_chart(path, summaries):
    """Write the fixed-evaluation chart without adding plotting dependencies."""
    aggregate = [item for item in summaries if item["opponent_id"] == "aggregate"]
    width, height = 1200, 560
    left, right, top = 100, 35, 75
    panel_height = 185
    colors = {"score_rate": "#60a5fa", "win_rate": "#34d399",
              "mean_crown_diff": "#f59e0b", "mean_tower_hp_diff": "#f472b6"}
    panels = (
        ("fixed schedule rates", ("score_rate", "win_rate"), (0.0, 1.0)),
        ("board advantage", ("mean_crown_diff", "mean_tower_hp_diff"), None),
    )
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{left}" y="42" fill="#f8fafc" font-family="monospace" font-size="24">Fixed evaluation schedule</text>',
    ]
    for panel_index, (label, keys, fixed_range) in enumerate(panels):
        y0 = top + panel_index * 230
        y1 = y0 + panel_height
        chart_width = width - left - right
        svg.append(f'<rect x="{left}" y="{y0}" width="{chart_width}" height="{panel_height}" fill="#111827" stroke="#334155"/>')
        svg.append(f'<text x="18" y="{y0 + 18}" fill="#cbd5e1" font-family="monospace" font-size="14">{label}</text>')
        points = [(item["round"], key, item.get(key)) for item in aggregate
                  for key in keys if isinstance(item.get(key), (int, float))]
        if not points:
            svg.append(f'<text x="{left + 18}" y="{y0 + 95}" fill="#64748b" font-family="monospace" font-size="14">no evaluations yet</text>')
            continue
        rounds = [point[0] for point in points]
        x_min, x_max = min(rounds), max(rounds)
        if x_min == x_max:
            x_max = x_min + 1
        if fixed_range is None:
            values = [float(point[2]) for point in points]
            y_min, y_max = min(values), max(values)
            if y_min == y_max:
                padding = max(abs(y_min) * 0.1, 0.1)
                y_min, y_max = y_min - padding, y_max + padding
        else:
            y_min, y_max = fixed_range
        for grid in range(5):
            y = y0 + grid * panel_height / 4
            value = y_max - grid * (y_max - y_min) / 4
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#1e293b"/>')
            svg.append(f'<text x="{left-8}" y="{y+5:.1f}" text-anchor="end" fill="#64748b" font-family="monospace" font-size="11">{value:.3g}</text>')
        for key_index, key in enumerate(keys):
            series = [(item["round"], float(item[key])) for item in aggregate
                      if isinstance(item.get(key), (int, float))]
            coords = []
            for round_value, value in series:
                x = left + (round_value - x_min) / (x_max - x_min) * chart_width
                y = y1 - (value - y_min) / (y_max - y_min) * panel_height
                coords.append(f"{x:.1f},{y:.1f}")
            if coords:
                color = colors[key]
                svg.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
                if len(coords) == 1:
                    x, y = coords[0].split(",")
                    svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
                svg.append(f'<text x="{left + key_index * 300}" y="{y1 + 25}" fill="{color}" font-family="monospace" font-size="12">{key}</text>')
        svg.append(f'<text x="{width-right}" y="{y1 + 25}" text-anchor="end" fill="#64748b" font-family="monospace" font-size="11">round {min(rounds)}..{max(rounds)}</text>')
    svg.append("</svg>")
    Path(path).write_text("\n".join(svg) + "\n", encoding="utf-8")


class FixedEvaluationCallback(BaseCallback):
    """Run an immutable seed/role/opponent schedule outside the PPO rollout."""

    def __init__(
        self,
        run_dir,
        env_class,
        algorithm_class,
        env_kwargs,
        rollout_decisions,
        every_rounds,
        seed_base,
        seed_count,
        battles,
        max_decisions,
        opponents,
        initial_policy_path,
        device,
        frozen_opponents=None,
        restore_training=True,
        deferred=False,
    ):
        super().__init__(verbose=0)
        self.root = Path(run_dir).resolve() / "evaluation"
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "baselines").mkdir(exist_ok=True)
        self.env_class = env_class
        self.algorithm_class = algorithm_class
        self.env_kwargs = dict(env_kwargs)
        self.rollout_decisions = int(rollout_decisions)
        self.every_rounds = int(every_rounds)
        self.seed_base = int(seed_base)
        self.seed_count = int(seed_count)
        self.battles = int(battles)
        self.max_decisions = int(max_decisions)
        if min(self.rollout_decisions, self.every_rounds, self.seed_count, self.max_decisions) <= 0 or self.seed_base < 0:
            raise ValueError("evaluation counts must be positive and seed_base nonnegative")
        if not 1 <= self.battles <= 16:
            raise ValueError("evaluation battles must be between 1 and 16")
        self.opponents = tuple(str(value) for value in opponents)
        self.initial_policy_path = Path(initial_policy_path).resolve()
        self.device = str(device)
        self.restore_training = bool(restore_training)
        self.deferred = bool(deferred)
        if self.deferred and self.restore_training:
            raise ValueError("deferred evaluation must not reset training environments")
        self.pending_checkpoints = {}
        self.checkpoint_identity = None
        self.elapsed_seconds = 0.0
        self.evaluated_rounds = set()
        self.summaries = []
        self._initial_model = None
        self._initial_spec = (
            checkpoint_spec(self.initial_policy_path,
                            f"{algorithm_class.__module__}:{algorithm_class.__name__}")
            if "initial_policy" in self.opponents else None
        )
        self.frozen_opponents = {
            name: checkpoint_spec(spec["path"], spec["algorithm"], spec.get("sha256"))
            for name, spec in (frozen_opponents or {}).items()
        }
        self._frozen_models = {}
        allowed = {"random", "initial_policy", *self.frozen_opponents}
        if not self.opponents or len(set(self.opponents)) != len(self.opponents):
            raise ValueError("evaluation opponents must be nonempty and unique")
        if any(name not in allowed for name in self.opponents):
            raise ValueError("unknown evaluation opponent")
        self._latest = {}
        self.matches_path = self.root / "matches.jsonl"
        self.csv_path = self.root / "summary.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS).writeheader()
        schedule = {
            "version": 2,
            "seed_base": self.seed_base,
            "seeds": [self.seed_base + index for index in range(self.seed_count)],
            "roles_per_seed": [0, 1],
            "opponents": list(self.opponents),
            "games_per_opponent": self.seed_count * 2,
            "deterministic_learner": True,
            "deterministic_frozen_policy": True,
            "every_rounds": self.every_rounds,
            "max_decisions_per_game": self.max_decisions,
            "frozen_opponents": self.frozen_opponents,
            "initial_policy": self._initial_spec,
            "environment": self.env_kwargs,
            "battles": self.battles,
            "timeout_policy": "incomplete; excluded from rates; invalid schedule",
            "execution": "after_training" if self.deferred else "immediate",
        }
        _atomic_json(self.root / "schedule.json", schedule)

    def metrics(self):
        return dict(self._latest)

    def summary(self):
        aggregate = next(
            (item for item in reversed(self.summaries)
             if item["opponent_id"] == "aggregate"),
            None,
        )
        return {
            "schedule": str(self.root / "schedule.json"),
            "matches": str(self.matches_path),
            "summary_csv": str(self.csv_path),
            "latest": aggregate,
            "evaluated_rounds": sorted(self.evaluated_rounds),
        }

    def _opponent(self, opponent_id, battles):
        from training_runtime import load_object, load_inference_model

        if opponent_id == "random":
            return None
        if opponent_id in self.frozen_opponents:
            spec = self.frozen_opponents[opponent_id]
            checkpoint_spec(spec["path"], spec["algorithm"], spec["sha256"])
            if opponent_id not in self._frozen_models:
                self._frozen_models[opponent_id] = load_inference_model(
                    load_object(spec["algorithm"]), spec["path"], self.device
                )
            return FixedPolicyOpponent(self._frozen_models[opponent_id], battles, opponent_id)
        if opponent_id != "initial_policy":
            raise ValueError(f"unknown fixed evaluation opponent {opponent_id!r}")
        if self._initial_model is None:
            checkpoint_spec(self.initial_policy_path, self._initial_spec["algorithm"], self._initial_spec["sha256"])
            self._initial_model = load_inference_model(
                self.algorithm_class, self.initial_policy_path, self.device
            )
        return FixedPolicyOpponent(
            self._initial_model, battles, opponent_id=opponent_id
        )

    @staticmethod
    def _result(winner, learner_role):
        if winner is None or int(winner) not in (0, 1):
            return "draw"
        return "win" if int(winner) == int(learner_role) else "loss"

    def _run_wave(self, round_number, opponent_id, role, seeds):
        count = len(seeds)
        opponent = self._opponent(opponent_id, count)
        kwargs = dict(self.env_kwargs)
        kwargs.update(
            battles=count,
            seed=int(seeds[0]),
            learner_roles=[int(role)] * count,
        )
        if opponent is not None:
            kwargs["opponent_policy"] = opponent
        else:
            kwargs.pop("opponent_policy", None)
        env = self.env_class(**kwargs)
        state = None
        episode_starts = np.ones(count, dtype=bool)
        rewards = np.zeros(count, dtype=np.float64)
        invalid_actions = np.zeros(count, dtype=np.int64)
        prediction_seconds = np.zeros(count, dtype=np.float64)
        search_transitions = np.zeros(count, dtype=np.int64)
        search_depth = np.zeros(count, dtype=np.int64)
        pending = set(range(count))
        matches = []
        last_infos = [{} for _ in range(count)]
        try:
            observations = env.reset()
            for decision in range(1, self.max_decisions + 1):
                prediction_started = time.monotonic()
                actions, state = self.model.predict(
                    observations,
                    state=state,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                prediction_elapsed = time.monotonic() - prediction_started
                diagnostics = getattr(self.model, 'last_search', [])
                for slot in pending:
                    prediction_seconds[slot] += prediction_elapsed / count
                    if len(diagnostics) == count:
                        search_transitions[slot] += diagnostics[slot]['transitions']
                        search_depth[slot] = max(search_depth[slot], diagnostics[slot]['depth'])
                observations, step_rewards, dones, infos = env.step(actions)
                rewards += np.asarray(step_rewards, dtype=np.float64)
                episode_starts = np.asarray(dones, dtype=bool)
                last_infos = infos
                for slot in tuple(pending):
                    if infos[slot].get("reward_components", {}).get("invalid", 0) < 0:
                        invalid_actions[slot] += 1
                    if not bool(dones[slot]):
                        continue
                    info = infos[slot]
                    crown_diff, tower_diff = _terminal_metrics(info)
                    match = {
                        "timestamp": _timestamp(),
                        "round": int(round_number),
                        "timesteps": int(self.num_timesteps),
                        "opponent_id": opponent_id,
                        "seed": int(seeds[slot]),
                        "learner_role": int(role),
                        "winner": info.get("winner"),
                        "result": self._result(info.get("winner"), role),
                        "crown_diff": crown_diff,
                        "tower_hp_diff": tower_diff,
                        "episode_reward": float(rewards[slot]),
                        "decisions": decision,
                        "invalid_actions": int(invalid_actions[slot]),
                        "timed_out": False,
                        'prediction_seconds_per_decision': float(prediction_seconds[slot] / decision),
                        'search_transitions_per_decision': float(search_transitions[slot] / decision),
                        'search_depth': int(search_depth[slot]),
                    }
                    matches.append(match)
                    pending.remove(slot)
                if not pending:
                    break
            for slot in sorted(pending):
                info = last_infos[slot]
                crown_diff, tower_diff = _terminal_metrics(info)
                matches.append({
                    "timestamp": _timestamp(),
                    "round": int(round_number),
                    "timesteps": int(self.num_timesteps),
                    "opponent_id": opponent_id,
                    "seed": int(seeds[slot]),
                    "learner_role": int(role),
                    "winner": None,
                    "result": "incomplete",
                    "crown_diff": crown_diff,
                    "tower_hp_diff": tower_diff,
                    "episode_reward": float(rewards[slot]),
                    "decisions": self.max_decisions,
                    "invalid_actions": int(invalid_actions[slot]),
                    "timed_out": True,
                    'prediction_seconds_per_decision': float(prediction_seconds[slot] / self.max_decisions),
                    'search_transitions_per_decision': float(search_transitions[slot] / self.max_decisions),
                    'search_depth': int(search_depth[slot]),
                })
        finally:
            env.close()
        return matches

    def _reset_training_environment(self):
        observations = self.training_env.reset()
        self.model._last_obs = observations
        self.model._last_episode_starts = np.ones(
            self.training_env.num_envs, dtype=bool
        )
        recurrent = getattr(self.model, "_last_lstm_states", None)
        if recurrent is not None:
            def zeros_like(value):
                if torch.is_tensor(value):
                    return torch.zeros_like(value)
                return np.zeros_like(value)

            self.model._last_lstm_states = type(recurrent)(
                tuple(zeros_like(value) for value in recurrent.pi),
                tuple(zeros_like(value) for value in recurrent.vf),
            )
        normalize = getattr(self.model, "_vec_normalize_env", None)
        if normalize is not None:
            self.model._last_original_obs = normalize.get_original_obs()

    def evaluate(self, round_number):
        try:
            with preserve_random_state():
                self._evaluate(round_number)
        except BaseException as error:
            _record_evaluation_failure(self.root, round_number, error)
            raise

    def _evaluate(self, round_number):
        round_number = int(round_number)
        if round_number in self.evaluated_rounds:
            return
        started = time.monotonic()
        print(
            f"[fixed-eval-start] round={round_number} opponents={','.join(self.opponents)} "
            f"seeds={self.seed_count} games={len(self.opponents) * self.seed_count * 2}",
            flush=True,
        )
        _atomic_json(self.root / "status.json", {
            "status": "running",
            "updated_at": _timestamp(),
            "round": round_number,
            "planned_games": len(self.opponents) * self.seed_count * 2,
        })
        # The native server has a single resident data-plane connection. Release
        # it before evaluation; reset below rebuilds all training slots afterward.
        if self.restore_training:
            try:
                self.training_env.env_method("release_batch")
            except (AttributeError, NotImplementedError):
                pass
        all_matches = []
        try:
            seeds = [self.seed_base + index for index in range(self.seed_count)]
            for opponent_id in self.opponents:
                for role in (0, 1):
                    for start in range(0, len(seeds), self.battles):
                        all_matches.extend(self._run_wave(
                            round_number,
                            opponent_id,
                            role,
                            seeds[start : start + self.battles],
                        ))
        finally:
            if self.restore_training:
                self._reset_training_environment()

        with self.matches_path.open("a", encoding="utf-8") as stream:
            for match in all_matches:
                stream.write(json.dumps(match, ensure_ascii=False, sort_keys=True) + "\n")
        elapsed = time.monotonic() - started
        self.elapsed_seconds += elapsed
        summaries = []
        for opponent_id in self.opponents:
            selected = [match for match in all_matches
                        if match["opponent_id"] == opponent_id]
            summaries.append(_summarize(
                selected, round_number, self.num_timesteps, opponent_id, elapsed
            ))
            for role in (0, 1):
                summaries.append(_summarize(
                    [match for match in selected if match["learner_role"] == role],
                    round_number, self.num_timesteps, f"{opponent_id}/p{role}", elapsed,
                ))
        summaries.append(_summarize(
            all_matches, round_number, self.num_timesteps, "aggregate", elapsed
        ))
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVALUATION_COLUMNS)
            writer.writerows(summaries)
        all_summaries = [*self.summaries, *summaries]
        aggregate = summaries[-1]
        latest = {
            "eval/round": round_number,
            "eval/games": aggregate["games"],
            "eval/win_rate": aggregate["win_rate"],
            "eval/score_rate": aggregate["score_rate"],
            "eval/crown_diff": aggregate["mean_crown_diff"],
            "eval/tower_hp_diff": aggregate["mean_tower_hp_diff"],
            "eval/timed_out": aggregate["timed_out"],
            "eval/total_elapsed_seconds": self.elapsed_seconds,
        }
        _atomic_json(self.root / "latest.json", {
            "updated_at": _timestamp(),
            "aggregate": aggregate,
            "by_opponent": summaries[:-1],
            "evaluated_rounds": sorted(self.evaluated_rounds | {round_number}),
        })
        charts = self.root.parent / "charts"
        charts.mkdir(exist_ok=True)
        _svg_chart(charts / "fixed_evaluation.svg", all_summaries)
        if self.checkpoint_identity is not None:
            spec = self.checkpoint_identity
            checkpoint_spec(spec["path"], spec["algorithm"], spec["sha256"])
        # Completion is the commit marker for every required output above.
        _atomic_json(self.root / "status.json", {
            "status": "completed",
            "updated_at": _timestamp(),
            "round": round_number,
            "timesteps": int(self.num_timesteps),
            "games": aggregate["games"],
            "score_rate": aggregate["score_rate"],
            "valid": aggregate["valid"],
            "timed_out": aggregate["timed_out"],
        })
        self.summaries = all_summaries
        self.evaluated_rounds.add(round_number)
        self._latest = latest
        print(
            f"[fixed-eval-complete] round={round_number} games={aggregate['games']} "
            f"win_rate={aggregate['win_rate']} "
            f"score_rate={aggregate['score_rate']} timed_out={aggregate['timed_out']} "
            f"duration={elapsed:.2f}s",
            flush=True,
        )

    def _on_training_start(self):
        self._schedule(int(self.model.num_timesteps // self.rollout_decisions))

    def _schedule(self, round_number):
        if self.deferred:
            directory = self.root / "checkpoints"
            directory.mkdir(exist_ok=True)
            path = directory / f"round_{round_number:06d}.zip"
            self.model.save(path)
            self.pending_checkpoints[round_number] = path
        else:
            self.evaluate(round_number)

    def _on_rollout_start(self):
        completed = int(self.num_timesteps // self.rollout_decisions)
        if completed > 0 and completed % self.every_rounds == 0:
            self._schedule(completed)

    def _on_step(self):
        return True

    def _on_training_end(self):
        from training_runtime import load_inference_model, completed_update_at_training_end

        if not completed_update_at_training_end(self.model, self.rollout_decisions):
            return

        completed = int(self.num_timesteps // self.rollout_decisions)
        self._schedule(completed)
        if self.deferred:
            original_model, original_timesteps = self.model, self.num_timesteps
            # No training follows this callback. Release the data plane once;
            # do not reset environments, league assignments, or recurrent memory.
            try:
                original_model.get_env().env_method("release_batch")
            except (AttributeError, NotImplementedError):
                pass
            try:
                with preserve_random_state():
                    for round_number, checkpoint in sorted(self.pending_checkpoints.items()):
                        self.model = load_inference_model(self.algorithm_class, checkpoint, self.device)
                        self.num_timesteps = self.model.num_timesteps
                        self.evaluate(round_number)
            finally:
                self.model, self.num_timesteps = original_model, original_timesteps


def evaluate_checkpoint(config, checkpoint, output_dir):
    """Re-evaluate a saved model without collecting PPO data or updating weights."""
    from training_runtime import load_object, load_inference_model

    output_dir = Path(output_dir).resolve()
    if (output_dir / "evaluation").exists():
        raise ValueError("use a fresh output directory for checkpoint evaluation")
    baseline = initial_policy_path(config, checkpoint)
    identity = checkpoint_spec(checkpoint, config["algorithm"])
    algorithm = load_object(config["algorithm"])
    with preserve_random_state():
        model = load_inference_model(algorithm, checkpoint, config["device"])
        if config.get('planning_enabled'):
            from dataclasses import replace
            overrides = {key: config[f'evaluation_search_{key}'] for key in ('budget', 'depth')
                         if f'evaluation_search_{key}' in config}
            model.search_settings = replace(model.search_settings, **overrides)
        checkpoint_spec(checkpoint, config["algorithm"], identity["sha256"])
        env_kwargs = dict(config.get("environment_kwargs", {}))
        env_kwargs.update(port=int(config["port"]), decision_ticks=int(config["decision_ticks"]))
        callback = FixedEvaluationCallback(
            run_dir=output_dir,
            env_class=load_object(config["environment"]),
            algorithm_class=algorithm,
            env_kwargs=env_kwargs,
            rollout_decisions=int(config["rollout_decisions"]),
            every_rounds=int(config["evaluation_every_rounds"]),
            seed_base=int(config["evaluation_seed_base"]),
            seed_count=int(config["evaluation_seed_count"]),
            battles=int(config["evaluation_battles"]),
            max_decisions=int(config["evaluation_max_decisions"]),
            opponents=config["evaluation_opponents"],
            initial_policy_path=baseline,
            device=config["device"],
            frozen_opponents=config.get("evaluation_frozen_opponents"),
            restore_training=False,
        )
        callback.init_callback(model)
        if config.get('planning_enabled'):
            from dataclasses import asdict
            _atomic_json(callback.root / 'planning_search.json', asdict(model.search_settings))
        callback.num_timesteps = model.num_timesteps
        callback.checkpoint_identity = identity
        try:
            _atomic_json(output_dir / "evaluation" / "checkpoint.json", identity)
        except BaseException as error:
            _record_evaluation_failure(callback.root, model.num_timesteps // int(config["rollout_decisions"]), error)
            raise
        callback.evaluate(model.num_timesteps // int(config["rollout_decisions"]))
        result = callback.summary()
        result["checkpoint"] = identity
        return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on its saved fixed schedule")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument('--search-budget', type=int, help='planning learner budget; zero disables search')
    parser.add_argument('--search-depth', type=int, help='planning learner maximum depth')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.device:
        config["device"] = args.device
    for key in ('budget', 'depth'):
        value = getattr(args, f'search_{key}')
        if value is not None:
            if not config.get('planning_enabled'):
                parser.error('search overrides require a planning checkpoint')
            config[f'evaluation_search_{key}'] = value
    result = evaluate_checkpoint(config, args.checkpoint, args.output_dir)
    return 0 if result["latest"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FixedEvaluationCallback", "FixedPolicyOpponent"]
