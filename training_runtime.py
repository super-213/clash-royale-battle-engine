"""Reusable SB3 training runtime with durable, agent-readable telemetry."""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor


METRIC_COLUMNS = (
    "phase",
    "round",
    "timesteps",
    "elapsed_seconds",
    "fps",
    "episodes",
    "episode_reward_mean_100",
    "episode_length_mean_100",
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/learning_rate",
    "train/tbptt_steps",
    "train/recurrent_sequence_max",
    "train/auxiliary_loss",
    "train/aux_tower_delta_loss",
    "train/aux_entity_death_loss",
    "train/aux_entity_position_loss",
    "train/aux_damage_source_loss",
    "train/aux_opponent_elixir_loss",
    "train/aux_opponent_hand_loss",
    "train/aux_opponent_next_card_loss",
    "train/aux_future_value_loss",
    "mps_allocated_mb",
    "mps_driver_allocated_mb",
    "self_play/games",
    "self_play/win_rate",
    "self_play/win_rate_100",
    "self_play/score_rate_100",
    "self_play/learner_elo",
    "self_play/snapshot_count",
    "eval/round",
    "eval/games",
    "eval/win_rate",
    "eval/score_rate",
    "eval/crown_diff",
    "eval/tower_hp_diff",
    "eval/timed_out",
    "exploration/slot_entropy", "exploration/position_entropy",
    "exploration/wait_probability", "exploration/wait_rate",
    *(f"exploration/card_{card}_rate" for card in range(1, 9)),
    "observation/entity_truncation_rate",
    "session_timesteps",
)


def resume_directory(checkpoint):
    return Path(str(Path(checkpoint).resolve()) + ".state")


def save_resume_state(checkpoint, model, config, pool=None, telemetry=None):
    directory = resume_directory(checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / "config.json", config)
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        rng["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_initialized():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, directory / "rng.pt")
    state = {"timesteps": model.num_timesteps, "model_sha256": _sha256(checkpoint)}
    baseline = config.get("_initial_policy_path")
    if baseline and Path(baseline).is_file():
        shutil.copyfile(baseline, directory / "initial_policy.zip")
        state["initial_policy_sha256"] = _sha256(directory / "initial_policy.zip")
    try:
        state["next_native_seed"] = model.get_env().get_attr("_seed_cursor")[0]
    except AttributeError:
        pass
    if telemetry is not None:
        state.update(episodes=telemetry.episode_count,
                     episode_rewards=list(telemetry.episode_rewards),
                     episode_lengths=list(telemetry.episode_lengths))
    _atomic_json(directory / "runtime.json", state)
    if pool is not None:
        pool.save_state(directory / "league")


def restore_resume_state(checkpoint, env, pool=None, telemetry=None):
    directory = resume_directory(checkpoint)
    saved = json.loads((directory / "runtime.json").read_text())
    if saved["model_sha256"] != _sha256(checkpoint):
        raise ValueError("resume model no longer matches its saved runtime state")
    if "next_native_seed" in saved:
        env.seed(saved["next_native_seed"])
    if pool is not None:
        pool.restore_state(directory / "league")
    if telemetry is not None:
        telemetry.episode_count = saved.get("episodes", 0)
        telemetry.episode_rewards.extend(saved.get("episode_rewards", []))
        telemetry.episode_lengths.extend(saved.get("episode_lengths", []))
    rng = torch.load(directory / "rng.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "mps" in rng and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])
    if "cuda" in rng and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(rng["cuda"])


class UpdatedCheckpointCallback(BaseCallback):
    """Save after PPO updates, at the next rollout boundary or training end."""

    def __init__(self, run_dir, config, pool=None):
        super().__init__()
        self.root, self.config, self.pool = Path(run_dir), config, pool
        self.telemetry = None
        self.saved_rounds = set()

    def _save_completed(self, force=False):
        completed = self.model.num_timesteps // int(self.config["rollout_decisions"])
        if completed <= 0 or completed in self.saved_rounds:
            return
        if not force and completed % int(self.config["checkpoint_every_rounds"]):
            return
        directory = self.root / "checkpoints"
        directory.mkdir(exist_ok=True)
        path = directory / f"model_{self.model.num_timesteps}_steps.zip"
        self.model.save(path)
        save_resume_state(path, self.model, self.config, self.pool, self.telemetry)
        self.saved_rounds.add(completed)

    def _on_rollout_start(self):
        self._save_completed()

    def _on_training_end(self):
        # Runs before deferred evaluation, so an evaluation failure cannot lose
        # the final completed optimizer update or its resumable league state.
        self._save_completed(force=True)

    def _on_step(self):
        return True


def utc_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_object(reference):
    """Load ``package.module:attribute`` without coupling the launcher to a model."""
    if ":" not in reference:
        raise ValueError(f"object reference must be module:attribute, got {reference!r}")
    module_name, attribute_name = reference.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise ImportError(f"{reference!r} does not exist") from error


def load_inference_model(algorithm_class, checkpoint, device):
    """Load a prediction-only model without retaining its training rollout.

    SB3 reconstructs rollout buffers on load even without an environment. A
    historical opponent needs neither a full rollout nor the saved optimizer
    moments. Resuming training must use the ordinary algorithm.load instead.
    """
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
    from fixed_evaluation import preserve_random_state

    kwargs = {"n_steps": 1} if issubclass(algorithm_class, OnPolicyAlgorithm) else {}
    with preserve_random_state():
        model = algorithm_class.load(checkpoint, device=device, **kwargs)
    if isinstance(model, OnPolicyAlgorithm):
        model.rollout_buffer = None
        model.policy.optimizer.state.clear()
    return model


def require_training_device(device):
    """Resolve a supported device and never silently demote MPS to CPU."""
    requested = str(device).lower()
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("This PyTorch build has no MPS support")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this process")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError("device must be 'mps' or 'cpu'")


def _json_value(value):
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class TrainingTelemetryCallback(BaseCallback):
    """Persist live PPO telemetry independently of terminal output."""

    def __init__(
        self,
        run_dir,
        rollout_decisions,
        planned_rounds,
        metrics_provider=None,
        verbose=0,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.rollout_decisions = int(rollout_decisions)
        self.planned_rounds = int(planned_rounds)
        self.started = time.monotonic()
        self.records = []
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_count = 0
        self.metrics_provider = metrics_provider
        self.exploration = []
        self.truncation = []
        self.last_rollout_metrics = {}
        self.last_recorded_timesteps = 0
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=METRIC_COLUMNS).writeheader()

    def _on_step(self):
        diagnostics = getattr(self.model.policy, "_rollout_diagnostics", None)
        if diagnostics is not None:
            self.exploration.append(diagnostics.cpu().numpy())
        for info in self.locals.get("infos", ()):
            if "entity_truncated" in info:
                self.truncation.append(float(info["entity_truncated"]))
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode:
                self.episode_count += 1
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(float(episode["l"]))
        return True

    def _logger_values(self):
        values = {}
        for name, value in self.model.logger.name_to_value.items():
            if name.startswith(("train/", "rollout/", "time/")):
                values[name] = _json_value(value)
        return values

    def _record(self, phase):
        elapsed = max(time.monotonic() - self.started, 1e-9)
        timesteps = int(self.num_timesteps)
        session_timesteps = timesteps - getattr(self, "initial_timesteps", 0)
        record = {
            "timestamp": utc_now(),
            "phase": phase,
            "round": int(math.ceil(timesteps / self.rollout_decisions)),
            "planned_rounds": self.planned_rounds,
            "timesteps": timesteps,
            "session_timesteps": session_timesteps,
            "elapsed_seconds": round(elapsed, 3),
            "fps": round(session_timesteps / elapsed, 3),
            "episodes": self.episode_count,
            "episode_reward_mean_100": (
                float(np.mean(self.episode_rewards)) if self.episode_rewards else None
            ),
            "episode_length_mean_100": (
                float(np.mean(self.episode_lengths)) if self.episode_lengths else None
            ),
        }
        if str(self.model.device).startswith("mps"):
            if hasattr(torch.mps, "current_allocated_memory"):
                record["mps_allocated_mb"] = round(
                    torch.mps.current_allocated_memory() / (1024 * 1024), 3
                )
            if hasattr(torch.mps, "driver_allocated_memory"):
                record["mps_driver_allocated_mb"] = round(
                    torch.mps.driver_allocated_memory() / (1024 * 1024), 3
                )
        if self.metrics_provider is not None:
            record.update(self.metrics_provider())
            record["fps"] = round(session_timesteps / max(elapsed - record.get("eval/total_elapsed_seconds", 0), 1e-9), 3)
        record.update(self.last_rollout_metrics)
        record.update(self._logger_values())
        self.records.append(record)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
            writer.writerow({column: record.get(column) for column in METRIC_COLUMNS})
        status = {
            "status": "running" if phase != "training_end" else "finalizing",
            "updated_at": record["timestamp"],
            "current_round": record["round"],
            "planned_rounds": self.planned_rounds,
            "progress_percent": round(
                min(100.0, 100.0 * record["round"] / max(self.planned_rounds, 1)), 2
            ),
            "latest_metrics": record,
            "paths": {
                "console_log": str(self.run_dir / "console.log"),
                "metrics_jsonl": str(self.jsonl_path),
                "metrics_csv": str(self.csv_path),
            },
        }
        _atomic_json(self.run_dir / "status.json", status)
        loss = record.get("train/loss")
        reward = record.get("episode_reward_mean_100")
        self_play = ""
        if record.get("self_play/games") is not None:
            self_play = (
                f" selfplay_games={record['self_play/games']} "
                f"win100={record.get('self_play/win_rate_100')} "
                f"elo={record.get('self_play/learner_elo'):.1f}"
            )
        evaluation = ""
        if record.get("eval/games") is not None:
            evaluation = (
                f" fixed_eval_round={record.get('eval/round')} "
                f"games={record['eval/games']} "
                f"score={record.get('eval/score_rate')}"
            )
        print(
            "[training-status] "
            f"round={record['round']}/{self.planned_rounds} "
            f"timesteps={timesteps} fps={record['fps']:.1f} "
            f"episodes={self.episode_count} "
            f"reward100={reward if reward is not None else 'n/a'} "
            f"loss={loss if loss is not None else 'pending'}"
            f"{self_play}{evaluation}",
            flush=True,
        )

    def _on_rollout_end(self):
        self.last_rollout_metrics = {}
        if self.exploration:
            values = np.concatenate(self.exploration).mean(axis=0)
            names = ("slot_entropy", "position_entropy", "wait_probability", "wait_rate",
                     *(f"card_{card}_rate" for card in range(1, 9)))
            self.last_rollout_metrics.update({f"exploration/{key}": float(value)
                                             for key, value in zip(names, values)})
        if self.truncation:
            self.last_rollout_metrics["observation/entity_truncation_rate"] = float(np.mean(self.truncation))
        self.exploration.clear()
        self.truncation.clear()

    def _on_rollout_start(self):
        if self.model.num_timesteps > self.last_recorded_timesteps:
            self._record("update_end")
            self.last_recorded_timesteps = self.model.num_timesteps

    def _on_training_start(self):
        self.initial_timesteps = self.model.num_timesteps
        self.last_recorded_timesteps = self.model.num_timesteps

    def _on_training_end(self):
        self._record("training_end")


def _finite_points(records, key):
    points = []
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            points.append((float(record["timesteps"]), float(value)))
    return points


def write_chart_svg(path, title, records, panels):
    """Write a dependency-free SVG dashboard with one scale per metric."""
    width = 1200
    panel_height = 220
    margin_left = 105
    margin_right = 35
    chart_width = width - margin_left - margin_right
    height = 85 + panel_height * len(panels)
    colors = ("#60a5fa", "#34d399", "#f59e0b", "#f472b6")
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{margin_left}" y="42" fill="#f8fafc" font-family="monospace" font-size="24">{escape(title)}</text>',
    ]
    for panel_index, (label, keys) in enumerate(panels):
        top = 70 + panel_index * panel_height
        bottom = top + 165
        all_points = [point for key in keys for point in _finite_points(records, key)]
        svg.append(
            f'<text x="18" y="{top + 18}" fill="#cbd5e1" font-family="monospace" font-size="14">{escape(label)}</text>'
        )
        svg.append(
            f'<rect x="{margin_left}" y="{top}" width="{chart_width}" height="165" fill="#111827" stroke="#334155"/>'
        )
        if not all_points:
            svg.append(
                f'<text x="{margin_left + 18}" y="{top + 88}" fill="#64748b" font-family="monospace" font-size="14">no samples yet</text>'
            )
            continue
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            padding = max(abs(y_min) * 0.05, 1.0)
            y_min -= padding
            y_max += padding
        for grid in range(5):
            y = top + grid * 165 / 4
            value = y_max - grid * (y_max - y_min) / 4
            svg.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#1e293b"/>')
            svg.append(f'<text x="{margin_left-8}" y="{y+5:.1f}" text-anchor="end" fill="#64748b" font-family="monospace" font-size="11">{value:.4g}</text>')
        for series_index, key in enumerate(keys):
            points = _finite_points(records, key)
            if not points:
                continue
            coordinates = []
            for x_value, y_value in points:
                x = margin_left + (x_value - x_min) / (x_max - x_min) * chart_width
                y = bottom - (y_value - y_min) / (y_max - y_min) * 165
                coordinates.append(f"{x:.1f},{y:.1f}")
            color = colors[series_index % len(colors)]
            svg.append(f'<polyline points="{" ".join(coordinates)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            if len(points) == 1:
                svg.append(f'<circle cx="{coordinates[0].split(",")[0]}" cy="{coordinates[0].split(",")[1]}" r="4" fill="{color}"/>')
            svg.append(f'<text x="{margin_left + series_index * 280}" y="{bottom + 25}" fill="{color}" font-family="monospace" font-size="12">{escape(key)}</text>')
        svg.append(f'<text x="{width-margin_right}" y="{bottom + 25}" text-anchor="end" fill="#64748b" font-family="monospace" font-size="11">timesteps {x_min:.0f}..{x_max:.0f}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def generate_charts(run_dir, records):
    charts = Path(run_dir) / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    write_chart_svg(charts / "exploration.svg", "Policy exploration", records, (
        ("raw entropy", ("exploration/slot_entropy", "exploration/position_entropy")),
        ("waiting", ("exploration/wait_probability", "exploration/wait_rate")),
        ("card use per learner decision", tuple(f"exploration/card_{card}_rate" for card in range(1, 9))),
        ("entity token truncation", ("observation/entity_truncation_rate",)),
    ))
    write_chart_svg(
        charts / "loss_curves.svg",
        "PPO loss metrics",
        records,
        (
            ("total loss", ("train/loss",)),
            ("policy loss", ("train/policy_gradient_loss",)),
            ("value loss", ("train/value_loss",)),
            ("entropy loss", ("train/entropy_loss",)),
        ),
    )
    write_chart_svg(
        charts / "auxiliary_losses.svg",
        "Auxiliary world-model losses",
        records,
        (
            ("combined", ("train/auxiliary_loss",)),
            ("next dynamics", (
                "train/aux_tower_delta_loss",
                "train/aux_damage_source_loss",
            )),
            ("entity dynamics", (
                "train/aux_entity_death_loss",
                "train/aux_entity_position_loss",
            )),
            ("opponent belief", (
                "train/aux_opponent_elixir_loss",
                "train/aux_opponent_hand_loss",
                "train/aux_opponent_next_card_loss",
            )),
            ("future scene value", ("train/aux_future_value_loss",)),
        ),
    )
    write_chart_svg(
        charts / "reward_curve.svg",
        "Episode quality",
        records,
        (
            ("mean reward (last 100)", ("episode_reward_mean_100",)),
            ("mean length (last 100)", ("episode_length_mean_100",)),
            ("explained variance", ("train/explained_variance",)),
        ),
    )
    write_chart_svg(
        charts / "training_speed.svg",
        "Training throughput and MPS memory",
        records,
        (
            ("decisions per second", ("fps",)),
            ("MPS allocated memory (MiB)", ("mps_allocated_mb",)),
            ("MPS driver memory (MiB)", ("mps_driver_allocated_mb",)),
        ),
    )
    write_chart_svg(
        charts / "optimization_diagnostics.svg",
        "PPO optimization diagnostics",
        records,
        (
            ("approximate KL", ("train/approx_kl",)),
            ("clip fraction", ("train/clip_fraction",)),
            ("learning rate", ("train/learning_rate",)),
        ),
    )
    write_chart_svg(
        charts / "self_play_progress.svg",
        "Self-play league progress",
        records,
        (
            ("win rate (last 100)", ("self_play/win_rate_100",)),
            ("score rate (last 100)", ("self_play/score_rate_100",)),
            ("learner Elo", ("self_play/learner_elo",)),
            ("historical snapshots", ("self_play/snapshot_count",)),
        ),
    )
    return tuple(sorted(charts.glob("*.svg")))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_metadata(model, model_path, config, elapsed, telemetry, status, source_sha256=None):
    parameter_count = sum(parameter.numel() for parameter in model.policy.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.policy.parameters() if parameter.requires_grad
    )
    extractor = getattr(model.policy, "features_extractor", None)
    metadata = {
        "status": status,
        "finished_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "elapsed_human": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        "planned_rounds": int(config["rounds"]),
        "completed_rounds": int(math.ceil(model.num_timesteps / config["rollout_decisions"])),
        "timesteps": int(model.num_timesteps),
        "resume_from": config.get("resume_from"),
        "episodes": telemetry.episode_count,
        "device": str(model.device),
        "algorithm": config["algorithm"],
        "policy": config["policy"],
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "architecture_version": getattr(extractor, "architecture_version", None),
        "observation_schema": getattr(extractor, "observation_schema", None),
        "battle_backend": getattr(extractor, "battle_backend", None),
        "model_file": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": _sha256(model_path),
        "source_sha256": source_sha256,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "dependencies": {name: importlib.metadata.version(name) for name in
                             ("stable-baselines3", "sb3-contrib", "gymnasium", "numpy")},
        },
    }
    training_env = model.get_env()
    deployment_rules = getattr(
        getattr(training_env, "unwrapped", training_env),
        "deployment_rules_version", None,
    )
    if deployment_rules is not None:
        metadata["deployment_rules_version"] = deployment_rules
    if config.get("recurrent_enabled"):
        metadata["recurrent"] = {
            "enabled": True,
            "tbptt_steps": int(config["tbptt_steps"]),
            "hidden_size": int(config["recurrent_hidden_size"]),
            "layers": int(config["recurrent_layers"]),
            "state_reset": "per VecEnv slot on episode_start",
        }
    if config.get("cross_modal_enabled"):
        metadata["cross_modal"] = {
            "enabled": True,
            "dimension": int(config["cross_modal_dim"]),
            "attention_heads": int(config["cross_modal_heads"]),
            "entity_layers": int(config["cross_modal_entity_layers"]),
            "interactions": [
                "hand queries entities/history/belief",
                "history queries current entities",
                "belief queries entities/history",
                "cards query spatial tokens through the action head",
                "card-conditioned placement logits",
            ],
        }
    if config.get("spatial_heatmap_enabled"):
        metadata["spatial_action_head"] = {
            "enabled": True,
            "feature_map": [32, 18],
            "backbone_channels": int(config["spatial_feature_channels"]),
            "feature_channels": int(config["cross_modal_dim"]),
            "conditional_distribution": "P(card) * P(tile | card)",
            "heatmaps_per_observation": 4,
        }
    if config.get("world_model_enabled"):
        action_conditioned = bool(getattr(model.policy, "action_conditioned", False))
        metadata["world_model"] = {
            "enabled": True,
            "action_conditioned": action_conditioned,
            "executed_action_schema": (
                "executed_card_xy_v1" if action_conditioned else None
            ),
            "dynamics_conditioning": (
                "state + accepted learner card identity + egocentric normalized x/y"
                if action_conditioned else "state only"
            ),
            "entity_target_schema": "native_entity_id_v1",
            "entity_association": "native ID lookup in the complete next state",
            "hidden_dimension": int(config["world_model_hidden_dim"]),
            "auxiliary_coefficient": float(config["auxiliary_coef"]),
            "future_horizon_steps": int(config["auxiliary_horizon"]),
            "future_horizon_seconds": round(
                int(config["auxiliary_horizon"])
                * int(config["decision_ticks"])
                * 0.05,
                3,
            ),
            "weights": dict(config["auxiliary_weights"]),
            "targets": [
                "next tower HP delta",
                "entity death",
                "entity next position",
                "next damage source class",
                "opponent public elixir interval",
                "opponent possible hand",
                "opponent possible next card",
                "future scene value",
            ],
        }
    if config.get("fixed_evaluation_enabled"):
        metadata["fixed_evaluation_config"] = {
            "every_rounds": int(config["evaluation_every_rounds"]),
            "seed_base": int(config["evaluation_seed_base"]),
            "seed_count": int(config["evaluation_seed_count"]),
            "roles_per_seed": [0, 1],
            "opponents": list(config["evaluation_opponents"]),
            "frozen_opponents": dict(config.get("evaluation_frozen_opponents", {})),
            "games_per_evaluation": (
                int(config["evaluation_seed_count"])
                * 2
                * len(config["evaluation_opponents"])
            ),
        }
    return metadata


def _write_summary_markdown(path, metadata, run_dir):
    lines = [
        "# Training summary",
        "",
        f"- Status: `{metadata['status']}`",
        f"- Duration: `{metadata['elapsed_human']}` ({metadata['elapsed_seconds']} seconds)",
        f"- Rounds: `{metadata['completed_rounds']}/{metadata['planned_rounds']}`",
        f"- Timesteps: `{metadata['timesteps']}`",
        f"- Episodes: `{metadata['episodes']}`",
        f"- Device: `{metadata['device']}`",
        f"- Algorithm: `{metadata['algorithm']}`",
        f"- Policy: `{metadata['policy']}`",
        f"- Parameters: `{metadata['parameters']:,}` (`{metadata['trainable_parameters']:,}` trainable)",
        f"- Model: `{metadata['model_file']}`",
        f"- SHA-256: `{metadata['model_sha256']}`",
        "",
    ]
    if "self_play" in metadata:
        self_play = metadata["self_play"]
        lines.extend((
            "## Self-play",
            "",
            f"- Games: `{self_play['self_play/games']}`",
            f"- Wins/losses/draws: `{self_play['self_play/wins']}` / `{self_play['self_play/losses']}` / `{self_play['self_play/draws']}`",
            f"- Recent win rate: `{self_play['self_play/win_rate_100']}`",
            f"- Learner Elo: `{self_play['self_play/learner_elo']:.3f}`",
            f"- Snapshots: `{self_play['self_play/snapshot_count']}`",
            "",
        ))
    if "recurrent" in metadata:
        recurrent = metadata["recurrent"]
        lines.extend((
            "## Recurrent memory",
            "",
            f"- LSTM: `{recurrent['layers']} x {recurrent['hidden_size']}`",
            f"- TBPTT steps: `{recurrent['tbptt_steps']}`",
            f"- State reset: `{recurrent['state_reset']}`",
            "",
        ))
    if "cross_modal" in metadata:
        cross_modal = metadata["cross_modal"]
        lines.extend((
            "## Cross-modal fusion",
            "",
            f"- Token dimension: `{cross_modal['dimension']}`",
            f"- Attention heads: `{cross_modal['attention_heads']}`",
            f"- Entity Transformer layers: `{cross_modal['entity_layers']}`",
            "- Action head: `card-token x placement-token`",
            "",
        ))
    if "spatial_action_head" in metadata:
        spatial = metadata["spatial_action_head"]
        lines.extend((
            "## Spatial action head",
            "",
            f"- CNN feature map: `{spatial['feature_map'][0]} x {spatial['feature_map'][1]}`",
            f"- CNN backbone channels: `{spatial['backbone_channels']}`",
            f"- Spatial token channels: `{spatial['feature_channels']}`",
            "- Heatmaps: `4 card-conditioned placement maps`",
            f"- Distribution: `{spatial['conditional_distribution']}`",
            "",
        ))
    if "world_model" in metadata:
        world = metadata["world_model"]
        lines.extend((
            "## Auxiliary world model",
            "",
            f"- Hidden dimension: `{world['hidden_dimension']}`",
            f"- Entity label schema: `{world['entity_target_schema']}`",
            f"- Entity association: `{world['entity_association']}`",
            f"- Action conditioned: `{world['action_conditioned']}`",
            f"- Executed action schema: `{world['executed_action_schema']}`",
            f"- Dynamics inputs: `{world['dynamics_conditioning']}`",
            f"- Total loss coefficient: `{world['auxiliary_coefficient']}`",
            f"- Future horizon: `{world['future_horizon_steps']} steps`",
            f"- Future horizon time: `{world['future_horizon_seconds']} seconds`",
            f"- Tasks: `{', '.join(world['targets'])}`",
            "",
        ))
    if "fixed_evaluation" in metadata:
        evaluation = metadata["fixed_evaluation"]
        latest = evaluation.get("latest") or {}
        lines.extend((
            "## Fixed evaluation",
            "",
            f"- Evaluated rounds: `{evaluation['evaluated_rounds']}`",
            f"- Latest games: `{latest.get('games')}`",
            f"- Latest win rate: `{latest.get('win_rate')}`",
            f"- Latest score rate: `{latest.get('score_rate')}`",
            f"- Timed out games: `{latest.get('timed_out')}`; valid: `{latest.get('valid')}`",
            f"- Schedule: `{evaluation['schedule']}`",
            f"- Match records: `{evaluation['matches']}`",
            "",
        ))
    lines.extend((
        "## Artifacts",
        "",
        f"- Live status: `{run_dir / 'status.json'}`",
        f"- Console log: `{run_dir / 'console.log'}`",
        f"- JSONL metrics: `{run_dir / 'metrics.jsonl'}`",
        f"- CSV metrics: `{run_dir / 'metrics.csv'}`",
        f"- SB3/TensorBoard data: `{run_dir / 'sb3'}`",
        f"- Charts: `{run_dir / 'charts'}`",
    ))
    if "self_play" in metadata:
        lines.append(f"- Self-play league: `{run_dir / 'self_play'}`")
    if "fixed_evaluation" in metadata:
        lines.append(f"- Fixed evaluation: `{run_dir / 'evaluation'}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_sb3_training(config, run_dir):
    """Train an SB3 algorithm from an external JSON profile."""
    # Model loading, logger creation and callback construction can all fail
    # before learn(). Own resources from the moment they are constructed.
    with ExitStack() as resources:
        return _run_sb3_training(config, run_dir, resources)


def _run_sb3_training(config, run_dir, resources):
    from training_experiments import source_hashes

    config = dict(config)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(path.name not in {"config.json", "console.log", "status.json"} for path in run_dir.iterdir()):
        raise ValueError("training output already contains run artifacts; use a fresh run directory")
    source_at_start = source_hashes()
    # Exclusive creation also prevents two trainers from claiming the same run.
    with (run_dir / "source_sha256.json").open("x", encoding="utf-8") as stream:
        json.dump(source_at_start, stream, indent=2, sort_keys=True)
        stream.write("\n")
    device = require_training_device(config["device"])
    battles = int(config["battles"])
    rollout_decisions = int(config["rollout_decisions"])
    if rollout_decisions % battles:
        raise ValueError("battles must divide rollout_decisions")
    rollout_steps = rollout_decisions // battles
    total_timesteps = int(config["rounds"]) * rollout_decisions
    env_class = load_object(config["environment"])
    algorithm_class = load_object(config["algorithm"])
    policy_class = load_object(config["policy"])
    self_play_pool = None
    self_play_callback = None
    evaluation_callback = None
    if config.get("self_play_enabled", False):
        from self_play import SelfPlayOpponentPool, SelfPlaySnapshotCallback

        self_play_pool = SelfPlayOpponentPool(
            run_dir=run_dir,
            algorithm_class=algorithm_class,
            device=device,
            battles=battles,
            seed=int(config["seed"]) + 700_001,
            random_ratio=float(config["self_play_random_ratio"]),
            current_ratio=float(config["self_play_current_ratio"]),
            historical_ratio=float(config["self_play_historical_ratio"]),
            snapshot_pool_size=int(config["self_play_pool_size"]),
            active_historical=int(config["self_play_active_historical"]),
            loaded_model_limit=int(config["self_play_loaded_model_limit"]),
            elo_k=float(config["self_play_elo_k"]),
        )
    env_kwargs = dict(config.get("environment_kwargs", {}))
    env_kwargs.update(
        battles=battles,
        port=int(config["port"]),
        decision_ticks=int(config["decision_ticks"]),
        seed=int(config["seed"]),
    )
    if self_play_pool is not None:
        env_kwargs["opponent_policy"] = self_play_pool
    env = VecMonitor(env_class(**env_kwargs), filename=str(run_dir / "episodes"))
    resources.callback(env.close)
    checkpoint = UpdatedCheckpointCallback(run_dir, config, self_play_pool)
    algorithm_kwargs = dict(config.get("algorithm_kwargs", {}))
    if config.get("recurrent_enabled", False):
        algorithm_kwargs["tbptt_steps"] = int(config["tbptt_steps"])
        policy_kwargs = dict(algorithm_kwargs.get("policy_kwargs", {}))
        policy_kwargs.update(
            lstm_hidden_size=int(config["recurrent_hidden_size"]),
            n_lstm_layers=int(config["recurrent_layers"]),
        )
        algorithm_kwargs["policy_kwargs"] = policy_kwargs
    if config.get("cross_modal_enabled", False):
        policy_kwargs = dict(algorithm_kwargs.get("policy_kwargs", {}))
        extractor_kwargs = dict(
            policy_kwargs.get("features_extractor_kwargs", {})
        )
        extractor_kwargs.update(
            cross_dim=int(config["cross_modal_dim"]),
            attention_heads=int(config["cross_modal_heads"]),
            entity_layers=int(config["cross_modal_entity_layers"]),
        )
        policy_kwargs["features_extractor_kwargs"] = extractor_kwargs
        algorithm_kwargs["policy_kwargs"] = policy_kwargs
    if config.get("spatial_heatmap_enabled", False):
        policy_kwargs = dict(algorithm_kwargs.get("policy_kwargs", {}))
        extractor_kwargs = dict(
            policy_kwargs.get("features_extractor_kwargs", {})
        )
        extractor_kwargs["spatial_channels"] = int(
            config["spatial_feature_channels"]
        )
        policy_kwargs["features_extractor_kwargs"] = extractor_kwargs
        algorithm_kwargs["policy_kwargs"] = policy_kwargs
    if config.get("world_model_enabled", False):
        policy_kwargs = dict(algorithm_kwargs.get("policy_kwargs", {}))
        policy_kwargs["world_model_hidden_dim"] = int(
            config["world_model_hidden_dim"]
        )
        algorithm_kwargs["policy_kwargs"] = policy_kwargs
        algorithm_kwargs.update(
            auxiliary_coef=float(config["auxiliary_coef"]),
            auxiliary_horizon=int(config["auxiliary_horizon"]),
            auxiliary_weights=dict(config["auxiliary_weights"]),
        )
    algorithm_kwargs.update(
        n_steps=rollout_steps,
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["epochs_per_round"]),
        learning_rate=float(config["learning_rate"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        ent_coef=float(config["entropy_coef"]),
        clip_range=float(config["clip_range"]),
        device=device,
        seed=int(config["seed"]),
        verbose=1,
    )
    resume_from = config.get("resume_from")
    if resume_from:
        saved_config = json.loads((resume_directory(resume_from) / "config.json").read_text())
        allowed_changes = {"rounds", "device", "port", "evaluation_port", "run_name", "run_dir",
                           "created_at", "resume_from", "auto_bootstrap", "_initial_policy_path"}
        for key in saved_config.keys() | config.keys():
            if key in allowed_changes:
                continue
            if config.get(key) != saved_config.get(key):
                raise ValueError(f"resume configuration changed: {key}")
        model = algorithm_class.load(resume_from, env=env, device=device)
    else:
        model = algorithm_class(policy_class, env, **algorithm_kwargs)
    initial_timesteps = model.num_timesteps
    remaining_timesteps = total_timesteps - initial_timesteps
    if remaining_timesteps <= 0:
        raise ValueError("target rounds must exceed completed checkpoint rounds")
    model.set_logger(configure(str(run_dir / "sb3"), ["stdout", "csv", "tensorboard"]))
    resources.callback(model.logger.close)
    callbacks = [checkpoint]
    if self_play_pool is not None:
        self_play_pool.set_current_model(model)
        bootstrap_path = self_play_pool.snapshots_dir / "snapshot_round_000000.zip"
        if not resume_from:
            model.save(bootstrap_path.with_suffix(""))
            self_play_pool.register_snapshot(
                "snapshot_round_000000", bootstrap_path, round_number=0
            )
        self_play_callback = SelfPlaySnapshotCallback(
            self_play_pool,
            rollout_decisions=rollout_decisions,
            every_rounds=int(config["self_play_snapshot_every_rounds"]),
            final_round=int(config["rounds"]),
        )
        self_play_callback.saved_rounds.add(0)
        callbacks.append(self_play_callback)
    if config.get("fixed_evaluation_enabled", False):
        from fixed_evaluation import FixedEvaluationCallback

        baseline_dir = run_dir / "evaluation" / "baselines"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / "initial_policy.zip"
        if resume_from:
            saved_baseline = resume_directory(resume_from) / "initial_policy.zip"
            if not saved_baseline.is_file():
                raise ValueError("resume state is missing its fixed initial-policy baseline")
            saved_runtime = json.loads((resume_directory(resume_from) / "runtime.json").read_text())
            expected_hash = saved_runtime.get("initial_policy_sha256")
            if expected_hash is not None and expected_hash != _sha256(saved_baseline):
                raise ValueError("resume initial-policy baseline no longer matches its saved runtime state")
            shutil.copyfile(saved_baseline, baseline_path)
        else:
            model.save(baseline_path.with_suffix(""))
        config["_initial_policy_path"] = str(baseline_path)
        evaluation_env_kwargs = dict(config.get("environment_kwargs", {}))
        evaluation_env_kwargs.update(
            port=int(config.get("evaluation_port") or config["port"]),
            decision_ticks=int(config["decision_ticks"]),
        )
        evaluation_callback = FixedEvaluationCallback(
            run_dir=run_dir,
            env_class=env_class,
            algorithm_class=algorithm_class,
            env_kwargs=evaluation_env_kwargs,
            rollout_decisions=rollout_decisions,
            every_rounds=int(config["evaluation_every_rounds"]),
            seed_base=int(config["evaluation_seed_base"]),
            seed_count=int(config["evaluation_seed_count"]),
            battles=int(config["evaluation_battles"]),
            max_decisions=int(config["evaluation_max_decisions"]),
            opponents=config["evaluation_opponents"],
            initial_policy_path=baseline_path,
            device=device,
            frozen_opponents=config.get("evaluation_frozen_opponents"),
            restore_training=False,
            deferred=not (config.get("evaluation_port") and int(config["evaluation_port"]) != int(config["port"])),
        )
        callbacks.append(evaluation_callback)
    def runtime_metrics():
        values = {}
        if self_play_pool is not None:
            values.update(self_play_pool.metrics())
        if evaluation_callback is not None:
            values.update(evaluation_callback.metrics())
        return values

    telemetry = TrainingTelemetryCallback(
        run_dir,
        rollout_decisions=rollout_decisions,
        planned_rounds=int(config["rounds"]),
        metrics_provider=runtime_metrics,
    )
    callbacks.append(telemetry)
    checkpoint.telemetry = telemetry
    if resume_from:
        restore_resume_state(resume_from, env, self_play_pool, telemetry)
        if self_play_callback is not None:
            self_play_callback.saved_rounds.update(entry["round"] for entry in self_play_pool.snapshots)
    started = time.monotonic()
    model_path = run_dir / "final_model.zip"
    try:
        print(
            f"[training-start] rounds={config['rounds']} timesteps={total_timesteps} "
            f"battles={battles} device={device} policy={config['policy']}",
            flush=True,
        )
        model.learn(
            total_timesteps=remaining_timesteps,
            reset_num_timesteps=not bool(resume_from),
            callback=CallbackList(callbacks),
            progress_bar=False,
            log_interval=1,
        )
        if source_hashes() != source_at_start:
            raise ValueError("training source changed during this run; use a fresh run")
        model.save(model_path.with_suffix(""))
        save_resume_state(model_path, model, config, self_play_pool, telemetry)
        elapsed = time.monotonic() - started
        generate_charts(run_dir, telemetry.records)
        metadata = _model_metadata(
            model, model_path, config, elapsed, telemetry, status="completed", source_sha256=source_at_start
        )
        if self_play_pool is not None:
            metadata["self_play"] = self_play_pool.summary()
        if evaluation_callback is not None:
            metadata["fixed_evaluation"] = evaluation_callback.summary()
            metadata["evaluation_elapsed_seconds"] = round(evaluation_callback.elapsed_seconds, 3)
        metadata["training_elapsed_seconds"] = round(elapsed - metadata.get("evaluation_elapsed_seconds", 0), 3)
        metadata["session_timesteps"] = model.num_timesteps - initial_timesteps
        _atomic_json(run_dir / "summary.json", metadata)
        _write_summary_markdown(run_dir / "summary.md", metadata, run_dir)
        _atomic_json(
            run_dir / "status.json",
            {
                "status": "completed",
                "updated_at": utc_now(),
                "current_round": metadata["completed_rounds"],
                "planned_rounds": metadata["planned_rounds"],
                "progress_percent": 100.0,
                "summary": metadata,
            },
        )
        print(
            f"[training-complete] duration={metadata['elapsed_human']} "
            f"rounds={metadata['completed_rounds']} timesteps={metadata['timesteps']} "
            f"parameters={metadata['parameters']:,} model={model_path}",
            flush=True,
        )
        return metadata
    except BaseException:
        try:
            elapsed = time.monotonic() - started
            interrupted_path = run_dir / "interrupted_model.zip"
            model.save(interrupted_path.with_suffix(""))
            if model.num_timesteps % rollout_decisions == 0 and not model.rollout_buffer.full:
                save_resume_state(interrupted_path, model, config, self_play_pool, telemetry)
            generate_charts(run_dir, telemetry.records)
            metadata = _model_metadata(
                model, interrupted_path, config, elapsed, telemetry, status="interrupted", source_sha256=source_at_start
            )
            if self_play_pool is not None:
                metadata["self_play"] = self_play_pool.summary()
            if evaluation_callback is not None:
                metadata["fixed_evaluation"] = evaluation_callback.summary()
            _atomic_json(run_dir / "summary.json", metadata)
            _write_summary_markdown(run_dir / "summary.md", metadata, run_dir)
            print(
                f"[training-interrupted] preserved={interrupted_path} "
                f"rounds={metadata['completed_rounds']} duration={metadata['elapsed_human']}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as preservation_error:
            print(
                f"[training-interrupted] could not preserve final state: {preservation_error}",
                file=sys.stderr,
                flush=True,
            )
        raise
