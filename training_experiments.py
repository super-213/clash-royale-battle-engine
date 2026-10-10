"""Plan, run and compare equal-interaction, multi-seed native PPO ablations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from fixed_evaluation import checkpoint_spec, evaluate_checkpoint
from one_click_train import load_profiles, prepare_config, validate_config
from training_runtime import _atomic_json, _sha256, load_object, utc_now


TASKS = (
    "tower_delta", "entity_death", "entity_position", "damage_source",
    "opponent_elixir", "opponent_hand", "opponent_next_card", "future_value",
)
CORE_VARIANTS = ("full", "no_aux", "state_only", "v4")
VARIANTS = (*CORE_VARIANTS, *(f"drop_{task}" for task in TASKS))
METRICS = ("win_rate", "score_rate", "mean_crown_diff", "mean_tower_hp_diff", "invalid_action_rate")


def source_hashes():
    root = Path(__file__).resolve().parent
    paths = [*root.glob("*.py"), *root.glob("native_engine/*.py"),
             *root.glob("training_profiles/*.json"), root / "standard_match.json",
             root / "requirements-training.txt",
             *(path for path in (root / "vendor/firstlight/probe").iterdir()
               if path.suffix in {".cpp", ".h", ".inc", ".S"} and not path.name.startswith("test_"))]
    return {str(path.relative_to(root)): _sha256(path)
            for path in sorted(paths)}


def variant_config(base, variant):
    config = copy.deepcopy(base)
    module = "native_engine.native_recurrent_model:"
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if not all(config.get(key) for key in (
        "recurrent_enabled", "cross_modal_enabled", "spatial_heatmap_enabled", "world_model_enabled",
    )) or not config["algorithm"].endswith(":NativeActionConditionedWorldModelPPO"):
        raise ValueError("ablation suite requires the v6 world-model profile")
    if variant == "no_aux":
        # Retain every parameter and its initialization, removing only auxiliary gradients.
        config["auxiliary_coef"] = 0.0
    elif variant == "state_only":
        config["algorithm"] = module + "NativeWorldModelPPO"
        config["policy"] = module + "NativeWorldModelRecurrentPolicy"
    elif variant == "v4":
        config["world_model_enabled"] = False
        config["algorithm"] = module + "NativeRecurrentPPO"
        config["policy"] = module + "NativeSpatialHeatmapRecurrentPolicy"
    elif variant.startswith("drop_"):
        config["auxiliary_weights"][variant.removeprefix("drop_")] = 0.0
    return config


def load_opponents(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    opponents = payload["opponents"]
    if not opponents:
        raise ValueError("at least one shared frozen checkpoint is required")
    result = {}
    for name, spec in opponents.items():
        if not name or name in {"random", "initial_policy", "aggregate"} or "/" in name:
            raise ValueError(f"invalid frozen opponent name: {name!r}")
        checkpoint = Path(spec["path"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        load_object(spec["algorithm"])
        result[name] = checkpoint_spec(checkpoint, spec["algorithm"], spec.get("sha256"))
    return result


def create_plan(base, suite_dir, opponents, total_decisions, seeds, variants=CORE_VARIANTS,
                test_seed_base=1_900_001, test_seed_count=64):
    suite_dir = Path(suite_dir).resolve()
    if suite_dir.exists():
        raise ValueError("suite directory already exists; use run/report for an existing plan")
    seeds, variants = list(seeds), list(variants)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("use at least three distinct nonnegative training seeds")
    if not variants or "full" not in variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be unique and include full")
    rollout = int(base["rollout_decisions"])
    if total_decisions <= 0 or total_decisions % rollout:
        raise ValueError("total decisions must be a positive multiple of rollout_decisions")
    if not opponents or test_seed_count <= 0:
        raise ValueError("shared opponents and positive test_seed_count are required")
    dev_seeds = set(range(int(base["evaluation_seed_base"]),
                          int(base["evaluation_seed_base"]) + int(base["evaluation_seed_count"])))
    test_seeds = set(range(test_seed_base, test_seed_base + test_seed_count))
    # Native training episodes allocate consecutive seeds. Include evaluation-induced resets.
    upper_training_seed = max(seeds) + total_decisions + (total_decisions // rollout + 2) * int(base["battles"])
    if min(dev_seeds | test_seeds) <= upper_training_seed or dev_seeds & test_seeds:
        raise ValueError("training, development and held-out test seed ranges must be disjoint")
    for spec in opponents.values():
        checkpoint_spec(spec["path"], spec["algorithm"], spec.get("sha256"))
    configs = []
    for variant in variants:
        for seed in seeds:
            config = variant_config(base, variant)
            config.update(
                rounds=total_decisions // rollout, seed=seed,
                fixed_evaluation_enabled=True,
                evaluation_opponents=["random", *opponents],
                evaluation_frozen_opponents=opponents,
                experiment_variant=variant,
            )
            validate_config(config)
            configs.append(config)
    suite_dir.mkdir(parents=True)
    frozen = {}
    for name, spec in opponents.items():
        # Store by hash so arbitrary display names cannot become filesystem paths.
        destination = suite_dir / "opponents" / f"{spec['sha256']}.zip"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copyfile(spec["path"], destination)
        frozen[name] = checkpoint_spec(destination, spec["algorithm"], spec["sha256"])
    runs = []
    for config in configs:
        run_name = f"{config['experiment_variant']}_seed_{config['seed']}"
        run_dir = suite_dir / "runs" / run_name
        run_dir.mkdir(parents=True)
        config.update(run_name=run_name, run_dir=str(run_dir), created_at=utc_now(),
                      evaluation_frozen_opponents=copy.deepcopy(frozen))
        _atomic_json(run_dir / "config.json", config)
        runs.append({"variant": config["experiment_variant"], "seed": config["seed"],
                     "run_dir": str(run_dir), "config": config})
    plan = {
        "version": 1, "created_at": utc_now(), "total_decisions_per_run": total_decisions,
        "total_training_decisions": total_decisions * len(runs), "training_seeds": seeds,
        "variants": variants, "frozen_opponents": frozen,
        "test_seed_base": test_seed_base, "test_seed_count": test_seed_count,
        "baseline_variant": "full", "runs": runs,
        "source_sha256": source_hashes(),
        "comparison": "paired training seeds; final checkpoints only; no best-reward selection",
    }
    _atomic_json(suite_dir / "plan.json", plan)
    _atomic_json(suite_dir / "status.json", {"status": "planned", "runs": len(runs)})
    return plan


def test_config(plan, config):
    result = copy.deepcopy(config)
    result.update(evaluation_seed_base=plan["test_seed_base"],
                  evaluation_seed_count=plan["test_seed_count"])
    return result


def verify_schedule(plan, config, schedule, *, held_out):
    expected = test_config(plan, config) if held_out else config
    fields = {
        "seeds": list(range(expected["evaluation_seed_base"],
                            expected["evaluation_seed_base"] + expected["evaluation_seed_count"])),
        "roles_per_seed": [0, 1], "opponents": expected["evaluation_opponents"],
        "frozen_opponents": plan["frozen_opponents"],
        "max_decisions_per_game": expected["evaluation_max_decisions"],
        "battles": expected["evaluation_battles"],
        "deterministic_learner": True, "deterministic_frozen_policy": True,
        "environment": {
            **expected.get("environment_kwargs", {}),
            "port": expected["port"], "decision_ticks": expected["decision_ticks"],
        },
    }
    for field, value in fields.items():
        if schedule.get(field) != value:
            raise ValueError(f"incomparable evaluation schedule: {field}")


def read_run_result(plan, run):
    root = Path(run["run_dir"])
    config = run["config"]
    if json.loads((root / "config.json").read_text()) != config:
        raise ValueError(f"config changed after planning: {root}")
    metadata = json.loads((root / "summary.json").read_text())
    if metadata.get("source_sha256") != plan["source_sha256"]:
        raise ValueError(f"training source does not match the experiment plan: {root}")
    if metadata["status"] != "completed" or metadata["timesteps"] != plan["total_decisions_per_run"]:
        raise ValueError(f"incomplete or unequal training budget: {root}")
    checkpoint_spec(root / "final_model.zip", config["algorithm"], metadata["model_sha256"])
    evaluation = root / "post_training" / "evaluation"
    checkpoint = json.loads((evaluation / "checkpoint.json").read_text())
    if checkpoint["sha256"] != metadata["model_sha256"] or checkpoint["algorithm"] != config["algorithm"]:
        raise ValueError("held-out evaluation does not correspond to the final checkpoint")
    schedule = json.loads((evaluation / "schedule.json").read_text())
    verify_schedule(plan, config, schedule, held_out=True)
    matches = [json.loads(line) for line in (evaluation / "matches.jsonl").read_text().splitlines()]
    expected_keys = {(opponent, seed, role) for opponent in config["evaluation_opponents"]
                     for seed in schedule["seeds"] for role in (0, 1)}
    keys = [(match["opponent_id"], match["seed"], match["learner_role"]) for match in matches]
    if set(keys) != expected_keys or len(keys) != len(expected_keys) or any(
        match["timed_out"] or match["result"] not in {"win", "loss", "draw"}
        or match["timesteps"] != plan["total_decisions_per_run"] for match in matches
    ):
        raise ValueError(f"held-out schedule incomplete, timed out or duplicated: {root}")
    from fixed_evaluation import _summarize

    rows = []
    selections = [("aggregate", matches)]
    for opponent in config["evaluation_opponents"]:
        selected = [match for match in matches if match["opponent_id"] == opponent]
        selections.append((opponent, selected))
        for role in (0, 1):
            selections.append((f"{opponent}/p{role}", [match for match in selected if match["learner_role"] == role]))
    for name, selected in selections:
        row = _summarize(selected, config["rounds"], metadata["timesteps"], name, 0)
        row.update(variant=run["variant"], training_seed=run["seed"],
                   parameters=metadata["parameters"], elapsed_seconds=metadata["elapsed_seconds"],
                   decisions_per_second=metadata.get("session_timesteps", metadata["timesteps"]) / max(1e-9, metadata.get("training_elapsed_seconds", metadata["elapsed_seconds"])))
        rows.append(row)
    return rows


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max()), "n": len(values)}


def build_report(suite_dir):
    suite_dir = Path(suite_dir).resolve()
    plan = json.loads((suite_dir / "plan.json").read_text())
    rows, incomplete = [], []
    for run in plan["runs"]:
        try:
            rows.extend(read_run_result(plan, run))
        except (OSError, ValueError, KeyError) as error:
            incomplete.append({"variant": run["variant"], "seed": run["seed"], "error": str(error)})
    groups, paired = [], []
    names = sorted({row["opponent_id"] for row in rows})
    for variant in plan["variants"]:
        for name in names:
            selected = [row for row in rows if row["variant"] == variant and row["opponent_id"] == name]
            if len(selected) != len(plan["training_seeds"]):
                continue
            group = {"variant": variant, "opponent_id": name, "training_seeds": len(selected)}
            for metric in (*METRICS, "parameters", "decisions_per_second"):
                values = [row[metric] for row in selected]
                if all(value is not None for value in values):
                    group[metric] = _statistics(values)
            groups.append(group)
    for variant in plan["variants"]:
        if variant == "full":
            continue
        for name in names:
            baseline = {row["training_seed"]: row for row in rows
                        if row["variant"] == "full" and row["opponent_id"] == name}
            other = {row["training_seed"]: row for row in rows
                     if row["variant"] == variant and row["opponent_id"] == name}
            if set(baseline) != set(plan["training_seeds"]) or set(other) != set(baseline):
                continue
            comparison = {"variant": variant, "opponent_id": name, "direction": "full minus variant"}
            for metric in METRICS:
                if any(baseline[seed][metric] is None or other[seed][metric] is None for seed in baseline):
                    continue
                deltas = [baseline[seed][metric] - other[seed][metric] for seed in sorted(baseline)]
                comparison[metric] = _statistics(deltas)
                comparison[metric]["per_seed"] = dict(zip(sorted(baseline), deltas))
            paired.append(comparison)
    report = {"valid": not incomplete, "incomplete_runs": incomplete,
              "groups": groups, "paired_differences": paired,
              "inference_unit": "training seed, not individual game",
              "note": "Descriptive statistics; positive mean alone is not evidence of significant improvement."}
    _atomic_json(suite_dir / "comparison.json", report)
    if rows:
        with (suite_dir / "per_seed.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Equal-budget ablation comparison", "",
             f"Valid complete suite: `{report['valid']}`. Training decisions per run: `{plan['total_decisions_per_run']}`.",
             "", "Statistics use training seeds as the independent unit. Final models use held-out battle seeds.",
             "", "| Variant | Opponent/role | Seeds | Score mean ± SD | Worst score | Win mean | Parameters |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in groups:
        score, win = group["score_rate"], group["win_rate"]
        lines.append(f"| {group['variant']} | {group['opponent_id']} | {group['training_seeds']} | "
                     f"{score['mean']:.4f} ± {score['std']:.4f} | {score['min']:.4f} | "
                     f"{win['mean']:.4f} | {group['parameters']['mean']:.0f} |")
    lines.extend(["", "Paired differences (full minus variant), all metrics and incomplete runs: `comparison.json`.",
                  "", report["note"]])
    (suite_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return report


def run_suite(suite_dir):
    suite_dir = Path(suite_dir).resolve()
    plan = json.loads((suite_dir / "plan.json").read_text())
    if source_hashes() != plan["source_sha256"]:
        raise ValueError("source changed after planning; create a new suite for the new implementation")
    # Check all configuration/checkpoint identities before starting any training.
    for spec in plan["frozen_opponents"].values():
        checkpoint_spec(spec["path"], spec["algorithm"], spec["sha256"])
    for run in plan["runs"]:
        if json.loads((Path(run["run_dir"]) / "config.json").read_text()) != run["config"]:
            raise ValueError("run configuration no longer matches the saved plan")
        validate_config(run["config"])
    try:
        for index, run in enumerate(plan["runs"], 1):
            root, config = Path(run["run_dir"]), run["config"]
            _atomic_json(suite_dir / "status.json", {
                "status": "running", "current_run": index, "runs": len(plan["runs"]),
                "variant": run["variant"], "seed": run["seed"], "updated_at": utc_now(),
            })
            print(f"[experiment] {index}/{len(plan['runs'])} {run['variant']} seed={run['seed']}", flush=True)
            if (root / "summary.json").exists():
                metadata = json.loads((root / "summary.json").read_text())
                if metadata["status"] != "completed" or metadata["timesteps"] != plan["total_decisions_per_run"]:
                    raise ValueError(f"interrupted run {root}; create a fresh suite rather than mixing restart budgets")
                checkpoint_spec(root / "final_model.zip", config["algorithm"], metadata["model_sha256"])
            else:
                from one_click_train import TeeStream
                import sys

                original_out, original_err = sys.stdout, sys.stderr
                output, errors = TeeStream(original_out, root / "console.log"), TeeStream(original_err, root / "console.log")
                try:
                    sys.stdout, sys.stderr = output, errors
                    metadata = load_object(config["trainer"])(config, root)
                finally:
                    sys.stdout, sys.stderr = original_out, original_err
                    output.close()
                    errors.close()
                if metadata["timesteps"] != plan["total_decisions_per_run"]:
                    raise ValueError("trainer exceeded the exact interaction budget")
            if not (root / "post_training" / "evaluation").exists():
                held_out_config = test_config(plan, config)
                (root / "post_training").mkdir(exist_ok=True)
                _atomic_json(root / "post_training" / "config.json", held_out_config)
                evaluate_checkpoint(held_out_config, root / "final_model.zip", root / "post_training")
            read_run_result(plan, run)
            build_report(suite_dir)
        report = build_report(suite_dir)
        _atomic_json(suite_dir / "status.json", {"status": "completed", "valid": report["valid"], "updated_at": utc_now()})
        return report
    except BaseException as error:
        build_report(suite_dir)
        _atomic_json(suite_dir / "status.json", {
            "status": "failed", "error": str(error), "error_type": type(error).__name__, "updated_at": utc_now(),
        })
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="write an inspectable experiment plan without training")
    plan_parser.add_argument("--suite-dir", type=Path, required=True)
    plan_parser.add_argument("--profile", default="native_hybrid_ppo")
    plan_parser.add_argument("--opponents", type=Path, required=True)
    plan_parser.add_argument("--total-decisions", type=int, required=True)
    plan_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    plan_parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(CORE_VARIANTS))
    plan_parser.add_argument("--all-variants", action="store_true")
    plan_parser.add_argument("--test-seed-base", type=int, default=1_900_001)
    plan_parser.add_argument("--test-seed-count", type=int, default=64)
    plan_parser.add_argument("--evaluation-seed-base", type=int)
    plan_parser.add_argument("--evaluation-seed-count", type=int)
    plan_parser.add_argument("--device", choices=("cpu", "mps"))
    for command in ("run", "report"):
        commands.add_parser(command).add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        profiles = {profile["profile_id"]: profile for profile in load_profiles()}
        base = prepare_config(profiles[args.profile], defaults_only=True)
        if args.device:
            base["device"] = args.device
        if args.evaluation_seed_base is not None:
            base["evaluation_seed_base"] = args.evaluation_seed_base
        if args.evaluation_seed_count is not None:
            base["evaluation_seed_count"] = args.evaluation_seed_count
        plan = create_plan(base, args.suite_dir, load_opponents(args.opponents), args.total_decisions,
                           args.seeds, VARIANTS if args.all_variants else args.variants,
                           args.test_seed_base, args.test_seed_count)
        print(f"Planned {len(plan['runs'])} runs, {plan['total_training_decisions']:,} learner decisions: {args.suite_dir / 'plan.json'}")
        return 0
    result = run_suite(args.suite_dir) if args.command == "run" else build_report(args.suite_dir)
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
