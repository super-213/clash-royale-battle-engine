#!/usr/bin/env python3
"""Interactive, profile-driven one-click training launcher."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from training_runtime import _atomic_json, load_object, utc_now, resume_directory


ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "training_profiles"
DEFAULT_RUNS_DIR = ROOT / "artifacts" / "training_runs"


class TeeStream:
    def __init__(self, terminal, path):
        self.terminal = terminal
        self.file = Path(path).open("a", encoding="utf-8", buffering=1)

    def write(self, value):
        self.terminal.write(value)
        self.file.write(value)
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def isatty(self):
        return self.terminal.isatty()


def load_profiles(directory=PROFILE_DIR):
    profiles = []
    for path in sorted(Path(directory).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for required in ("name", "trainer", "environment", "algorithm", "policy", "defaults"):
            if required not in payload:
                raise ValueError(f"{path}: missing {required}")
        payload["profile_id"] = path.stem
        payload["profile_path"] = str(path.resolve())
        profiles.append(payload)
    if not profiles:
        raise RuntimeError(f"no training profiles found in {directory}")
    return profiles


def ask(prompt, default, convert=str, validate=lambda value: True):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        try:
            value = convert(raw) if raw else default
        except (TypeError, ValueError):
            print("输入格式无效，请重试。")
            continue
        if validate(value):
            return value
        print("输入超出允许范围，请重试。")


def ask_yes_no(prompt, default=True):
    marker = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{marker}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "是"):
            return True
        if raw in ("n", "no", "否"):
            return False
        print("请输入 y 或 n。")


def choose_profile(profiles):
    print("\n可用训练配置：")
    for index, profile in enumerate(profiles, 1):
        print(f"  {index}. {profile['name']} — {profile.get('description', '')}")
    selected = ask("选择配置编号", 1, int, lambda value: 1 <= value <= len(profiles))
    return profiles[selected - 1]


def unique_run_dir(base, name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "training"
    candidate = Path(base) / sanitized
    suffix = 2
    while candidate.exists():
        candidate = Path(base) / f"{sanitized}_{suffix}"
        suffix += 1
    return candidate.resolve()


def engine_available(port):
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
            return True
    except OSError:
        return False


def prepare_config(profile, defaults_only=False):
    config = dict(profile["defaults"])
    config.update({
        key: value
        for key, value in profile.items()
        if key not in ("defaults", "name", "description", "profile_path")
    })
    config["profile_name"] = profile["name"]
    config["profile_path"] = profile["profile_path"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{timestamp}_{profile['profile_id']}"
    if defaults_only:
        config["run_name"] = default_name
        config["auto_bootstrap"] = True
        return config

    print("\n基础训练设置（直接回车使用默认值）：")
    config["run_name"] = ask("运行名称", default_name)
    config["rounds"] = ask("训练轮次（1轮 = 1次 rollout + PPO 更新）", config["rounds"], int, lambda value: value > 0)
    config["battles"] = ask("并行战斗数量", config["battles"], int, lambda value: value > 0)
    config["decision_ticks"] = ask("每次决策推进的 native ticks", config["decision_ticks"], int, lambda value: value > 0)
    config["seed"] = ask("随机种子", config["seed"], int)
    config["device"] = ask("训练设备（mps/cpu）", config["device"], str, lambda value: value in ("mps", "cpu"))
    config["port"] = ask("原生引擎端口", config["port"], int, lambda value: 1 <= value <= 65535)
    config["checkpoint_every_rounds"] = ask("每多少轮保存检查点", config["checkpoint_every_rounds"], int, lambda value: value > 0)
    config["self_play_enabled"] = ask_yes_no(
        "启用自博弈与历史对手池", config.get("self_play_enabled", True)
    )
    if config["self_play_enabled"]:
        config["self_play_snapshot_every_rounds"] = ask(
            "每多少轮加入一个历史对手快照",
            config["self_play_snapshot_every_rounds"],
            int,
            lambda value: value > 0,
        )
        config["self_play_pool_size"] = ask(
            "历史对手池最大快照数",
            config["self_play_pool_size"],
            int,
            lambda value: value > 0,
        )
    config["fixed_evaluation_enabled"] = ask_yes_no(
        "启用独立固定评估赛程", config.get("fixed_evaluation_enabled", True)
    )
    if config["fixed_evaluation_enabled"]:
        config["evaluation_every_rounds"] = ask(
            "每多少轮运行固定评估",
            config["evaluation_every_rounds"], int, lambda value: value > 0,
        )
        config["evaluation_seed_count"] = ask(
            "每个对手的固定 seed 数（每个 seed 双方角色各一局）",
            config["evaluation_seed_count"], int, lambda value: value > 0,
        )
    config["auto_bootstrap"] = ask_yes_no("端口未就绪时自动运行 bootstrap_emulator.py", True)

    if ask_yes_no("修改 PPO 高级参数", False):
        print("\n高级设置：")
        config["rollout_decisions"] = ask("每轮决策总数", config["rollout_decisions"], int, lambda value: value > 0)
        config["batch_size"] = ask("PPO minibatch 大小", config["batch_size"], int, lambda value: value > 0)
        config["epochs_per_round"] = ask("每轮 PPO epochs", config["epochs_per_round"], int, lambda value: value > 0)
        if config.get("recurrent_enabled"):
            config["tbptt_steps"] = ask(
                "TBPTT 序列长度", config["tbptt_steps"], int,
                lambda value: value > 0,
            )
            config["recurrent_hidden_size"] = ask(
                "持续 LSTM hidden size", config["recurrent_hidden_size"], int,
                lambda value: value > 0,
            )
            config["recurrent_layers"] = ask(
                "持续 LSTM 层数", config["recurrent_layers"], int,
                lambda value: value > 0,
            )
        if config.get("cross_modal_enabled"):
            config["cross_modal_dim"] = ask(
                "Cross-attention token 维度", config["cross_modal_dim"], int,
                lambda value: value > 0,
            )
            config["cross_modal_heads"] = ask(
                "Cross-attention 头数", config["cross_modal_heads"], int,
                lambda value: value > 0,
            )
            config["cross_modal_entity_layers"] = ask(
                "实体 Transformer 层数",
                config["cross_modal_entity_layers"], int,
                lambda value: value > 0,
            )
        if config.get("spatial_heatmap_enabled"):
            config["spatial_feature_channels"] = ask(
                "全分辨率 CNN backbone channels",
                config["spatial_feature_channels"], int,
                lambda value: value > 0 and value % 8 == 0,
            )
        if config.get("world_model_enabled"):
            config["world_model_hidden_dim"] = ask(
                "世界模型 hidden dimension",
                config["world_model_hidden_dim"], int,
                lambda value: value > 0,
            )
            config["auxiliary_coef"] = ask(
                "辅助总损失系数", config["auxiliary_coef"], float,
                lambda value: value >= 0,
            )
            config["auxiliary_horizon"] = ask(
                "未来局面预测步数", config["auxiliary_horizon"], int,
                lambda value: value > 0,
            )
        config["learning_rate"] = ask("学习率", config["learning_rate"], float, lambda value: value > 0)
        config["gamma"] = ask("Gamma", config["gamma"], float, lambda value: 0 < value <= 1)
        config["gae_lambda"] = ask("GAE lambda", config["gae_lambda"], float, lambda value: 0 < value <= 1)
        config["entropy_coef"] = ask("熵系数", config["entropy_coef"], float, lambda value: value >= 0)
        config["clip_range"] = ask("PPO clip range", config["clip_range"], float, lambda value: value > 0)
        if config.get("self_play_enabled"):
            config["self_play_random_ratio"] = ask(
                "随机对手比例", config["self_play_random_ratio"], float,
                lambda value: 0 <= value <= 1,
            )
            config["self_play_current_ratio"] = ask(
                "当前策略镜像比例", config["self_play_current_ratio"], float,
                lambda value: 0 <= value <= 1,
            )
            config["self_play_historical_ratio"] = ask(
                "历史快照比例", config["self_play_historical_ratio"], float,
                lambda value: 0 <= value <= 1,
            )
    return config


def validate_config(config):
    if int(config["rounds"]) <= 0 or not 1 <= int(config["battles"]) <= 16:
        raise ValueError("rounds must be positive and battles must be between 1 and 16")
    if int(config["batch_size"]) <= 1 or int(config["checkpoint_every_rounds"]) <= 0:
        raise ValueError("batch_size must exceed one and checkpoint interval must be positive")
    if config.get("evaluation_port") is not None and not 1 <= int(config["evaluation_port"]) <= 65535:
        raise ValueError("evaluation_port must be between 1 and 65535")
    if int(config["rollout_decisions"]) % int(config["battles"]):
        raise ValueError("并行战斗数量必须能整除每轮决策总数")
    rollout_steps = int(config["rollout_decisions"]) // int(config["battles"])
    if rollout_steps % int(config["batch_size"]):
        # SB3 needs batch_size to divide the full rollout buffer, not n_steps.
        full_buffer = rollout_steps * int(config["battles"])
        if full_buffer % int(config["batch_size"]):
            raise ValueError("PPO batch_size 必须能整除每轮决策总数")
    if config.get("self_play_enabled"):
        ratio_sum = sum(float(config[key]) for key in (
            "self_play_random_ratio",
            "self_play_current_ratio",
            "self_play_historical_ratio",
        ))
        if not abs(ratio_sum - 1.0) <= 1e-6:
            raise ValueError("随机、当前策略和历史快照的对手比例之和必须为 1")
    if config.get("recurrent_enabled"):
        rollout_steps = int(config["rollout_decisions"]) // int(config["battles"])
        if int(config["tbptt_steps"]) > rollout_steps:
            raise ValueError("TBPTT 序列长度不能超过每个环境的 rollout steps")
    if config.get("cross_modal_enabled"):
        if int(config["cross_modal_dim"]) % int(config["cross_modal_heads"]):
            raise ValueError("Cross-attention token 维度必须能被头数整除")
    if config.get("spatial_heatmap_enabled"):
        if int(config["spatial_feature_channels"]) % 8:
            raise ValueError("空间 feature channels 必须能被 8 整除")
    if config.get("world_model_enabled"):
        if float(config["auxiliary_coef"]) < 0:
            raise ValueError("辅助总损失系数不能为负数")
        if int(config["auxiliary_horizon"]) >= int(config["tbptt_steps"]):
            raise ValueError("未来局面预测步数必须小于 TBPTT 序列长度")
        if any(float(value) < 0 for value in config["auxiliary_weights"].values()):
            raise ValueError("辅助任务权重不能为负数")
    if config.get("fixed_evaluation_enabled"):
        if int(config["evaluation_every_rounds"]) <= 0:
            raise ValueError("固定评估间隔必须为正数")
        if int(config["evaluation_seed_count"]) <= 0:
            raise ValueError("固定评估 seed 数必须为正数")
        if not 1 <= int(config["evaluation_battles"]) <= 16:
            raise ValueError("固定评估并行战斗数必须在 1 到 16 之间")
        if int(config["evaluation_max_decisions"]) <= 0:
            raise ValueError("固定评估最大决策数必须为正数")
        allowed = {"random", "initial_policy", *config.get("evaluation_frozen_opponents", {})}
        opponents = list(config["evaluation_opponents"])
        if not opponents or len(opponents) != len(set(opponents)):
            raise ValueError("固定评估对手必须非空且不能重复")
        if any(opponent not in allowed for opponent in opponents):
            raise ValueError("固定评估对手只支持 random 和 initial_policy 或已注册的冻结 checkpoint")
        from fixed_evaluation import checkpoint_spec
        for name, spec in config.get("evaluation_frozen_opponents", {}).items():
            if name in {"random", "initial_policy", "aggregate"} or "/" in name:
                raise ValueError("冻结评估对手名称不能使用保留名称或包含斜线")
            checkpoint_spec(spec["path"], spec["algorithm"], spec.get("sha256"))


def print_plan(config, run_dir):
    total = int(config["rounds"]) * int(config["rollout_decisions"])
    print("\n训练计划：")
    print(f"  配置/模型 : {config['profile_name']} / {config['policy']}")
    print(f"  轮次       : {config['rounds']}")
    print(f"  总决策数   : {total:,}")
    print(f"  并行环境   : {config['battles']}")
    print(f"  设备       : {config['device']}")
    print(f"  检查点间隔 : {config['checkpoint_every_rounds']} 轮")
    if config.get("recurrent_enabled"):
        print(
            f"  持续记忆   : LSTM {config['recurrent_layers']}×"
            f"{config['recurrent_hidden_size']}，TBPTT={config['tbptt_steps']}"
        )
    if config.get("cross_modal_enabled"):
        print(
            f"  跨模态融合 : {config['cross_modal_heads']} heads × "
            f"{config['cross_modal_dim']} dim，"
            f"entity layers={config['cross_modal_entity_layers']}"
        )
    if config.get("spatial_heatmap_enabled"):
        print(
            "  空间动作头   : 32×18 full-resolution heatmap，"
            f"CNN backbone channels={config['spatial_feature_channels']}"
        )
    if config.get("world_model_enabled"):
        print(
            "  辅助世界模型 : "
            f"hidden={config['world_model_hidden_dim']}，"
            f"coef={config['auxiliary_coef']}，"
            f"future horizon={config['auxiliary_horizon']} steps/"
            f"{config['auxiliary_horizon'] * config['decision_ticks'] * 0.05:.2f}s"
        )
    if config.get("self_play_enabled"):
        print(
            "  自博弈对手 : "
            f"随机 {config['self_play_random_ratio']:.0%} / "
            f"当前 {config['self_play_current_ratio']:.0%} / "
            f"历史 {config['self_play_historical_ratio']:.0%}"
        )
        print(
            f"  对手池     : 每 {config['self_play_snapshot_every_rounds']} 轮快照，"
            f"最多 {config['self_play_pool_size']} 个"
        )
    if config.get("fixed_evaluation_enabled"):
        games = (
            config["evaluation_seed_count"] * 2
            * len(config["evaluation_opponents"])
        )
        print(
            f"  固定评估   : round 0/每 {config['evaluation_every_rounds']} 轮/最终轮，"
            f"{games} 局，seed×双角色，对手={','.join(config['evaluation_opponents'])}"
        )
    print(f"  输出目录   : {run_dir}")


def ensure_engine(config):
    if engine_available(config["port"]):
        return
    if not config.get("auto_bootstrap"):
        raise RuntimeError(
            f"127.0.0.1:{config['port']} 没有可用引擎；请先运行 python bootstrap_emulator.py"
        )
    print("[engine] 端口未就绪，正在运行 bootstrap_emulator.py ...", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "bootstrap_emulator.py"),
        "--serial", os.environ.get("CLASH_TRAIN_SERIAL", "emulator-5554"),
        "--port", str(config["port"]),
    ], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="profile id; interactive selection when omitted")
    parser.add_argument("--defaults", action="store_true", help="accept profile defaults without prompts")
    parser.add_argument("--dry-run", action="store_true", help="write and validate config without training")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--experiment-suite", type=Path, help="execute a saved equal-budget experiment plan")
    parser.add_argument("--resume", type=Path, help="resume an updated checkpoint with its .zip.state bundle")
    parser.add_argument("--rounds", type=int, help="target total PPO rounds, including completed rounds")
    arguments = parser.parse_args(argv)
    if arguments.experiment_suite:
        from training_experiments import run_suite

        plan = json.loads((arguments.experiment_suite / "plan.json").read_text())
        for run in plan["runs"]:
            validate_config(run["config"])
        print(f"实验矩阵：{len(plan['runs'])} 次运行，共 {plan['total_training_decisions']:,} 个训练决策")
        if arguments.dry_run:
            return 0
        ensure_engine(plan["runs"][0]["config"])
        result = run_suite(arguments.experiment_suite)
        return 0 if result["valid"] else 2
    if arguments.resume:
        checkpoint = arguments.resume.resolve()
        config = json.loads((resume_directory(checkpoint) / "config.json").read_text())
        config["resume_from"] = str(checkpoint)
        config["run_name"] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_resumed"
        if arguments.rounds is not None:
            config["rounds"] = arguments.rounds
        return launch_config(config, arguments)
    profiles = load_profiles()
    if arguments.profile:
        matches = [profile for profile in profiles if profile["profile_id"] == arguments.profile]
        if not matches:
            parser.error(f"unknown profile {arguments.profile!r}")
        profile = matches[0]
    elif arguments.defaults:
        profile = profiles[0]
    else:
        profile = choose_profile(profiles)
    config = prepare_config(profile, defaults_only=arguments.defaults)
    if arguments.rounds is not None:
        config["rounds"] = arguments.rounds
    return launch_config(config, arguments)


def launch_config(config, arguments):
    validate_config(config)
    run_dir = unique_run_dir(arguments.runs_dir, config["run_name"])
    print_plan(config, run_dir)
    if not arguments.defaults and not ask_yes_no("确认开始训练", True):
        print("已取消。")
        return 0

    run_dir.mkdir(parents=True)
    config["run_dir"] = str(run_dir)
    config["created_at"] = utc_now()
    _atomic_json(run_dir / "config.json", config)
    _atomic_json(
        run_dir / "status.json",
        {"status": "preparing", "updated_at": utc_now(), "run_dir": str(run_dir)},
    )
    Path(arguments.runs_dir).mkdir(parents=True, exist_ok=True)
    _atomic_json(
        Path(arguments.runs_dir).resolve() / "latest.json",
        {
            "run_dir": str(run_dir),
            "status_file": str(run_dir / "status.json"),
            "config_file": str(run_dir / "config.json"),
            "created_at": config["created_at"],
        },
    )
    if arguments.dry_run:
        _atomic_json(
            run_dir / "status.json",
            {
                "status": "dry_run_completed",
                "updated_at": utc_now(),
                "run_dir": str(run_dir),
                "config_file": str(run_dir / "config.json"),
            },
        )
        print(f"配置验证通过：{run_dir / 'config.json'}")
        return 0

    original_stdout, original_stderr = sys.stdout, sys.stderr
    tee_stdout = TeeStream(original_stdout, run_dir / "console.log")
    tee_stderr = TeeStream(original_stderr, run_dir / "console.log")
    sys.stdout, sys.stderr = tee_stdout, tee_stderr
    try:
        ensure_engine(config)
        trainer = load_object(config["trainer"])
        trainer(config, run_dir)
        return 0
    except BaseException as error:
        _atomic_json(
            run_dir / "status.json",
            {
                "status": "failed",
                "updated_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "console_log": str(run_dir / "console.log"),
            },
        )
        print(f"[training-failed] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        tee_stdout.close()
        tee_stderr.close()


if __name__ == "__main__":
    raise SystemExit(main())
