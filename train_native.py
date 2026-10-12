"""Direct CLI for the same profile-driven trainer used by train.sh."""

import argparse
import json
import torch
from pathlib import Path

from one_click_train import load_profiles, prepare_config, validate_config, ensure_engine
from training_runtime import require_training_device, resume_directory, _atomic_json, load_object


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', default='native_hybrid_ppo')
    parser.add_argument('--port', type=int)
    parser.add_argument('--evaluation-port', type=int, help='independent resident; otherwise evaluations run after training')
    parser.add_argument('--battles', type=int)
    parser.add_argument('--decision-ticks', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--total-timesteps', type=int, help='target total budget, a multiple of rollout_decisions')
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--device', choices=('mps', 'cpu'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/native_training'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--no-self-play', action='store_true')
    parser.add_argument('--no-evaluation', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.resume:
        config = json.loads((resume_directory(args.resume) / 'config.json').read_text())
        config['resume_from'] = str(args.resume.resolve())
    else:
        profile = next((item for item in load_profiles() if item['profile_id'] == args.profile), None)
        if profile is None:
            parser.error(f'unknown profile: {args.profile}')
        config = prepare_config(profile, defaults_only=True)
    for field in ('port', 'evaluation_port', 'battles', 'decision_ticks', 'seed', 'device', 'rounds'):
        value = getattr(args, field)
        if value is not None:
            config[field] = value
    if args.total_timesteps is not None:
        if args.rounds is not None:
            parser.error('choose --rounds or --total-timesteps')
        if args.total_timesteps <= 0 or args.total_timesteps % config['rollout_decisions']:
            parser.error('total-timesteps must be a positive multiple of rollout_decisions')
        config['rounds'] = args.total_timesteps // config['rollout_decisions']
    if args.no_self_play:
        config['self_play_enabled'] = False
    if args.no_evaluation:
        config['fixed_evaluation_enabled'] = False
    validate_config(config)
    if args.output.exists():
        parser.error('use a fresh output directory')
    args.output.mkdir(parents=True)
    config['run_dir'] = str(args.output.resolve())
    _atomic_json(args.output / 'config.json', config)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    ensure_engine(config)
    load_object(config['trainer'])(config, args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
