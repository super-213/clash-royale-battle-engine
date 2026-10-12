"""Frozen multi-step public dynamics validation from a saved sequence replay.

This measures predictions along recorded joint actions, not counterfactual
ranking or strength. Use fixed_evaluation with different search budgets for
strength comparisons. A held-out replay must be collected in a separate run.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .planning_policy import NativePlanningAgent
from .planning_replay import PlanningStep, SequenceReplay
from .planning_training import PlanningLearner, single_observation
from .planning_rules import canonical_actions, canonical_mask


def collect_replay(agent, config, decisions=1024, battles=4, seed=2900001):
    """Frozen policy, independent seeds, random opponents; zero training updates."""
    from training_runtime import load_object
    if decisions < 1 or not 1 <= battles <= 16 or decisions % battles:
        raise ValueError('collection decisions must be positive and divisible by battles')
    training_upper = int(config['seed']) + int(config['rounds']) * (int(config['rollout_decisions']) + int(config['battles'])) + int(config['battles'])
    if seed <= training_upper:
        raise ValueError('collection seeds must exceed the complete training seed range')
    eval_begin = config['evaluation_seed_base']
    if seed < eval_begin + config['evaluation_seed_count'] and seed + decisions + battles >= eval_begin:
        raise ValueError('collection seeds overlap fixed evaluation seeds')
    kwargs = dict(config.get('environment_kwargs', {}))
    kwargs.update(battles=battles, port=config['port'], decision_ticks=config['decision_ticks'],
                  seed=seed, planning_labels=True)
    kwargs.pop('opponent_policy', None)
    env = load_object(config['environment'])(**kwargs)
    replay = SequenceReplay(decisions)
    state = None; starts = np.ones(battles, dtype=bool)
    try:
        observations = env.reset()
        for _ in range(decisions // battles):
            actions, state = agent.predict(observations, state, starts, deterministic=True)
            identities = canonical_actions(observations['hand'], actions)
            records = copy.deepcopy(agent.last_search)
            after, rewards, dones, infos = env.step(actions)
            for row, info in enumerate(infos):
                label = info.get('planning', {})
                if label.get('schema') != 'public_planning_transition_v1':
                    raise ValueError('collection requires planning transition labels')
                replay.append(row, PlanningStep({key: value[row].copy() for key, value in observations.items()},
                              label['next_observation'], int(identities[row]), label['opponent_action'],
                              label['accepted'], float(rewards[row]), bool(dones[row]), label['ticks'], records[row],
                              agent.version, seed=int(info['seed']), role=int(info['learner_role'][1:])))
            observations, starts = after, dones
    finally:
        env.close()
    return replay


@torch.no_grad()
def validate_replay(agent, replay, horizons=(1, 4, 8), samples=64, outcome_samples=4, seed=2900001, burn_in=16):
    if samples < 1 or outcome_samples < 1 or burn_in < 0 or not horizons or any(h < 1 for h in horizons):
        raise ValueError('invalid multi-step validation settings')
    model = agent.policy.eval()
    rng = np.random.default_rng(seed)
    sequences = replay.sample(rng, samples, max(horizons), burn_in)
    metrics = {h: [] for h in horizons}
    for prefix, sequence in sequences:
        memory = PlanningLearner.burn_in(model, prefix)
        root = model.encode(single_observation(sequence[0].observation), memory)
        predictions = {h: [] for h in horizons}
        for _ in range(outcome_samples):
            state = root
            cumulative = 0.
            for index, step in enumerate(sequence):
                probabilities = model.chance_distribution(state, [step.action], [step.opponent_action]).softmax(-1)[0].cpu().numpy().astype(float)
                probabilities /= probabilities.sum()
                code = int(rng.choice(len(probabilities), p=probabilities))
                state, output = model.dynamics(state, [step.action], [step.opponent_action], [code], ticks=[step.ticks])
                cumulative += agent.search_settings.discount ** index * float(output['reward'][0])
                horizon = index + 1
                if horizon in predictions:
                    predictions[horizon].append(dict(
                        spatial=output['spatial'].sigmoid()[0].cpu().numpy(),
                        scalar=state.scalar[0].cpu().numpy(), terminal=float(output['terminal'].sigmoid()[0]),
                        mask=model.predicted_mask(state)[0].cpu().numpy().astype(float),
                        cumulative_reward=cumulative))
        for horizon in horizons:
            if len(sequence) < horizon:
                continue
            target = sequence[horizon - 1]
            group = predictions[horizon]
            if not group:
                continue
            spatial = np.mean([item['spatial'] for item in group], axis=0)
            scalar = np.mean([item['scalar'] for item in group], axis=0)
            terminal = float(np.mean([item['terminal'] for item in group]))
            mask = np.mean([item['mask'] for item in group], axis=0) > .5
            truth = canonical_mask(single_observation(target.next_observation))[0]
            reward = sum(agent.search_settings.discount ** k * step.reward for k, step in enumerate(sequence[:horizon]))
            metrics[horizon].append(dict(
                spatial_mae=float(np.abs(spatial - target.next_observation['spatial']).mean()),
                tower_hp_mae=float(np.abs(scalar[9:11] - target.next_observation['scalar'][9:11]).mean()),
                own_elixir_mae=float(abs(scalar[0] - target.next_observation['scalar'][0]) * 10),
                cumulative_reward_mae=float(abs(np.mean([item['cumulative_reward'] for item in group]) - reward)),
                terminal_brier=(terminal - float(target.done)) ** 2,
                legality_iou=float((mask & truth).sum() / max(1, (mask | truth).sum()))))
    return dict(schema='public_planning_validation_v1', seed=seed, outcome_samples=outcome_samples, burn_in=burn_in,
                scope='Recorded joint-action public dynamics; no counterfactual or strength claim.',
                horizons={str(h): dict(samples=len(rows), **{
                    key: float(np.mean([row[key] for row in rows])) for key in rows[0]})
                          if rows else dict(samples=0) for h, rows in metrics.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'evaluate'), nargs='?', default='evaluate')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--replay', type=Path, help='trusted replay from an independent collection run')
    parser.add_argument('--config', type=Path, help='saved config.json, required for native collection')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--horizons', type=int, nargs='+', default=[1, 4, 8])
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--outcomes', type=int, default=4)
    parser.add_argument('--decisions', type=int, default=1024)
    parser.add_argument('--battles', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2900001)
    parser.add_argument('--search-budget', type=int, help='collection search budget override')
    args = parser.parse_args(argv)
    from training_runtime import _atomic_json, _sha256
    agent = NativePlanningAgent.load(args.checkpoint, device=args.device)
    if args.mode == 'collect':
        if args.config is None:
            parser.error('collection requires --config')
        if args.output.exists():
            parser.error('use a fresh collection output path')
        from dataclasses import replace
        if args.search_budget is not None:
            agent.search_settings = replace(agent.search_settings, budget=args.search_budget)
        config = json.loads(args.config.read_text())
        if config['decision_ticks'] != agent.policy.decision_ticks:
            raise ValueError('collection timestep differs from trained dynamics')
        replay = collect_replay(agent, config, args.decisions, args.battles, args.seed)
        metadata = dict(seed=args.seed, decisions=args.decisions, battles=args.battles, opponent='legal_random',
                        checkpoint_sha256=_sha256(args.checkpoint), training_updates=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            torch.save(dict(schema='planning_validation_replay_v1', replay=replay.state_dict(), metadata=metadata), stream)
        print(json.dumps(metadata, indent=2))
        return 0
    if args.replay is None:
        parser.error('evaluation requires --replay')
    payload = torch.load(args.replay, map_location='cpu', weights_only=False)
    if payload.get('schema') not in ('planning_training_state_v1', 'planning_validation_replay_v1'):
        raise ValueError('unsupported validation replay')
    result = validate_replay(agent, SequenceReplay.from_state_dict(payload['replay']),
                             args.horizons, args.samples, args.outcomes)
    result.update(checkpoint_sha256=_sha256(args.checkpoint), replay_sha256=_sha256(args.replay),
                  collection=payload.get('metadata', {'scope': 'training replay; not held-out validation'}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
