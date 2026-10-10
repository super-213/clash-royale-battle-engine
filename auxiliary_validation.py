"""Collect independent native trajectories and score frozen auxiliary predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from policy_inspection import checkpoint_config, load_policy
from training_runtime import load_object, _atomic_json, _sha256
from fixed_evaluation import checkpoint_spec, preserve_random_state


def scene_value(scalar):
    return scalar[..., 9] - scalar[..., 10] + .25 * (scalar[..., 2] - scalar[..., 3])


def collect_dataset(config, model, output, seed_base=2_900_001, seed_count=8,
                    decisions=128, battles=8, port=None, collection_checkpoint=None):
    if min(seed_count, decisions, battles) <= 0 or battles > 16:
        raise ValueError('positive seeds/decisions and 1..16 battles required')
    if not hasattr(model.policy, 'auxiliary_predictions'):
        raise ValueError('checkpoint has no auxiliary prediction head')
    output = Path(output)
    if output.exists():
        raise ValueError('use a fresh dataset path')
    training_upper = (int(config.get('seed', 0)) + int(config['rounds']) * int(config['rollout_decisions'])
                      + int(config['battles']))
    validation_upper = seed_base + seed_count * (decisions * battles + battles)
    if seed_base <= training_upper:
        raise ValueError('validation seeds must be above the complete training seed range')
    reserved = [(config['evaluation_seed_base'], config['evaluation_seed_count'])]
    plan_path = Path(config.get('run_dir', '.')).parent.parent / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        reserved.append((plan['test_seed_base'], plan['test_seed_count']))
    if any(seed_base < base + count and validation_upper >= base for base, count in reserved):
        raise ValueError('validation seeds overlap development or held-out test seeds')
    rows = []
    horizon = int(config['auxiliary_horizon'])
    for wave in range(0, seed_count, battles):
        count = min(battles, seed_count - wave)
        kwargs = dict(config.get('environment_kwargs', {}))
        kwargs.update(battles=count, seed=seed_base + wave * (decisions * battles + battles), learner_roles='alternating',
                      port=port or config['port'], decision_ticks=config['decision_ticks'])
        kwargs.pop('opponent_policy', None)
        env = load_object(config['environment'])(**kwargs)
        state, starts = None, np.ones(count, dtype=bool)
        wave_rows = []
        try:
            observation = env.reset()
            for decision in range(decisions):
                if state is None:
                    lstm = model.policy.lstm_actor
                    prior = tuple(np.zeros((lstm.num_layers, count, lstm.hidden_size), np.float32) for _ in range(2))
                else:
                    prior = state
                actions, state = model.predict(observation, state=prior, episode_start=starts, deterministic=True)
                after, _, dones, infos = env.step(actions)
                step_rows = []
                for slot, info in enumerate(infos):
                    scalar = info.get('terminal_observation', {}).get('scalar', after['scalar'][slot])
                    row = {f'obs_{key}': value[slot].copy() for key, value in observation.items()}
                    row.update(wave=wave, decision=decision,
                               episode_start=starts[slot], executed_action=info['executed_action'],
                               entity_targets=info['entity_targets'], transition_targets=info['transition_targets'],
                               seed=info['seed'], role=int(info['learner_role'][1:]),
                               next_scene=scene_value(scalar), done=bool(dones[slot]),
                               future_targets=np.zeros(2, dtype=np.float32))
                    step_rows.append(row)
                wave_rows.append(step_rows)
                observation, starts = after, dones
            # Pure labels may cross chunks, never episodes. An exact t+h terminal
            # observation is valid; termination before that horizon is excluded.
            for step, step_rows in enumerate(wave_rows):
                endpoint = step + horizon - 1
                if endpoint >= len(wave_rows):
                    continue
                for slot, row in enumerate(step_rows):
                    if not any(wave_rows[k][slot]['done'] for k in range(step, endpoint)):
                        row['future_targets'][:] = (wave_rows[endpoint][slot]['next_scene'], 1)
            rows.extend(row for step_rows in wave_rows for row in step_rows)
        finally:
            env.close()
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]
              if key not in {'next_scene', 'done'}}
    weights = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        weights.update(name.encode())
        weights.update(value.detach().cpu().contiguous().numpy().tobytes())
    arrays['metadata'] = np.asarray(json.dumps({
        'schema': 'native_auxiliary_validation_v1', 'observation_schema': env.observation_schema,
        'transition_target_schema': 'public_damage_result_v2',
        'seed_base': seed_base, 'initial_seed_count': seed_count, 'samples': len(rows),
        'auxiliary_horizon': horizon, 'decision_ticks': config['decision_ticks'],
        'opponent': 'legal_random', 'learner': 'deterministic_collection_checkpoint',
        'collection_policy': config['policy'], 'collection_weights_sha256': weights.hexdigest(),
        'collection_checkpoint': collection_checkpoint,
        'scope': 'Public dynamics and public belief reconstruction; no true hidden-state labels.',
    }))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    return output


def score_dataset(config, model, dataset, batch_size=64):
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if not hasattr(model.policy, 'auxiliary_predictions'):
        raise ValueError('checkpoint has no auxiliary prediction head')
    model.policy.set_training_mode(False)
    collected = {}
    with np.load(dataset, allow_pickle=False) as archive:
        # Materialize each compressed array once, rather than once per minibatch.
        data = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(data['metadata']))
    if metadata['observation_schema'] != model.policy.features_extractor.observation_schema:
        raise ValueError('dataset observation schema differs from checkpoint')
    if metadata['transition_target_schema'] != 'public_damage_result_v2':
        raise ValueError('dataset transition target schema differs from training')
    if metadata['decision_ticks'] != config['decision_ticks']:
        raise ValueError('dataset decision interval differs from checkpoint')
    if metadata['auxiliary_horizon'] != config['auxiliary_horizon']:
        raise ValueError('dataset prediction horizon differs from checkpoint')
    with torch.no_grad():
        # Rebuild THIS checkpoint's recurrent memory on recorded observations.
        # Collection-model hidden states are not portable between checkpoints.
        states = None
        prior_wave = None
        groups = dict.fromkeys(zip(data['wave'].tolist(), data['decision'].tolist()))
        for wave, decision in groups:
            batch = np.flatnonzero((data['wave'] == wave) & (data['decision'] == decision))
            if wave != prior_wave:
                lstm = model.policy.lstm_actor
                states = tuple(torch.zeros(lstm.num_layers, len(batch), lstm.hidden_size,
                                           device=model.device) for _ in range(2))
                prior_wave = wave
            next_states = [[], []]
            for offset in range(0, len(batch), batch_size):
                chosen = batch[offset:offset + batch_size]
                current = {key[4:]: torch.as_tensor(data[key][chosen], device=model.device)
                           for key in data if key.startswith('obs_')}
                initial = tuple(value[:, offset:offset + len(chosen)] for value in states)
                starts = torch.as_tensor(data['episode_start'][chosen], dtype=torch.float32, device=model.device)
                _, next_state = model.policy.get_distribution(current, initial, starts)
                for index, value in enumerate(next_state):
                    next_states[index].append(value)
                actions = torch.as_tensor(data['executed_action'][chosen], device=model.device)
                predictions = model.policy.auxiliary_predictions(
                    executed_actions=actions if model.policy.action_conditioned else None)
                for key, value in predictions.items():
                    collected.setdefault(key, []).append(value.cpu())
            states = tuple(torch.cat(values, dim=1) for values in next_states)
    pred = {key: torch.cat(value) for key, value in collected.items()}
    transition = torch.tensor(data['transition_targets'])
    targets = torch.tensor(data['entity_targets'])
    valid = (targets[..., 3] > .5) & (torch.tensor(data['obs_entity_mask']) > .5)
    survivor = valid & (targets[..., 0] < .5)
    future = torch.tensor(data['future_targets'])
    scalar = torch.tensor(data['obs_scalar'])
    belief = torch.tensor(data['obs_public_belief'])
    def mean(value, mask=None):
        selected = value if mask is None else value[mask]
        return float(selected.mean()) if selected.numel() else None
    death = pred['entity_death'] >= 0
    actual_death = targets[..., 0] > .5
    next_targets = belief[:, 8:16]
    next_valid = next_targets.sum(-1) > 0
    metrics = {
        'tower_delta_mae': mean((pred['tower_delta'] - transition[:, :2]).abs().mean(-1), transition[:, 7] > .5),
        'damage_result_accuracy': mean((pred['damage_source'].argmax(-1) == transition[:, 2:7].argmax(-1)).float(), transition[:, 7] > .5),
        'entity_disappearance_accuracy': mean((death == actual_death).float(), valid),
        'entity_disappearance_bce': mean(F.binary_cross_entropy_with_logits(pred['entity_death'], targets[..., 0], reduction='none'), valid),
        'entity_position_delta_mae': mean((pred['entity_position_delta'] - targets[..., 1:3]).abs().mean(-1), survivor),
        'future_scene_mae': mean((pred['future_value'] - future[:, 0]).abs(), future[:, 1] > .5),
        'public_elixir_bin_accuracy': mean((pred['opponent_elixir'].argmax(-1) == (scalar[:, 1]*5).long().clamp(0, 4)).float()),
        'public_hand_belief_bce': mean(F.binary_cross_entropy_with_logits(pred['opponent_hand'], belief[:, :8], reduction='none')),
        'public_next_card_cross_entropy': mean(-(next_targets * F.log_softmax(pred['opponent_next_card'], -1)).sum(-1), next_valid),
        'entity_labels': int(valid.sum()), 'surviving_entity_labels': int(survivor.sum()),
        'future_labels': int((future[:, 1] > .5).sum()),
    }
    by_role = {}
    for role in (0, 1):
        selected = torch.tensor(data['role'] == role)
        by_role[f'p{role}'] = {'samples': int(selected.sum()),
                              'tower_delta_mae': mean((pred['tower_delta'] - transition[:, :2]).abs().mean(-1), selected)}
    return {'dataset': metadata, 'dataset_sha256': _sha256(dataset), 'metrics': metrics, 'by_role': by_role,
            'inference_only': True, 'training_updates': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('collect', 'evaluate'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps'))
    parser.add_argument('--port', type=int)
    parser.add_argument('--seed-base', type=int, default=2_900_001)
    parser.add_argument('--seed-count', type=int, default=8)
    parser.add_argument('--decisions', type=int, default=128)
    parser.add_argument('--battles', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    config = checkpoint_config(args.checkpoint, args.config)
    model = load_policy(args.checkpoint, config, args.device)
    with preserve_random_state():
        if args.mode == 'collect':
            collect_dataset(config, model, args.dataset, args.seed_base, args.seed_count,
                            args.decisions, args.battles, args.port,
                            checkpoint_spec(args.checkpoint, config['algorithm']))
            print(args.dataset.resolve())
        else:
            result = score_dataset(config, model, args.dataset, args.batch_size)
            result['checkpoint'] = checkpoint_spec(args.checkpoint, config['algorithm'])
            output = args.output or args.dataset.with_suffix('.metrics.json')
            _atomic_json(output, result)
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
