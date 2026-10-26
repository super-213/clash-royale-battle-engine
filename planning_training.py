"""Sequence learning and search-improved native self play, independent of PPO."""
from __future__ import annotations

from pathlib import Path
import copy
import json
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .native_search import SampledSearch
from .planning_policy import NativePlanningAgent
from .planning_replay import PlanningStep, SequenceReplay
from .planning_rules import canonical_actions, canonical_mask, proposal_log_probabilities


def validate_planning_config(config):
    positive = ('planning_hidden_dim', 'planning_spatial_dim', 'planning_chance_codes',
                'planning_depth', 'planning_candidates', 'planning_internal_candidates',
                'planning_outcome_samples', 'planning_replay_capacity', 'planning_unroll',
                'planning_n_steps', 'planning_updates', 'planning_batch_size')
    for key in (*positive, 'planning_budget', 'planning_burn_in', 'planning_reanalyze_every', 'planning_warmup_rounds'):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key in positive else 0):
            raise ValueError(f'invalid {key}')
    for key in ('planning_exploration', 'planning_temperature', 'planning_grad_clip'):
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f'invalid {key}')
    if config['planning_chance_codes'] < 2 or min(config['planning_candidates'], config['planning_internal_candidates']) < 2:
        raise ValueError('planning needs at least two codes and candidates')
    if (isinstance(config['cross_modal_heads'], bool) or not isinstance(config['cross_modal_heads'], int)
            or config['cross_modal_heads'] <= 0):
        raise ValueError('planning attention heads must be a positive integer')
    if config['planning_spatial_dim'] % 8 or config['planning_spatial_dim'] % config['cross_modal_heads']:
        raise ValueError('planning spatial dimension must divide by 8 and attention heads')
    if not 0 < float(config['planning_target_tau']) <= 1:
        raise ValueError('invalid planning target tau')
    if config['planning_replay_capacity'] < config['battles'] * (config['planning_unroll'] + 1):
        raise ValueError('planning replay is too small for the configured sequences')


def agent_from_config(config):
    return NativePlanningAgent(
        model_settings=dict(hidden_dim=config['planning_hidden_dim'], spatial_dim=config['planning_spatial_dim'],
                            chance_codes=config['planning_chance_codes'], decision_ticks=config['decision_ticks'],
                            entity_layers=config['cross_modal_entity_layers'], attention_heads=config['cross_modal_heads']),
        search_settings=dict(budget=config['planning_budget'], depth=config['planning_depth'],
                             candidates=config['planning_candidates'], internal_candidates=config['planning_internal_candidates'],
                             outcome_samples=config['planning_outcome_samples'], discount=config['gamma'],
                             exploration=config['planning_exploration'], temperature=config['planning_temperature']),
        device=config['device'], seed=config['seed'])


def single_observation(observation):
    return {key: np.asarray(value)[None] for key, value in observation.items()}


class PlanningLearner:
    def __init__(self, agent, learning_rate=.0003, discount=.997, n_steps=5, target_tau=.01, grad_clip=1.):
        self.agent = agent
        self.target = copy.deepcopy(agent.policy).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(agent.policy.parameters(), lr=learning_rate)
        self.discount, self.n_steps = discount, n_steps
        self.target_tau, self.grad_clip = target_tau, grad_clip
        self.updates = 0

    @staticmethod
    def burn_in(model, prefix):
        memory = None
        with torch.no_grad():
            for step in prefix:
                memory = model.encode(single_observation(step.observation), memory).memory
        return memory

    def train_batch(self, samples, unroll):
        model = self.agent.policy
        model.train()
        totals = {}
        valid_count = 0
        normalizer = sum(min(unroll, len(sequence)) for _prefix, sequence in samples)
        if not normalizer:
            raise ValueError('empty planning minibatch')
        self.optimizer.zero_grad()
        for prefix, sequence in samples:
            if not sequence:
                continue
            memory = self.burn_in(model, prefix)
            predicted = model.encode(single_observation(sequence[0].observation), memory)
            with torch.no_grad():
                target_memory = self.burn_in(self.target, prefix)
                actual = [self.target.encode(single_observation(sequence[0].observation), target_memory)]
                for step in sequence:
                    actual.append(self.target.encode(single_observation(step.next_observation), actual[-1].memory))
                values = [float(self.target.evaluate(state)['value'][0]) for state in actual]
            for offset, step in enumerate(sequence[:unroll]):
                outputs = model.evaluate(predicted)
                device = model.device
                candidates = torch.as_tensor(step.search['candidates'], device=device, dtype=torch.long)
                probabilities = torch.as_tensor(step.search['probabilities'], device=device)
                # Conditional loss on the actual proposed set; absence from a
                # sampled set is not a negative label for every other action.
                actual_mask = torch.as_tensor(canonical_mask(single_observation(step.observation))[0], device=device)
                policy_loss = -(probabilities * proposal_log_probabilities(outputs['policy'][0], candidates, actual_mask)).sum()
                return_target, weight = 0., 1.
                endpoint = offset
                for endpoint in range(offset, min(len(sequence), offset + self.n_steps)):
                    future = sequence[endpoint]
                    return_target += weight * future.reward
                    weight *= self.discount
                    if future.done:
                        weight = 0.; break
                return_target += weight * values[endpoint + 1]
                value_loss = F.smooth_l1_loss(outputs['value'], outputs['value'].new_tensor([return_target]))
                opponent_loss = -outputs['opponent'][0, step.opponent_action]
                successor, dynamics = model.dynamics(predicted, [step.action], [step.opponent_action],
                                                      target=actual[offset + 1], ticks=[step.ticks])
                next_actual = actual[offset + 1]
                prior_log = F.log_softmax(dynamics['prior'], -1)
                posterior_log = F.log_softmax(dynamics['posterior'], -1)
                # Balanced KL: prior learns outcomes; posterior stays supported
                # by the online prior instead of encoding an unavailable oracle.
                posterior_p = posterior_log.exp()
                kl = .8 * (posterior_p.detach() * (posterior_log.detach() - prior_log)).sum(-1).mean()
                kl = kl + .2 * (posterior_p * (posterior_log - prior_log.detach())).sum(-1).mean()
                latent_loss = F.mse_loss(successor.memory, next_actual.memory)
                latent_loss = latent_loss + F.mse_loss(successor.board, next_actual.board)
                reward_loss = F.smooth_l1_loss(dynamics['reward'], dynamics['reward'].new_tensor([step.reward]))
                terminal_loss = F.binary_cross_entropy_with_logits(
                    dynamics['terminal'], dynamics['terminal'].new_tensor([float(step.done)]))
                success_loss = F.binary_cross_entropy_with_logits(
                    dynamics['success'], dynamics['success'].new_tensor([float(step.accepted)]))
                next_loss = F.cross_entropy(dynamics['next_card'], next_actual.next_card.argmax(-1))
                scalar_loss = F.smooth_l1_loss(successor.scalar, next_actual.scalar)
                belief_loss = F.binary_cross_entropy(successor.belief.clamp(1e-6, 1 - 1e-6), next_actual.belief)
                spatial_loss = F.binary_cross_entropy_with_logits(
                    dynamics['spatial'], torch.as_tensor(step.next_observation['spatial'][None], device=device))
                next_mask = torch.as_tensor(canonical_mask(single_observation(step.next_observation))[:, 1:],
                                            device=device, dtype=torch.float32)
                # Affordability/hand constraints are handled by explicit rules.
                # Train tile validity only for cards actually available next.
                available = (next_actual.hand > .5) & (next_actual.scalar[:, :1] + 1e-6 >= model.costs)
                tile_error = F.binary_cross_entropy_with_logits(dynamics['mask'], next_mask, reduction='none')
                tile_error = tile_error.reshape(-1, 8, 576).mean(-1)
                mask_loss = (tile_error * available).sum() / available.sum().clamp_min(1)
                losses = dict(policy=policy_loss, value=value_loss, opponent=opponent_loss,
                              latent=latent_loss, chance=kl, reward=reward_loss, terminal=terminal_loss,
                              success=success_loss, next_card=next_loss, scalar=scalar_loss,
                              belief=belief_loss, spatial=spatial_loss, legality=mask_loss)
                loss = sum(losses.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite planning loss')
                # Short tails contribute only real steps, never padded weight.
                (loss / normalizer).backward(retain_graph=offset + 1 < min(unroll, len(sequence)))
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach())
                valid_count += 1
                predicted = successor
        if not valid_count:
            raise ValueError('empty planning minibatch')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), model.parameters()):
                target.lerp_(source, self.target_tau)
            for target, source in zip(self.target.buffers(), model.buffers()):
                target.copy_(source)
        self.updates += 1
        return {**{key: value / valid_count for key, value in totals.items()},
                'grad_norm': float(norm), 'valid_steps': valid_count}

    @torch.no_grad()
    def reanalyze(self, samples):
        self.agent.policy.eval()
        for prefix, sequence in samples:
            if not sequence:
                continue
            memory = self.burn_in(self.agent.policy, prefix)
            step = sequence[0]
            root = self.agent.policy.encode(single_observation(step.observation), memory)
            masks = torch.as_tensor(canonical_mask(single_observation(step.observation)), device=self.agent.device)
            rng = np.random.default_rng(int(self.agent.rng.integers(2**32)))
            _, records = SampledSearch(self.agent.policy, self.agent.search_settings).run(root, masks, [rng])
            step.search = records[0]
            step.model_version = self.agent.version

    def state_dict(self):
        return dict(target=self.target.state_dict(), optimizer=self.optimizer.state_dict(), updates=self.updates)

    def load_state_dict(self, payload):
        self.target.load_state_dict(payload['target'])
        self.optimizer.load_state_dict(payload['optimizer'])
        self.updates = payload['updates']


def save_training_state(path, agent, learner, replay, config, env, pool=None):
    from training_runtime import resume_directory, _atomic_json, _sha256
    directory = resume_directory(path)
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json(directory / 'config.json', config)
    # state_dict excludes active mappings. Saving never modifies live replay;
    # loading starts fresh native episodes, detached from all saved trajectories.
    payload = dict(schema='planning_training_state_v1', learner=learner.state_dict(), replay=replay.state_dict(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                   next_seed=env.get_attr('_seed_cursor')[0])
    if torch.backends.mps.is_available():
        payload['mps_rng'] = torch.mps.get_rng_state()
    temporary = directory / 'training.pt.tmp'
    torch.save(payload, temporary); temporary.replace(directory / 'training.pt')
    if pool is not None:
        pool.save_state(directory / 'league')
    from shutil import copyfile
    copyfile(config['_initial_policy_path'], directory / 'initial_policy.zip')
    _atomic_json(directory / 'runtime.json', dict(schema='planning_resume_v1', model_sha256=_sha256(path),
                                                training_sha256=_sha256(directory / 'training.pt'),
                                                config_sha256=_sha256(directory / 'config.json'),
                                                initial_policy_sha256=_sha256(directory / 'initial_policy.zip'),
                                                league_sha256=_sha256(directory / 'league/state.json') if pool is not None else None))


def restore_training_state(path, learner, config, env, pool=None):
    from training_runtime import resume_directory, _sha256
    directory = resume_directory(path)
    manifest = json.loads((directory / 'runtime.json').read_text())
    if (manifest.get('schema') != 'planning_resume_v1' or manifest['model_sha256'] != _sha256(path)
            or manifest['training_sha256'] != _sha256(directory / 'training.pt')
            or manifest['config_sha256'] != _sha256(directory / 'config.json')
            or manifest['initial_policy_sha256'] != _sha256(directory / 'initial_policy.zip')
            or (pool is not None and manifest['league_sha256'] != _sha256(directory / 'league/state.json'))):
        raise ValueError('planning resume state identity mismatch')
    saved_config = json.loads((directory / 'config.json').read_text())
    allowed = {'rounds', 'device', 'port', 'evaluation_port', 'run_name', 'run_dir', 'created_at', 'resume_from',
               'auto_bootstrap', '_initial_policy_path', 'evaluation_initial_policy_path'}
    if any(saved_config.get(key) != config.get(key) for key in saved_config.keys() | config.keys() if key not in allowed):
        raise ValueError('planning resume configuration changed')
    payload = torch.load(directory / 'training.pt', map_location=learner.agent.device, weights_only=False)
    if payload['schema'] != 'planning_training_state_v1':
        raise ValueError('unsupported planning training state')
    learner.load_state_dict(payload['learner'])
    env.seed(payload['next_seed'])
    if pool is not None:
        pool.restore_state(directory / 'league')
    random.setstate(payload['python_rng']); np.random.set_state(payload['numpy_rng'])
    torch.set_rng_state(payload['torch_rng'].cpu())
    if 'mps_rng' in payload and torch.backends.mps.is_available():
        torch.mps.set_rng_state(payload['mps_rng'].cpu())
    return SequenceReplay.from_state_dict(payload['replay'])


def run_planning_training(config, run_dir):
    """Profile trainer; deferred evaluation shares the singleton native engine."""
    from training_runtime import load_object, require_training_device, _atomic_json, utc_now
    from self_play import SelfPlayOpponentPool
    from fixed_evaluation import evaluate_checkpoint, initial_policy_path
    import shutil

    validate_planning_config(config)
    require_training_device(config['device'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    agent = (NativePlanningAgent.load(config['resume_from'], device=config['device'])
             if config.get('resume_from') else agent_from_config(config))
    learner = PlanningLearner(agent, config['learning_rate'], config['gamma'], config['planning_n_steps'],
                              config['planning_target_tau'], config['planning_grad_clip'])
    pool = None
    if config.get('self_play_enabled'):
        pool = SelfPlayOpponentPool(run_dir, NativePlanningAgent, config['device'], config['battles'],
                                   seed=config['seed'] + 700001,
                                   random_ratio=config['self_play_random_ratio'], current_ratio=config['self_play_current_ratio'],
                                   historical_ratio=config['self_play_historical_ratio'],
                                   snapshot_pool_size=config['self_play_pool_size'],
                                   active_historical=config['self_play_active_historical'],
                                   loaded_model_limit=config['self_play_loaded_model_limit'], elo_k=config['self_play_elo_k'],
                                   mode=config.get('self_play_mode', 'league'))
        pool.set_current_model(agent)
    kwargs = dict(config.get('environment_kwargs', {}))
    kwargs.update(battles=config['battles'], port=config['port'], seed=config['seed'],
                  decision_ticks=config['decision_ticks'], planning_labels=True, opponent_policy=pool)
    env = load_object(config['environment'])(**kwargs)
    agent.training_env = env
    replay = SequenceReplay(config['planning_replay_capacity'])
    completed_rounds = agent.num_timesteps // config['rollout_decisions']
    if completed_rounds >= config['rounds']:
        env.close(); raise ValueError('target rounds must exceed completed rounds')
    if config.get('resume_from'):
        replay = restore_training_state(config['resume_from'], learner, config, env, pool)
    initial = run_dir / 'evaluation' / 'baselines' / 'initial_policy.zip'
    initial.parent.mkdir(parents=True, exist_ok=True)
    if config.get('resume_from'):
        # Preserve the actual original model, even when no evaluation is enabled.
        source = initial_policy_path({**config, 'evaluation_opponents': ['initial_policy']}, config['resume_from'])
        shutil.copyfile(source, initial)
    else:
        if config.get('planning_teacher'):
            from policy_inspection import checkpoint_config, load_policy
            teacher_config = checkpoint_config(config['planning_teacher'])
            teacher = load_policy(config['planning_teacher'], teacher_config, config['device'])
            copied = agent.policy.initialize_encoder(teacher)
            learner.target.load_state_dict(agent.policy.state_dict())
            _atomic_json(run_dir / 'teacher_initialization.json', dict(checkpoint=config['planning_teacher'], tensors=copied))
        agent.save(initial)
    config['_initial_policy_path'] = str(initial.resolve())
    _atomic_json(run_dir / 'config.json', config)
    if pool is not None and not pool.snapshots:
        pool.register_snapshot('round_000000', initial, 0)
    initial_checkpoint = run_dir / 'checkpoints' / 'round_000000.zip'
    if not completed_rounds:
        agent.save(initial_checkpoint)
    scheduled = [(0, initial_checkpoint)] if not completed_rounds else []
    started = time.monotonic()
    status = dict(status='running', updated_at=utc_now(), algorithm='public_stochastic_planning_v1')
    history = [[] for _ in range(config['battles'])]
    try:
        observations = env.reset()
        state = None; starts = np.ones(config['battles'], dtype=bool)
        for round_number in range(completed_rounds + 1, config['rounds'] + 1):
            # Reconstruct real memories under the updated network. Search states
            # are discarded; no predicted branch is committed to real history.
            if round_number > completed_rounds + 1:
                memories = []
                with torch.no_grad():
                    for episode in history:
                        memory = None
                        for obs in episode:
                            memory = agent.policy.encode(single_observation(obs), memory).memory
                        memories.append(np.zeros((1, agent.policy.hidden_dim), np.float32) if memory is None
                                        else memory.cpu().numpy())
                state = (np.concatenate(memories, axis=0)[None],)
            agent.search_enabled = round_number > config['planning_warmup_rounds']
            search_steps = search_depth = attempted_plays = accepted_plays = terminal_games = 0
            reward_total = 0.
            for _ in range(config['rollout_decisions'] // config['battles']):
                actions, state = agent.predict(observations, state, starts, deterministic=False)
                records = copy.deepcopy(agent.last_search)  # Mirror predict may overwrite diagnostics.
                identities = canonical_actions(observations['hand'], actions)
                after, rewards, dones, infos = env.step(actions)
                for slot, info in enumerate(infos):
                    labels = info.get('planning', {})
                    if labels.get('schema') != 'public_planning_transition_v1':
                        raise ValueError('environment did not provide planning transition labels')
                    obs = {key: value[slot].copy() for key, value in observations.items()}
                    replay.append(slot, PlanningStep(obs, labels['next_observation'], int(identities[slot]),
                                  int(labels['opponent_action']), labels['accepted'], float(rewards[slot]),
                                  bool(dones[slot]), labels['ticks'], records[slot], agent.version,
                                  info.get('opponent_id', 'random'), seed=int(info['seed']),
                                  role=int(info['learner_role'][1:])))
                    history[slot].append(obs)
                    if dones[slot]:
                        history[slot].clear()
                    search_steps += records[slot]['transitions']; search_depth = max(search_depth, records[slot]['depth'])
                    attempted_plays += int(identities[slot] > 0)
                    accepted_plays += int(labels['accepted'])
                    terminal_games += int(dones[slot])
                    reward_total += float(rewards[slot])
                observations, starts = after, dones
                agent.num_timesteps += config['battles']
            metrics = {}
            for _ in range(config['planning_updates']):
                samples = replay.sample(agent.rng, config['planning_batch_size'],
                                        config['planning_unroll'] + config['planning_n_steps'], config['planning_burn_in'])
                if config['planning_reanalyze_every'] and (learner.updates + 1) % config['planning_reanalyze_every'] == 0:
                    learner.reanalyze(samples[:1])
                metrics = learner.train_batch(samples, config['planning_unroll'])
            agent.version += 1
            agent.policy.eval()
            if pool is not None:
                pool.set_learner_generation(round_number)
            elapsed = time.monotonic() - started
            status.update(round=round_number, timesteps=agent.num_timesteps, replay_steps=len(replay),
                          elapsed_seconds=elapsed, search_transitions=search_steps, search_depth=search_depth,
                          attempted_plays=attempted_plays, accepted_plays=accepted_plays, terminal_games=terminal_games,
                          reward_total=reward_total,
                          updates=learner.updates, losses=metrics, updated_at=utc_now())
            _atomic_json(run_dir / 'status.json', status)
            with (run_dir / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(status, allow_nan=False) + '\n')
            print(f"[planning] round={round_number} timesteps={agent.num_timesteps} depth={search_depth} losses={metrics}", flush=True)
            checkpoint_due = round_number % config['checkpoint_every_rounds'] == 0 or round_number == config['rounds']
            snapshot_due = pool is not None and round_number % config['self_play_snapshot_every_rounds'] == 0
            evaluation_due = config.get('fixed_evaluation_enabled') and (
                round_number % config['evaluation_every_rounds'] == 0 or round_number == config['rounds'])
            if checkpoint_due or snapshot_due or evaluation_due:
                path = run_dir / 'checkpoints' / f'round_{round_number:06d}.zip'
                agent.search_enabled = True
                agent.save(path)
                if snapshot_due:
                    pool.register_snapshot(f'round_{round_number:06d}', path, round_number)
                if evaluation_due:
                    scheduled.append((round_number, path))
                save_training_state(path, agent, learner, replay, config, env, pool)
        agent.search_enabled = True
        final = run_dir / 'final_model.zip'
        agent.save(final)
        save_training_state(final, agent, learner, replay, config, env, pool)
    except BaseException as error:
        status.update(status='failed', error=repr(error), updated_at=utc_now())
        _atomic_json(run_dir / 'status.json', status)
        raise
    finally:
        env.close()
    if config.get('fixed_evaluation_enabled'):
        try:
            for round_number, checkpoint in scheduled:
                evaluate_checkpoint(config, checkpoint, run_dir / 'fixed_evaluations' / f'round_{round_number:06d}')
        except BaseException as error:
            status.update(status='evaluation_failed', model=str(final.resolve()), error=repr(error), updated_at=utc_now())
            _atomic_json(run_dir / 'status.json', status)
            raise
    status.update(status='completed', model=str(final.resolve()), updated_at=utc_now())
    _atomic_json(run_dir / 'status.json', status)
    return status
