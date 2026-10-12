"""Batched sampled PUCT with opponent/chance expectation nodes.

No minimax sign flips: rewards stay in the root player's frame. Opponent draws
depend on the current public state, never the concurrently selected own action.
The budget counts every model transition, not just completed simulations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from .planning_model import PlanningState
from .planning_rules import ACTION_COUNT, candidate_set, proposal_log_probabilities


@dataclass
class SearchSettings:
    budget: int = 64
    depth: int = 6
    candidates: int = 16
    internal_candidates: int = 8
    outcome_samples: int = 2
    exploration: float = 1.5
    discount: float = .997
    temperature: float = 1.
    terminal_threshold: float = .95

    def __post_init__(self):
        for name in ('budget', 'depth', 'candidates', 'internal_candidates', 'outcome_samples'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (2 if 'candidates' in name else 0):
                raise ValueError(f'invalid search {name}')
        if self.depth < 1 or not 0 < self.discount <= 1:
            raise ValueError('invalid search depth or discount')
        if self.outcome_samples < 1:
            raise ValueError('outcome_samples must be positive')
        if not math.isfinite(self.exploration) or self.exploration <= 0:
            raise ValueError('search exploration must be finite and positive')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('search temperature must be finite and positive')
        if not 0 < self.terminal_threshold <= 1:
            raise ValueError('invalid terminal threshold')


@dataclass
class Edge:
    action: int
    prior: float
    visits: int = 0
    total: float = 0.
    children: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)

    @property
    def value(self):
        return self.total / self.visits if self.visits else 0.


@dataclass
class Node:
    state: PlanningState
    value: float
    opponent: np.ndarray
    edges: list


class SampledSearch:
    def __init__(self, model, settings=None):
        self.model = model
        self.settings = settings or SearchSettings()

    def nodes(self, states, masks, rngs, root=False):
        outputs = self.model.evaluate(states)
        logits = outputs['policy'].masked_fill(~masks, -1e9)
        own = logits.softmax(-1).cpu().numpy()
        opponent = outputs['opponent'].softmax(-1).cpu().numpy()
        values = outputs['value'].cpu().numpy()
        nodes = []
        for row, rng in enumerate(rngs):
            limit = self.settings.candidates if root else self.settings.internal_candidates
            actions = candidate_set(own[row], masks[row].cpu().numpy(), limit, rng)
            prior = proposal_log_probabilities(logits[row], torch.as_tensor(actions, device=logits.device)).exp().cpu().numpy()
            nodes.append(Node(states.select(slice(row, row + 1)).detach(), float(values[row]), opponent[row],
                              [Edge(int(action), float(p)) for action, p in zip(actions, prior)]))
        return nodes

    def select(self, node):
        visits = sum(edge.visits for edge in node.edges)
        known = [edge.value for edge in node.edges if edge.visits]
        low, high = (min(known), max(known)) if known else (0., 0.)
        def score(edge):
            q = edge.value if edge.visits else node.value
            normalized = (q - low) / (high - low) if high - low > 1e-6 else 0.
            return normalized + self.settings.exploration * edge.prior * math.sqrt(visits + 1) / (edge.visits + 1)
        return max(node.edges, key=score)

    def backup(self, path, value):
        for edge, reward, continuation in reversed(path):
            value = reward + self.settings.discount * continuation * value
            edge.visits += 1
            edge.total += value

    @torch.no_grad()
    def run(self, states, root_masks, rngs, deterministic=False):
        roots = self.nodes(states, root_masks, rngs, root=True)
        used = np.zeros(len(roots), dtype=np.int64)
        max_depth = np.zeros(len(roots), dtype=np.int64)
        paths = [[] for _ in roots]
        current = roots.copy()
        # At most one new transition per root per loop; leaves from independent
        # native battles are combined into a single neural dynamics call.
        while np.any(used < self.settings.budget):
            requests = []
            for row, root in enumerate(roots):
                if used[row] >= self.settings.budget:
                    continue
                node = current[row]
                edge = self.select(node)
                opponent_p = node.opponent.astype(np.float64)
                opponent_p /= opponent_p.sum()
                scenario = (edge.scenarios[int(rngs[row].integers(len(edge.scenarios)))]
                            if len(edge.scenarios) >= self.settings.outcome_samples else None)
                enemy = scenario[0] if scenario else int(rngs[row].choice(ACTION_COUNT, p=opponent_p))
                requests.append((row, node, edge, enemy, scenario))
            if not requests:
                break
            batch = PlanningState.stack([item[1].state for item in requests])
            own = [item[2].action for item in requests]
            enemy = [item[3] for item in requests]
            chance_p = self.model.chance_distribution(batch, own, enemy).softmax(-1).cpu().numpy()
            codes = []
            for offset, (row, _node, _edge, _enemy, scenario) in enumerate(requests):
                p = chance_p[offset].astype(np.float64); p /= p.sum()
                codes.append(scenario[1] if scenario else int(rngs[row].choice(len(p), p=p)))
            successor, outcomes = self.model.dynamics(batch, own, enemy, codes)
            masks = self.model.predicted_mask(successor)
            children = self.nodes(successor, masks, [rngs[item[0]] for item in requests])
            rewards = outcomes['reward'].cpu().numpy()
            terminals = outcomes['terminal'].sigmoid().cpu().numpy()
            for offset, (row, node, edge, enemy_action, scenario) in enumerate(requests):
                used[row] += 1
                key = (enemy_action, codes[offset])
                if scenario is None:
                    # Equal-weight Monte Carlo samples, including duplicates.
                    # A finite outcome bank lets finite-budget search go deep.
                    edge.scenarios.append(key)
                previous = edge.children.get(key)
                # Deterministic conditional dynamics: identical public node,
                # joint action and outcome code identify the same child.
                if previous is None:
                    child = children[offset]
                    terminal = float(terminals[offset]) >= self.settings.terminal_threshold
                    edge.children[key] = (child, float(rewards[offset]), terminal)
                else:
                    child, _, _ = previous
                child, reward, terminal = edge.children[key]
                paths[row].append((edge, reward, 0. if terminal else 1.))
                depth = len(paths[row]); max_depth[row] = max(max_depth[row], depth)
                if terminal or depth >= self.settings.depth or previous is None:
                    self.backup(paths[row], 0. if terminal else child.value)
                    paths[row] = []; current[row] = roots[row]
                else:
                    current[row] = child
                if used[row] >= self.settings.budget and paths[row]:
                    self.backup(paths[row], child.value)
                    paths[row] = []; current[row] = roots[row]
        decisions, records = [], []
        for row, root in enumerate(roots):
            counts = np.asarray([edge.visits for edge in root.edges], dtype=np.float64)
            if counts.sum():
                logs = np.where(counts > 0, np.log(np.maximum(counts, 1)) / self.settings.temperature, -np.inf)
                weights = np.exp(logs - logs.max())
                target = weights / weights.sum()
            else:
                target = np.asarray([edge.prior for edge in root.edges])
                target /= target.sum()
            chosen = int(np.argmax(counts if counts.sum() else target)) if deterministic else int(
                rngs[row].choice(len(target), p=target))
            decisions.append(root.edges[chosen].action)
            records.append(dict(candidates=np.asarray([edge.action for edge in root.edges], dtype=np.int64),
                                probabilities=target.astype(np.float32), visits=counts.astype(np.int64),
                                values=np.asarray([edge.value for edge in root.edges], dtype=np.float32),
                                priors=np.asarray([edge.prior for edge in root.edges], dtype=np.float32),
                                root_value=float(sum(edge.total for edge in root.edges) / max(counts.sum(), 1))
                                if counts.sum() else root.value,
                                transitions=int(used[row]), depth=int(max_depth[row])))
        return np.asarray(decisions, dtype=np.int64), records
