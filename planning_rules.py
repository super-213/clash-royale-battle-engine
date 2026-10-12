"""Public action contracts shared by planning, replay and native collection.

Tree actions use card identity, never a hand slot: 0 is wait, otherwise
1 + (card_token - 1) * 576 + y * 18 + x, in the learner's coordinates.
"""
from __future__ import annotations

import numpy as np
import torch

from .native_vec_env import ARENA_HEIGHT, ARENA_WIDTH, CARD_IDS, SPELL_IDS

TILES = ARENA_HEIGHT * ARENA_WIDTH
ACTION_COUNT = 1 + len(CARD_IDS) * TILES
CARD_COSTS = (3., 3., 3., 3., 4., 5., 4., 4.)
SPELL_TOKENS = tuple(index + 1 for index, card in enumerate(CARD_IDS) if card in SPELL_IDS)


def canonical_actions(hand, actions):
    """Convert validated public [slot,y,x] actions to stable identities."""
    hand = np.asarray(hand)
    actions = np.asarray(actions)
    if actions.shape != (len(hand), 3) or not np.isfinite(actions).all():
        raise ValueError('invalid planning actions')
    if not np.equal(actions, np.floor(actions)).all():
        raise ValueError('planning actions must be integral')
    slots, y, x = actions.astype(np.int64).T
    if np.any((slots < 0) | (slots > 4) | (y < 0) | (y >= ARENA_HEIGHT)
              | (x < 0) | (x >= ARENA_WIDTH)):
        raise ValueError('planning action out of range')
    cards = hand[np.arange(len(hand)), np.maximum(slots - 1, 0)]
    if np.any((slots > 0) & ((cards < 1) | (cards > len(CARD_IDS)))):
        raise ValueError('planning action selects an empty hand slot')
    if np.any((slots == 0) & ((x != 0) | (y != 0))):
        raise ValueError('wait must use zero coordinates')
    return np.where(slots == 0, 0, 1 + (cards - 1) * TILES + y * ARENA_WIDTH + x).astype(np.int64)


def public_actions(hand, identities):
    hand, identities = np.asarray(hand), np.asarray(identities, dtype=np.int64)
    result = np.zeros((len(hand), 3), dtype=np.int64)
    for row, action in enumerate(identities):
        if not 0 <= action < ACTION_COUNT:
            raise ValueError('invalid canonical action')
        if action == 0:
            continue
        card, tile = divmod(int(action) - 1, TILES)
        slots = np.flatnonzero(hand[row, :4] == card + 1)
        if not len(slots):
            raise ValueError('candidate card is absent from the observed hand')
        result[row] = (slots[0] + 1, tile // ARENA_WIDTH, tile % ARENA_WIDTH)
    return result


def canonical_mask(observations):
    hand = np.asarray(observations['hand'])
    masks = np.asarray(observations['action_mask']).reshape(len(hand), 5, TILES)
    result = np.zeros((len(hand), ACTION_COUNT), dtype=bool)
    result[:, 0] = True
    for row in range(len(hand)):
        for slot, card in enumerate(hand[row, :4], 1):
            if card:
                begin = 1 + (int(card) - 1) * TILES
                result[row, begin:begin + TILES] |= masks[row, slot].astype(bool)
    return result


def executed_identity(features, mirror=False):
    """Native accepted action to canonical action; rotate opponent coordinates."""
    features = np.asarray(features)
    if features.shape != (11,) or not np.isfinite(features).all():
        raise ValueError('invalid executed action features')
    if features[0] > .5:
        return 0
    card = int(features[1:9].argmax())
    x, y = features[-2:]
    if mirror:
        x, y = 1 - x, 1 - y
    tile_x = int(np.clip(np.floor(x * ARENA_WIDTH), 0, ARENA_WIDTH - 1))
    tile_y = int(np.clip(np.floor(y * ARENA_HEIGHT), 0, ARENA_HEIGHT - 1))
    return 1 + card * TILES + tile_y * ARENA_WIDTH + tile_x


def action_features(actions, device=None):
    actions = torch.as_tensor(actions, device=device, dtype=torch.long)
    cards = torch.where(actions == 0, 0, (actions - 1) // TILES + 1)
    tiles = (actions - 1).clamp_min(0) % TILES
    coordinates = torch.stack(((tiles % ARENA_WIDTH + .5) / ARENA_WIDTH,
                               (tiles // ARENA_WIDTH + .5) / ARENA_HEIGHT), -1)
    coordinates = coordinates * (actions != 0).unsqueeze(-1)
    return torch.cat((torch.nn.functional.one_hot(cards, 9).float(), coordinates), -1)


def proposal_log_probabilities(logits, candidates, legal=None):
    """Stratified restriction: preserve card mass while restricting its tiles.

    Plain renormalization over 16 tiles and wait would give wait almost all mass
    even when the original card distribution strongly prefers playing. This
    operation is also used by search-policy distillation, so targets and loss
    share exactly the same conditional action-set semantics.
    """
    if legal is not None:
        logits = logits.masked_fill(~legal, -1e9)
    groups = torch.where(candidates == 0, 0, (candidates - 1) // TILES + 1)
    result = logits[candidates].clone()
    for group in groups.unique().tolist():
        if group == 0:
            continue
        selected = groups == group
        begin = 1 + (group - 1) * TILES
        result[selected] = (logits[candidates[selected]] - torch.logsumexp(logits[candidates[selected]], 0)
                            + torch.logsumexp(logits[begin:begin + TILES], 0))
    return torch.log_softmax(result, 0)


def candidate_set(probabilities, legal, limit, rng=None):
    """Per-card diverse proposals, wait and best action; stochastic exploration.

    Proposals are a restricted action set, not a proof about unproposed moves.
    The learner uses a conditional policy loss on the recorded set.
    """
    p, legal = np.asarray(probabilities, dtype=np.float64), np.asarray(legal, dtype=bool)
    if p.shape != (ACTION_COUNT,) or legal.shape != p.shape or limit < 2:
        raise ValueError('invalid candidate request')
    valid = np.flatnonzero(legal)
    if not len(valid) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('invalid candidate probabilities or support')
    chosen = [0] if legal[0] else []
    best = int(valid[np.argmax(p[valid])])
    if best not in chosen:
        chosen.append(best)
    # One distinct high-probability placement per playable card before filling.
    for card in range(8):
        indices = valid[(valid >= 1 + card * TILES) & (valid < 1 + (card + 1) * TILES)]
        if len(indices) and len(chosen) < limit:
            action = int(indices[np.argmax(p[indices])])
            if action not in chosen:
                chosen.append(action)
    remaining = np.setdiff1d(valid, chosen)
    if rng is not None and len(remaining) and len(chosen) < limit:
        # Reserve one uniform proposal so low-prior actions remain discoverable.
        action = int(rng.choice(remaining))
        chosen.append(action)
        remaining = remaining[remaining != action]
    for action in remaining[np.argsort(-p[remaining], kind='stable')]:
        if len(chosen) >= limit:
            break
        # Avoid spending the entire budget on adjacent points of one card.
        tile = (int(action) - 1) % TILES
        if action and any(other and (other - 1) // TILES == (action - 1) // TILES
                          and abs((other - 1) % TILES // ARENA_WIDTH - tile // ARENA_WIDTH) <= 1
                          and abs((other - 1) % ARENA_WIDTH - tile % ARENA_WIDTH) <= 1
                          for other in chosen):
            continue
        chosen.append(int(action))
    return np.asarray(chosen, dtype=np.int64)
