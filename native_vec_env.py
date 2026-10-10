"""Stable-Baselines3 VecEnv backed by one 16-slot native batch connection.

The important property of this adapter is that one policy decision produces one
``Batch.step`` call for every live battle.  It is not a collection of synchronous
single-battle environments hidden behind ``DummyVecEnv``.

The observation is intentionally split along the model boundaries used by the
training project:

* ``spatial`` is a 16-channel arena tensor for a CNN;
* ``entities`` plus ``entity_mask`` are public entity tokens for attention;
* ``action_history`` is an ordered public event sequence for an RNN;
* ``scalar`` and ``hand`` carry the non-spatial current context;
* ``action_mask`` is the conditional slot/position legality mask.

This module exposes only public opponent information.  In particular, the
opponent elixir scalar is reconstructed from public card plays rather than read
from the opponent's native player record.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations
from typing import Any, Callable, Sequence

from gymnasium import spaces
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from policy_contracts import validated_actions

from .batch import Batch
from .engine import Play


ARENA_HEIGHT = 32
ARENA_WIDTH = 18
SPATIAL_CHANNELS = 16
MAX_ENTITIES = 128
ENTITY_FEATURES = 24
HISTORY_LENGTH = 32
HISTORY_FEATURES = 8
TEMPORAL_FEATURES = 20
ENTITY_TARGET_SCHEMA = "native_entity_id_v1"
EXECUTED_ACTION_SCHEMA = "executed_card_xy_v1"
EXECUTED_ACTION_FEATURES = 11  # wait, eight card identities, normalized x/y
TRANSITION_TARGET_SCHEMA = "public_damage_result_v2"
NEXT_SCENE_TARGET_SCHEMA = "public_scene_value_v1"
FIRST_PLAYABLE_TICK = 90
TICK_SECONDS = 0.05
MATCH_SECONDS = 300.0

CARD_IDS = (
    26000000,  # Knight
    26000001,  # Archers
    26000005,  # Minions
    28000001,  # Arrows
    28000000,  # Fireball
    26000003,  # Giant
    26000014,  # Musketeer
    26000018,  # Mini P.E.K.K.A
)
CARD_TOKENS = {card_id: index + 1 for index, card_id in enumerate(CARD_IDS)}
SPELL_IDS = frozenset((28000000, 28000001))
_INITIAL_CYCLE_STATES = tuple(
    (frozenset(set(CARD_IDS).difference(queue)), queue)
    for queue in permutations(CARD_IDS, 4)
)


def scene_value(scalar):
    return scalar[..., 9] - scalar[..., 10] + .25 * (scalar[..., 2] - scalar[..., 3])


def _executed_action_features(owner, plays, receipts, events):
    """Encode the accepted learner play in its own arena coordinates.

    Failed or masked deployments have the same dynamics input as waiting.
    Card identity comes from the native receipt, before any hand cycling/reset.
    """
    features = np.zeros(EXECUTED_ACTION_FEATURES, dtype=np.float32)
    features[0] = 1.0
    for play, receipt in zip(plays, receipts):
        if play.owner != owner or receipt.status != 0:
            continue
        if not any(event.owner == owner and event.card == receipt.card_id for event in events):
            continue
        card_token = CARD_TOKENS.get(int(receipt.card_id))
        if card_token is None:
            raise ValueError("executed card is outside the fixed training deck")
        x, y = play.x, play.y
        if owner == 1:
            x, y = 18000 - x, 32000 - y
        features[0] = 0.0
        features[card_token] = 1.0
        features[-2:] = (x / 18000.0, y / 32000.0)
        break
    return features


def native_observation_space() -> spaces.Dict:
    """Return the version-1 native training observation contract."""
    return spaces.Dict({
        "spatial": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(SPATIAL_CHANNELS, ARENA_HEIGHT, ARENA_WIDTH),
            dtype=np.float32,
        ),
        "entities": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_ENTITIES, ENTITY_FEATURES),
            dtype=np.float32,
        ),
        "entity_mask": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(MAX_ENTITIES,),
            dtype=np.float32,
        ),
        "hand": spaces.Box(low=0, high=len(CARD_IDS), shape=(5,), dtype=np.int32),
        "scalar": spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32),
        "public_belief": spaces.Box(low=0.0, high=1.0, shape=(20,), dtype=np.float32),
        "action_history": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(HISTORY_LENGTH, HISTORY_FEATURES),
            dtype=np.float32,
        ),
        "temporal": spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(TEMPORAL_FEATURES,),
            dtype=np.float32,
        ),
        "action_mask": spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, ARENA_HEIGHT, ARENA_WIDTH),
            dtype=np.float32,
        ),
    })


@dataclass(slots=True)
class _HistoryEvent:
    tick: int
    owner: int
    card: int
    cost: int
    x: int = 0
    y: int = 0


@dataclass(slots=True)
class _Slot:
    index: int
    learner_owner: int = 0
    episode: int = 0
    seed: int = 0
    state: Any = None
    previous_state: Any = None
    initial_tower_hp: tuple[float, float] = (1.0, 1.0)
    initial_towers: tuple[Any, ...] = ()
    history: list[_HistoryEvent] = field(default_factory=list)
    ledger: list[_HistoryEvent] = field(default_factory=list)
    elixir_ticks: list[list[int]] = field(default_factory=lambda: [[], []])
    elixir_values: list[list[float]] = field(default_factory=lambda: [[], []])
    cycle_states: list[dict] = field(default_factory=list)
    cycle_beliefs: list[tuple] = field(default_factory=list)


def _selected(indices, count):
    if indices is None:
        return list(range(count))
    if isinstance(indices, (int, np.integer)):
        return [int(indices)]
    return [int(index) for index in indices]


def _tower_hp(state, owner):
    return float(sum(
        max(float(entity.hp or 0), 0.0)
        for entity in state.entities
        if int(entity.owner) == owner
        and int(entity.card) == -1
        and entity.entity_type == "building"
    ))


def _ordered_entities(state):
    """Keep label rows aligned with the public entity tokens."""
    return sorted(state.entities, key=lambda entity: int(entity.id))


def _entity_transition_targets(before, after, owner):
    """Training-only [death, dx, dy, valid] rows, matched by native ID.

    Look up survivors in the entire next state, including entities outside the
    observation's token cap. Native IDs are episode-local and never features.
    """
    targets = np.zeros((MAX_ENTITIES, 4), dtype=np.float32)
    if before.generation != after.generation:
        raise ValueError("entity targets require states from the same episode")
    future = {int(entity.id): entity for entity in after.entities}
    for row, entity in enumerate(_ordered_entities(before)[:MAX_ENTITIES]):
        # Preserve the existing exclusion of stationary buildings.
        if entity.entity_type == "building":
            continue
        targets[row, 3] = 1.0
        survivor = future.get(int(entity.id))
        if survivor is None:
            targets[row, 0] = 1.0
            continue
        # Use the same clipped, egocentric coordinates as entity tokens.
        for column, axis, scale in ((1, "x", 18000.0), (2, "y", 32000.0)):
            old_position = float(getattr(entity, axis)) / scale
            new_position = float(getattr(survivor, axis)) / scale
            if owner == 1:
                old_position, new_position = 1 - old_position, 1 - new_position
            targets[row, column] = (
                min(max(new_position, 0.0), 1.0)
                - min(max(old_position, 0.0), 1.0)
            )
    return targets


def _public_transition_targets(before, after, owner, initial_tower_hp):
    """[tower deltas (2), damage result scores (5), valid], including terminal.

    Unit damage uses ID-aligned HP loss, not net army HP or projectile expiry.
    Scores are fractions of each side's tower/army maximum HP, not attacker IDs.
    """
    if before.generation != after.generation:
        raise ValueError("transition targets require states from the same episode")
    deltas, unit_damage = [], []
    future = {int(entity.id): entity for entity in after.entities}
    for side in (owner, 1 - owner):
        deltas.append((_tower_hp(after, side) - _tower_hp(before, side))
                      / initial_tower_hp[side])
        army = [entity for entity in before.entities
                if int(entity.owner) == side and int(entity.card) != -1
                and entity.entity_type != "projectile" and entity.hp is not None]
        lost = 0.0
        for entity in army:
            survivor = future.get(int(entity.id))
            next_hp = max(0.0, float(survivor.hp or 0)) if survivor is not None else 0.0
            lost += max(0.0, float(entity.hp) - next_hp)
        scale = sum(max(float(entity.max_hp or entity.hp), 1.0) for entity in army)
        unit_damage.append(lost / max(scale, 1.0))
    return np.asarray((*np.clip(deltas, -1, 1), 0.0,
                       max(0.0, -deltas[0]), max(0.0, -deltas[1]),
                       *np.clip(unit_damage, 0, 1), 1.0), dtype=np.float32)


def _tower_position(entity, owner):
    if owner == 1:
        return 18000 - int(entity.x), 32000 - int(entity.y)
    return int(entity.x), int(entity.y)


def _troop_deployment_mask(state, owner, initial_towers):
    """Standard 18x32 arena, verified against native deployment snapping.

    Coordinates are egocentric: our king is near y=3, the river is y=15..16.
    Only towers observed at episode start can unlock enemy territory. A missing
    tower in a partial initial fixture must not be treated as a destroyed tower.
    """
    mask = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=bool)
    mask[0, 6:12] = True
    mask[1:14, :] = True
    mask[14, 1:17] = True
    live_towers = {
        entity.id: entity for entity in state.entities
        if entity.card == -1 and entity.entity_type == "building"
        and (entity.hp is None or entity.hp > 0)
    }
    for tower in live_towers.values():
        if tower.owner != owner:
            continue
        x, y = _tower_position(tower, owner)
        if (x, y) == (9000, 3000):
            mask[1:5, 7:11] = False
        elif y == 6500 and x in (3500, 14500):
            left = 2 if x == 3500 else 13
            mask[5:8, left:left + 3] = False

    for tower in initial_towers:
        if tower.owner == owner or tower.id in live_towers:
            continue
        x, y = _tower_position(tower, owner)
        if y != 25500 or x not in (3500, 14500):
            continue
        if x == 3500:
            mask[15:17, 3] = True
            mask[17, 1:9] = True
            mask[18:21, :9] = True
        else:
            mask[15:17, 14] = True
            mask[17, 9:17] = True
            mask[18:21, 9:] = True
    return mask


def _regenerated_elixir(start_seconds, end_seconds):
    regenerated = 0.0
    for left, right, seconds_per_elixir in (
        (0.0, 120.0, 2.8),
        (120.0, 240.0, 1.4),
        (240.0, float("inf"), 2.8 / 3.0),
    ):
        duration = max(0.0, min(end_seconds, right) - max(start_seconds, left))
        regenerated += duration / seconds_per_elixir
    return regenerated


def _advance_cycle_states(states, card):
    transitioned = {}
    for (hand, queue), weight in states.items():
        if card not in hand:
            continue
        next_state = (
            frozenset((*hand.difference((card,)), queue[0])),
            (*queue[1:], card),
        )
        transitioned[next_state] = transitioned.get(next_state, 0) + weight
    return transitioned


def _cycle_summary(states):
    hand_belief = np.zeros(len(CARD_IDS), dtype=np.float32)
    next_belief = np.zeros(len(CARD_IDS), dtype=np.float32)
    total = float(sum(states.values()))
    if total:
        for (hand, queue), weight in states.items():
            probability = weight / total
            for card in hand:
                hand_belief[CARD_TOKENS[card] - 1] += probability
            next_belief[CARD_TOKENS[queue[0]] - 1] += probability
    confidence = 1.0 - (
        np.log(max(len(states), 1)) / np.log(len(_INITIAL_CYCLE_STATES))
    )
    return tuple(hand_belief), tuple(next_belief), float(confidence)


@lru_cache(maxsize=8192)
def _cycle_belief(plays):
    """Reference reconstruction from a complete public play sequence."""
    states = {state: 1 for state in _INITIAL_CYCLE_STATES}
    for card in plays:
        states = _advance_cycle_states(states, card)
        if not states:
            break
    return _cycle_summary(states)


class NativeBatchVecEnv(VecEnv):
    """Sixteen native battles sharing one resident batch data-plane request.

    ``opponent_policy`` may be an SB3-like object with ``predict`` or a callable
    accepting ``(observation, action_mask, slot_index)``.  With no policy, a
    deterministic per-slot RNG samples legal public actions.
    """

    backend_name = "native_batch_v1"
    observation_schema = "native_model_v3"
    deployment_rules_version = "standard_1v1_dynamic_v1"
    render_mode = None

    def __init__(
        self,
        battles: int = 16,
        port: int = 26789,
        decision_ticks: int = 10,
        seed: int = 0,
        learner_roles: str | Sequence[int] = "alternating",
        opponent_policy: Any = None,
        batch_factory: Callable[..., Any] | None = None,
    ):
        if not 1 <= int(battles) <= 16:
            raise ValueError("battles must be between 1 and 16")
        if not 1 <= int(decision_ticks) <= 1_000_000:
            raise ValueError("decision_ticks must be between 1 and 1,000,000")
        self.battles = int(battles)
        self.port = int(port)
        self.decision_ticks = int(decision_ticks)
        self.base_seed = int(seed)
        self.opponent_policy = opponent_policy
        self._opponent_state = None
        self._opponent_episode_starts = np.ones(self.battles, dtype=bool)
        self._batch_factory = batch_factory or Batch
        self.batch = None
        self.waiting = False
        self.closed = False
        self._pending_actions = None
        self._seed_cursor = self.base_seed
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence((self.base_seed, slot, 1)))
            for slot in range(self.battles)
        ]
        self._role_mode, self._fixed_roles = self._parse_roles(learner_roles)
        self.slots = [_Slot(index=index) for index in range(self.battles)]
        super().__init__(
            self.battles,
            native_observation_space(),
            spaces.MultiDiscrete([5, ARENA_HEIGHT, ARENA_WIDTH]),
        )
        self.seed(self.base_seed)

    def seed(self, seed: int | None = None):
        values = super().seed(seed)
        self.base_seed = int(values[0])
        self._seed_cursor = self.base_seed
        self._rngs = [
            np.random.default_rng(np.random.SeedSequence((self.base_seed, slot, 1)))
            for slot in range(self.battles)
        ]
        return values

    def _parse_roles(self, roles):
        if isinstance(roles, str):
            if roles not in ("p0", "p1", "alternating"):
                raise ValueError("learner_roles must be p0, p1, alternating, or a role list")
            return roles, None
        fixed = tuple(int(role) for role in roles)
        if len(fixed) != self.battles or any(role not in (0, 1) for role in fixed):
            raise ValueError("learner role list must contain one 0/1 value per battle")
        return "fixed", fixed

    def _role_for(self, slot):
        if self._fixed_roles is not None:
            return self._fixed_roles[slot.index]
        if self._role_mode == "p0":
            return 0
        if self._role_mode == "p1":
            return 1
        return (slot.index + slot.episode) % 2

    def _make_batch(self):
        return self._batch_factory(battles=self.battles, port=self.port)

    def reset(self):
        if self.closed:
            raise RuntimeError("cannot reset a closed NativeBatchVecEnv")
        if self.waiting:
            raise RuntimeError("cannot reset while an asynchronous step is pending")
        if self.batch is None:
            self.batch = self._make_batch()
        requested = self._seeds
        first_seed = self._seed_cursor if requested[0] is None else int(requested[0])
        expected = [first_seed + index for index in range(self.battles)]
        if any(value is not None and int(value) != expected[index]
               for index, value in enumerate(requested)):
            raise ValueError("Batch.reset requires consecutive per-slot seeds")
        states = self.batch.reset(seed=first_seed)
        if len(states) != self.battles:
            raise RuntimeError(f"batch reset returned {len(states)} of {self.battles} states")
        for slot, state, episode_seed in zip(self.slots, states, expected):
            self._begin_episode(slot, state, episode_seed)
        self._seed_cursor = first_seed + self.battles
        self.reset_infos = [self._reset_info(slot) for slot in self.slots]
        self._reset_seeds()
        self._reset_options()
        return self._stack_observations()

    def _begin_episode(self, slot, state, seed):
        slot.seed = int(seed)
        slot.learner_owner = self._role_for(slot)
        slot.state = state
        slot.previous_state = None
        self._opponent_episode_starts[slot.index] = True
        slot.history.clear()
        slot.ledger.clear()
        slot.elixir_ticks = [[], []]
        slot.elixir_values = [[], []]
        slot.cycle_states = [dict.fromkeys(_INITIAL_CYCLE_STATES, 1) for _ in range(2)]
        slot.cycle_beliefs = [_cycle_belief(()) for _ in range(2)]
        slot.initial_tower_hp = tuple(
            max(_tower_hp(state, owner), 1.0) for owner in (0, 1)
        )
        slot.initial_towers = tuple(
            entity for entity in state.entities
            if entity.card == -1 and entity.entity_type == "building"
            and entity.hp is not None and entity.hp > 0
        )
        if hasattr(self.opponent_policy, "on_episode_start"):
            self.opponent_policy.on_episode_start(
                slot_index=slot.index,
                episode=slot.episode,
                seed=slot.seed,
                learner_role=slot.learner_owner,
            )

    def _reset_info(self, slot):
        info = {
            "battle_backend": self.backend_name,
            "observation_schema": self.observation_schema,
            "deployment_rules_version": self.deployment_rules_version,
            "slot": slot.index,
            "seed": slot.seed,
            "learner_role": f"p{slot.learner_owner}",
            "engine_port": self.port,
        }
        if hasattr(self.opponent_policy, "opponent_metadata"):
            info.update(self.opponent_policy.opponent_metadata(slot.index))
        return info

    def step_async(self, actions):
        if self.batch is None:
            raise RuntimeError("reset the VecEnv before stepping it")
        if self.waiting:
            raise RuntimeError("step_async called while another step is pending")
        values = validated_actions(actions, self.battles, "learner")
        self._pending_actions = values.copy()
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("step_wait called without step_async")
        actions = self._pending_actions
        self._pending_actions = None
        self.waiting = False

        before = tuple(slot.state for slot in self.slots)
        opponent_observations = [
            self._observation(slot, owner=1 - slot.learner_owner)
            for slot in self.slots
        ]
        opponent_actions = self._opponent_actions(opponent_observations)
        groups = []
        submitted = []
        learner_attempted = []
        for slot, learner_action, opponent_action in zip(
            self.slots, actions, opponent_actions
        ):
            learner_play, attempted = self._action_to_play(
                slot, slot.learner_owner, learner_action
            )
            opponent_owner = 1 - slot.learner_owner
            opponent_play, _ = self._action_to_play(
                slot, opponent_owner, opponent_action
            )
            plays = tuple(play for play in (learner_play, opponent_play) if play is not None)
            groups.append(plays)
            submitted.append(plays)
            learner_attempted.append(attempted)

        states = self.batch.step(groups, ticks=self.decision_ticks)
        if len(states) != self.battles:
            raise RuntimeError(f"batch step returned {len(states)} of {self.battles} states")
        if len(self.batch.last_receipts) != self.battles:
            raise RuntimeError("batch step returned an incomplete receipt group list")

        rewards = np.zeros(self.battles, dtype=np.float32)
        dones = np.zeros(self.battles, dtype=bool)
        infos = []
        terminal_observations = {}
        done_slots = []
        done_seeds = []
        for slot, old_state, new_state, attempted, plays, receipts in zip(
            self.slots,
            before,
            states,
            learner_attempted,
            submitted,
            self.batch.last_receipts,
        ):
            self._record_plays(slot, new_state, plays, receipts)
            slot.previous_state = old_state
            slot.state = new_state
            accepted = self._learner_action_accepted(
                slot.learner_owner, plays, receipts
            )
            reward, reward_components = self._reward(
                slot, old_state, new_state, attempted and not accepted
            )
            rewards[slot.index] = reward
            dones[slot.index] = bool(new_state.finalized)
            learner = slot.learner_owner
            opponent = 1 - learner
            info = {
                **self._reset_info(slot),
                "tick": int(new_state.tick),
                "winner": new_state.result if new_state.finalized else None,
                "learner_crowns": int(new_state.crowns[learner]),
                "opponent_crowns": int(new_state.crowns[opponent]),
                "crown_diff": float(
                    new_state.crowns[learner] - new_state.crowns[opponent]
                ),
                "tower_hp_diff": float(
                    _tower_hp(new_state, learner) / slot.initial_tower_hp[learner]
                    - _tower_hp(new_state, opponent) / slot.initial_tower_hp[opponent]
                ),
                "deployment_succeeded": bool(accepted),
                "no_op": not attempted,
                "entity_truncated": len(old_state.entities) > MAX_ENTITIES,
                "entity_count": len(old_state.entities),
                "reward_components": reward_components,
                "TimeLimit.truncated": False,
                "entity_target_schema": ENTITY_TARGET_SCHEMA,
                "entity_targets": _entity_transition_targets(
                    old_state, new_state, learner
                ),
                "transition_target_schema": TRANSITION_TARGET_SCHEMA,
                "transition_targets": _public_transition_targets(
                    old_state, new_state, learner, slot.initial_tower_hp
                ),
                "executed_action_schema": EXECUTED_ACTION_SCHEMA,
                "executed_action": _executed_action_features(
                    learner, plays, receipts, new_state.plays
                ),
            }
            if new_state.finalized:
                if hasattr(self.opponent_policy, "on_episode_end"):
                    info.update(self.opponent_policy.on_episode_end(
                        slot_index=slot.index,
                        winner=new_state.result,
                        learner_role=slot.learner_owner,
                    ))
                terminal_observations[slot.index] = self._observation(slot)
                info["terminal_observation"] = terminal_observations[slot.index]
                done_slots.append(slot.index)
                slot.episode += 1
                done_seed = self._seed_cursor
                self._seed_cursor += 1
                done_seeds.append(done_seed)
            infos.append(info)

        if done_slots:
            reset_states = self.batch.reset_slots(done_slots, done_seeds)
            if len(reset_states) != len(done_slots):
                raise RuntimeError("batch partial reset returned an incomplete state list")
            for index, state, episode_seed in zip(done_slots, reset_states, done_seeds):
                self._begin_episode(self.slots[index], state, episode_seed)
                self.reset_infos[index] = self._reset_info(self.slots[index])

        observations = self._stack_observations()
        for index, info in enumerate(infos):
            scalar = (terminal_observations[index]['scalar'] if dones[index]
                      else observations['scalar'][index])
            info['next_scene_target_schema'] = NEXT_SCENE_TARGET_SCHEMA
            info['next_scene_value'] = float(scene_value(scalar))
        return observations, rewards, dones, infos

    def _action_to_play(self, slot, owner, action):
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        if values.size != 3:
            return None, True
        policy_slot, y, x = (int(value) for value in values)
        attempted = policy_slot != 0
        if policy_slot == 0:
            return None, False
        mask = self._action_mask(slot.state, owner, slot.initial_towers)
        if not (0 <= policy_slot < 5 and 0 <= y < ARENA_HEIGHT and 0 <= x < ARENA_WIDTH):
            return None, attempted
        if not mask[policy_slot, y, x]:
            return None, attempted
        player = slot.state.players[owner]
        if policy_slot - 1 >= len(player.hand):
            return None, attempted
        card = player.hand[policy_slot - 1]
        world_x = int((x + 0.5) * 1000)
        world_y = int((y + 0.5) * 1000)
        if owner == 1:
            world_x = 18000 - world_x
            world_y = 32000 - world_y
        return Play(owner=owner, slot=int(card.slot), x=world_x, y=world_y), attempted

    def _opponent_actions(self, observations):
        policy = self.opponent_policy
        if policy is not None and hasattr(policy, "predict"):
            batched = {
                key: np.stack([observation[key] for observation in observations])
                for key in self.observation_space.spaces
            }
            if isinstance(policy, BaseAlgorithm):
                # A directly supplied recurrent SB3 model needs the same
                # per-slot memory/reset contract as our league wrappers.
                actions, next_state = policy.predict(
                    batched, state=self._opponent_state,
                    episode_start=self._opponent_episode_starts,
                    deterministic=False,
                )
                actions = validated_actions(actions, self.battles, "opponent policy")
                self._opponent_state = next_state
                self._opponent_episode_starts[:] = False
            else:
                actions = np.asarray(policy.predict(batched, deterministic=False)[0])
            return validated_actions(actions, self.battles, "opponent policy")
        return validated_actions(np.stack([
            self._opponent_action(slot, observation, observation["action_mask"])
            for slot, observation in zip(self.slots, observations)
        ]), self.battles, "opponent policy")

    def _opponent_action(self, slot, observation, action_mask):
        policy = self.opponent_policy
        if policy is not None:
            return np.asarray(policy(observation, action_mask, slot.index))
        legal = np.argwhere(action_mask[1:] > 0)
        if len(legal) == 0 or self._rngs[slot.index].random() < 0.2:
            return np.asarray((0, 0, 0), dtype=np.int64)
        card_offset, y, x = legal[self._rngs[slot.index].integers(len(legal))]
        return np.asarray((int(card_offset) + 1, int(y), int(x)), dtype=np.int64)

    @staticmethod
    def _learner_action_accepted(owner, plays, receipts):
        for play, receipt in zip(plays, receipts):
            if play.owner == owner:
                return receipt.status == 0
        return False

    def _record_plays(self, slot, state, submitted, receipts):
        coordinates: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for play, receipt in zip(submitted, receipts):
            if receipt.status == 0:
                coordinates.setdefault((receipt.owner, receipt.card_id), []).append(
                    (play.x, play.y)
                )
        for event in state.plays:
            positions = coordinates.get((event.owner, event.card), [])
            x, y = positions.pop(0) if positions else (0, 0)
            recorded = _HistoryEvent(
                tick=int(event.tick),
                owner=int(event.owner),
                card=int(event.card),
                cost=int(event.cost),
                x=int(x),
                y=int(y),
            )
            owner = recorded.owner
            ticks = slot.elixir_ticks[owner]
            if ticks and recorded.tick < ticks[-1]:
                raise ValueError("public play ticks must be nondecreasing per owner")
            slot.history.append(recorded)
            slot.ledger.append(recorded)
            value = max(0.0, self._public_elixir(slot, owner, recorded.tick) - recorded.cost)
            slot.elixir_ticks[owner].append(recorded.tick)
            slot.elixir_values[owner].append(value)
            slot.cycle_states[owner] = _advance_cycle_states(slot.cycle_states[owner], recorded.card)
            slot.cycle_beliefs[owner] = _cycle_summary(slot.cycle_states[owner])
        del slot.history[:-HISTORY_LENGTH]

    def _reward(self, slot, before, after, invalid):
        owner = slot.learner_owner
        enemy = 1 - owner
        own_scale = slot.initial_tower_hp[owner]
        enemy_scale = slot.initial_tower_hp[enemy]
        before_tower = (
            _tower_hp(before, owner) / own_scale
            - _tower_hp(before, enemy) / enemy_scale
        )
        after_tower = (
            _tower_hp(after, owner) / own_scale
            - _tower_hp(after, enemy) / enemy_scale
        )
        tower = 5.0 * (after_tower - before_tower)
        crown = 2.0 * (
            (after.crowns[owner] - before.crowns[owner])
            - (after.crowns[enemy] - before.crowns[enemy])
        )
        terminal = 0.0
        if after.finalized:
            if after.result == owner:
                terminal = 10.0
            elif after.result == enemy:
                terminal = -10.0
        invalid_value = -0.05 if invalid else 0.0
        components = {
            "tower": float(tower),
            "crown": float(crown),
            "terminal": float(terminal),
            "invalid": float(invalid_value),
        }
        return float(sum(components.values())), components

    def _action_mask(self, state, owner, initial_towers=()):
        mask = np.zeros((5, ARENA_HEIGHT, ARENA_WIDTH), dtype=np.float32)
        mask[0, 0, 0] = 1.0
        if state is None or state.finalized:
            return mask
        troop_mask = _troop_deployment_mask(state, owner, initial_towers)
        player = state.players[owner]
        for policy_slot, card in enumerate(player.hand[:4], start=1):
            if int(card.cost) * 10_000 > int(player.elixir):
                continue
            if int(card.card) in SPELL_IDS:
                mask[policy_slot, :, :] = 1.0
            else:
                mask[policy_slot] = troop_mask
        return mask

    def _public_elixir(self, slot, owner, tick):
        # Values are committed only at public plays. Binary search preserves
        # historical before-state queries used by temporal features, without
        # replaying the full match ledger on every observation.
        ticks = slot.elixir_ticks[owner]
        index = bisect_right(ticks, tick) - 1
        estimate = slot.elixir_values[owner][index] if index >= 0 else 5.0
        last_seconds = (max(0.0, (ticks[index] - FIRST_PLAYABLE_TICK) * TICK_SECONDS)
                        if index >= 0 else 0.0)
        now_seconds = max(0.0, (int(tick) - FIRST_PLAYABLE_TICK) * TICK_SECONDS)
        return min(10.0, estimate + _regenerated_elixir(last_seconds, now_seconds))

    @staticmethod
    def _combat_entities(state, owner):
        return {
            int(entity.id): entity
            for entity in state.entities
            if int(entity.owner) == owner and int(entity.card) != -1
        }

    def _temporal_features(self, slot, owner):
        """Describe continuous state changes since the previous decision."""
        before = slot.previous_state
        after = slot.state
        if before is None or after is None:
            return np.zeros(TEMPORAL_FEATURES, dtype=np.float32)
        enemy = 1 - owner
        own_before = self._combat_entities(before, owner)
        own_after = self._combat_entities(after, owner)
        enemy_before = self._combat_entities(before, enemy)
        enemy_after = self._combat_entities(after, enemy)

        def entity_changes(left, right):
            left_ids, right_ids = set(left), set(right)
            spawned = len(right_ids - left_ids)
            died = len(left_ids - right_ids)
            hp_before = sum(max(float(entity.hp or 0), 0.0) for entity in left.values())
            hp_after = sum(max(float(entity.hp or 0), 0.0) for entity in right.values())
            targeted_before = sum(entity.target_id is not None for entity in left.values())
            targeted_after = sum(entity.target_id is not None for entity in right.values())
            return spawned, died, hp_after - hp_before, targeted_after - targeted_before

        own_spawned, own_died, own_hp_delta, own_target_delta = entity_changes(
            own_before, own_after
        )
        enemy_spawned, enemy_died, enemy_hp_delta, enemy_target_delta = entity_changes(
            enemy_before, enemy_after
        )
        own_plays = sum(int(event.owner) == owner for event in after.plays)
        enemy_plays = sum(int(event.owner) == enemy for event in after.plays)
        before_enemy_elixir = self._public_elixir(slot, enemy, before.tick)
        after_enemy_elixir = self._public_elixir(slot, enemy, after.tick)
        own_elixir_delta = (
            float(after.players[owner].elixir) - float(before.players[owner].elixir)
        ) / 100_000.0
        enemy_elixir_delta = (after_enemy_elixir - before_enemy_elixir) / 10.0
        return np.asarray((
            1.0,
            np.clip((int(after.tick) - int(before.tick)) / 100.0, 0, 1),
            np.clip(
                (_tower_hp(after, owner) - _tower_hp(before, owner))
                / slot.initial_tower_hp[owner],
                -1,
                1,
            ),
            np.clip(
                (_tower_hp(after, enemy) - _tower_hp(before, enemy))
                / slot.initial_tower_hp[enemy],
                -1,
                1,
            ),
            np.clip(own_elixir_delta, -1, 1),
            np.clip(enemy_elixir_delta, -1, 1),
            np.clip(own_spawned / 8.0, 0, 1),
            np.clip(enemy_spawned / 8.0, 0, 1),
            np.clip(own_died / 8.0, 0, 1),
            np.clip(enemy_died / 8.0, 0, 1),
            np.clip(own_hp_delta / 20_000.0, -1, 1),
            np.clip(enemy_hp_delta / 20_000.0, -1, 1),
            np.clip(own_plays / 4.0, 0, 1),
            np.clip(enemy_plays / 4.0, 0, 1),
            np.clip(
                (after.crowns[owner] - before.crowns[owner]) / 3.0, -1, 1
            ),
            np.clip(
                (after.crowns[enemy] - before.crowns[enemy]) / 3.0, -1, 1
            ),
            np.clip(own_target_delta / 8.0, -1, 1),
            np.clip(enemy_target_delta / 8.0, -1, 1),
            np.clip((len(own_after) - len(own_before)) / 8.0, -1, 1),
            np.clip((len(enemy_after) - len(enemy_before)) / 8.0, -1, 1),
        ), dtype=np.float32)

    def _observation(self, slot, owner=None):
        owner = slot.learner_owner if owner is None else int(owner)
        state = slot.state
        spatial = np.zeros(
            (SPATIAL_CHANNELS, ARENA_HEIGHT, ARENA_WIDTH), dtype=np.float32
        )
        entities = np.zeros((MAX_ENTITIES, ENTITY_FEATURES), dtype=np.float32)
        entity_mask = np.zeros(MAX_ENTITIES, dtype=np.float32)
        ordered_entities = _ordered_entities(state)
        targets_by_id = {int(entity.id): entity for entity in ordered_entities}
        for row, entity in enumerate(ordered_entities):
            enemy = int(entity.owner) != owner
            x = float(entity.x) / 1000.0
            y = float(entity.y) / 1000.0
            if owner == 1:
                x, y = ARENA_WIDTH - x, ARENA_HEIGHT - y
            cell_x = int(np.clip(np.floor(x), 0, ARENA_WIDTH - 1))
            cell_y = int(np.clip(np.floor(y), 0, ARENA_HEIGHT - 1))
            hp = max(float(entity.hp or 0.0), 0.0)
            max_hp = max(float(entity.max_hp or 0.0), 0.0)
            hp_ratio = np.clip(hp / max_hp, 0.0, 1.0) if max_hp else 0.0
            kind = {"troop": 0, "building": 1, "projectile": 2}.get(
                entity.entity_type, 0
            )
            side = 1 if enemy else 0
            spatial[side, cell_y, cell_x] = 1.0
            spatial[2 + side, cell_y, cell_x] = max(
                spatial[2 + side, cell_y, cell_x], hp_ratio
            )
            spatial[4 + kind * 2 + side, cell_y, cell_x] = 1.0
            deployment = np.clip(float(entity.deployment_remaining_ms or 0) / 3000.0, 0, 1)
            cooldown = np.clip(float(entity.attack_cooldown_ms or 0) / 5000.0, 0, 1)
            spatial[10 + side, cell_y, cell_x] = max(
                spatial[10 + side, cell_y, cell_x], deployment
            )
            spatial[12 + side, cell_y, cell_x] = max(
                spatial[12 + side, cell_y, cell_x], 1.0 - cooldown
            )
            spatial[14 + side, cell_y, cell_x] = max(
                spatial[14 + side, cell_y, cell_x], float(entity.target_id is not None)
            )

            if row >= MAX_ENTITIES:
                continue

            token = entities[row]
            token[0] = 1.0
            token[1] = float(enemy)
            token[2 + kind] = 1.0
            card_token = CARD_TOKENS.get(int(entity.card), 0)
            token[5 + card_token] = 1.0
            token[14] = np.clip(x / ARENA_WIDTH, 0, 1)
            token[15] = np.clip(y / ARENA_HEIGHT, 0, 1)
            token[16] = hp_ratio
            token[17] = np.clip(np.log1p(hp) / 10.0, 0, 1)
            token[18] = deployment
            token[19] = cooldown
            token[20] = float(entity.target_id is not None)
            if entity.target_id is not None:
                target = targets_by_id.get(int(entity.target_id))
                if target is not None:
                    target_x = float(target.x) / 1000.0
                    target_y = float(target.y) / 1000.0
                    if owner == 1:
                        target_x = ARENA_WIDTH - target_x
                        target_y = ARENA_HEIGHT - target_y
                    token[21] = np.clip((target_x - x) / ARENA_WIDTH, -1, 1)
                    token[22] = np.clip((target_y - y) / ARENA_HEIGHT, -1, 1)
            token[23] = float(bool(entity.is_spell))
            entity_mask[row] = 1.0

        player = state.players[owner]
        hand_cards = [CARD_TOKENS.get(int(card.card), 0) for card in player.hand[:4]]
        hand_cards.extend([0] * (4 - len(hand_cards)))
        next_card = CARD_TOKENS.get(int(player.cycle[0]), 0) if player.cycle else 0
        hand = np.asarray((*hand_cards, next_card), dtype=np.int32)

        elapsed = max(0.0, (int(state.tick) - FIRST_PLAYABLE_TICK) * TICK_SECONDS)
        phase = np.zeros(3, dtype=np.float32)
        phase[0 if elapsed < 120 else 1 if elapsed < 240 else 2] = 1.0
        own_tower = np.clip(
            _tower_hp(state, owner) / slot.initial_tower_hp[owner], 0, 1
        )
        enemy = 1 - owner
        enemy_tower = np.clip(
            _tower_hp(state, enemy) / slot.initial_tower_hp[enemy], 0, 1
        )
        scalar = np.asarray((
            np.clip(float(player.elixir) / 100_000.0, 0, 1),
            self._public_elixir(slot, enemy, state.tick) / 10.0,
            np.clip(float(state.crowns[owner]) / 3.0, 0, 1),
            np.clip(float(state.crowns[enemy]) / 3.0, 0, 1),
            np.clip(elapsed / MATCH_SECONDS, 0, 1),
            *phase,
            float(owner == 1),
            own_tower,
            enemy_tower,
            min(len(slot.history) / HISTORY_LENGTH, 1.0),
        ), dtype=np.float32)

        # Update once per public event; do not replay the entire match's card
        # sequence on every learner/opponent observation.
        hand_belief, next_belief, confidence = slot.cycle_beliefs[enemy]
        last_play_ages = []
        for physical_owner in (owner, enemy):
            ticks = slot.elixir_ticks[physical_owner]
            last_tick = ticks[-1] if ticks else FIRST_PLAYABLE_TICK
            last_play_ages.append(np.clip(
                (state.tick - last_tick) * TICK_SECONDS / 60.0, 0, 1
            ))
        public_belief = np.asarray((
            *hand_belief,
            *next_belief,
            min(len(slot.elixir_ticks[enemy]) / 4.0, 1.0),
            confidence,
            *last_play_ages,
        ), dtype=np.float32)

        history = np.zeros((HISTORY_LENGTH, HISTORY_FEATURES), dtype=np.float32)
        for row, event in enumerate(slot.history[-HISTORY_LENGTH:]):
            event_x = float(event.x) / 1000.0
            event_y = float(event.y) / 1000.0
            if owner == 1 and (event.x or event.y):
                event_x = ARENA_WIDTH - event_x
                event_y = ARENA_HEIGHT - event_y
            event_seconds = max(
                0.0, (event.tick - FIRST_PLAYABLE_TICK) * TICK_SECONDS
            )
            history[row] = (
                1.0,
                float(event.owner != owner),
                CARD_TOKENS.get(event.card, 0) / len(CARD_IDS),
                np.clip(event_x / ARENA_WIDTH, 0, 1),
                np.clip(event_y / ARENA_HEIGHT, 0, 1),
                np.clip(event_seconds / MATCH_SECONDS, 0, 1),
                np.clip((elapsed - event_seconds) / 60.0, 0, 1),
                np.clip(event.cost / 10.0, 0, 1),
            )

        return {
            "spatial": spatial,
            "entities": entities,
            "entity_mask": entity_mask,
            "hand": hand,
            "scalar": scalar,
            "public_belief": public_belief,
            "action_history": history,
            "temporal": self._temporal_features(slot, owner),
            "action_mask": self._action_mask(state, owner, slot.initial_towers),
        }

    def _stack_observations(self):
        observations = [self._observation(slot) for slot in self.slots]
        return {
            key: np.stack([observation[key] for observation in observations])
            for key in self.observation_space.spaces
        }

    def close(self):
        if self.closed:
            return
        if self.batch is not None:
            self.batch.close()
        if hasattr(self.opponent_policy, "close"):
            self.opponent_policy.close()
        self.batch = None
        self.closed = True
        self.waiting = False
        self._pending_actions = None

    def release_batch(self):
        """Release the singleton native data plane without closing this VecEnv.

        Fixed evaluation temporarily owns the resident engine slots.  A later
        ``reset()`` reconfigures this environment's slots and resumes training.
        """
        if self.waiting:
            raise RuntimeError("cannot release batch while a step is pending")
        if self.batch is not None:
            self.batch.close()

    def get_attr(self, attr_name, indices=None):
        selected = _selected(indices, self.battles)
        if attr_name == "render_mode":
            return [self.render_mode for _ in selected]
        if hasattr(self, attr_name):
            return [getattr(self, attr_name) for _ in selected]
        values = []
        for index in selected:
            slot = self.slots[index]
            if not hasattr(slot, attr_name):
                raise AttributeError(attr_name)
            values.append(getattr(slot, attr_name))
        return values

    def set_attr(self, attr_name, value, indices=None):
        selected = _selected(indices, self.battles)
        for index in selected:
            slot = self.slots[index]
            if not hasattr(slot, attr_name):
                raise AttributeError(attr_name)
            setattr(slot, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        selected = _selected(indices, self.battles)
        if method_name == "release_batch":
            self.release_batch()
            return [None for _ in selected]
        if method_name == "action_masks":
            return [
                self._action_mask(
                    self.slots[index].state,
                    self.slots[index].learner_owner,
                    self.slots[index].initial_towers,
                )
                for index in selected
            ]
        if method_name == "get_state":
            return [self.slots[index].state for index in selected]
        raise AttributeError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in _selected(indices, self.battles)]


__all__ = [
    "CARD_IDS",
    "NativeBatchVecEnv",
    "native_observation_space",
]
