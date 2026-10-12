"""Resident batch client: several battles in one lane, stepped together."""
from dataclasses import dataclass, replace
import json
import socket
import struct
from pathlib import Path

from .protocol import State


VERSION = 1
MAX_BATTLES = 16
MAX_ACTIONS = 8
MAX_PAYLOAD = 65536
HANDSHAKE = struct.Struct('<4sHHHH')
HEADER = struct.Struct('<4sHHQI')
REQUEST_ENTRY = struct.Struct('<HHI')
RESPONSE_ENTRY = struct.Struct('<HHHHI')
RECEIPT = struct.Struct('<BBHHHiIIiiiii')


@dataclass(frozen=True, slots=True)
class ActionReceipt:
    kind: int
    owner: int
    action_index: int
    status: int
    execute_tick: int
    card_id: int
    card_parameter: int
    deck_slot: int
    cost: int
    form_code: int
    argument0: int
    argument1: int


def _decode_receipt(data):
    values = RECEIPT.unpack(data)
    if values[4]:
        raise RuntimeError('batch receipt reserved field is set')
    return ActionReceipt(*values[:4], *values[5:])


def _receive(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(min(1 << 16, count - len(data)))
        if not chunk:
            raise EOFError(f'battle engine closed at {len(data)}/{count} bytes')
        data += chunk
    return bytes(data)


def _encode_play(play):
    action = bytearray(20)
    action[0] = 1  # deploy
    action[1] = play.owner
    struct.pack_into('<H', action, 2, 1)  # execute on the next tick
    struct.pack_into('<i', action, 4, play.slot)
    struct.pack_into('<i', action, 8, play.x)
    struct.pack_into('<i', action, 12, play.y)
    return action


def stop_resident(port=26789):
    """Return a lane to single-battle mode after a batch session."""
    with socket.create_connection(('127.0.0.1', port), timeout=10) as sock:
        with sock.makefile('rb') as session:
            sock.sendall(b'session-v1\n')
            if not json.loads(session.readline()).get('ok'):
                return False
            sock.sendall(b'multi-stop\n')
            return json.loads(session.readline()).get('ok', False)


class Batch:
    """Several independent battles in one lane, advanced by one round trip per step.

    `states = batch.step(actions, ticks)` returns one State per battle. `actions[i]`
    is an iterable of Play for battle i; pass `()` for no actions.
    """

    def __init__(self, battles=16, port=26789, timeout=30):
        if not 1 <= battles <= MAX_BATTLES:
            raise ValueError(f'battles must be between 1 and {MAX_BATTLES}')
        self.battles = battles
        self.address = ('127.0.0.1', port)
        self.timeout = timeout
        self.connection = None
        self.sequence = 1
        self.last_receipts = tuple(() for _ in range(battles))
        self.states = tuple(None for _ in range(battles))

    def reset(self, seed=1):
        self.close()
        self.states = tuple(None for _ in range(self.battles))
        self.last_receipts = tuple(() for _ in range(self.battles))
        slots = tuple(range(self.battles))
        seeds = tuple(seed + battle for battle in slots)
        self._configure_slots(slots, seeds)
        self._open()
        states = self._advance(90, ())  # first playable boundary, as Engine.reset
        self._validate_reset_ticks(states)
        self.last_receipts = tuple(() for _ in range(self.battles))
        return states

    def reset_slots(self, slots, seeds):
        """Reset selected battles without advancing the surviving slots.

        The resident control server owns one connection at a time, so the
        binary data plane is briefly closed while the selected managers are
        reconstructed and advanced to tick 90.  A capture-only batch frame
        (zero ticks, zero actions) then obtains their compact observations.
        """
        slots = tuple(int(slot) for slot in slots)
        seeds = tuple(int(seed) for seed in seeds)
        if not slots:
            return ()
        if len(slots) != len(seeds):
            raise ValueError('slots and seeds must have the same length')
        if len(set(slots)) != len(slots):
            raise ValueError('slots must be unique')
        if any(slot < 0 or slot >= self.battles for slot in slots):
            raise ValueError(f'slots must be between 0 and {self.battles - 1}')
        if not self.connection or any(state is None for state in self.states):
            raise RuntimeError('reset the batch before resetting individual slots')

        self.close()
        self._configure_slots(slots, seeds, advance=90)
        current = list(self.states)
        for slot in slots:
            current[slot] = None
        self.states = tuple(current)
        self._open()
        refreshed = self._exchange(
            tuple((slot, (), 0) for slot in slots)
        )
        self._validate_reset_ticks(refreshed)
        receipts = list(self.last_receipts)
        for slot in slots:
            receipts[slot] = ()
        self.last_receipts = tuple(receipts)
        return refreshed

    def _validate_reset_ticks(self, states):
        if any(state.tick != 90 or state.finalized for state in states):
            self.close()
            raise RuntimeError('batch reset missed the first playable tick')

    def step(self, actions=(), ticks=5):
        return self._advance(ticks, actions)

    def close(self):
        if self.connection:
            self.connection.close()
        self.connection = None

    def _configure_slots(self, slots, seeds, advance=0):
        match = json.loads(Path(__file__).with_name('standard_match.json').read_text())
        with socket.create_connection(self.address, timeout=self.timeout) as sock:
            with sock.makefile('rb') as session:
                sock.sendall(b'session-v1\n')
                if not self._read(session).get('ok'):
                    raise RuntimeError('control session rejected')
                for battle, seed in zip(slots, seeds):
                    match['rndSeed'] = seed
                    config = json.dumps(match, separators=(',', ':'))
                    sock.sendall(f'env {battle} configure {config}\n'.encode())
                    reply = self._read(session)
                    if not reply.get('ok'):
                        raise RuntimeError(f'battle {battle} configure failed: {reply}')
                    if advance:
                        sock.sendall(f'env {battle} step {advance}\n'.encode())
                        reply = self._read(session)
                        if not reply.get('ok'):
                            raise RuntimeError(
                                f'battle {battle} initial advance failed: {reply}'
                            )

    def _read(self, session):
        line = session.readline()
        if not line:
            raise ConnectionError(
                f'no battle engine responded on port {self.address[1]}; '
                f'start a lane and pass its port')
        return json.loads(line)

    def _open(self):
        self.connection = socket.create_connection(self.address, timeout=self.timeout)
        try:
            self.connection.sendall(b'batch-fast-v1\n')
            magic, version, status, capacity, reserved = HANDSHAKE.unpack(
                _receive(self.connection, HANDSHAKE.size))
            if (magic, version, status, reserved) != (b'CRBH', VERSION, 0, 0) or capacity < self.battles:
                raise RuntimeError(
                    f'batch handshake rejected: version={version} status={status} capacity={capacity}')
        except BaseException:
            self.close()
            raise

    def _advance(self, ticks, actions):
        if not self.connection:
            raise RuntimeError('reset the batch before stepping it')
        if not 1 <= ticks <= 1_000_000:
            raise ValueError('ticks must be between 1 and 1,000,000')
        groups = tuple(tuple(group) for group in actions) if actions else tuple(() for _ in range(self.battles))
        if len(groups) != self.battles:
            raise ValueError(f'expected {self.battles} action groups, got {len(groups)}')
        if any(len(group) > MAX_ACTIONS for group in groups):
            raise ValueError(f'each battle accepts at most {MAX_ACTIONS} actions')
        return self._exchange(tuple(
            (battle, plays, ticks) for battle, plays in enumerate(groups)
        ))

    def _exchange(self, entries):
        if not self.connection:
            raise RuntimeError('reset the batch before using the data plane')
        if not entries or len(entries) > self.battles:
            raise ValueError(f'expected between 1 and {self.battles} batch entries')
        slots = tuple(entry[0] for entry in entries)
        if len(set(slots)) != len(slots):
            raise ValueError('batch entry slots must be unique')
        for slot, plays, ticks in entries:
            if not 0 <= slot < self.battles:
                raise ValueError(f'battle slot {slot} is out of range')
            if not 0 <= ticks <= 1_000_000:
                raise ValueError('ticks must be between 0 and 1,000,000')
            if ticks == 0 and plays:
                raise ValueError('capture-only entries cannot contain actions')
            if len(plays) > MAX_ACTIONS:
                raise ValueError(f'each battle accepts at most {MAX_ACTIONS} actions')

        header = HEADER.pack(b'CRBQ', VERSION, 0, self.sequence, len(entries))
        body = bytearray()
        for battle, plays, ticks in entries:
            body += REQUEST_ENTRY.pack(battle, len(plays), ticks)
            for play in plays:
                body += _encode_play(play)
        try:
            self.connection.sendall(header + body)
            response = HEADER.unpack(_receive(self.connection, HEADER.size))
            expected = (b'CRBS', VERSION, 0, self.sequence, len(entries))
            if response != expected:
                raise RuntimeError(f'batch response header mismatch: {response!r}')
            states = []
            all_receipts = []
            for expected_slot, plays, ticks in entries:
                slot, status, count, flags, size = RESPONSE_ENTRY.unpack(
                    _receive(self.connection, RESPONSE_ENTRY.size))
                if slot != expected_slot or count > MAX_ACTIONS or flags or size > MAX_PAYLOAD:
                    raise RuntimeError(
                        f'invalid batch entry: slot={slot} status={status} '
                        f'receipts={count} flags={flags} size={size}')
                receipts = tuple(_decode_receipt(_receive(self.connection, RECEIPT.size))
                                 for _ in range(count))
                payload = json.loads(_receive(self.connection, size))
                if status:
                    raise RuntimeError(f'battle {slot} failed: {payload.get("error", payload)}')
                if count != len(plays) or any(
                    (receipt.kind, receipt.owner, receipt.action_index) != (1, play.owner, index)
                    for index, (play, receipt) in enumerate(zip(plays, receipts))
                ):
                    raise RuntimeError(f'battle {slot} action receipts do not match the request')
                state = State.decode(payload)
                previous = self.states[slot]
                if previous is not None:
                    advanced = state.tick - previous.tick
                    if state.generation != previous.generation or not 0 <= advanced <= ticks or (
                        not state.finalized and advanced != ticks
                    ):
                        raise RuntimeError(f'battle {slot} returned a stale generation or invalid tick advancement')
                    if state.next_sequence < previous.next_sequence:
                        raise RuntimeError(f'battle {slot} returned a regressing event cursor')
                    # Minimal resident captures begin at the current bind's
                    # epoch. With one slot there is no bind switch, so the
                    # epoch also contains already delivered plays. Preserve
                    # the incremental State.plays contract by native sequence,
                    # never by tick/card identity (same-tick plays are valid).
                    state = replace(state, plays=tuple(event for event in state.plays
                                                       if event.sequence >= previous.next_sequence))
                states.append(state)
                all_receipts.append(receipts)
            self.sequence += 1
            current_states = list(self.states)
            current_receipts = list(self.last_receipts)
            for slot, state, receipts in zip(slots, states, all_receipts):
                current_states[slot] = state
                current_receipts[slot] = receipts
            self.states = tuple(current_states)
            self.last_receipts = tuple(current_receipts)
            return tuple(states)
        except BaseException:
            self.close()
            raise
