"""Fail closed on interrupted sessions and mismatched native responses."""
import importlib
import io
import json
import unittest
from unittest.mock import patch

from test_batch import PACKAGE_NAME, batch_module

engine_module = importlib.import_module(f'{PACKAGE_NAME}.engine')
protocol = importlib.import_module(f'{PACKAGE_NAME}.protocol')
Engine, RenderedEngine, Play = engine_module.Engine, engine_module.RenderedEngine, engine_module.Play


def payload(tick=90, generation=1, finalized=False):
    return dict(generation=generation, tick=tick, finalized=finalized, result=None,
                crowns=[0, 0], players=[dict(owner=owner, elixir=50000, hand=[], cycle=[])
                                       for owner in (0, 1)], entities=[], plays=[], next=0)


class Socket:
    def __init__(self, data=b''):
        self.data = io.BytesIO(data)
        self.stream = io.BytesIO(data)
        self.closed = False
        self.sent = []

    def makefile(self, mode):
        return self.stream

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, count):
        return self.data.read(count)

    def close(self):
        self.closed = True


class ClientFailureTest(unittest.TestCase):
    def test_truncated_handshake_closes_both_client_types(self):
        for cls in (Engine, batch_module.Batch):
            connection = Socket(b'bad')
            client = cls()
            with self.subTest(client=cls.__name__), patch.object(
                    engine_module.socket, 'create_connection', return_value=connection):
                with self.assertRaises((ConnectionError, EOFError)):
                    client.connect() if cls is Engine else client._open()
            self.assertTrue(connection.closed)
            self.assertIsNone(client.connection)
            if cls is Engine:
                self.assertTrue(connection.stream.closed)

    def test_keyboard_interrupt_closes_session_without_replaying_write(self):
        connection = Socket()
        client = Engine()
        client.connection, client.stream = connection, connection.stream
        with patch.object(client, '_receive', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                client.request('step 5')
        self.assertEqual(connection.sent, [b'step 5\n'])
        self.assertTrue(connection.closed)
        batch = batch_module.Batch(battles=1)
        batch.connection = Socket()
        connection = batch.connection
        with patch.object(batch_module, '_receive', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                batch.step(ticks=5)
        self.assertEqual(len(connection.sent), 1)
        self.assertTrue(connection.closed)

    def test_reset_rejections_never_publish_old_or_incomplete_state(self):
        for responses, calls in (([{'ok': False, 'error': 'configure failed'}], 1),
                                 ([{'ok': True}, {'ok': False, 'error': 'step failed'}], 2),
                                 ([{'ok': True}, {'ok': True}], 3)):
            engine = Engine()
            engine.state = protocol.State.decode(payload())
            with self.subTest(calls=calls), patch.object(engine, 'request', side_effect=responses) as request:
                with patch.object(engine, '_capture', return_value=protocol.State.decode(payload(tick=0))):
                    with self.assertRaises(RuntimeError):
                        engine.reset()
                self.assertEqual(request.call_count, min(calls, 2))
                self.assertIsNone(engine.state)
        with patch.object(engine, 'request', return_value={'ok': True}), patch.object(
                engine, '_capture', return_value=protocol.State.decode(payload())):
            self.assertEqual(engine.reset().tick, 90)

    def test_single_and_rendered_clients_reject_stale_or_impossible_steps(self):
        for cls in (Engine, RenderedEngine):
            for response in (payload(tick=95, generation=2), payload(tick=94), payload(tick=96)):
                engine = cls()
                engine.state = protocol.State.decode(payload())
                with self.subTest(client=cls.__name__, response=response), patch.object(
                        engine, 'request', return_value=response if cls is Engine else {'ok': True}), patch.object(
                        engine, '_capture', return_value=protocol.State.decode(response)):
                    with self.assertRaisesRegex(RuntimeError, 'generation|tick'):
                        engine.step(ticks=5)
                self.assertIsNone(engine.state)
            engine.state = protocol.State.decode(payload())
            with patch.object(engine, 'request', return_value=payload(tick=93, finalized=True)
                              if cls is Engine else {'ok': True}), patch.object(
                    engine, '_capture', return_value=protocol.State.decode(payload(tick=93, finalized=True))):
                self.assertEqual(engine.step(ticks=5).advanced, 3)

    def test_invalid_action_is_rejected_before_sending(self):
        for cls in (Engine, RenderedEngine):
            engine = cls()
            with patch.object(engine, 'request') as request:
                with self.assertRaises(RuntimeError):
                    engine.step()
                engine.state = protocol.State.decode(payload())
                for action in (Play(-1, 1, 0, 0), Play(2, 1, 0, 0), Play(0, 4, 0, 0)):
                    with self.assertRaises(ValueError):
                        engine.step([action])
                request.assert_not_called()

    def test_batch_reset_rejects_wrong_playable_boundary(self):
        batch = batch_module.Batch(battles=1)
        for state in (payload(tick=0), payload(finalized=True)):
            connection = Socket()
            batch.connection = connection
            with self.assertRaisesRegex(RuntimeError, 'playable tick'):
                batch._validate_reset_ticks((protocol.State.decode(state),))
            self.assertTrue(connection.closed)

    def batch_response(self, state, receipts=()):
        batch = batch_module.Batch(battles=1)
        batch.states = (protocol.State.decode(payload()),)
        raw = json.dumps(state).encode()
        frame = (batch_module.HEADER.pack(b'CRBS', 1, 0, 1, 1)
                 + batch_module.RESPONSE_ENTRY.pack(0, 0, len(receipts), 0, len(raw))
                 + b''.join(receipts) + raw)
        batch.connection = Socket(frame)
        return batch

    @staticmethod
    def receipt(owner=0, index=0, kind=1):
        return batch_module.RECEIPT.pack(kind, owner, index, 0, 0, 91, 26000000, 0, 0, 3, 0, 0, 0)

    def test_batch_requires_receipts_in_request_order_for_both_owners(self):
        actions = ((Play(0, 1, 1000, 1000), Play(1, 1, 1000, 1000)),)
        for receipts in ((), (self.receipt(),),
                         (self.receipt(owner=1), self.receipt(owner=0, index=1)),
                         (self.receipt(index=1), self.receipt(owner=1, index=0)),
                         (self.receipt(kind=2), self.receipt(owner=1, index=1))):
            batch = self.batch_response(payload(tick=95), receipts)
            previous, connection = batch.states, batch.connection
            with self.subTest(receipts=len(receipts)), self.assertRaisesRegex(RuntimeError, 'receipts'):
                batch.step(actions, ticks=5)
            self.assertEqual(batch.states, previous)
            self.assertTrue(connection.closed)
            self.assertIsNone(batch.connection)
        batch = self.batch_response(payload(tick=95), (self.receipt(), self.receipt(owner=1, index=1)))
        self.assertEqual(batch.step(actions, ticks=5)[0].tick, 95)
        self.assertEqual([receipt.owner for receipt in batch.last_receipts[0]], [0, 1])
        batch.close()

    def test_batch_rejects_generation_change_and_invalid_tick_progress(self):
        for response in (payload(tick=95, generation=2), payload(tick=94), payload(tick=96)):
            batch = self.batch_response(response)
            previous, connection = batch.states, batch.connection
            with self.subTest(response=response), self.assertRaisesRegex(RuntimeError, 'generation|tick'):
                batch.step(ticks=5)
            self.assertEqual(batch.states, previous)
            self.assertTrue(connection.closed)
        batch = self.batch_response(payload(tick=93, finalized=True))
        self.assertEqual(batch.step(ticks=5)[0].tick, 93)
        batch.close()

    def test_single_slot_epoch_repeated_plays_are_delivered_once_by_sequence(self):
        previous = payload()
        previous.update(plays=[[1, 81, 0, 26000000, 3], [2, 90, 1, 26000001, 3]], next=3)
        response = payload(tick=95)
        response.update(plays=[*previous['plays'], [3, 91, 0, 26000000, 3]], next=4)
        batch = self.batch_response(response)
        batch.states = (protocol.State.decode(previous),)
        state = batch.step(ticks=5)[0]
        self.assertEqual([event.sequence for event in state.plays], [3])
        self.assertEqual(state.next_sequence, 4)
        batch.close()
        response['next'] = 2
        batch = self.batch_response(response)
        batch.states = (protocol.State.decode(previous),)
        with self.assertRaisesRegex(RuntimeError, 'event cursor'):
            batch.step(ticks=5)


if __name__ == '__main__':
    unittest.main()
