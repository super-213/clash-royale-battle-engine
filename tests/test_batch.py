import importlib
import importlib.util
import json
from pathlib import Path
import socket
import struct
import sys
import threading
import unittest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "native_engine_test_package"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    spec.loader.exec_module(package)

batch_module = importlib.import_module(f"{PACKAGE_NAME}.batch")
Batch = batch_module.Batch


class FakeResidentServer:
    HANDSHAKE = struct.Struct("<4sHHHH")
    HEADER = struct.Struct("<4sHHQI")
    REQUEST_ENTRY = struct.Struct("<HHI")
    RESPONSE_ENTRY = struct.Struct("<HHHHI")

    def __init__(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(0.1)
        self.port = self.listener.getsockname()[1]
        self.slots = {}
        self.generation = 0
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    @staticmethod
    def _receive(connection, size):
        data = bytearray()
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return bytes(data)

    @staticmethod
    def _send_json(connection, payload):
        connection.sendall(json.dumps(payload, separators=(",", ":")).encode() + b"\n")

    def _state(self, slot):
        record = self.slots[slot]
        return {
            "generation": record["generation"],
            "tick": record["tick"],
            "finalized": False,
            "result": None,
            "crowns": [0, 0],
            "players": [
                {"owner": 0, "elixir": 50000, "hand": [], "cycle": []},
                {"owner": 1, "elixir": 50000, "hand": [], "cycle": []},
            ],
            "entities": [],
            "plays": [],
            "next": 0,
        }

    def _handle_control(self, connection, stream):
        self._send_json(connection, {"ok": True})
        for raw_line in stream:
            command = raw_line.decode().rstrip("\n")
            prefix, nested = command.split(" ", 1)
            self.assert_equal(prefix, "env")
            slot_text, nested = nested.split(" ", 1)
            slot = int(slot_text)
            if nested.startswith("configure "):
                config = json.loads(nested[len("configure "):])
                self.generation += 1
                self.slots[slot] = {
                    "generation": self.generation,
                    "seed": config["rndSeed"],
                    "tick": 0,
                }
                self._send_json(connection, {"ok": True})
            elif nested.startswith("step "):
                ticks = int(nested[len("step "):])
                self.slots[slot]["tick"] += ticks
                self._send_json(connection, {"ok": True})
            else:
                raise AssertionError(f"unexpected control command: {command}")

    def _handle_batch(self, connection):
        connection.sendall(self.HANDSHAKE.pack(b"CRBH", 1, 0, 16, 0))
        while True:
            raw_header = self._receive(connection, self.HEADER.size)
            if raw_header is None:
                return
            magic, version, reserved, sequence, count = self.HEADER.unpack(raw_header)
            self.assert_equal((magic, version, reserved), (b"CRBQ", 1, 0))
            requests = []
            for _ in range(count):
                raw_entry = self._receive(connection, self.REQUEST_ENTRY.size)
                slot, action_count, ticks = self.REQUEST_ENTRY.unpack(raw_entry)
                actions = self._receive(connection, action_count * 20)
                self.assert_equal(len(actions), action_count * 20)
                self.assert_equal(ticks != 0 or action_count == 0, True)
                requests.append((slot, ticks))

            connection.sendall(self.HEADER.pack(b"CRBS", 1, 0, sequence, count))
            for slot, ticks in requests:
                self.slots[slot]["tick"] += ticks
                payload = json.dumps(
                    self._state(slot), separators=(",", ":")
                ).encode()
                connection.sendall(
                    self.RESPONSE_ENTRY.pack(slot, 0, 0, 0, len(payload)) + payload
                )

    @staticmethod
    def assert_equal(actual, expected):
        if actual != expected:
            raise AssertionError(f"{actual!r} != {expected!r}")

    def _serve(self):
        try:
            while not self.stopping.is_set():
                try:
                    connection, _ = self.listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    stream = connection.makefile("rb")
                    with stream:
                        mode = stream.readline().decode().rstrip("\n")
                        if mode == "session-v1":
                            self._handle_control(connection, stream)
                        elif mode == "batch-fast-v1":
                            self._handle_batch(connection)
                        else:
                            raise AssertionError(f"unexpected session mode: {mode}")
        except OSError:
            if not self.stopping.is_set():
                self.error = sys.exc_info()[1]
        except Exception as error:  # surfaced by stop() in the test thread
            self.error = error

    def stop(self):
        self.stopping.set()
        self.listener.close()
        self.thread.join(timeout=2)
        if self.error is not None:
            raise self.error


class BatchResetSlotsTest(unittest.TestCase):
    def setUp(self):
        self.server = FakeResidentServer()
        self.batch = Batch(battles=2, port=self.server.port, timeout=2)

    def tearDown(self):
        self.batch.close()
        self.server.stop()

    def test_partial_reset_preserves_other_slot(self):
        initial = self.batch.reset(seed=10)
        self.assertEqual([state.tick for state in initial], [90, 90])
        self.assertEqual(
            [self.server.slots[index]["seed"] for index in range(2)],
            [10, 11],
        )

        stepped = self.batch.step(ticks=10)
        self.assertEqual([state.tick for state in stepped], [100, 100])
        surviving_generation = stepped[0].generation

        refreshed = self.batch.reset_slots([1], [99])
        self.assertEqual(len(refreshed), 1)
        self.assertEqual(refreshed[0].tick, 90)
        self.assertEqual(self.server.slots[1]["seed"], 99)
        self.assertEqual(self.batch.states[0].tick, 100)
        self.assertEqual(self.batch.states[0].generation, surviving_generation)
        self.assertEqual(self.batch.states[1].tick, 90)

        resumed = self.batch.step(ticks=10)
        self.assertEqual([state.tick for state in resumed], [110, 100])
        self.assertEqual(resumed[0].generation, surviving_generation)

    def test_partial_reset_validates_arguments(self):
        self.batch.reset(seed=1)
        with self.assertRaisesRegex(ValueError, "same length"):
            self.batch.reset_slots([0], [])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.batch.reset_slots([0, 0], [2, 3])
        with self.assertRaisesRegex(ValueError, "between"):
            self.batch.reset_slots([2], [2])


if __name__ == "__main__":
    unittest.main()
