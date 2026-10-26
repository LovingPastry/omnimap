"""Caller deadline and single-flight behavior under a genuinely blocked RPC."""
import sys
import threading
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"info_flow"))
from bounded_query import SingleFlightQuery


class QueryTests(unittest.TestCase):
    def test_timeout_retains_live_slot_and_discards_late_result(self):
        client = SingleFlightQuery()
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def blocked(request):
            entered.set()
            release.wait()
            finished.set()
            return "late"
        try:
            with self.assertRaises(TimeoutError):
                client.call(blocked, {}, .02)
            self.assertTrue(entered.is_set())
            for _ in range(3):
                with self.assertRaisesRegex(RuntimeError, "still running"):
                    client.call(lambda _: self.fail("replacement started"), {}, .02)
        finally:
            release.set()
        self.assertTrue(finished.wait(1))
        self.assertTrue(client._active["done"].wait(1))
        self.assertEqual(client.call(lambda _: "fresh", {}, .5), "fresh")

    def test_request_owned_error_propagated_and_slot_recovers(self):
        client = SingleFlightQuery()
        source = {"values": [1]}
        def fail(request):
            request["values"].append(2)
            raise ValueError("transport failure")
        with self.assertRaisesRegex(ValueError, "transport failure"):
            client.call(fail, source, .5)
        self.assertEqual(source, {"values": [1]})
        self.assertEqual(client.call(lambda x: x["values"], source, .5), [1])
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                client.call(lambda _: None, {}, timeout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
