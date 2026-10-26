"""Bound caller latency without accumulating blocked RPC workers."""

import copy
import math
import threading


class SingleFlightQuery:
    """One outstanding read-only query, including after caller timeout.

    Python cannot cancel an arbitrary blocking transport. A timed-out worker
    remains the sole outstanding query until it actually returns; its result
    is discarded. Callers fail promptly rather than spawning replacements.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = None

    def call(self, query, request, timeout):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("query timeout must be finite and positive")
        with self._lock:
            if self._active is not None and not self._active["done"].is_set():
                raise RuntimeError("previous preview query is still running")
            job = {"done": threading.Event()}
            request = copy.deepcopy(request)
            self._active = job

            def run():
                try:
                    job["result"] = query(request)
                except BaseException as exc:
                    job["error"] = exc
                finally:
                    job["done"].set()

            worker = threading.Thread(target=run, name="kinematic-preview-query", daemon=True)
            try:
                worker.start()
            except BaseException:
                self._active = None
                raise
        if not job["done"].wait(timeout):
            raise TimeoutError("preview query deadline exceeded")
        if "error" in job:
            raise job["error"]
        return job["result"]
