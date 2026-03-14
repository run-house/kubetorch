"""
Per-key read-write locks for the metadata server.

Provides fine-grained locking so operations on different keys can run in parallel,
while ensuring thread-safety for concurrent access to the same key.

"""

from contextlib import contextmanager
from threading import Condition, Lock


class RWLock:
    """Allows multiple concurrent readers OR a single exclusive writer."""

    def __init__(self):
        self._lock = Lock()
        self._cond = Condition(self._lock)
        self._readers = 0
        self._writer = False

    def read_acquire(self, timeout: float = 1.0) -> bool:
        """Acquire read lock. Returns True if acquired, False on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: not self._writer, timeout=timeout):
                return False
            self._readers += 1
            return True

    def read_release(self):
        """Release read lock."""
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self, timeout: float = 1.0) -> bool:
        """Acquire write lock. Returns True if acquired, False on timeout."""
        with self._cond:
            if not self._cond.wait_for(
                lambda: not self._writer and self._readers == 0, timeout=timeout
            ):
                return False
            self._writer = True
            return True

    def write_release(self):
        """Release write lock."""
        with self._cond:
            self._writer = False
            self._cond.notify_all()


class PerKeyRWLock:
    """Per-key read-write lock manager.

    Creates a separate RWLock for each key, allowing operations on different
    keys to proceed in parallel.

    """

    def __init__(self, timeout: float = 1.0):
        self._locks: dict[str, RWLock] = {}
        self._global_lock = Lock()  # Only for creating new key locks
        self._timeout = timeout

    def _get_lock(self, key: str) -> RWLock:
        """Get or create a lock for the given key."""
        if key not in self._locks:
            with self._global_lock:
                if key not in self._locks:  # Double-check after acquiring lock
                    self._locks[key] = RWLock()
        return self._locks[key]

    @contextmanager
    def read(self, key: str):
        """Context manager for read access to a key."""
        lock = self._get_lock(key)
        if not lock.read_acquire(self._timeout):
            raise TimeoutError(
                f"Read lock timeout for key '{key}' - server may be overloaded"
            )
        try:
            yield
        finally:
            lock.read_release()

    @contextmanager
    def write(self, key: str):
        """Context manager for write access to a key."""
        lock = self._get_lock(key)
        if not lock.write_acquire(self._timeout):
            raise TimeoutError(
                f"Write lock timeout for key '{key}' - server may be overloaded"
            )
        try:
            yield
        finally:
            lock.write_release()

    def cleanup_unused(self, active_keys: set):
        """Remove locks for keys that no longer exist. Call periodically."""
        with self._global_lock:
            stale = [k for k in self._locks if k not in active_keys]
            for k in stale:
                del self._locks[k]


class SimpleLock:
    """Simple lock wrapper with timeout for non-per-key use cases."""

    def __init__(self, timeout: float = 1.0):
        self._lock = Lock()
        self._timeout = timeout

    def __enter__(self):
        acquired = self._lock.acquire(timeout=self._timeout)
        if not acquired:
            raise TimeoutError("Lock acquisition timeout - server may be overloaded")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
