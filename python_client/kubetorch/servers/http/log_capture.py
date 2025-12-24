"""
Log Capture for log streaming.

Captures ALL stdout/stderr from the main process and subprocesses:
1. Pushes structured logs to log store (async batched) for querying
2. Forwards logs to original stdout/stderr (kubectl logs + user handlers)

Subprocess Log Capture:
- Subprocesses can send logs via the subprocess_queue (multiprocessing.Queue)
- Use get_subprocess_queue() to get the queue and push log entries
- Entry format: {"message": str, "level": str, "request_id": str, "extra_labels": dict}
"""

import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from queue import Empty
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class LogCapture:
    """
    Captures ALL stdout/stderr from the main process and:
    1. Pushes structured logs to log store (async batched)
    2. Forwards logs to original stdout/stderr (kubectl logs + user handlers)
    3. Collects logs from subprocesses via a multiprocessing.Queue

    Can also run in "queue mode" for subprocesses, where logs are pushed to a queue
    instead of being buffered and sent to Loki.
    """

    def __init__(
        self,
        log_store_url: str = None,
        labels: Dict[str, str] = None,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        # Queue mode: push to queue instead of Loki (for subprocesses)
        output_queue: mp.Queue = None,
        get_request_id_fn=None,
    ):
        """
        Initialize log capture.

        Args:
            log_store_url: Base URL for log store (e.g., http://kubetorch-data-store.namespace:3100)
            labels: Base labels for all logs (service, pod_name, namespace)
            batch_size: Number of log entries to batch before pushing
            flush_interval: Seconds between automatic flushes
            output_queue: If provided, push logs to this queue instead of Loki (subprocess mode)
            get_request_id_fn: Function to get current request_id (required for subprocess mode)
        """
        self.log_store_url = log_store_url
        self.labels = labels or {}
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Queue mode for subprocesses
        self._output_queue = output_queue
        self._get_request_id_fn = get_request_id_fn
        self._queue_mode = output_queue is not None

        # Buffer for log store push (not used in queue mode)
        self._buffer = []
        self._buffer_lock = threading.Lock()

        # Original streams (for forwarding)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Queue for subprocess log collection (multiprocessing-safe) - only in main process mode
        # Use Manager queue to avoid context mismatch issues between fork/spawn
        # (PyTorch CUDA changes start method to 'spawn' after LogCapture init)
        if self._queue_mode:
            self._subprocess_queue = None
            self._manager = None
        else:
            self._manager = mp.Manager()
            self._subprocess_queue = self._manager.Queue()

        # Background threads (not used in queue mode)
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self._subprocess_collector_thread: Optional[threading.Thread] = None

        # Track if started
        self._started = False

    def start(self):
        """Start log capture - call early in process startup."""
        if self._started:
            return

        # Replace stdout/stderr with interceptors
        sys.stdout = _StreamInterceptor(
            original=self._original_stdout,
            log_capture=self,
            stream_name="stdout",
        )
        sys.stderr = _StreamInterceptor(
            original=self._original_stderr,
            log_capture=self,
            stream_name="stderr",
        )

        # Redirect root logger to use our interceptor
        self._setup_logging_handler()

        # In queue mode (subprocess), we don't need background threads
        if not self._queue_mode:
            # Start background flush thread
            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._flush_thread.start()

            # Start subprocess queue collector
            self._subprocess_collector_thread = threading.Thread(target=self._collect_subprocess_logs, daemon=True)
            self._subprocess_collector_thread.start()

            logger.debug(f"LogCapture started - pushing to {self.log_store_url}")

        self._started = True

    def stop(self):
        """Stop log capture and flush remaining logs."""
        if not self._started:
            return

        self._stop_event.set()
        if not self._queue_mode:
            self._flush_now()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        # Shutdown manager if we created one
        if self._manager is not None:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

        self._started = False

    def flush(self):
        """Flush buffered logs to log store immediately."""
        self._flush_now()

    def ensure_handler(self):
        """Ensure our logging handler is still on the root logger.

        Call this after user code runs (e.g., after load_callable()) in case
        the user's code reconfigured logging with dictConfig() and removed our handler.
        """
        self._ensure_logging_handler()

    def get_subprocess_queue(self) -> mp.Queue:
        """
        Get the queue for subprocesses to push logs to.

        Subprocesses should push entries in the format:
        {"message": str, "level": str, "request_id": str, "extra_labels": dict}
        """
        return self._subprocess_queue

    def add_log(
        self,
        message: str,
        level: str = "INFO",
        request_id: str = "-",
        extra_labels: Optional[Dict[str, str]] = None,
        name: str = "print_redirect",
        asctime: Optional[str] = None,
    ):
        """Add a log entry to the buffer (or queue in subprocess mode).

        Args:
            message: The log message text
            level: Log level (INFO, ERROR, etc.)
            request_id: Request ID for filtering
            extra_labels: Additional Loki labels
            name: Logger name ("print_redirect" for stdout/stderr, logger name for logging)
            asctime: Formatted timestamp (auto-generated if not provided)
        """
        # In queue mode, push directly to queue for main process to handle
        if self._queue_mode:
            try:
                self._output_queue.put_nowait(
                    {
                        "message": message,
                        "level": level,
                        "request_id": request_id,
                        "extra_labels": extra_labels,
                    }
                )
            except Exception:
                pass  # Don't block on queue errors
            return

        timestamp_ns = time.time_ns()

        # Generate asctime if not provided
        if asctime is None:
            from datetime import datetime

            asctime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build labels for this log line
        labels = {**self.labels}
        if extra_labels:
            labels.update(extra_labels)
        labels["level"] = level
        labels["request_id"] = request_id

        with self._buffer_lock:
            self._buffer.append(
                {
                    "labels": labels,
                    "timestamp": timestamp_ns,
                    "message": message,
                    "name": name,
                    "asctime": asctime,
                    "levelname": level,
                }
            )
            if len(self._buffer) >= self.batch_size:
                self._flush_now()

    def _setup_logging_handler(self):
        """Add handler to root logger that feeds into our capture."""
        self._ensure_logging_handler()

    def _ensure_logging_handler(self):
        """Ensure our handler is on the root logger.

        This is idempotent and can be called multiple times (e.g., after user code
        reconfigures logging with dictConfig() which might remove our handler).
        """
        # Check if our handler is already there
        for h in logging.root.handlers:
            if isinstance(h, _LogCaptureHandler) and h.log_capture is self:
                return  # Already present

        # Add our handler
        handler = _LogCaptureHandler(self)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)

    def _flush_loop(self):
        """Background thread: flush buffer periodically."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self._flush_now()

    def _flush_now(self):
        """Push buffered logs to log store."""
        import json

        with self._buffer_lock:
            if not self._buffer:
                return
            batch, self._buffer = self._buffer, []

        # Group by labels for efficient push (Loki format)
        streams: Dict[tuple, Dict] = {}
        for entry in batch:
            label_key = tuple(sorted(entry["labels"].items()))
            if label_key not in streams:
                streams[label_key] = {"stream": entry["labels"], "values": []}

            # Format message as JSON for client parsing (distinguishes logs from prints)
            json_message = json.dumps(
                {
                    "message": entry["message"],
                    "name": entry.get("name", "print_redirect"),
                    "levelname": entry.get("levelname", "INFO"),
                    "asctime": entry.get("asctime", ""),
                }
            )
            streams[label_key]["values"].append([str(entry["timestamp"]), json_message])

        payload = {"streams": list(streams.values())}

        try:
            requests.post(
                f"{self.log_store_url}/loki/api/v1/push",
                json=payload,
                timeout=5,
            )
        except Exception as e:
            # Log to original stderr (don't recurse)
            self._original_stderr.write(f"Failed to push logs to log store: {e}\n")

    def _collect_subprocess_logs(self):
        """Collect logs from subprocess queue and add to buffer."""
        while not self._stop_event.is_set():
            try:
                entry = self._subprocess_queue.get(timeout=0.5)
                # Entry should have: message, level, request_id, extra_labels
                self.add_log(
                    message=entry.get("message", ""),
                    level=entry.get("level", "INFO"),
                    request_id=entry.get("request_id", "-"),
                    extra_labels=entry.get("extra_labels"),
                )
            except Empty:
                continue
            except Exception:
                pass  # Don't crash on malformed entries


class _StreamInterceptor:
    """Intercepts writes to stdout/stderr."""

    def __init__(self, original, log_capture: LogCapture, stream_name: str):
        self.original = original
        self.log_capture = log_capture
        self.stream_name = stream_name

    def _is_from_logging(self):
        """Check if the current write call is coming from the logging system."""
        frame = sys._getframe()
        while frame:
            if frame.f_globals.get("__name__", "").startswith("logging"):
                return True
            frame = frame.f_back
        return False

    def write(self, msg: str):
        # Always forward to original (kubectl logs, user handlers)
        self.original.write(msg)

        # Skip if from logging system (already captured via handler)
        if self._is_from_logging():
            return

        # Also capture for log store (skip empty lines)
        if msg.strip():
            level = "ERROR" if self.stream_name == "stderr" else "INFO"
            # Get request_id from context if available
            try:
                from .utils import request_id_ctx_var

                request_id = request_id_ctx_var.get("-")
            except Exception:
                request_id = "-"
            self.log_capture.add_log(msg.strip(), level=level, request_id=request_id)

    def flush(self):
        self.original.flush()

    def isatty(self):
        if hasattr(self.original, "isatty"):
            return self.original.isatty()
        return False

    def fileno(self):
        if hasattr(self.original, "fileno"):
            return self.original.fileno()
        raise OSError("Stream does not support fileno()")

    @property
    def encoding(self):
        if hasattr(self.original, "encoding"):
            return self.original.encoding
        return "utf-8"

    # Proxy other attributes
    def __getattr__(self, name):
        return getattr(self.original, name)


class _LogCaptureHandler(logging.Handler):
    """Logging handler that feeds into LogCapture."""

    def __init__(self, log_capture: LogCapture):
        super().__init__()
        self.log_capture = log_capture
        # Create a formatter for asctime
        self._time_formatter = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S")

    def emit(self, record):
        # Get request_id from record if set by filter, otherwise from context variable.
        # Filters on root logger don't run for records that propagate from child loggers,
        # so we need to check the context variable directly.
        request_id = getattr(record, "request_id", None)
        if request_id is None or request_id == "-":
            try:
                from .utils import request_id_ctx_var

                request_id = request_id_ctx_var.get("-")
            except Exception:
                request_id = "-"

        # Format timestamp
        asctime = self._time_formatter.formatTime(record, self._time_formatter.datefmt)

        self.log_capture.add_log(
            message=self.format(record),
            level=record.levelname,
            request_id=request_id,
            name=record.name,
            asctime=asctime,
        )


# Global instance for easy access
_log_capture: Optional[LogCapture] = None


def get_log_capture() -> Optional[LogCapture]:
    """Get the global LogCapture instance."""
    return _log_capture


def get_subprocess_queue() -> Optional[mp.Queue]:
    """
    Get the subprocess queue for sending logs from subprocesses.

    Returns None if LogCapture is not initialized.

    Usage in subprocess:
        queue = get_subprocess_queue()
        if queue:
            queue.put({
                "message": "Log message",
                "level": "INFO",
                "request_id": "-",
                "extra_labels": {"source": "pds"}
            })
    """
    if _log_capture is not None:
        return _log_capture.get_subprocess_queue()
    return None


def init_log_capture(
    log_store_url: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Optional[LogCapture]:
    """
    Initialize and start global log capture.

    Automatically constructs log store URL and labels from environment if not provided.
    """
    global _log_capture

    if _log_capture is not None:
        return _log_capture

    # Get log store URL from environment
    if log_store_url is None:
        log_store_host = os.environ.get("LOG_STORE_HOST")
        log_store_port = os.environ.get("LOG_STORE_PORT", "3100")
        if not log_store_host:
            # Default to data store service
            namespace = os.environ.get("POD_NAMESPACE", "default")
            log_store_host = f"kubetorch-data-store.{namespace}.svc.cluster.local"
        log_store_url = f"http://{log_store_host}:{log_store_port}"

    # Get labels from environment
    if labels is None:
        labels = {
            "service": os.environ.get("KT_SERVICE", "unknown"),
            "pod": os.environ.get("POD_NAME", "unknown"),
            "namespace": os.environ.get("POD_NAMESPACE", "default"),
        }

    _log_capture = LogCapture(log_store_url=log_store_url, labels=labels)
    _log_capture.start()
    return _log_capture


def stop_log_capture():
    """Stop the global log capture."""
    global _log_capture
    if _log_capture is not None:
        _log_capture.stop()
        _log_capture = None


def create_subprocess_log_capture(output_queue: mp.Queue) -> Optional[LogCapture]:
    """
    Create a LogCapture instance for use in a subprocess.

    In subprocess mode, logs are pushed to the output_queue instead of Loki.
    The main process's LogCapture will collect from this queue and push to Loki.

    Args:
        output_queue: Queue to push log entries to (from main process's LogCapture)

    Returns:
        LogCapture instance in queue mode, or None if queue is None
    """
    if output_queue is None:
        return None

    log_capture = LogCapture(output_queue=output_queue)
    log_capture.start()
    return log_capture
