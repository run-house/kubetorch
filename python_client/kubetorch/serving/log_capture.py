"""
Log capture for streaming logs to Loki and kubectl logs.

_StreamInterceptor: Captures print() statements
_LogCaptureHandler: Captures logger calls, writes to stdout

StreamHandlers are redirected to original_stdout to prevent double-capture.
Subprocesses use queue mode to forward logs to main process.
"""

import logging
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
from queue import Empty
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Silence httpx/httpcore INFO logs to prevent feedback loop
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class LogCapture:
    """
    Captures stdout/stderr and logger output, pushes to Loki.

    Two capture mechanisms (no coordination needed between them):
    1. _StreamInterceptor: wraps sys.stdout/stderr, captures print() statements
    2. _LogCaptureHandler: logging handler that captures logger calls AND writes to stdout

    Can run in "queue mode" for subprocesses, where logs are pushed to a queue
    instead of Loki.
    """

    def __init__(
        self,
        log_store_url: str = None,
        labels: Dict[str, str] = None,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        output_queue: mp.Queue = None,
    ):
        self.log_store_url = log_store_url
        self.labels = labels or {}
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        # Queue mode for subprocesses
        self._output_queue = output_queue
        self._queue_mode = output_queue is not None

        # Buffer for log store push
        self._buffer = []
        self._buffer_lock = threading.Lock()

        # Original streams (for forwarding to kubectl logs)
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Subprocess queue for multiprocessing
        if self._queue_mode:
            self._subprocess_queue = None
            self._manager = None
        else:
            self._manager = mp.Manager()
            self._subprocess_queue = self._manager.Queue()

        # Background threads
        self._stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self._subprocess_collector_thread: Optional[threading.Thread] = None
        self._started = False
        self._session: Optional[httpx.Client] = None

    def start(self):
        """Start log capture."""
        if self._started:
            return

        # Replace stdout/stderr with interceptors
        sys.stdout = _StreamInterceptor(self._original_stdout, self, "stdout")
        sys.stderr = _StreamInterceptor(self._original_stderr, self, "stderr")

        # Redirect any existing StreamHandlers to original_stdout to prevent double-capture
        self._redirect_stream_handlers()

        # Add our logging handler
        self._setup_logging_handler()

        if not self._queue_mode:
            self._session = httpx.Client(timeout=5.0)
            self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
            self._flush_thread.start()
            self._subprocess_collector_thread = threading.Thread(target=self._collect_subprocess_logs, daemon=True)
            self._subprocess_collector_thread.start()

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

        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

        if self._manager:
            try:
                self._manager.shutdown()
            except Exception:
                pass
            self._manager = None

        self._started = False

    def flush(self):
        """Flush buffered logs to log store immediately."""
        self._flush_now()

    def _redirect_stream_handlers(self):
        """Redirect StreamHandlers to original_stdout to prevent double-capture."""
        for handler in logging.root.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, _LogCaptureHandler):
                # Check if stream is stdout/stderr (original OR interceptor)
                is_stdout = handler.stream in (sys.stdout, self._original_stdout)
                is_stderr = handler.stream in (sys.stderr, self._original_stderr)
                if is_stdout:
                    handler.stream = self._original_stdout
                elif is_stderr:
                    handler.stream = self._original_stderr

    def _setup_logging_handler(self):
        """Add handler to root logger."""
        self._ensure_logging_handler()

    def _ensure_logging_handler(self):
        """Ensure our handler is on the root logger (idempotent)."""
        for h in logging.root.handlers:
            if isinstance(h, _LogCaptureHandler) and h.log_capture is self:
                return

        handler = _LogCaptureHandler(self, self._original_stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.root.addHandler(handler)

    def ensure_handler(self):
        """Re-add handler and redirect any new StreamHandlers added by user code."""
        self._redirect_stream_handlers()
        self._ensure_logging_handler()

    def get_subprocess_queue(self) -> mp.Queue:
        """Get queue for subprocess log forwarding."""
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
        """Add a log entry to the buffer (or queue in subprocess mode)."""
        if self._queue_mode:
            try:
                self._output_queue.put_nowait(
                    {
                        "message": message,
                        "level": level,
                        "request_id": request_id,
                        "extra_labels": extra_labels,
                        "name": name,
                    }
                )
            except Exception:
                pass
            return

        timestamp_ns = time.time_ns()
        if asctime is None:
            from datetime import datetime

            asctime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    def _flush_loop(self):
        """Background flush thread."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self._flush_now()

    def _flush_now(self):
        """Push buffered logs to Loki."""
        import json

        with self._buffer_lock:
            if not self._buffer:
                return
            batch, self._buffer = self._buffer, []

        streams: Dict[tuple, Dict] = {}
        for entry in batch:
            label_key = tuple(sorted(entry["labels"].items()))
            if label_key not in streams:
                streams[label_key] = {"stream": entry["labels"], "values": []}

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
            if self._session:
                self._session.post(f"{self.log_store_url}/loki/api/v1/push", json=payload)
        except Exception as e:
            self._original_stderr.write(f"Failed to push logs: {e}\n")

    def _collect_subprocess_logs(self):
        """Collect logs from subprocess queue."""
        while not self._stop_event.is_set():
            try:
                entry = self._subprocess_queue.get(timeout=0.5)
                self.add_log(
                    message=entry.get("message", ""),
                    level=entry.get("level", "INFO"),
                    request_id=entry.get("request_id", "-"),
                    extra_labels=entry.get("extra_labels"),
                    name=entry.get("name", "print_redirect"),
                )
            except Empty:
                continue
            except Exception:
                pass


class _StreamInterceptor:
    """Intercepts stdout/stderr for print() capture."""

    def __init__(self, original, log_capture: LogCapture, stream_name: str):
        self.original = original
        self.log_capture = log_capture
        self.stream_name = stream_name

    def write(self, msg: str):
        self.original.write(msg)
        self.original.flush()

        if msg.strip():
            level = "ERROR" if self.stream_name == "stderr" else "INFO"
            try:
                from .utils import request_id_ctx_var

                request_id = request_id_ctx_var.get("-")
            except Exception:
                request_id = "-"
            self.log_capture.add_log(msg.strip(), level=level, request_id=request_id)

    def flush(self):
        self.original.flush()

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()

    def fileno(self):
        if hasattr(self.original, "fileno"):
            return self.original.fileno()
        raise OSError("Stream does not support fileno()")

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self.original, name)


class _LogCaptureHandler(logging.Handler):
    """Captures logger calls, writes to stdout, pushes to Loki. Sets flag to prevent duplicates."""

    def __init__(self, log_capture: LogCapture, original_stdout):
        super().__init__()
        self.log_capture = log_capture
        self.original_stdout = original_stdout
        self._time_formatter = logging.Formatter(datefmt="%Y-%m-%d %H:%M:%S")

    def emit(self, record):
        try:
            formatted = self.format(record)
            self.original_stdout.write(formatted + "\n")
            self.original_stdout.flush()

            request_id = getattr(record, "request_id", None)
            if request_id is None or request_id == "-":
                try:
                    from .utils import request_id_ctx_var

                    request_id = request_id_ctx_var.get("-")
                except Exception:
                    request_id = "-"

            asctime = self._time_formatter.formatTime(record, self._time_formatter.datefmt)
            self.log_capture.add_log(
                message=record.getMessage(),
                level=record.levelname,
                request_id=request_id,
                name=record.name,
                asctime=asctime,
            )
        except Exception:
            self.handleError(record)


# Global instance
_log_capture: Optional[LogCapture] = None


def get_log_capture() -> Optional[LogCapture]:
    """Get the global LogCapture instance."""
    return _log_capture


def get_subprocess_queue() -> Optional[mp.Queue]:
    """Get subprocess queue for log forwarding."""
    if _log_capture is not None:
        return _log_capture.get_subprocess_queue()
    return None


def init_log_capture(
    log_store_url: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Optional[LogCapture]:
    """Initialize and start global log capture."""
    global _log_capture

    if _log_capture is not None:
        return _log_capture

    if log_store_url is None:
        log_store_host = os.environ.get("LOG_STORE_HOST")
        log_store_port = os.environ.get("LOG_STORE_PORT", "3100")
        if not log_store_host:
            namespace = os.environ.get("POD_NAMESPACE", "default")
            log_store_host = f"kubetorch-data-store.{namespace}.svc.cluster.local"
        log_store_url = f"http://{log_store_host}:{log_store_port}"

    if labels is None:
        pod_name = os.environ.get("POD_NAME") or os.environ.get("HOSTNAME") or socket.gethostname()
        labels = {
            "service": os.environ.get("KT_SERVICE", "unknown"),
            "pod": pod_name,
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
    """Create LogCapture for subprocess (queue mode)."""
    global _log_capture

    if output_queue is None:
        return None

    log_capture = LogCapture(output_queue=output_queue)
    log_capture.start()
    _log_capture = log_capture
    return log_capture
