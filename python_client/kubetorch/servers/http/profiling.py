import contextlib
import logging
import subprocess
import time
import urllib.parse
from pathlib import Path

from pyroscope import tag_wrapper

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def pyspy_profiler(pid, out_path, rate=100, timeout=5):
    """Context manager to run py-spy and ensure its output file is flushed before exit."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "py-spy",
        "record",
        "--pid",
        str(pid),
        "--rate",
        str(rate),
        "--format",
        "speedscope",
        "-o",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd)
    try:
        yield
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("py-spy did not exit cleanly after terminate()")

        # Wait up to timeout seconds for file to appear and be nonempty
        start = time.time()
        while (not out_path.exists() or out_path.stat().st_size == 0) and (time.time() - start) < timeout:
            time.sleep(0.2)

        if not out_path.exists() or out_path.stat().st_size == 0:
            logger.error(f"py-spy output file not found or empty after {timeout}s: {out_path}")


def run_with_optional_profile(
    fn,
    *args,
    profiler: str = None,
    request_id: str = None,
    callable_name: str = None,
    profile_type: str = None,
    **kwargs,
):
    """Run the function with optional profiling, using Pyroscope as the backend."""
    if profiler not in ("pyspy", "torch"):
        return fn(*args, **kwargs), None

    tags = {
        "request_id": request_id,
        "callable_name": callable_name or fn.__name__,
    }

    # Attach tags for this request — this doesn’t reconfigure or restart the profiler
    with tag_wrapper(tags):
        logger.info(f"Running {callable_name} with profiler (tags={tags})")
        result = fn(*args, **kwargs)

    query_expr = f'process_cpu{{application_name="kubetorch",request_id="{request_id}"}}'
    encoded_query = urllib.parse.quote(query_expr)
    profile_url = f"?query={encoded_query}"

    return result, {"type": profiler, "viewer": "pyroscope", "url": profile_url}
