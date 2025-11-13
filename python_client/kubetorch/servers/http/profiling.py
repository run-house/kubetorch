import contextlib
import logging
import os
import signal
import subprocess
import time

from kubetorch.servers.http.utils import PYSPY_SAMPLE_RATE_HZ, SUPPORTED_PROFILERS

logger = logging.getLogger(__name__)


# Using a high duration (e.g., 5000s) since we don't know the exact fn runtime.
# We terminate the process manually at the end of the execution.
@contextlib.contextmanager
def pyspy_profiler(output: dict, request_id: str, pid=None, rate=PYSPY_SAMPLE_RATE_HZ):
    pid_to_profile = pid or os.getpid()
    out_path = f"/tmp/{request_id}.raw"

    cmd = [
        "py-spy",
        "record",
        "--pid",
        str(pid_to_profile),
        "--rate",
        str(rate),
        "--format",
        "raw",
        "--output",
        out_path,
        "--nonblocking",
    ]

    pyspy_process = None

    try:
        logger.debug(f"Starting py-spy to profile PID {pid_to_profile} to {out_path}...")

        pyspy_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        yield pyspy_process

    except subprocess.TimeoutExpired:
        logger.error("Timeout waiting for py-spy process to terminate and flush output.")
    except FileNotFoundError:
        logger.error(f"py-spy output file not found at {out_path}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"py-spy profiling failed (Exit Code {e.returncode}).")
        raise

    finally:
        # This handles the case where the function call finished BUT the process
        # was not stopped gracefully (e.g., due to an exception propagating)
        if pyspy_process and pyspy_process.poll() is None:
            logger.debug("Terminating py-spy process for cleanup...")
            # Send SIGINT again as a fallback, then wait/kill
            pyspy_process.send_signal(signal.SIGINT)
            try:
                # Wait for graceful shutdown and file flush
                pyspy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("py-spy did not terminate gracefully, killing...")
                pyspy_process.kill()
                pyspy_process.wait()

        time.sleep(0.2)

        # read the output file and clean up
        if os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    content = f.read()
                    if content.strip():
                        output["result"] = content
                        logger.debug(f"Successfully read {len(content)} bytes from {out_path}")
                    else:
                        logger.warning(f"py-spy output file {out_path} is empty")
            except Exception as e:
                logger.error(f"Error reading py-spy output: {e}")
            finally:
                os.remove(out_path)
        else:
            logger.error(f"py-spy output file not found at {out_path}")


def run_with_profile(
    fn,
    *args,
    profiler: str = None,
    request_id: str = None,
    callable_name: str = None,
    **kwargs,
):
    """Run the function with optional profiling, using Pyroscope as the backend."""
    if profiler not in SUPPORTED_PROFILERS:
        return fn(*args, **kwargs), None

    if profiler == "torch":
        from torch.profiler import profile, ProfilerActivity, record_function

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
            callable_name = callable_name or fn.__name__
            with record_function(callable_name):
                result = fn(*args, **kwargs)

        table = prof.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total",
            row_limit=1,
            max_src_column_width=40,
            max_name_column_width=30,
            max_shapes_column_width=40,
        )
        return result, table
    if profiler == "pyspy":

        pyspy_output_data = {"result": None}

        with pyspy_profiler(output=pyspy_output_data, request_id=request_id) as pyspy_proc:
            try:
                logger.debug(f"Running {callable_name} with pyspy profiler")
                result = fn(*args, **kwargs)
            finally:
                if pyspy_proc and pyspy_proc.poll() is None:
                    logger.debug("Function finished. Terminating py-spy sampling now to collect trace.")
                    pyspy_proc.send_signal(signal.SIGINT)
                    try:
                        # Wait for process to exit gracefully after SIGINT
                        pyspy_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pyspy_proc.kill()

        pyspy_json_data = pyspy_output_data.get("result")

        if pyspy_json_data is None:
            logger.warning("pyspy profiler ran, but captured no output.")

        return result, pyspy_json_data
