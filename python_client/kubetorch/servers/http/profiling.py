import contextlib
import json
import logging
import os
import signal
import subprocess
import tempfile
import time

from kubetorch.globals import ProfilerConfig
from kubetorch.servers.http.constants import PYSPY_SAMPLE_RATE_HZ

logger = logging.getLogger(__name__)


# Using a high duration (e.g., 5000s) since we don't know the exact fn runtime.
# We terminate the process manually at the end of the execution.
@contextlib.contextmanager
def pyspy_profiler(output: dict, output_format: str, pid=None, rate=PYSPY_SAMPLE_RATE_HZ):
    pid_to_profile = pid or os.getpid()

    # Determine the file extension based on output format
    if output_format == "flamegraph":
        ext = ".svg"
    elif output_format == "raw":
        ext = ".raw"
    else:  # "speedscope", "chrometrace"
        ext = ".json"

    try:
        with tempfile.NamedTemporaryFile(mode="r+", suffix=ext, delete=True) as temp_file:
            temp_out_path = temp_file.name

            cmd = [
                "py-spy",
                "record",
                "--pid",
                str(pid_to_profile),
                "--rate",
                str(rate),
                "--format",
                output_format,
                "--output",
                temp_out_path,
                "--nonblocking",
            ]

            logger.debug(f"Starting py-spy to profile PID {pid_to_profile} to {temp_out_path}")

            pyspy_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

            # This is the point where the profiler is yielding control back to the caller
            yield pyspy_process

            # Handle the case where the caller returned but py-spy is still running
            if pyspy_process and pyspy_process.poll() is None:
                logger.debug("Terminating py-spy process for cleanup...")
                pyspy_process.send_signal(signal.SIGINT)
                try:
                    pyspy_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("py-spy did not terminate gracefully, killing...")
                    pyspy_process.kill()
                    pyspy_process.wait()

            # Give a small buffer time for I/O flush
            time.sleep(0.2)

            # Read the output file contents
            try:
                # Seek to the start of the file before reading,
                # as the file pointer might be at the end after py-spy wrote to it.
                temp_file.seek(0)
                content = temp_file.read()

                if content.strip():
                    output["result"] = content
                    logger.debug(f"Successfully read {len(content)} bytes from {temp_out_path}")
                else:
                    logger.warning(f"py-spy output file {temp_out_path} is empty")

            except Exception as e:
                logger.error(f"Error reading py-spy output from {temp_out_path}: {e}")

    except subprocess.TimeoutExpired:
        logger.error("Timeout waiting for py-spy process to terminate and flush output.")
    except FileNotFoundError:
        # This catches errors if the 'py-spy' executable itself is not found
        logger.error("py-spy executable not found in PATH.")
    except subprocess.CalledProcessError as e:
        logger.error(f"py-spy profiling failed (Exit Code {e.returncode}).")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def run_with_profile(
    fn,
    *args,
    profiler: ProfilerConfig = None,
    callable_name: str = None,
    **kwargs,
):
    """Run the function with optional profiling, using Pyroscope as the backend."""

    profiler_type = profiler.profiler_type
    output_format = profiler.output_format

    if profiler_type == "pytorch":
        import torch
        from torch.profiler import profile, ProfilerActivity, record_function

        analyze_stack_traces = profiler.analyze_stack_traces
        group_by_input_shape = profiler.group_by_input_shape
        group_by_stack_n = profiler.group_by_stack_n

        if analyze_stack_traces:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
                record_shapes=True,
            ) as prof:
                result = fn(*args, **kwargs)
        elif output_format == "memory_timeline":
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True
            ) as prof:
                result = fn(*args, **kwargs)
        else:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
                callable_name = callable_name or fn.__name__
                with record_function(callable_name):
                    result = fn(*args, **kwargs)

        profiler_output = None

        if output_format == "chrometrace":
            with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp:
                trace_path = tmp.name
                prof.export_chrome_trace(trace_path)
                with open(trace_path, "rb") as f:
                    profiler_output = json.loads(f.read())
        elif output_format == "memory_timeline":
            file_suffix = profiler.memory_timeline_output_type
            with tempfile.NamedTemporaryFile(delete=True, suffix=f".{file_suffix}") as timeline_tmp:
                memory_timeline_path = timeline_tmp.name
                prof.export_memory_timeline(memory_timeline_path)
                with open(memory_timeline_path, "rb") as f:
                    profiler_output = f.read().decode("utf-8")
        elif output_format == "stacks":
            with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as stacks_tmp:
                stacks_path = stacks_tmp.name
                prof.export_stacks(path=stacks_path, metric="self_cuda_time_total")
                with open(stacks_path, "rb") as f:
                    profiler_output = f.read().decode("utf-8")
        elif output_format == "table":
            prof = prof.key_averages(group_by_input_shape=group_by_input_shape, group_by_stack_n=group_by_stack_n)
            profiler_output = prof.table(sort_by=profiler.table_sort_by)

        return result, profiler_output

    if profiler_type == "pyspy":

        pyspy_output_data = {"result": None}

        with pyspy_profiler(output=pyspy_output_data, output_format=output_format) as pyspy_proc:
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

        profiler_output = pyspy_output_data.get("result")

        if profiler_output is None:
            logger.warning("pyspy profiler captured no output.")

        return result, profiler_output
