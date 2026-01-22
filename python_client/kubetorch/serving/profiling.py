import contextlib
import json
import logging
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Union

import cloudpickle

from kubetorch.constants import PYSPY_SAMPLE_RATE_HZ
from kubetorch.globals import ProfilerConfig

logger = logging.getLogger(__name__)

# Minimal script template for subprocess profiling - keeps call stack clean
_SUBPROCESS_SCRIPT = """\
import cloudpickle
from pathlib import Path
import time

with open("{args_path}", "rb") as f:
    fn, args, kwargs = cloudpickle.load(f)

Path("{ready_path}").touch()
while Path("{ready_path}").exists():
    time.sleep(0.01)

try:
    result = fn(*args, **kwargs)
    with open("{result_path}", "wb") as f:
        cloudpickle.dump(("success", result), f)
except Exception as e:
    import traceback
    with open("{result_path}", "wb") as f:
        cloudpickle.dump(("error", str(e), traceback.format_exc()), f)
"""

# Torch profiler subprocess script - runs user function in isolated process
_TORCH_SUBPROCESS_SCRIPT = """\
import cloudpickle
from pathlib import Path
import time
import json
import tempfile

with open("{args_path}", "rb") as f:
    fn, args, kwargs, profiler_config = cloudpickle.load(f)

Path("{ready_path}").touch()
while Path("{ready_path}").exists():
    time.sleep(0.01)

try:
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # Common profiler options for more detailed capture
    common_kwargs = {{
        "activities": activities,
        "profile_memory": True,
        "record_shapes": True,  # Record tensor shapes for more context
        "with_flops": True,  # Record FLOPs for operations
    }}

    if profiler_config.get("analyze_stack_traces"):
        # Enable detailed stack traces with Python line numbers
        prof = profile(
            **common_kwargs,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )
    else:
        prof = profile(**common_kwargs)

    callable_name = profiler_config.get("callable_name", "unknown")

    # Run function with profiler
    if profiler_config.get("analyze_stack_traces"):
        with prof:
            result = fn(*args, **kwargs)
    else:
        with prof:
            with record_function(callable_name):
                result = fn(*args, **kwargs)

    # Export profiler output
    profiler_output = None
    output_format = profiler_config.get("output_format", "chrometrace")
    group_by_input_shape = profiler_config.get("group_by_input_shape", False)
    group_by_stack_n = profiler_config.get("group_by_stack_n")
    table_sort_by = profiler_config.get("table_sort_by")

    if output_format == "chrometrace":
        with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            with open(tmp.name, "rb") as f:
                profiler_output = json.loads(f.read())
    elif output_format == "stacks":
        # Export stack traces - this format shows more detailed stack information
        stacks_output = prof.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n  # Use the group_by_stack_n set in the profiler config
        ).table(sort_by="self_cpu_time_total", row_limit=100)
        profiler_output = {{"format": "stacks", "content": stacks_output}}
    elif output_format == "table":
        # Export table format with optional sort key
        sort_by = table_sort_by if table_sort_by else "cpu_time_total"
        table_output = prof.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n
        ).table(sort_by=sort_by, row_limit=-1)
        profiler_output = table_output

    with open("{result_path}", "wb") as f:
        cloudpickle.dump(("success", result, profiler_output), f)

except Exception as e:
    import traceback
    with open("{result_path}", "wb") as f:
        cloudpickle.dump(("error", str(e), traceback.format_exc()), f)
"""

# Torch profiler subprocess script for async functions
_TORCH_ASYNC_SUBPROCESS_SCRIPT = """\
import cloudpickle
from pathlib import Path
import time
import json
import tempfile
import asyncio

with open("{args_path}", "rb") as f:
    fn, args, kwargs, profiler_config = cloudpickle.load(f)

Path("{ready_path}").touch()
while Path("{ready_path}").exists():
    time.sleep(0.01)

async def _run_async():
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # Common profiler options for more detailed capture
    common_kwargs = {{
        "activities": activities,
        "profile_memory": True,
        "record_shapes": True,  # Record tensor shapes for more context
        "with_flops": True,  # Record FLOPs for operations
    }}

    if profiler_config.get("analyze_stack_traces"):
        # Enable detailed stack traces with Python line numbers
        prof = profile(
            **common_kwargs,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )
    else:
        prof = profile(**common_kwargs)

    callable_name = profiler_config.get("callable_name", "unknown")

    # Run async function with profiler
    if profiler_config.get("analyze_stack_traces"):
        with prof:
            result = await fn(*args, **kwargs)
    else:
        with prof:
            with record_function(callable_name):
                result = await fn(*args, **kwargs)

    # Export profiler output
    profiler_output = None
    output_format = profiler_config.get("output_format", "chrometrace")
    group_by_input_shape = profiler_config.get("group_by_input_shape", False)
    group_by_stack_n = profiler_config.get("group_by_stack_n")
    table_sort_by = profiler_config.get("table_sort_by")

    if output_format == "chrometrace":
        with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            with open(tmp.name, "rb") as f:
                profiler_output = json.loads(f.read())
    elif output_format == "stacks":
        # Export stack traces - this format shows more detailed stack information
        stacks_output = prof.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n  # Use the group_by_stack_n set in the profiler config
        ).table(sort_by="self_cpu_time_total", row_limit=100)
        profiler_output = {{"format": "stacks", "content": stacks_output}}
    elif output_format == "table":
        # Export table format with optional sort key
        sort_by = table_sort_by if table_sort_by else "cpu_time_total"
        table_output = prof.key_averages(
            group_by_input_shape=group_by_input_shape,
            group_by_stack_n=group_by_stack_n
        ).table(sort_by=sort_by, row_limit=-1)
        profiler_output = table_output

    return result, profiler_output

try:
    result, profiler_output = asyncio.run(_run_async())
    with open("{result_path}", "wb") as f:
        cloudpickle.dump(("success", result, profiler_output), f)
except Exception as e:
    import traceback
    with open("{result_path}", "wb") as f:
        cloudpickle.dump(("error", str(e), traceback.format_exc()), f)
"""


def _run_in_subprocess_with_pyspy(fn, args, kwargs, output_format, rate=PYSPY_SAMPLE_RATE_HZ):
    """Run function in a clean subprocess and profile with py-spy.

    Uses a fresh Python interpreter so the call stack contains ONLY the user function.
    Function, args, and kwargs are pickled to the subprocess.
    """
    import sys

    # Create temp files for IPC
    args_fd, args_path = tempfile.mkstemp(suffix=".pkl")
    result_fd, result_path = tempfile.mkstemp(suffix=".pkl")
    ready_fd, ready_path = tempfile.mkstemp(suffix=".ready")

    os.close(args_fd)
    os.close(result_fd)
    os.close(ready_fd)
    os.remove(ready_path)  # Subprocess will create this to signal ready

    # Pickle function and args
    with open(args_path, "wb") as f:
        cloudpickle.dump((fn, args, kwargs), f)

    script = _SUBPROCESS_SCRIPT.format(
        args_path=args_path,
        ready_path=ready_path,
        result_path=result_path,
    )

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for ready signal
        timeout = 60
        start = time.time()
        while not os.path.exists(ready_path):
            time.sleep(0.01)
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                raise RuntimeError(f"Subprocess startup failed: {stderr}")
            if time.time() - start > timeout:
                proc.terminate()
                raise RuntimeError(f"Subprocess timed out after {timeout}s")

        pyspy_output = {"result": None}

        with pyspy_profiler(output=pyspy_output, output_format=output_format, pid=proc.pid, rate=rate):
            os.remove(ready_path)  # Signal to start
            proc.wait()

        if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"No result from subprocess: {stderr}")

        with open(result_path, "rb") as f:
            status, *rest = cloudpickle.load(f)

        if status == "error":
            raise RuntimeError(f"Function failed: {rest[0]}\n{rest[1]}")

        return rest[0], pyspy_output.get("result")

    finally:
        for path in [args_path, result_path, ready_path]:
            if os.path.exists(path):
                os.remove(path)


def _run_in_subprocess_with_torch(
    fn, args, kwargs, profiler: ProfilerConfig, callable_name: str, is_async: bool = False
):
    """Run function in a clean subprocess and profile with torch profiler.

    Uses a fresh Python interpreter so the call stack contains ONLY the user function.
    Function, args, kwargs, and profiler config are pickled to the subprocess.
    """
    import sys

    # Create temp files for IPC
    args_fd, args_path = tempfile.mkstemp(suffix=".pkl")
    result_fd, result_path = tempfile.mkstemp(suffix=".pkl")
    ready_fd, ready_path = tempfile.mkstemp(suffix=".ready")

    os.close(args_fd)
    os.close(result_fd)
    os.close(ready_fd)
    os.remove(ready_path)  # Subprocess will create this to signal ready

    # Prepare profiler config dict for subprocess
    profiler_config_dict = {
        "analyze_stack_traces": profiler.analyze_stack_traces,
        "output_format": profiler.output_format,
        "callable_name": callable_name or (fn.__name__ if hasattr(fn, "__name__") else "unknown"),
        "group_by_input_shape": profiler.group_by_input_shape,
        "group_by_stack_n": profiler.group_by_stack_n,
        "table_sort_by": profiler.table_sort_by,
        "consolidate_table": profiler.consolidate_table,
    }

    # Pickle function, args, and profiler config
    with open(args_path, "wb") as f:
        cloudpickle.dump((fn, args, kwargs, profiler_config_dict), f)

    # Use async script for async functions
    script_template = _TORCH_ASYNC_SUBPROCESS_SCRIPT if is_async else _TORCH_SUBPROCESS_SCRIPT
    script = script_template.format(
        args_path=args_path,
        ready_path=ready_path,
        result_path=result_path,
    )

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for ready signal
        timeout = 60
        start = time.time()
        while not os.path.exists(ready_path):
            time.sleep(0.01)
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode() if proc.stderr else ""
                raise RuntimeError(f"Subprocess startup failed: {stderr}")
            if time.time() - start > timeout:
                proc.terminate()
                raise RuntimeError(f"Subprocess timed out after {timeout}s")

        # Signal to start and wait for completion
        os.remove(ready_path)
        proc.wait()

        if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"No result from subprocess: {stderr}")

        with open(result_path, "rb") as f:
            status, *rest = cloudpickle.load(f)

        if status == "error":
            raise RuntimeError(f"Function failed: {rest[0]}\n{rest[1]}")

        # rest[0] is result, rest[1] is profiler_output
        return rest[0], rest[1]

    finally:
        for path in [args_path, result_path, ready_path]:
            if os.path.exists(path):
                os.remove(path)


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

            pyspy_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

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

            # Log any stderr from py-spy
            if pyspy_process.stderr:
                stderr_output = pyspy_process.stderr.read()
                if stderr_output:
                    logger.warning(f"py-spy stderr: {stderr_output}")

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


def _run_pytorch_profile_impl(fn, args, kwargs, profiler: ProfilerConfig, callable_name: str):
    """Shared implementation for PyTorch profiling. Returns (prof, result)."""
    try:
        import torch
        from torch.profiler import profile, ProfilerActivity, record_function
    except ImportError:
        raise ImportError(
            "PyTorch profiler requires torch to be installed, but it is not available in this environment. "
            "Please install torch in your image."
        )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    if profiler.analyze_stack_traces:
        prof = profile(
            activities=activities,
            profile_memory=True,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )
    else:
        prof = profile(activities=activities, profile_memory=True)

    return prof, record_function, callable_name or (fn.__name__ if hasattr(fn, "__name__") else "unknown")


def _export_pytorch_profile(prof, output_format: str):
    """Export PyTorch profiler results to the specified format."""
    if output_format == "chrometrace":
        with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            with open(tmp.name, "rb") as f:
                return json.loads(f.read())
    return None


async def run_pytorch_profile_async(
    fn,
    *args,
    profiler: ProfilerConfig = None,
    callable_name: str = None,
    **kwargs,
):
    """Run an async function with PyTorch profiling in a clean subprocess.

    Uses subprocess isolation so the call stack contains ONLY the user function.
    """
    logger.debug(f"Running async '{callable_name}' with torch profiler")

    # Run in subprocess with is_async=True - the subprocess handles asyncio.run()
    result, profiler_output = _run_in_subprocess_with_torch(fn, args, kwargs, profiler, callable_name, is_async=True)

    if profiler_output is None:
        logger.warning("torch profiler captured no output.")

    return result, profiler_output


def run_with_profile(
    fn,
    *args,
    profiler: ProfilerConfig = None,
    callable_name: str = None,
    **kwargs,
):
    """Run the function with optional profiling."""

    # Skip profiling if disabled (e.g., invalid profiler type)
    if profiler._disabled:
        result = fn(*args, **kwargs)
        return result, None

    profiler_type = profiler.profiler_type
    output_format = profiler.output_format

    if profiler_type == "pytorch":
        # Run in clean subprocess so profiling shows ONLY user code
        logger.debug(f"Running '{callable_name}' with torch profiler")

        result, profiler_output = _run_in_subprocess_with_torch(fn, args, kwargs, profiler, callable_name)

        if profiler_output is None:
            logger.warning("torch profiler captured no output.")

        return result, profiler_output

    elif profiler_type == "pyspy":
        # Run in clean subprocess so flamegraph shows ONLY user code
        logger.debug(f"Running '{callable_name}' with py-spy profiler")

        result, profiler_output = _run_in_subprocess_with_pyspy(fn, args, kwargs, output_format=output_format)

        if profiler_output is None:
            logger.warning("pyspy profiler captured no output.")

        return result, profiler_output

    else:
        logger.warning(f"Unsupported profiler type {profiler}, running without profiling")
        result = fn(*args, **kwargs)
        return result, None


def generate_profiler_output_filename(filename: str, file_suffix: str, service_name: str, request_id: str) -> str:
    if filename:
        if filename.endswith(file_suffix):
            return filename
        return f"{filename}{file_suffix}"
    else:
        # Note: add request id to prevent collisions (The running ts will be a part of the file metadata)
        return f"{service_name}_{request_id}{file_suffix}"


def _is_valid_profiler_output(profiler_output: str, output_format: str) -> bool:
    """Check if the profiler output is valid and contains actual profiling data."""
    if not profiler_output:
        return False

    # Check for py-spy error messages in SVG output
    if output_format == "flamegraph" and "ERROR:" in profiler_output:
        return False

    # Check for empty/minimal SVG (no actual flame data)
    if output_format == "flamegraph" and 'height="60"' in profiler_output:
        # Height of 60 typically means no frames were captured
        return False

    return True


def parse_profiler_output_helper(
    single_call_output: dict,
    profiler: ProfilerConfig,
    service_name: str,
    request_id: str,
    file_name_suffix: str = None,
):
    profiler_output = single_call_output.pop("profiler_output", None)
    fn_output = single_call_output.pop("fn_output")

    if not profiler_output:
        logger.warning(f"No profiling information found for service '{service_name}'.")
        return fn_output, None

    # Validate the profiler output before saving
    if not _is_valid_profiler_output(profiler_output, profiler.output_format):
        logger.warning(
            f"Profiler captured no valid samples for service '{service_name}'. "
            "This usually means the function completed too quickly or was mostly sleeping/waiting. "
            "For meaningful profiles, ensure the function does CPU-bound work."
        )
        return fn_output, None

    # For table output format, return the profiler output directly instead of saving to file
    if profiler.output_format == "table":
        return fn_output, profiler_output

    profiler_output_path = profiler.output_path
    profiler_output_filename = profiler.output_filename
    profiler_output_suffix = profiler.output_file_suffix()

    if not profiler_output_path:
        profiler_output_path = str(Path.cwd())

    file_suffix = f"_{file_name_suffix}.{profiler_output_suffix}" if file_name_suffix else f".{profiler_output_suffix}"
    profiler_output_filename = generate_profiler_output_filename(
        filename=profiler_output_filename,
        file_suffix=file_suffix,
        service_name=service_name,
        request_id=request_id,
    )

    output_full_path = Path(profiler_output_path) / Path(profiler_output_filename)

    with open(output_full_path, "w+") as output_file:
        if isinstance(profiler_output, str):
            output_file.write(profiler_output)
        else:
            output_file.write(json.dumps(profiler_output))
        logger.info(f"Profiler output saved to: {output_full_path}")

    return fn_output, None


def parse_profiler_output(
    call_output: Union[dict, List[dict]],
    profiler: ProfilerConfig,
    service_name: str,
    request_id: str,
    distribution_type: str = None,
):
    filename_suffix = distribution_type if distribution_type else None
    if isinstance(call_output, list):
        fn_outputs = []
        profiler_outputs = []  # in case profiler output is a table
        output_index = 0
        for output in call_output:
            indexed_suffix = f"{filename_suffix}_{output_index}" if filename_suffix else str(output_index)
            fn_output, profiler_output = parse_profiler_output_helper(
                single_call_output=output,
                profiler=profiler,
                service_name=service_name,
                request_id=request_id,
                file_name_suffix=indexed_suffix,
            )
            output_index += 1
            fn_outputs.append(fn_output)

            if profiler.output_format == "table":
                profiler_outputs.append(profiler_output)

        if profiler.output_format == "table":
            if profiler.consolidate_table:
                # consolidate the distributed tables into one table
                profiler_outputs = "\n\n".join(profiler_outputs)
                return fn_outputs, profiler_outputs

            return fn_outputs, profiler_outputs

        return fn_outputs

    fn_output, profiler_output = parse_profiler_output_helper(
        single_call_output=call_output,
        profiler=profiler,
        service_name=service_name,
        request_id=request_id,
        file_name_suffix=filename_suffix,
    )

    # For table output format, return both function output and profiler output
    if profiler.output_format == "table":
        return fn_output, profiler_output

    return fn_output
