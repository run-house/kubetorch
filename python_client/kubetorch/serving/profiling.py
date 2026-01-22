import base64
import contextlib
import json
import logging
import os

import pickle
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Union

from kubetorch.constants import PYSPY_SAMPLE_RATE_HZ
from kubetorch.globals import ProfilerConfig

logger = logging.getLogger(__name__)

# Optimized script template for py-spy profiling (uses stdin/stdout/stderr).
# Reads raw data from stdin (pickle or JSON based on env var), metadata from env vars.
PYSPY_EXEC_SCRIPT_TEMPLATE = """
import sys
import os
import json
import pickle
import time

# Read raw data from stdin (format determined by env var)
serialization = os.environ.get('KT_SERIALIZATION', 'pickle')
raw_data = sys.stdin.buffer.read()
if serialization == 'pickle':
    params = pickle.loads(raw_data)
else:
    params = json.loads(raw_data.decode('utf-8'))
args = params.get('args', [])
kwargs = params.get('kwargs', {})

# Read metadata from environment variables (no pickling needed)
method_name = os.environ.get('KT_PROFILE_METHOD_NAME') or None
is_async = os.environ.get('KT_PROFILE_IS_ASYNC', 'false').lower() == 'true'

# Load callable using the same mechanism as the main process (via environment variables)
from kubetorch.serving.http_server import load_callable
callable_obj = load_callable(distributed_subprocess=True)

# Get the user method
if method_name:
    fn = getattr(callable_obj, method_name)
else:
    fn = callable_obj

# Signal ready via stderr - now that imports are warmed up
sys.stderr.write('ready\\n')
sys.stderr.flush()

# Small delay to ensure py-spy has attached before we start the actual work
time.sleep(0.1)

try:
    # Execute function at module level - appears at top of flamegraph
    if is_async:
        import asyncio
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _result = _loop.run_until_complete(fn(*args, **kwargs))
        _loop.close()
    else:
        _result = fn(*args, **kwargs)

    # Give py-spy time to capture profile
    time.sleep(0.5)

    # Write result to stdout
    pickle.dump(('success', _result), sys.stdout.buffer)
    sys.stdout.buffer.flush()
except Exception as _e:
    import traceback
    pickle.dump(('error', str(_e), traceback.format_exc()), sys.stdout.buffer)
    sys.stdout.buffer.flush()
"""

# Optimized script template for torch profiling.
# Runs at module level for clean stack traces without threading overhead.
# Reads raw data from stdin (pickle or JSON based on env var), metadata from env vars.
TORCH_EXEC_SCRIPT_TEMPLATE = """
import sys
import os
import json
import pickle

# Read raw data from stdin (format determined by env var)
serialization = os.environ.get('KT_SERIALIZATION', 'pickle')
raw_data = sys.stdin.buffer.read()
if serialization == 'pickle':
    params = pickle.loads(raw_data)
else:
    params = json.loads(raw_data.decode('utf-8'))
args = params.get('args', [])
kwargs = params.get('kwargs', {})

# Read metadata from environment variables (no pickling needed)
method_name = os.environ.get('KT_PROFILE_METHOD_NAME') or None
is_async = os.environ.get('KT_PROFILE_IS_ASYNC', 'false').lower() == 'true'
callable_name = os.environ.get('KT_PROFILE_CALLABLE_NAME', 'unknown')
profiler_config_json = os.environ.get('KT_PROFILE_CONFIG', '{}')
profiler_config = json.loads(profiler_config_json)

# Load callable using the same mechanism as the main process (via environment variables)
from kubetorch.serving.http_server import load_callable
callable_obj = load_callable(distributed_subprocess=True)

# Get the user method
if method_name:
    fn = getattr(callable_obj, method_name)
else:
    fn = callable_obj

try:
    import torch
    from torch.profiler import profile, ProfilerActivity, record_function

    # Setup profiler at module level (clean stack trace)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    common_kwargs = {
        "activities": activities,
        "profile_memory": True,
        "record_shapes": True,
    }

    analyze_stack_traces = profiler_config.get('analyze_stack_traces', False)
    output_format = profiler_config.get('output_format', 'chrometrace')

    if analyze_stack_traces:
        prof = profile(
            **common_kwargs,
            with_stack=True,
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        )
    else:
        prof = profile(**common_kwargs)

    # Execute function at module level with profiler - appears at top of flamegraph
    if analyze_stack_traces:
        with prof:
            if is_async:
                import asyncio
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                _result = _loop.run_until_complete(fn(*args, **kwargs))
                _loop.close()
            else:
                _result = fn(*args, **kwargs)
    else:
        with prof:
            with record_function(callable_name):
                if is_async:
                    import asyncio
                    _loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(_loop)
                    _result = _loop.run_until_complete(fn(*args, **kwargs))
                    _loop.close()
                else:
                    _result = fn(*args, **kwargs)

    # Export profiler output inline (avoid import overhead in stack)
    profiler_output = None
    if output_format == "chrometrace":
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp:
            prof.export_chrome_trace(tmp.name)
            with open(tmp.name, "rb") as f:
                profiler_output = json.loads(f.read())
    elif output_format == "stacks":
        group_by_input_shape = profiler_config.get('group_by_input_shape', False)
        group_by_stack_n = profiler_config.get('group_by_stack_n', 0)
        stacks_output = prof.key_averages(
            group_by_input_shape=group_by_input_shape, group_by_stack_n=group_by_stack_n
        ).table(sort_by="self_cpu_time_total", row_limit=100)
        profiler_output = {"format": "stacks", "content": stacks_output}
    elif output_format == "table":
        group_by_input_shape = profiler_config.get('group_by_input_shape', False)
        group_by_stack_n = profiler_config.get('group_by_stack_n', 0)
        table_sort_by = profiler_config.get('table_sort_by') or "cpu_time_total"
        table_output = prof.key_averages(
            group_by_input_shape=group_by_input_shape, group_by_stack_n=group_by_stack_n
        ).table(sort_by=table_sort_by, row_limit=-1)
        profiler_output = table_output

    # Write result to stdout (only pickle the result, not args)
    pickle.dump(('success', _result, profiler_output), sys.stdout.buffer)
    sys.stdout.buffer.flush()

except Exception as _e:
    import traceback
    pickle.dump(('error', str(_e), traceback.format_exc()), sys.stdout.buffer)
    sys.stdout.buffer.flush()
"""


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
        raise


async def run_with_torch_profiler(
    stdin_data, serialization, method_name, is_async, profiler_config: ProfilerConfig, running_loop
):
    # Torch profiler - run in subprocess for clean stack traces (no threading overhead)
    # Pass raw data via stdin (pickle or JSON), metadata via env vars

    callable_name = (
        f"{os.environ['KT_CLS_OR_FN_NAME']}.{method_name}" if method_name else os.environ["KT_CLS_OR_FN_NAME"]
    )
    proc = None

    try:
        # Base64 encode the script to avoid shell escaping issues with -c
        script_bytes = TORCH_EXEC_SCRIPT_TEMPLATE.encode("utf-8")
        script_b64 = base64.b64encode(script_bytes).decode("ascii")

        # Prepare environment with metadata
        env = os.environ.copy()
        env["KT_SERIALIZATION"] = serialization
        env["KT_PROFILE_METHOD_NAME"] = method_name or ""
        env["KT_PROFILE_IS_ASYNC"] = "true" if is_async else "false"
        env["KT_PROFILE_CALLABLE_NAME"] = callable_name
        env["KT_PROFILE_CONFIG"] = json.dumps(
            {
                "analyze_stack_traces": profiler_config.analyze_stack_traces,
                "output_format": profiler_config.output_format,
                "group_by_input_shape": profiler_config.group_by_input_shape,
                "group_by_stack_n": profiler_config.group_by_stack_n,
                "table_sort_by": profiler_config.table_sort_by,
            }
        )

        # Start subprocess with -c flag
        proc = subprocess.Popen(
            [sys.executable, "-c", f"import base64;exec(base64.b64decode('{script_b64}'))"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Write raw data to stdin and close it
        proc.stdin.write(stdin_data)
        proc.stdin.close()

        # Wait for result (run in executor to not block event loop)
        def wait_and_get_result():
            stdout = proc.stdout.read()
            proc.wait()
            if proc.returncode != 0:
                stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
                stdout_text = stdout.decode("utf-8", errors="replace") if stdout else "No stdout"
                raise RuntimeError(
                    f"Subprocess failed with return code {proc.returncode}:\n"
                    f"STDOUT: {stdout_text}\nSTDERR: {stderr_text}"
                )
            if not stdout:
                raise RuntimeError("Subprocess completed but produced no output")
            # Parse pickled result from stdout (contains fn_output and profiler_output)
            status, *result_data = pickle.loads(stdout)
            if status == "error":
                exc_str, tb = result_data
                raise RuntimeError(f"Function failed in subprocess: {exc_str}\n{tb}")
            return result_data[0], result_data[1]  # fn_output, profiler_output

        fn_output, profiler_output = await running_loop.run_in_executor(None, wait_and_get_result)
        result = {"fn_output": fn_output, "profiler_output": profiler_output}
        return result

    except Exception:
        # Cleanup subprocess if still running
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        raise


async def run_with_pyspy_profiler(
    stdin_data, serialization, method_name, is_async, profiler_config: ProfilerConfig, running_loop
):
    # py-spy profiler - run in subprocess for clean stack traces
    # Pass raw data via stdin (pickle or JSON), metadata via env vars

    output_format = profiler_config.output_format
    pyspy_output = {"result": None}
    proc = None

    try:
        # Base64 encode the script to avoid shell escaping issues with -c
        script_bytes = PYSPY_EXEC_SCRIPT_TEMPLATE.encode("utf-8")
        script_b64 = base64.b64encode(script_bytes).decode("ascii")

        # Prepare environment with metadata
        env = os.environ.copy()
        env["KT_SERIALIZATION"] = serialization
        env["KT_PROFILE_METHOD_NAME"] = method_name or ""
        env["KT_PROFILE_IS_ASYNC"] = "true" if is_async else "false"

        # Start subprocess with -c flag, reading script from base64
        proc = subprocess.Popen(
            [sys.executable, "-c", f"import base64;exec(base64.b64decode('{script_b64}'))"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Write raw data to stdin and close it
        proc.stdin.write(stdin_data)
        proc.stdin.close()

        # Wait for ready signal on stderr (run in executor to not block event loop)
        def wait_for_ready(timeout=5.0):
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Read one line from stderr (non-blocking check)
                line = proc.stderr.readline()
                if line:
                    if line.strip() == b"ready":
                        return True
                # Check if process died
                if proc.poll() is not None:
                    return False
                time.sleep(0.01)
            return False

        ready = await running_loop.run_in_executor(None, wait_for_ready, 5.0)
        if not ready:
            raise RuntimeError("Subprocess did not become ready for profiling in time")

        # Profile the subprocess and get result from stdout
        def wait_and_get_result():
            # Don't use communicate() since stdin is already closed
            # Read stdout directly and wait for process
            stdout = proc.stdout.read()
            proc.wait()
            if proc.returncode != 0:
                stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
                stdout_text = stdout.decode("utf-8", errors="replace") if stdout else "No stdout"
                raise RuntimeError(
                    f"Subprocess failed with return code {proc.returncode}:\n"
                    f"STDOUT: {stdout_text}\nSTDERR: {stderr_text}"
                )
            if not stdout:
                raise RuntimeError("Subprocess completed but produced no output")
            # Parse pickled result from stdout
            status, *result_data = pickle.loads(stdout)
            if status == "error":
                exc_str, tb = result_data
                raise RuntimeError(f"Function failed in subprocess: {exc_str}\n{tb}")
            return result_data[0]

        with pyspy_profiler(output=pyspy_output, output_format=output_format, pid=proc.pid):
            fn_output = await running_loop.run_in_executor(None, wait_and_get_result)

        profiler_output = pyspy_output.get("result")
        result = {"fn_output": fn_output, "profiler_output": profiler_output}
        return result

    except Exception:
        # Cleanup subprocess if still running
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
        raise


def generate_profiler_output_filename(filename: str, file_suffix: str, service_name: str, request_id: str) -> str:
    if filename:
        if filename.endswith(file_suffix):
            return filename
        return f"{filename}{file_suffix}"
    else:
        # Note: add request id to prevent collisions (The running ts will be a part of the file metadata)
        return f"{service_name}_{request_id}{file_suffix}"


def is_valid_profiler_output(profiler_output: str, output_format: str) -> bool:
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
    if not is_valid_profiler_output(profiler_output, profiler.output_format):
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
