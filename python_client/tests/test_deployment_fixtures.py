import asyncio
import os

# Mimic CI for this test suite even locally, to ensure that
# resources are created with the branch name prefix
os.environ["CI"] = "true"

import pytest

from pydantic import BaseModel

from .utils import SlowNumpyArray, summer, TestModel


class OSInfoRequest(BaseModel):
    method: str


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "150"
    # Only set TEST_COMPUTE_TYPE if it's not already set (to allow override from environment)
    if "TEST_COMPUTE_TYPE" not in os.environ:
        os.environ["TEST_COMPUTE_TYPE"] = "deployment"
    yield


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_fn_sync_reload_by_name_only(remote_fn):
    import kubetorch as kt

    # Note: set to `get_if_exists` to `False` in order to stop any fallback and only look for an exact name match
    service_name = remote_fn.service_name
    some_reloaded_fn = kt.fn(summer, name=service_name, get_if_exists=True)
    assert some_reloaded_fn(2, 2) == 4

    another_reloaded_fn = kt.fn(name=service_name, get_if_exists=True)
    assert another_reloaded_fn(2, 3) == 5

    # Uncomment to manually test pdb
    # remote_fn(1, 2, pdb=True)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_fn_sync_reload_by_name_with_prefixes(remote_fn):
    import kubetorch as kt
    from kubetorch.utils import current_git_branch, validate_username

    # Note: bc this is running in CI, the service name will have a git branch prefix
    current_branch = current_git_branch()
    branch_prefix = validate_username(current_branch)
    # Note: username prefix required for CI tests where TEST_HASH is set in job yaml
    username_prefix = kt.config.username

    another_reloaded_fn = kt.fn(name=remote_fn.name, reload_prefixes=[branch_prefix, username_prefix])
    assert another_reloaded_fn(2, 3) == 5

    # Note: Different way to load function and using a list of prefixes
    compute_type = os.getenv("TEST_COMPUTE_TYPE")
    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    compute_prefix = f"{username_prefix}-{service_name_prefix}"
    reloaded_fn = kt.fn(summer, reload_prefixes=[branch_prefix, username_prefix, compute_prefix])
    assert reloaded_fn(2, 2) == 4


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_send_fn_to_compute_with_reload(remote_fn):
    import kubetorch as kt

    service_name = remote_fn.service_name
    another_reloaded_fn = kt.fn(summer, name=service_name).to(
        remote_fn.compute,
        get_if_exists=True,
    )
    assert another_reloaded_fn(1, 2, use_tqdm=True) == 3


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_skip_fn_deploy_if_already_exists(remote_fn):
    import kubetorch as kt

    name = remote_fn.name
    reloaded_fn = kt.fn(summer, name=name).to(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
        ),
        get_if_exists=True,
    )
    assert reloaded_fn.service_name == remote_fn.service_name
    assert reloaded_fn.compute.pods() == remote_fn.compute.pods()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_failure_to_reload_non_existent_fn(remote_fn):
    import kubetorch as kt

    with pytest.raises(ValueError):
        kt.fn(name="some-random-fn-name")

    with pytest.raises(ValueError):
        kt.fn(name=remote_fn.name, reload_prefixes=["fake-tag"])

    # Note: get_if_exists=False is not allowed without function object
    with pytest.raises(ValueError):
        kt.fn(name=remote_fn.name, get_if_exists=False)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_serialization_formats(remote_fn, remote_cls):
    """Test that both JSON and pickle serialization work correctly."""
    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    if compute_type == "ray":
        pytest.skip("Skipping serialization tests for Ray compute type")

    # Test function serialization
    print("Testing function serialization...")

    # Test pickle serialization
    result_pickle = remote_fn(2, 3, serialization="pickle")
    assert result_pickle == 5

    # Test pickle serialization with Pydantic model
    result_pickle = remote_fn(
        TestModel(name="test_a", value=42),
        TestModel(name="test_b", value=42),
        serialization="pickle",
    )

    assert isinstance(result_pickle, TestModel)
    assert result_pickle.name == "sum_result"
    assert result_pickle.value == 84

    # Test class method serialization
    print("Testing class method serialization...")

    # Test pickle serialization
    cpu_count_pickle = remote_cls.cpu_count(serialization="pickle")
    assert isinstance(cpu_count_pickle, int)

    requests = [
        OSInfoRequest(method="uname"),
        OSInfoRequest(method="cpu_count"),
        OSInfoRequest(method="getpid"),
    ]
    result_pickle = remote_cls.os_info(requests, serialization="pickle")
    assert isinstance(result_pickle, list)
    assert all(isinstance(r, BaseModel) for r in result_pickle)
    assert all(r.name in ["uname", "cpu_count", "getpid"] for r in result_pickle)
    assert result_pickle[1].value == str(cpu_count_pickle)

    # Test module-level serialization setting
    print("Testing module-level serialization setting...")

    try:
        # Set remote_fn to use pickle by default
        original_serialization = remote_fn.serialization
        remote_fn.serialization = "pickle"

        # Now calls should use pickle by default
        result_pickle = remote_fn(TestModel(name="test_a", value=42), TestModel(name="test_b", value=42))
        assert isinstance(result_pickle, TestModel)
        assert result_pickle.name == "sum_result"
        assert result_pickle.value == 84

        # Can still override for individual calls
        result_override = remote_fn(3, 4, serialization="json")
        assert result_override == 7
    except Exception as e:
        raise e
    finally:
        remote_fn.serialization = original_serialization

    try:
        # Test class serialization setting
        original_serialization = remote_cls.serialization
        remote_cls.serialization = "pickle"

        # Test that class methods now use pickle by default
        requests = [OSInfoRequest(method="cpu_count")]
        result_pickle = remote_cls.os_info(requests, serialization="pickle")
        assert isinstance(result_pickle, list)
        assert isinstance(result_pickle[0], BaseModel)
        assert result_pickle[0].name == "cpu_count"
        assert result_pickle[0].value == str(cpu_count_pickle)

        # Can still override for individual calls
        result_override = remote_cls.size_minus_cpus(serialization="json")
        assert isinstance(result_override, int)
    except Exception as e:
        raise e
    finally:
        remote_cls.serialization = original_serialization


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_sync_basic(remote_cls):
    remote_cpu_count = remote_cls.cpu_count()
    assert remote_cls.print_and_log(1) == "Hello from the cluster! [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]"
    assert remote_cls.size_minus_cpus() == 10 - remote_cpu_count


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_sync_reload_by_name_only(remote_cls):
    import kubetorch as kt

    service_name = remote_cls.service_name
    reloaded_cls = kt.cls(SlowNumpyArray, name=service_name, get_if_exists=True)
    assert reloaded_cls.cpu_count() == remote_cls.cpu_count()

    another_reloaded_cls = kt.cls(name=service_name, get_if_exists=True)
    assert another_reloaded_cls.cpu_count() == reloaded_cls.cpu_count()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_sync_reload_by_name_with_prefixes(remote_cls):
    import kubetorch as kt
    from kubetorch.utils import current_git_branch, validate_username

    # Note: service name will have the git branch prefix
    current_branch = current_git_branch()
    branch_prefix = validate_username(current_branch)
    # Note: username prefix required for CI tests where TEST_HASH is set in job yaml
    username_prefix = kt.config.username

    reloaded_cls = kt.cls(
        SlowNumpyArray,
        name=remote_cls.name,
        reload_prefixes=[branch_prefix, username_prefix],
    )
    assert reloaded_cls.cpu_count() == remote_cls.cpu_count()

    compute_type = os.getenv("TEST_COMPUTE_TYPE")
    compute_prefix = f"{username_prefix}-{compute_type}"
    another_reloaded_cls = kt.cls(
        name=remote_cls.name,
        reload_prefixes=[branch_prefix, username_prefix, compute_prefix],
    )
    assert another_reloaded_cls.cpu_count() == remote_cls.cpu_count()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_send_cls_to_compute_with_reload(remote_cls):
    import kubetorch as kt

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    if compute_type == "ray":
        pytest.skip("Skipping cls reload tests for Ray compute type")

    service_name = remote_cls.service_name
    reloaded_cls = kt.cls(SlowNumpyArray, name=service_name).to(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
        ),
        get_if_exists=True,
    )
    assert reloaded_cls.cpu_count() == remote_cls.cpu_count()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_skip_cls_deploy_if_already_exists(remote_cls):
    import kubetorch as kt

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    if compute_type == "ray":
        pytest.skip("Skipping if already exists tests for Ray compute type")

    name = remote_cls.name
    reloaded_cls = kt.cls(SlowNumpyArray, name=name).to(
        kt.Compute(
            cpus=".01",
            gpu_anti_affinity=True,
            launch_timeout=300,
        ),
        get_if_exists=True,
    )
    assert reloaded_cls.service_name == remote_cls.service_name
    assert reloaded_cls.compute.pods() == remote_cls.compute.pods()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_fn_async_call(remote_fn):
    async def assert_async_call(fn, *args, **kwargs):
        coroutine = fn(*args, **kwargs)
        assert asyncio.iscoroutine(coroutine)
        result = await coroutine
        assert result == 3

    # use async_ flag in method call
    await assert_async_call(remote_fn, 1, 2, async_=True)

    # use async_ fn property
    try:
        remote_fn.async_ = True
        await assert_async_call(remote_fn, 1, 2)
    finally:
        remote_fn.async_ = False


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cls_async_call(remote_cls):
    async def assert_async_call(fn, *args, **kwargs):
        coroutine = fn(*args, **kwargs)
        assert asyncio.iscoroutine(coroutine)
        result = await coroutine
        assert isinstance(result, int)

    # use async_ flag in method call
    await assert_async_call(remote_cls.cpu_count, async_=True)

    # use async_ cls property
    try:
        remote_cls.async_ = True
        await assert_async_call(remote_cls.cpu_count)
    finally:
        remote_cls.async_ = False


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_async_fn_call(remote_async_fn):
    # basic async function call
    coroutine = remote_async_fn(1, 2)
    assert asyncio.iscoroutine(coroutine)
    result = await coroutine
    assert result == 3

    # run multiple async function calls and assert that all start times are less than the earliest end time
    async def run_tasks():
        tasks = []
        for i in range(3):
            tasks.append(remote_async_fn(1, 2, return_times=True))
        results = await asyncio.gather(*tasks)
        return results

    results = await run_tasks()

    start_times = [r[0] for r in results]
    end_times = [r[1] for r in results]
    assert max(start_times) < min(end_times)


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_async_to():
    import kubetorch as kt

    from .conftest import get_compute
    from .utils import summer

    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    compute = get_compute(compute_type)

    service_name_prefix = os.getenv("SERVICE_NAME_PREFIX", compute_type)
    name = f"{service_name_prefix}-summer-async-to"
    fn = await kt.fn(summer, name=name).to_async(compute)

    # check running function normally
    assert fn(1, 2) == 3

    # check running function in async mode
    coroutine = fn(1, 2, async_=True)
    assert asyncio.iscoroutine(coroutine)
    result = await coroutine
    assert result == 3


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_debug_modes(remote_cls, remote_fn, capsys):
    """Test various debug modes and verify debug server cleanup.

    This test verifies that:
    1. The PDB WebSocket server starts and accepts connections via kt debug locally
    2. PDB prompt is sent to the client and stepping works
    3. Client can send commands ('c' to continue) and the method completes
    4. Server cleanup works correctly after debugging
    5. In-cluster WebSocket connection works via kubectl exec
    """
    import asyncio
    import re
    import subprocess

    import kubetorch as kt

    # Get test pod info for running WebSocket tests from inside the cluster
    test_pod = remote_fn.compute.pod_names()[0]
    test_namespace = remote_fn.compute.namespace

    # Test 1: Call method with breakpoint() inside, connect with kt debug locally
    print("\n=== Test 1: Method with breakpoint() - local `kt debug` ===")

    # Start the call in a background task (it will block at breakpoint)
    async def call_breakpoint_method():
        return await remote_cls.method_with_breakpoint(async_=True, stream_logs=True)

    task = asyncio.create_task(call_breakpoint_method())

    # Give it time to hit the breakpoint and print the kt debug command
    await asyncio.sleep(6)

    # Capture the output to get the kt debug command
    captured = capsys.readouterr()
    output = captured.out + captured.err
    print(f"Captured output:\n{output}")

    # Verify the kt debug command is printed
    assert "kt debug " in output, "kt debug command should be printed"
    assert "--mode pdb" in output, "Should use pdb mode by default"
    assert "--pod-ip" in output, "Should include pod IP for in-cluster usage"

    # Extract the full kt debug command
    kt_debug_match = re.search(
        r"kt debug ([\w-]+) --port (\d+) --namespace ([\w-]+) --mode pdb(?:\s+--pod-ip ([\d.]+))?", output
    )
    assert kt_debug_match, f"Should find complete kt debug command in output: {output}"
    debug_pod = kt_debug_match.group(1)
    debug_port = int(kt_debug_match.group(2))
    debug_namespace = kt_debug_match.group(3)
    pod_ip = kt_debug_match.group(4)
    print(f"Found: pod={debug_pod}, port={debug_port}, namespace={debug_namespace}, pod_ip={pod_ip}")

    # Run kt debug locally as a subprocess - send 'c\n' to continue after getting prompt
    kt_debug_cmd = [
        "kt",
        "debug",
        debug_pod,
        "--port",
        str(debug_port),
        "--namespace",
        debug_namespace,
        "--mode",
        "pdb",
    ]
    print(f"Running: {' '.join(kt_debug_cmd)}")

    import fcntl
    import os

    proc = subprocess.Popen(
        kt_debug_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Use binary mode for non-blocking I/O compatibility
    )

    # Set stdout to non-blocking mode
    fd = proc.stdout.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    # Read output until we see PDB prompt, then send 'c' to continue
    import time

    output_buffer = b""
    start_time = time.time()
    timeout = 30
    got_prompt = False

    while time.time() - start_time < timeout:
        try:
            chunk = proc.stdout.read()
            if chunk:
                output_buffer += chunk
                print(f"kt debug output: {repr(chunk)}")
                if b"(Pdb)" in output_buffer and not got_prompt:
                    got_prompt = True
                    print("Got PDB prompt, sending 'c' to continue...")
                    proc.stdin.write(b"c\n")
                    proc.stdin.flush()
                    # Give it a moment to process and exit
                    await asyncio.sleep(1)
                    break
        except BlockingIOError:
            # No data available yet
            pass

        if proc.poll() is not None:
            # Process exited, read any remaining output
            try:
                chunk = proc.stdout.read()
                if chunk:
                    output_buffer += chunk
                    print(f"kt debug final output: {repr(chunk)}")
            except BlockingIOError:
                pass
            break

        await asyncio.sleep(0.1)

    # Clean up subprocess
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)

    stderr_output = b""
    try:
        stderr_output = proc.stderr.read() or b""
    except BlockingIOError:
        pass
    print(f"stderr: {stderr_output.decode('utf-8', errors='replace')}")

    output_str = output_buffer.decode("utf-8", errors="replace")
    assert got_prompt, f"Should have received PDB prompt. Output: {output_str}"

    # Wait for the task to complete (it should have continued after we sent 'c')
    try:
        result = await asyncio.wait_for(task, timeout=10.0)
        print(f"Method returned: {result}")
        assert "Breakpoint method executed" in result, f"Unexpected result: {result}"
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise AssertionError("Method did not complete after sending 'c' command")

    print("Test 1 PASSED: PDB works correctly with local kt debug!")

    # Verify server cleanup: make a normal call to ensure service isn't locked up
    print("\n=== Verifying server cleanup after Test 1 ===")
    await remote_cls.to_async(remote_cls.compute)
    result = remote_cls.print_and_log(0)
    assert "Hello from the cluster!" in result
    print("Server cleanup verified - service is responsive!")

    # Test 1b: Call method with breakpoint() again, connect from inside cluster via kubectl exec
    print("\n=== Test 1b: Method with breakpoint() - in-cluster kt debug ===")

    task1b = asyncio.create_task(call_breakpoint_method())

    # Give it time to hit the breakpoint
    await asyncio.sleep(3)

    # Capture the output to get the full kt debug command
    captured = capsys.readouterr()
    output = captured.out + captured.err
    print(f"Captured output:\n{output}")

    # Extract the full kt debug command including --pod-ip
    kt_debug_match = re.search(
        r"kt debug ([\w-]+) --port (\d+) --namespace ([\w-]+) --mode pdb --pod-ip ([\d.]+)", output
    )
    assert kt_debug_match, f"Should find complete kt debug command with pod-ip in output: {output}"
    debug_pod = kt_debug_match.group(1)
    debug_port = kt_debug_match.group(2)
    debug_namespace = kt_debug_match.group(3)
    pod_ip = kt_debug_match.group(4)
    print(f"Found: pod={debug_pod}, port={debug_port}, namespace={debug_namespace}, pod_ip={pod_ip}")

    # Run kt debug from inside the test pod, piping 'c' to continue
    # Use --pod-ip since we're connecting from inside the cluster
    kt_debug_cmd = f"echo 'c' | kt debug {debug_pod} --port {debug_port} --namespace {debug_namespace} --mode pdb --pod-ip {pod_ip}"
    exec_cmd = ["kubectl", "exec", "-n", test_namespace, test_pod, "--", "bash", "-c", kt_debug_cmd]
    print(f"Running from pod {test_pod}: {kt_debug_cmd}")

    proc = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=30)
    print(f"kt debug stdout: {proc.stdout}")
    print(f"kt debug stderr: {proc.stderr}")

    # Verify we got the PDB prompt (indicates successful connection)
    assert (
        "(Pdb)" in proc.stdout or "Connected" in proc.stdout
    ), f"Should see PDB prompt or connection message: {proc.stdout}"

    # Wait for the task to complete
    try:
        result = await asyncio.wait_for(task1b, timeout=10.0)
        print(f"Method returned: {result}")
        assert "Breakpoint method executed" in result, f"Unexpected result: {result}"
    except asyncio.TimeoutError:
        task1b.cancel()
        try:
            await task1b
        except asyncio.CancelledError:
            pass
        raise AssertionError("Method did not complete after sending 'c' command")

    print("Test 1b PASSED: PDB works correctly with in-cluster kt debug!")

    # Verify server cleanup again
    print("\n=== Verifying server cleanup after Test 1b ===")
    await remote_cls.to_async(remote_cls.compute)
    result = remote_cls.print_and_log(1)
    assert "Hello from the cluster!" in result
    print("Server cleanup verified - service is responsive!")

    # Test 2: Call existing method with debug=True
    print("\n=== Test 2: Method call with debug=True ===")

    async def call_with_debug_true():
        return await remote_cls.print_and_log(2, debug=True, async_=True, stream_logs=True)

    task2 = asyncio.create_task(call_with_debug_true())

    # Give it time to hit the breakpoint
    await asyncio.sleep(3)

    # Capture the output
    captured = capsys.readouterr()
    output = captured.out + captured.err
    print(f"Captured output:\n{output}")

    # Verify the kt debug command is printed
    assert "kt debug" in output, "kt debug command should be printed for debug=True"
    assert "--mode pdb" in output, "Should use pdb mode by default"

    # Extract the full kt debug command
    kt_debug_match = re.search(
        r"kt debug ([\w-]+) --port (\d+) --namespace ([\w-]+) --mode pdb --pod-ip ([\d.]+)", output
    )
    assert kt_debug_match, f"Should find complete kt debug command with pod-ip: {output}"
    debug_pod = kt_debug_match.group(1)
    debug_port = kt_debug_match.group(2)
    debug_namespace = kt_debug_match.group(3)
    pod_ip = kt_debug_match.group(4)

    # Run kt debug from inside the test pod, piping 'c' to continue
    kt_debug_cmd = f"echo 'c' | kt debug {debug_pod} --port {debug_port} --namespace {debug_namespace} --mode pdb --pod-ip {pod_ip}"
    exec_cmd = ["kubectl", "exec", "-n", test_namespace, test_pod, "--", "bash", "-c", kt_debug_cmd]
    print(f"Running from pod {test_pod}: {kt_debug_cmd}")

    proc = subprocess.run(exec_cmd, capture_output=True, text=True, timeout=30)
    print(f"kt debug stdout: {proc.stdout}")
    print(f"kt debug stderr: {proc.stderr}")

    # Verify we got the PDB prompt (indicates successful connection)
    assert (
        "(Pdb)" in proc.stdout or "Connected" in proc.stdout
    ), f"Should see PDB prompt or connection message: {proc.stdout}"

    # Wait for the task to complete
    try:
        result = await asyncio.wait_for(task2, timeout=10.0)
        print(f"Method returned: {result}")
        assert "Hello from the cluster!" in result
    except asyncio.TimeoutError:
        task2.cancel()
        try:
            await task2
        except asyncio.CancelledError:
            pass
        raise AssertionError("Method did not complete after sending 'c' command")

    print("Test 2 PASSED: PDB WebSocket works correctly with debug=True!")

    # Test server cleanup
    print("\n=== Testing server cleanup after debug=True ===")
    await remote_cls.to_async(remote_cls.compute)
    result = remote_cls.print_and_log(3)
    assert "Hello from the cluster!" in result

    # Test 3: Call with debug=kt.DebugConfig(mode="pdb-ui")
    print("\n=== Test 3: Method call with debug=kt.DebugConfig(mode='pdb-ui') ===")

    async def call_with_debug_config():
        return await remote_cls.print_and_log(4, debug=kt.DebugConfig(mode="pdb-ui"), async_=True, stream_logs=True)

    task3 = asyncio.create_task(call_with_debug_config())

    # Give it time to set up the debug server
    await asyncio.sleep(5)

    # pdb-ui mode uses web-pdb which has a different interface (HTTP, not WebSocket for stdin/stdout)
    # We just verify it doesn't crash and can be cleaned up
    captured = capsys.readouterr()
    output = captured.out + captured.err
    print(f"Captured output:\n{output}")

    # Should see pdb-ui mode in the kt debug command
    assert "kt debug" in output, "kt debug command should be printed"
    assert "--mode pdb-ui" in output, "Should use pdb-ui mode when specified"

    # Cancel the task (pdb-ui requires a browser, which we can't test here)
    task3.cancel()
    try:
        await task3
    except asyncio.CancelledError:
        pass
    except Exception as e:
        # Expected: server might return 502 if web-pdb blocks
        print(f"Task 3 ended with: {type(e).__name__}: {e}")

    # Final server cleanup test
    print("\n=== Testing server cleanup after debug=DebugConfig(mode='pdb-ui') ===")
    await remote_cls.to_async(remote_cls.compute)
    result = remote_cls.print_and_log(4)
    assert "Hello from the cluster!" in result

    print("\n=== All debug tests passed! ===")
