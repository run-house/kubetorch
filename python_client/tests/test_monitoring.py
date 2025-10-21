import asyncio
import time

import pytest
from kubetorch.globals import service_url

from kubetorch.utils import capture_stdout

from .utils import create_random_name_prefix, service_deployer_with_logs


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_monitoring_default_log_streaming(remote_monitoring_fn):
    size = 3
    out = ""
    with capture_stdout() as stdout:
        results = remote_monitoring_fn(size)
        await asyncio.sleep(4)  # wait for the logs to finish streaming
        out = out + str(stdout)

    assert len(results) == size

    for i in range(size):
        assert f"INFO | Hello from the cluster logs! {i}" in out
        assert f"Hello from the cluster stdout! {i}" in out


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_monitoring_default_log_streaming_autoscaled(remote_logs_fn_autoscaled):
    size = 3
    out = ""
    with capture_stdout() as stdout:
        results = remote_logs_fn_autoscaled(n=size)
        await asyncio.sleep(4)  # wait for the logs to finish streaming
        out = out + str(stdout)

        assert results == f"Hello from cluster logs! was logged {size} times"

    for i in range(size):
        assert f"Hello from cluster logs! {i}" in out


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_monitoring_no_log_streaming(remote_monitoring_fn):
    size = 3
    out = ""
    with capture_stdout() as stdout:
        results = remote_monitoring_fn(size, stream_logs=False)
        out = out + str(stdout)

    assert len(results) == size

    for i in range(size):
        assert f"INFO | Hello from the cluster logs! {i}" not in out
        assert f"Hello from the cluster stdout! {i}" not in out


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_monitoring_query_metrics(remote_cls):
    """
    Test to query the specific metric from Prometheus and validate it
    """
    from prometheus_api_client import PrometheusConnect

    remote_cls_pod_names = remote_cls.compute.pod_names()

    # Define a Prometheus queries(for example, check the number of running pods)
    queries = dict()
    for pod_name in remote_cls_pod_names:
        queries[
            f"memory usage {pod_name}"
        ] = f"container_cpu_usage_seconds_total{{pod='{pod_name}'}}"
        queries[
            f"cpu usage {pod_name}"
        ] = f"container_cpu_usage_seconds_total{{pod='{pod_name}'}}"

    base_url = service_url()
    prom = PrometheusConnect(
        url=f"{base_url}/prometheus",
        disable_ssl=True,
    )
    # Perform the queries
    for query_name, query in queries.items():
        response = prom.custom_query(query=query)
        assert response
        response = response[0].get("value")
        assert len(response) == 2
        try:
            query_res = float(response[1])
            assert query_res
        except Exception:
            assert False  # got unexpected value


@pytest.mark.level("minimal")
def test_monitoring_nested_service_log_streaming():
    import re

    import kubetorch as kt

    prefix = create_random_name_prefix()
    name = f"{prefix}-to-to"
    child_name = f"{prefix}-to-to-child"

    deployer_fn_with_logs = kt.fn(service_deployer_with_logs, name).to(
        kt.Compute(
            cpus=".1",
            image=kt.images.Debian(),
            gpu_anti_affinity=True,
        ).autoscale(min_replicas=1)
    )

    out = ""
    expected_result = 5

    with capture_stdout() as stdout:
        summer_result = deployer_fn_with_logs(child_name)
        time.sleep(4)  # wait for the logs to finish streaming
        out = out + str(stdout)

    assert summer_result == expected_result

    for i in range(expected_result):
        # added re.DOTALL, so .* will match across newlines.
        assert re.search(
            rf"({name}).*This is the {i}th log from the parent service", out, re.DOTALL
        ), f"Missing parent log {i}"
        assert re.search(
            rf"({child_name}).*This {i}th log from nested service", out, re.DOTALL
        ), f"Missing child log {i}"


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_monitoring_with_custom_structlog():
    """Test that log streaming works correctly with custom structlog configuration"""
    import kubetorch as kt

    # Import from our test module which sets up complex logging at module level
    from tests.assets.complex_logging.worker_functions import LoggingTestWorker

    # Deploy class with structlog dependency - single deployment for all tests
    img = kt.Image().pip_install(["structlog"])
    compute = kt.Compute(cpus="0.1", image=img)

    # Test initialization logs
    with capture_stdout() as stdout:
        remote_worker = kt.cls(LoggingTestWorker).to(compute)
        init_out = str(stdout)

    assert "LoggingTestWorker initialized" in init_out

    # Test the main processing method
    out = ""
    num_iterations = 3

    with capture_stdout() as stdout:
        results = remote_worker.process_with_logs(num_iterations)
        await asyncio.sleep(1)  # Wait for logs to stream
        out = str(stdout)

    # Verify results
    assert results == [i * 2 for i in range(num_iterations)]

    # Verify all print statements were captured and streamed
    for i in range(num_iterations):
        assert f"Processing iteration {i}" in out, f"Missing print for iteration {i}"
        assert (
            f"Module logger info: Processing item {i}" in out
        ), f"Missing module logger info {i}"
        assert (
            f"Local logger info: Item {i} in progress" in out
        ), f"Missing local logger info {i}"
        assert (
            f"Local logger warning: Check item {i}" in out
        ), f"Missing logger warning {i}"
        assert f"Class logger: Processing {i}" in out, f"Missing class logger {i}"

    assert f"Completed {num_iterations} iterations" in out
    assert "Final results:" in out

    # Test nested logging method
    with capture_stdout() as stdout:
        nested_result = remote_worker.nested_logging_test()
        await asyncio.sleep(2)  # Wait for logs to stream
        nested_out = str(stdout)

    assert nested_result == "inner_result"
    assert "Starting nested test" in nested_out
    assert "Log from inner function" in nested_out
    assert "Print from inner function" in nested_out
    assert "Nested test complete" in nested_out
    assert "Class logger confirms:" in nested_out

    # Test error logging method (should succeed without error)
    with capture_stdout() as stdout:
        success_result = remote_worker.process_with_errors(should_fail=False)
        await asyncio.sleep(2)  # Wait for logs to stream
        error_out = str(stdout)

    assert success_result == "success"
    assert "Starting process with potential errors" in error_out
    assert "Process completed successfully" in error_out
