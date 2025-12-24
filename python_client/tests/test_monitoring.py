import asyncio
import re
import time

import pytest
from kubetorch.globals import service_url

from kubetorch.utils import capture_stdout

from .utils import create_random_name_prefix, service_deployer_with_logs, summer


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
        queries[f"memory usage {pod_name}"] = f"container_cpu_usage_seconds_total{{pod='{pod_name}'}}"
        queries[f"cpu usage {pod_name}"] = f"container_cpu_usage_seconds_total{{pod='{pod_name}'}}"

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

    # Test initialization logs and event streaming during launch
    # stream_logs=True is passed explicitly to ensure launch logs stream in CI (where KT_STREAM_LOGS=FALSE)
    with capture_stdout() as stdout:
        remote_worker = kt.cls(LoggingTestWorker).to(compute, stream_logs=True)
        await asyncio.sleep(2)  # wait for launch logs and events to stream
        init_out = str(stdout)

    # Verify initialization logs from the class are captured
    assert "LoggingTestWorker initialized" in init_out

    # Verify K8s events are streamed during launch (events like Scheduled, Pulling, Started)
    # These come from the controller's event watcher pushing to Loki
    service_name = remote_worker.service_name
    assert f"({service_name} events)" in init_out, f"Missing events prefix in launch output. Got: {init_out[:500]}"

    # Check for at least one common K8s event reason (Scheduled, Pulling, Pulled, Created, or Started)
    event_reasons = ["Scheduled", "Pulling", "Pulled", "Created", "Started"]
    found_event = any(reason in init_out for reason in event_reasons)
    assert (
        found_event
    ), f"No K8s event reasons found in launch output. Expected one of {event_reasons}. Got: {init_out[:500]}"

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
        assert f"Module logger info: Processing item {i}" in out, f"Missing module logger info {i}"
        assert f"Local logger info: Item {i} in progress" in out, f"Missing local logger info {i}"
        assert f"Local logger warning: Check item {i}" in out, f"Missing logger warning {i}"
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


def test_metrics_config_helper(service_name, metrics_config, a, b, expected_result, pod_name=None):
    import kubetorch as kt

    reloaded_fn = kt.fn(summer, name=service_name, get_if_exists=True)
    out = ""
    # Use short sleep - metrics interval is set to 5s in test cases, so 10s is enough for 2 pushes
    fn_sleep_time = 10
    with capture_stdout() as stdout:
        sum_result = reloaded_fn(a=a, b=b, sleep_time=fn_sleep_time, stream_metrics=metrics_config)
        time.sleep(3)  # wait for the logs to finish streaming
        out = out + str(stdout)

    assert sum_result == expected_result

    # if stream_metrics == false, make sure we don't stream metrics
    if isinstance(metrics_config, bool) and not metrics_config:
        assert "CPU: " not in out
        assert "Memory: " not in out
    else:
        # test the following use-cases:
        # 1. No metrics_config is provided: use the default metrics_config, where metrics_config.scope = resource
        # 2. metrics_config is provided as bool, it's value == True
        # 3. metrics_config is provided as MetricsConfig() instance, with metrics_config.scope == "resource"

        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        clean_out = ansi_escape.sub("", out)

        if metrics_config is None or isinstance(metrics_config, bool) or metrics_config.scope == "resource":
            pattern = re.compile(rf"\({re.escape(service_name)} metrics\).*CPU:\s*[\d.]+.*Memory:\s*[\d.]+MiB")
        # 4. metrics_config is provided as MetricsConfig() instance, with metrics_config.scope == "pod"
        else:
            pattern = re.compile(rf"\({re.escape(pod_name)} metrics\).*CPU:\s*[\d.]+.*Memory:\s*[\d.]+MiB")

        assert re.search(pattern, clean_out)


@pytest.mark.level("minimal")
def test_metrics_config(remote_fn):
    import kubetorch as kt

    service_name = remote_fn.service_name
    pod_name = remote_fn.compute.pod_names()[0]

    test_args = [
        (1, 2, 3, kt.MetricsConfig(interval=5)),  # resource scope (default) with short interval
        (6, 7, 13, kt.MetricsConfig(scope="pod", interval=5)),  # pod scope with short interval
    ]

    for a, b, expected_result, metrics_config in test_args:
        test_metrics_config_helper(
            service_name=service_name,
            metrics_config=metrics_config,
            a=a,
            b=b,
            expected_result=expected_result,
            pod_name=pod_name,
        )

    # Test stream_metrics=False (no metrics expected, so short interval is fine)
    test_metrics_config_helper(
        service_name=service_name, metrics_config=False, a=5, b=3, expected_result=8, pod_name=pod_name
    )

    # Test with explicit MetricsConfig using short interval (verifies metrics streaming works)
    test_metrics_config_helper(
        service_name=service_name,
        metrics_config=kt.MetricsConfig(interval=5),
        a=3,
        b=2,
        expected_result=5,
        pod_name=pod_name,
    )
