import asyncio
import re
import time

import pytest

from kubetorch import Cls as kt_cls, Fn as kt_fn
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


def test_metrics_config_helper(service_name, metrics_config, a, b, expected_result):
    import kubetorch as kt

    reloaded_fn = kt.fn(summer, name=service_name, get_if_exists=True)
    out = ""
    fn_sleep_time = 45
    with capture_stdout() as stdout:
        sum_result = reloaded_fn(a=a, b=b, sleep_time=fn_sleep_time, stream_metrics=metrics_config)
        time.sleep(7)  # wait for the logs to finish streaming
        out = out + str(stdout)

    assert sum_result == expected_result

    # if stream_metrics == false, make sure we don't stream metrics
    if isinstance(metrics_config, bool) and not metrics_config:
        assert "[METRICS]" not in out
    else:
        # test the following use-cases:
        # 1. No metrics_config is provided: use the default metrics_config, where metrics_config.scope = resource
        # 2. metrics_config is provided as bool, it's value == True
        # 3. metrics_config is provided as MetricsConfig() instance, with metrics_config.scope == "resource"
        if metrics_config is None or isinstance(metrics_config, bool) or metrics_config.scope == "resource":
            pattern = re.compile(
                r"^\[METRICS\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|\s*CPU:\s*[\d.]+.\s*\|\s*Memory:\s*[\d.]+MiB\s*$",
                re.MULTILINE,
            )
        # 4. metrics_config is provided as MetricsConfig() instance, with metrics_config.scope == "pod"
        else:
            pattern = re.compile(
                r"^\[METRICS\]\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|\s*pod:\s*\S+\s*\|\s*CPU:\s*[\d.]+.\s*\|\s*Memory:\s*[\d.]+MiB\s*$",
                re.MULTILINE,
            )

        assert re.search(pattern, out)


@pytest.mark.level("minimal")
def test_metrics_config(remote_fn):
    import kubetorch as kt

    service_name = remote_fn.service_name

    test_args = [
        (1, 2, 3, kt.MetricsConfig(interval=35)),  # new interval + a=1, b=2, expected_result=3
        (6, 7, 13, kt.MetricsConfig(scope="pod")),  # new scope + a=6, b=7, expected_result=13
        (
            15,
            2,
            17,
            kt.MetricsConfig(scope="pod", interval=35),
        ),  # new interval and new scope + a=15, b=2, expected_result=17
    ]

    for a, b, expected_result, metrics_config in test_args:
        test_metrics_config_helper(
            service_name=service_name, metrics_config=metrics_config, a=a, b=b, expected_result=expected_result
        )

    # passing stream_metrics as bool
    test_metrics_config_helper(service_name=service_name, metrics_config=True, a=2, b=2, expected_result=4)

    test_metrics_config_helper(service_name=service_name, metrics_config=False, a=5, b=3, expected_result=8)

    # passing stream_metrics as None
    test_metrics_config_helper(service_name=service_name, metrics_config=None, a=3, b=2, expected_result=5)


def check_only_user_fn_name_in_pyspy_output(fn_name: str, output: str):
    import re

    clean = re.sub(r"\x1b\[[0-9;]*m", "", output)

    # Extract profiling table rows: lines that start with a number and contain a pipe
    rows = re.findall(r"^\s*\d+\.\d+\s+\|\s+.*$", clean, flags=re.MULTILINE)

    if not rows:
        return False

    for row in rows:
        # Ensure only user's function name appears in the function frame column
        if fn_name not in row:
            return False

    return True


def check_only_user_fn_name_in_torch_output(fn_name: str, output: str):
    lines = output.splitlines()

    # Detect table section
    start = None
    end = None

    for i, line in enumerate(lines):
        if line.strip().startswith("---") and start is None:
            start = i + 1
        elif line.strip().startswith("---") and start is not None:
            end = i
            break

    # Extract only the data rows inside the table
    data_rows = lines[start:end]

    for row in data_rows:
        if fn_name not in row:
            return False

    return True


def get_remote_profiling_compute(compute_type: str, autoscale_replicas: int = 1):
    from .conftest import get_compute

    compute = get_compute(compute_type=compute_type, autoscale_replicas=autoscale_replicas)
    compute.image.pip_install(["numpy"])

    return compute


async def test_pyspy_profiling_helper_fn(profiler_remote_fn: kt_fn, is_async: bool):
    out = ""
    with capture_stdout() as stdout:
        if is_async:
            fn_result = await profiler_remote_fn(profiler="pyspy")
        else:
            fn_result = profiler_remote_fn(profiler="pyspy")
        out = out + str(stdout)

    assert fn_result == "matrix_dot_np ran successfully!"
    assert "================  py-spy Profiling Output ================" in out
    assert f"Estimated CPU usage for '{profiler_remote_fn.module_name}'" in out
    assert "Est. Time (s) | % of Total Func. Time | Function Frame" in out
    assert check_only_user_fn_name_in_pyspy_output(fn_name=profiler_remote_fn.module_name, output=out)


async def test_pyspy_profiling_helper_cls(profiler_remote_cls: kt_cls, is_async: bool):
    out = ""
    with capture_stdout() as stdout:
        if is_async:
            method_result = await profiler_remote_cls.dot_np(profiler="pyspy")
        else:
            method_result = profiler_remote_cls.dot_np(profiler="pyspy")
        out = out + str(stdout)

    assert method_result == "dot_np in Matrix class instance ran successfully!"
    assert "================  py-spy Profiling Output ================" in out
    assert f"Estimated CPU usage for '{profiler_remote_cls.module_name}'" in out
    assert "Est. Time (s) | % of Total Func. Time | Function Frame" in out
    assert check_only_user_fn_name_in_pyspy_output(fn_name=profiler_remote_cls.module_name, output=out)


async def test_torch_profiling_helper_fn(profiler_remote_fn: kt_fn, is_async: bool):
    out = ""
    with capture_stdout() as stdout:
        if is_async:
            fn_result = await profiler_remote_fn(profiler="torch")
        else:
            fn_result = profiler_remote_fn(profiler="torch")
        out = out + str(stdout)

    assert fn_result == "matrix_dot_torch ran successfully!"
    assert "================  PyTorch Profiling Output ================'" in out
    output_col_names = [
        "Name",
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
        "Self CUDA",
        "Self CUDA %",
        "CUDA total",
        "CUDA time avg",
        "CPU Mem",
        "Self CPU Mem",
        "CUDA Mem",
        "Self CUDA Mem",
    ]
    for col_name in output_col_names:
        assert col_name in out
    assert check_only_user_fn_name_in_torch_output(fn_name=profiler_remote_fn.module_name, output=out)


async def test_torch_profiling_helper_cls(profiler_remote_cls: kt_cls, is_async: bool):
    out = ""
    with capture_stdout() as stdout:
        if is_async:
            fn_result = await profiler_remote_cls.dot_torch(profiler="torch")
        else:
            fn_result = profiler_remote_cls.dot_torch(profiler="torch")
        out = out + str(stdout)

    assert fn_result == "matrix_dot_torch ran successfully!"
    assert "================  PyTorch Profiling Output ================'" in out
    output_col_names = [
        "Name",
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
        "Self CUDA",
        "Self CUDA %",
        "CUDA total",
        "CUDA time avg",
        "CPU Mem",
        "Self CPU Mem",
        "CUDA Mem",
        "Self CUDA Mem",
    ]
    for col_name in output_col_names:
        assert col_name in out
    assert check_only_user_fn_name_in_torch_output(fn_name=profiler_remote_cls.module_name, output=out)


@pytest.mark.level("minimal")
def test_profiling_pyspy_fn():
    import kubetorch as kt

    from .utils import matrix_dot_np

    # 1. single pod
    compute = get_remote_profiling_compute(compute_type="deployment")

    # async
    remote_fn = kt.fn(matrix_dot_np, name="fn-pyspy-async-single").to_async(compute)
    test_pyspy_profiling_helper_fn(remote_fn, is_async=True)

    # sync
    remote_fn = kt.fn(matrix_dot_np, name="fn-pyspy-sync-single").to(compute)
    test_pyspy_profiling_helper_fn(remote_fn, is_async=False)

    # 2. multi pod
    compute = get_remote_profiling_compute(compute_type="knative", autoscale_replicas=2)

    # async
    remote_fn = kt.fn(matrix_dot_np, name="fn-pyspy-async-multi").to_async(compute)
    test_pyspy_profiling_helper_fn(remote_fn, is_async=True)

    # sync
    remote_fn = kt.fn(matrix_dot_np, name="fn-pyspy-sync-multi").to(compute)
    test_pyspy_profiling_helper_fn(remote_fn, is_async=False)


@pytest.mark.level("minimal")
def test_profiling_pyspy_cls():
    import kubetorch as kt

    from .utils import Matrix

    # 1. single pod
    compute = get_remote_profiling_compute(compute_type="deployment")

    # async
    remote_cls = kt.cls(Matrix, name="cls-pyspy-async-single").to_async(compute)
    test_pyspy_profiling_helper_cls(remote_cls, is_async=True)

    # sync
    remote_cls = kt.cls(Matrix, name="cls-pyspy-sync-single").to(compute)
    test_pyspy_profiling_helper_cls(remote_cls, is_async=False)

    # 2. multi pod
    compute = get_remote_profiling_compute(compute_type="knative", autoscale_replicas=2)

    # async
    remote_cls = kt.cls(Matrix, name="cls-pyspy-async-multi").to_async(compute)
    test_pyspy_profiling_helper_cls(remote_cls, is_async=True)

    # sync
    remote_cls = kt.cls(Matrix, name="fn-pyspy-sync-multi").to(compute)
    test_pyspy_profiling_helper_cls(remote_cls, is_async=False)


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_profiling_torch_fn():
    import kubetorch as kt

    from .utils import matrix_dot_torch

    gpu = kt.Compute(cpus=0.1, image=kt.images.Pytorch2312().pip_install(["numpy"]), gpus=1)

    # async
    remote_fn_00 = kt.fn(matrix_dot_torch, name="fn-torch-async-single").to_async(gpu)
    test_torch_profiling_helper_fn(remote_fn_00, is_async=True)

    # sync
    remote_fn_01 = kt.fn(matrix_dot_torch, name="fn-torch-sync-single").to(gpu)
    test_torch_profiling_helper_fn(remote_fn_01, is_async=False)

    # 2. multi pod
    gpus = gpu.autoscale(min_scale=2, initial_scale=2)

    # async
    remote_fn_02 = kt.fn(matrix_dot_torch, name="fn-torch-async-multi").to_async(gpus)
    test_torch_profiling_helper_fn(remote_fn_02, is_async=True)

    # sync
    remote_fn_03 = kt.fn(matrix_dot_torch, name="fn-torch-sync-multi").to(gpus)
    test_torch_profiling_helper_fn(remote_fn_03, is_async=False)


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_profiling_torch_cls():
    import kubetorch as kt

    from .utils import Matrix_GPU

    gpu = kt.Compute(cpus=0.1, image=kt.images.Pytorch2312().pip_install(["numpy"]), gpus=1)

    # async
    remote_cls_00 = kt.cls(Matrix_GPU, name="cls-torch-async-single").to_async(gpu)
    test_torch_profiling_helper_cls(remote_cls_00, is_async=True)

    # sync
    remote_cls_01 = kt.cls(Matrix_GPU, name="cls-torch-sync-single").to(gpu)
    test_torch_profiling_helper_cls(remote_cls_01, is_async=False)

    # 2. multi pod
    gpus = gpu.autoscale(min_scale=2, initial_scale=2)

    # async
    remote_cls_02 = kt.cls(Matrix_GPU, name="cls-torch-async-multi").to_async(gpus)
    test_torch_profiling_helper_cls(remote_cls_02, is_async=True)

    # sync
    remote_cls_03 = kt.cls(Matrix_GPU, name="cls-torch-sync-multi").to(gpus)
    test_torch_profiling_helper_cls(remote_cls_03, is_async=False)


@pytest.mark.level("minimal")
async def test_unsupported_profiler():
    import kubetorch as kt

    from .utils import matrix_dot_np

    compute = get_remote_profiling_compute(compute_type="deployment")
    remote_fn = await kt.fn(matrix_dot_np).to_async(compute)
    out = ""
    with capture_stdout() as stdout:
        fn_result = remote_fn(profiler="unsupported-profiler")
        out = out + str(stdout)

    assert fn_result == "matrix_dot_np ran successfully!"
    assert "================  py-spy Profiling Output ================" not in out
    assert "================  PyTorch Profiling Output ================'" not in out
