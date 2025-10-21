import datetime
import os
import re
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path

import kubetorch as kt

import pytest
from kubetorch.cli import app
from kubetorch.cli_utils import get_ingress_host, load_ingress

from typer.testing import CliRunner

from tests.utils import create_random_name_prefix, get_tests_namespace, random_string
from .conftest import get_test_hash


def strip_ansi_codes(text):
    """Remove ANSI escape sequences from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def remote_fn_for_teardown(secret: kt.Secret = None):
    import kubetorch as kt

    from .utils import summer

    secrets = [secret] if secret else None

    compute = kt.Compute(
        cpus=".01",
        gpu_anti_affinity=True,
        launch_timeout=300,
        allowed_serialization=["json", "pickle"],
        secrets=secrets,
    )
    name = f"td-summer-{random_string(3)}"
    fn = kt.fn(summer, name=name).to(compute)
    return fn


def validate_logs_fn_service_info(
    list_output: str, service_name: str, compute_type: str
):
    table_column_names = [
        "SERVICE",
        "TYPE",
        "STATUS",
        "# OF PODS",
        "POD NAMES",
        "VOLUMES",
        "LAST STATUS CHANGE",
        "TTL",
        "CREATOR",
        "QUEUE",
        "CPUs",
        "MEMORY",
        "GPUs",
    ]
    for column_name in table_column_names:
        assert column_name in list_output

    expected_status = "Ready"
    expected_num_of_pods = "1"
    expected_pod_prefix = service_name
    expected_creator = kt.config.username
    service_expected_info = (
        f".*{re.escape(service_name)}.*"
        f"{re.escape(compute_type)}.*"
        f"{re.escape(expected_status)}.*"
        f"{re.escape(expected_num_of_pods)}.*"
        f"{re.escape(expected_pod_prefix)}.*"
        f"{re.escape(expected_creator)}.*"
    )
    assert re.search(service_expected_info, list_output, re.DOTALL)


def validate_teardown_output(
    teardown_output: str, service_name: str, force_delete: bool = False
):
    assert "The following resources will be deleted:\n" in teardown_output
    assert f"• Deployment: {service_name}\n• Service: {service_name}" in teardown_output

    assert (
        "Force deleting resources..."
        if force_delete
        else "Deleting resources..." in teardown_output
    )

    assert (
        f"✓ Deleted deployment {service_name}\n✓ Deleted service {service_name}"
        in teardown_output
    )

    if force_delete:
        time.sleep(3)

    list_result = runner.invoke(app, ["list"], color=False, env={"COLUMNS": "200"})

    # make sure that the service is not displayed in kt list after it has been deleted
    assert list_result.exit_code == 0
    assert service_name not in list_result.stdout


runner = CliRunner()


###############################
####### kt status tests #######
###############################


@pytest.mark.level("minimal")
def test_status_cli_single_pod(remote_cls):
    service_name = remote_cls.service_name
    result = runner.invoke(app, ["status", service_name], color=False)
    assert result.exit_code == 0
    status_output = result.stdout

    # Remove newline characters and ANSI escape sequences for easier assertions
    status_output = status_output.replace("\n", "")
    status_output = strip_ansi_codes(status_output)
    assert service_name in status_output
    assert "Created by:" in status_output
    assert f"Kubetorch version: {kt.__version__}" in status_output
    assert "GPU: " not in status_output
    assert "GPU type: " not in status_output
    assert "CPU: " in status_output
    assert "Pods (1):" in status_output

    remote_cls_pod_names = remote_cls.compute.pod_names()
    assert remote_cls_pod_names


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_status_cli_multi_pod(remote_logs_fn_autoscaled):
    import kubetorch as kt

    service_name = remote_logs_fn_autoscaled.service_name
    result = runner.invoke(app, ["status", service_name], color=False)
    assert result.exit_code == 0

    status_output = result.stdout
    # Remove newline characters and ANSI escape sequences for easier assertions
    status_output = status_output.replace("\n", "")
    status_output = strip_ansi_codes(status_output)
    assert service_name in status_output
    assert "Created by:" in status_output
    assert f"Kubetorch version: {kt.__version__}" in status_output
    assert "GPU: " not in status_output
    assert "GPU type: " not in status_output
    assert "CPU: " in status_output
    assert "Pods (2):" in status_output

    remote_cls_pod_names = remote_logs_fn_autoscaled.compute.pod_names()
    assert remote_cls_pod_names
    assert len(remote_cls_pod_names) == 2

    for pod in remote_cls_pod_names:
        assert pod in status_output


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_status_cli_gpu_single_pod(a10g_gpu):
    import kubetorch as kt

    from .utils import SlowNumpyArray

    remote_cls = kt.cls(SlowNumpyArray).to(compute=a10g_gpu, init_args={"size": 10})

    service_name = remote_cls.service_name

    result = runner.invoke(app, ["status", service_name], color=False)
    assert result.exit_code == 0

    status_output = result.stdout
    # Remove newline characters and ANSI escape sequences for easier assertions
    status_output = status_output.replace("\n", "")
    status_output = strip_ansi_codes(status_output)
    assert service_name in status_output
    assert "Created by:" in status_output
    assert f"Kubetorch version: {kt.__version__}" in status_output
    assert "GPU: " in status_output
    assert "GPU type: " in status_output
    assert "CPU: " in status_output
    assert "Pods (1):" in status_output

    remote_cls_pod_names = remote_cls.compute.pod_names()
    assert remote_cls_pod_names


@pytest.mark.gpu_test
@pytest.mark.level("minimal")
def test_status_cli_gpu_multi_pod(a10g_gpu_autoscale):
    import kubetorch as kt

    from .utils import SlowNumpyArray

    service_name = f"status-two-gpu-{datetime.datetime.now().strftime('%H%M%S')}"[:40]

    remote_cls = kt.cls(SlowNumpyArray, name=service_name).to(
        compute=a10g_gpu_autoscale, init_args={"size": 10}
    )

    service_name = remote_cls.service_name
    result = runner.invoke(app, ["status", service_name], color=False)
    assert result.exit_code == 0

    status_output = result.stdout
    # Remove newline characters and ANSI escape sequences for easier assertions
    status_output = status_output.replace("\n", "")
    status_output = strip_ansi_codes(status_output)
    assert service_name in status_output
    assert "Created by:" in status_output
    assert f"Kubetorch version: {kt.__version__}" in status_output
    assert "GPU: " in status_output
    assert "GPU type: " in status_output
    assert "CPU: " in status_output
    assert "Pods (2):" in status_output

    remote_cls_pod_names = remote_cls.compute.pod_names()
    assert remote_cls_pod_names
    assert len(remote_cls_pod_names) == 2

    for pod in remote_cls_pod_names:
        assert pod in status_output


@pytest.mark.level("minimal")
def test_status_cli_wrong_flags():
    service_name = "wrong_name"

    # wrong name
    result = runner.invoke(app, ["status", service_name], color=False)
    assert result.exit_code == 1
    assert "Failed to load service" in result.stdout

    # wrong name in an existing namespace
    result = runner.invoke(app, ["status", service_name, "-n", "default"], color=False)
    assert result.exit_code == 1
    assert "Failed to load service" in result.stdout
    assert "in namespace" in result.stdout

    # don't pass name
    result = runner.invoke(app, ["status"], color=False)
    assert result.exit_code == 2
    assert "Missing argument 'NAME'" in result.stderr


########################################################################################################################
#################################################### kt logs tests #####################################################
# *NOTE*: We expect log output length to be less the 200 and not less than 200 (default tail length per pod is 100,
# and we have 2 pods) because some logs are split into 2 lines, due the lack of space (the width of the output
# terminal is not big enough). Moreover, we parse the output in order to get only the logs. (By parsing the results in
# the tests, we remove additional info that is printed, such as `looking for service <SERVICE_NAME>` etc).
########################################################################################################################


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_basic(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 10

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    service_name = remote_logs_fn.service_name

    result = runner.invoke(app, ["logs", service_name], color=False)
    assert result.exit_code == 0

    logs_output = result.stdout
    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 111


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_basic_no_username(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 10

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    function_name = remote_logs_fn.name
    service_name = remote_logs_fn.service_name

    result = runner.invoke(app, ["logs", function_name], color=False)
    assert result.exit_code == 0

    logs_output = result.stdout
    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 111


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_single_pod_index(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 1

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    service_name = remote_logs_fn.service_name
    pod_name = remote_logs_fn.compute.pod_names()[0]

    result = runner.invoke(app, ["logs", service_name, "-p", "0"], color=False)
    assert result.exit_code == 0

    logs_output = result.stdout
    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output
    assert pod_name in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 111


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_single_pod_name(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 1

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    service_name = remote_logs_fn.service_name
    pod_name = remote_logs_fn.compute.pod_names()[0]

    result = runner.invoke(app, ["logs", service_name, "-p", pod_name], color=False)
    assert result.exit_code == 0

    logs_output = result.stdout
    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output
    assert pod_name in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 111


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_single_pod_tail(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 1

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    service_name = remote_logs_fn.service_name
    result = runner.invoke(app, ["logs", service_name, "-t", "10"], color=False)
    assert result.exit_code == 0
    logs_output = result.stdout

    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output
    assert remote_logs_fn.service_name in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 15


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_single_pod_namespace(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 1

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    service_name = remote_logs_fn.service_name

    result = runner.invoke(app, ["logs", service_name, "-n", "default"], color=False)
    assert result.exit_code == 0
    logs_output = result.stdout

    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output
    assert remote_logs_fn.service_name in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 111


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_basic_multi_pod(remote_logs_fn_autoscaled):
    log_msg = "Tests logs msg"
    log_amount = 10

    remote_logs_fn_autoscaled(log_msg, log_amount)

    assert len(remote_logs_fn_autoscaled.compute.pod_names()) == 2

    service_name = remote_logs_fn_autoscaled.service_name

    result = runner.invoke(app, ["logs", service_name], color=False)
    assert result.exit_code == 0
    logs_output = result.stdout

    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 200


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_multi_pod_index(remote_logs_fn_autoscaled):
    log_msg = "Tests logs msg"
    log_amount = 10

    remote_logs_fn_autoscaled(log_msg, log_amount)

    assert len(remote_logs_fn_autoscaled.compute.pod_names()) == 2

    service_name = remote_logs_fn_autoscaled.service_name
    pod_names = remote_logs_fn_autoscaled.compute.pod_names()

    result = runner.invoke(app, ["logs", service_name, "-p", "1"], color=False)
    assert result.exit_code == 0
    logs_output = result.stdout

    assert logs_output
    assert service_name in logs_output
    assert all(p for p in pod_names if p in logs_output)

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 200


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_multi_pod_name(remote_logs_fn_autoscaled):
    log_msg = "Tests logs msg"
    log_amount = 10

    remote_logs_fn_autoscaled(log_msg, log_amount)

    assert len(remote_logs_fn_autoscaled.compute.pod_names()) == 2

    service_name = remote_logs_fn_autoscaled.service_name
    pod_names = remote_logs_fn_autoscaled.compute.pod_names()

    result = runner.invoke(app, ["logs", service_name, "-p", pod_names[1]], color=False)
    assert result.exit_code == 0
    logs_output = result.stdout

    assert logs_output
    assert service_name in logs_output
    assert all(p for p in pod_names if p in logs_output)

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 200


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_multi_pod_tail(remote_logs_fn_autoscaled):
    log_msg = "Tests logs msg"
    log_amount = 50

    remote_logs_fn_autoscaled(log_msg, log_amount)

    assert len(remote_logs_fn_autoscaled.compute.pod_names()) == 2

    service_name = remote_logs_fn_autoscaled.service_name

    result = runner.invoke(
        app, ["logs", service_name, "-t", "20"], color=False, env={"COLUMNS": "200"}
    )
    assert result.exit_code == 0
    logs_output = result.stdout

    assert service_name in logs_output
    # assert latest logs appear in logs tail output
    assert f"{log_msg} 49" in logs_output

    # assert more early logs don't appear in logs tail output
    assert f"{log_msg} 0" not in logs_output
    assert f"{log_msg} 1" not in logs_output
    assert f"{log_msg} 2" not in logs_output
    assert remote_logs_fn_autoscaled.service_name in logs_output

    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 30


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_multi_pod_namespace(remote_logs_fn_autoscaled):
    log_msg = "Tests logs msg"
    log_amount = 10

    remote_logs_fn_autoscaled(log_msg, log_amount)
    assert len(remote_logs_fn_autoscaled.compute.pod_names()) == 2

    service_name = remote_logs_fn_autoscaled.service_name

    result = runner.invoke(app, ["logs", service_name, "-n", "default"], color=False)
    assert result.exit_code == 0
    logs_output = result.stdout

    assert service_name in logs_output
    for i in range(log_amount):
        assert f"{log_msg} {i}" in logs_output

    assert remote_logs_fn_autoscaled.service_name in logs_output
    parsed_logs = logs_output.strip().replace("\n\n", "\n").split("\n")[5:-1]
    assert len(parsed_logs) <= 200


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_non_existing_service():
    non_existing_service_name = "no-such-service"

    result = runner.invoke(app, ["logs", non_existing_service_name], color=False)
    assert result.exit_code == 1
    logs_output = result.stdout
    assert f"Failed to load service {non_existing_service_name}" in logs_output


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_single_pod_wrong_index(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 1

    remote_logs_fn(log_msg, log_amount)

    assert len(remote_logs_fn.compute.pod_names()) == 1

    service_name = remote_logs_fn.service_name
    cmd = ["logs", service_name, "-p", "1"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 1
    logs_output = result.stdout

    pod_index = cmd[-1]

    assert f"Pod index {pod_index} is out of range" in logs_output


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_logs_cli_single_pod_wrong_name(remote_logs_fn):
    log_msg = "Tests logs msg"
    log_amount = 1

    remote_logs_fn(log_msg, log_amount)

    non_existing_pod_name = "wrong_pod_name"

    service_name = remote_logs_fn.service_name
    cmd = ["logs", service_name, "-p", non_existing_pod_name]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 1
    logs_output = result.stdout

    # Using regex because the GitHub stdout is causing a line break in the middle of the error message.
    regex_expression = f".*{remote_logs_fn.service_name} does not have an associated pod called.*{non_existing_pod_name}.*"
    assert re.search(regex_expression, logs_output, re.DOTALL)


###############################
######## kt list tests ########
###############################
@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_list_basic(remote_logs_fn):
    service_name = remote_logs_fn.service_name
    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")

    # sort by timestamp, so we will get the latest services first, in case there are few CI jobs running simultaneously
    result = runner.invoke(app, ["list", "--sort"], env={"COLUMNS": "400"}, color=False)
    assert result.exit_code == 0
    list_output = result.stdout
    validate_logs_fn_service_info(
        list_output=list_output, service_name=service_name, compute_type=compute_type
    )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_list_ns_with_services(remote_logs_fn):
    service_name = remote_logs_fn.service_name
    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")

    namespace_flag_names = ["-n", "--namespace"]
    for ns_flag in namespace_flag_names:
        # sort by timestamp, so we will get the latest services first, in case there are few CI jobs running simultaneously
        result = runner.invoke(
            app,
            ["list", "--sort", ns_flag, remote_logs_fn.namespace],
            env={"COLUMNS": "400"},
            color=False,
        )
        assert result.exit_code == 0
        list_output = result.stdout
        validate_logs_fn_service_info(
            list_output=list_output,
            service_name=service_name,
            compute_type=compute_type,
        )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_list_ns_without_services(remote_logs_fn):
    result = runner.invoke(
        app, ["list", "-n", "no-such-ns"], env={"COLUMNS": "200"}, color=False
    )
    assert result.exit_code == 0
    list_output = result.stdout
    assert "No services found in no-such-ns namespace" in list_output


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_list_existing_prefix(remote_logs_fn):
    service_name = remote_logs_fn.service_name
    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    service_prefix = service_name[:10]  # prefix = first 10 chars of the service name

    tag_flag_names = ["-t", "--tag"]
    for tag_flag in tag_flag_names:
        # sort by timestamp, so we will get the latest services first, in case there are few CI jobs running simultaneously
        result = runner.invoke(
            app,
            ["list", "--sort", tag_flag, service_prefix],
            env={"COLUMNS": "200"},
            color=False,
        )
        assert result.exit_code == 0
        list_output = result.stdout
        validate_logs_fn_service_info(
            list_output=list_output,
            service_name=service_name,
            compute_type=compute_type,
        )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_list_non_existing_prefix(remote_logs_fn):
    result = runner.invoke(
        app, ["list", "-t", "no-such-prefix"], env={"COLUMNS": "200"}, color=False
    )
    assert result.exit_code == 0
    list_output = result.stdout
    assert "No services found in default namespace" in list_output


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_list_multiple_flags(remote_logs_fn):
    service_name = remote_logs_fn.service_name
    compute_type = os.getenv("TEST_COMPUTE_TYPE", "deployment")
    existing_service_prefix = service_name[
        :10
    ]  # prefix = first 10 chars of the service name
    existing_ns = remote_logs_fn.namespace
    wrong_service_prefix = "no-such-prefix"
    wrong_ns = "no-such-ns"

    # correct ns, correct service prefix
    result = runner.invoke(
        app,
        ["list", "--sort", "-t", existing_service_prefix, "-n", existing_ns],
        env={"COLUMNS": "200"},
        color=False,
    )
    assert result.exit_code == 0
    list_output = result.stdout
    validate_logs_fn_service_info(
        list_output=list_output, service_name=service_name, compute_type=compute_type
    )

    # wrong ns, correct service prefix
    result = runner.invoke(
        app,
        ["list", "-t", existing_service_prefix, "-n", wrong_ns],
        env={"COLUMNS": "200"},
        color=False,
    )
    list_output = result.stdout
    assert result.exit_code == 0
    assert f"No services found in {wrong_ns} namespace" in list_output

    # correct ns, wrong service prefix
    result = runner.invoke(
        app,
        ["list", "-t", wrong_service_prefix, "-n", existing_ns],
        env={"COLUMNS": "200"},
        color=False,
    )
    list_output = result.stdout
    assert result.exit_code == 0
    assert f"No services found in {existing_ns} namespace" in list_output


###############################
####### kt config tests #######
###############################


@pytest.mark.level("unit")
def test_cli_kt_config():
    import json

    get_whole_config_cmds = [["config"], ["config", "list"]]

    for cmd in get_whole_config_cmds:

        result = runner.invoke(app, cmd, color=False)
        assert result.exit_code == 0

        config_list_output = result.stdout

        # parsing the config_list_output str, so we could convert the str output to a dict using json.loads()
        replacements_dict = {
            "\n": "",
            "None": "null",
            "True": "true",
            "False": "false",
            "'": '"',
        }
        for k, v in replacements_dict.items():
            config_list_output = config_list_output.replace(k, v)
        config_list_output = json.loads(config_list_output)

        kt_config = kt.globals.config

        for config_key in config_list_output.keys():
            assert config_list_output.get(config_key) == getattr(
                kt_config, config_key, None
            )


@pytest.mark.level("unit")
def test_cli_kt_config_get():
    # get supported key
    result = runner.invoke(app, ["config", "get", "username"], color=False)
    assert result.exit_code == 0
    assert result.stdout.strip() == kt.config.username

    # get un-supported key
    result = runner.invoke(
        app, ["config", "get", "noSuchKey"], env={"COLUMNS": "200"}, color=False
    )  # added {"COLUMNS": "200"} to make sure the error will be printed in one line
    assert result.exit_code == 2
    result_output = result.stdout or result.stderr
    assert "Error" in result_output
    assert (
        "Invalid value for '[KEY]': Valid keys are: api_url, cluster_config, install_namespace, install_url, license_key, log_verbosity, namespace, queue, stream_logs, tracing_enabled, username"
    ) in result_output


@pytest.mark.level("unit")
def test_cli_kt_config_set():
    from kubetorch.utils import LogVerbosity

    # making hard copy, so this value won't change during `kt config set`.
    original_username = kt.config.username  # str type value
    original_stream_logs = kt.config.stream_logs  # bool type value
    original_log_verbosity = kt.config.log_verbosity  # LogVerbosity (enum) value

    try:
        # Part A: set supported keys
        new_values = {
            "username": "sashab1",
            "stream_logs": "False",
            "log_verbosity": LogVerbosity.CRITICAL.value,
        }
        for key, value in new_values.items():
            result = runner.invoke(app, ["config", "set", key, value], color=False)
            assert result.exit_code == 0
            assert str(getattr(kt.config, key, None)) == value

        # Part B: set un-supported keys
        result = runner.invoke(
            app,
            ["config", "set", "noSuchKey", "noSuchValue"],
            env={"COLUMNS": "200"},
            color=False,
        )  # added {"COLUMNS": "200"} to make sure the error will be printed in one line
        assert result.exit_code == 2
        result_output = result.stdout or result.stderr
        assert "Error" in result_output
        assert (
            "Invalid value for '[KEY]': Valid keys are: api_url, cluster_config, install_namespace, install_url, license_key, log_verbosity, namespace, queue, stream_logs, tracing_enabled, username"
            in result_output
        )

        # Part C: set supported key, but don't provide value
        result = runner.invoke(app, ["config", "set", "username"], color=False)
        assert result.exit_code == 1
        assert "Both key and value are required for 'set'" in result.stdout

        kt.config.username = original_username

        # Part D: set supported key, but provide a value of a wrong type
        # D.1: unsupported username
        invalid_username = "1"
        result = runner.invoke(
            app, ["config", "set", "username", invalid_username], color=False
        )
        assert result.exit_code == 1
        assert (
            f"Error setting username: {invalid_username} must be a valid k8s name"
            in result.stdout
        )

        # D.2: provide username that is in a reserved names
        reserved_names = ["kt", "kubetorch", "knative"]
        for name in reserved_names:
            result = runner.invoke(
                app,
                ["config", "set", "username", name],
                color=False,
                env={"COLUMNS": "200"},
            )
            assert result.exit_code == 1
            assert (
                f"Error setting username: {name} is one of the reserved names: {', '.join(reserved_names)}"
                in result.stdout
            )

        # D.3: provide stream_logs that is not a boolean
        result = runner.invoke(
            app, ["config", "set", "stream_logs", "notBoolVal"], color=False
        )
        assert result.exit_code == 1
        assert (
            "Error setting stream_logs: stream_logs must be a boolean value"
            in result.stdout
        )

        # D.4: provide stream_logs that is not a boolean
        result = runner.invoke(
            app,
            ["config", "set", "log_verbosity", "BadLogVerbosityVal"],
            env={"COLUMNS": "200"},
            color=False,
        )  # added {"COLUMNS": "200"} to make sure the error will be printed in one line
        assert result.exit_code == 1
        assert (
            "Error setting log_verbosity: Invalid log verbosity value. Must be one of: 'debug', 'info', 'critical'."
            in result.stdout
        )

    finally:
        # set the config keys to their original values, even if the test fails
        kt.config.username = original_username
        kt.config.stream_logs = original_stream_logs
        kt.config.log_verbosity = original_log_verbosity
        kt.config.write()


@pytest.mark.level("unit")
def test_cli_kt_config_unset():
    original_username = kt.config.username  # str type value
    original_stream_logs = kt.config.stream_logs  # bool type value
    original_log_verbosity = kt.config.log_verbosity  # LogVerbosity (enum) value
    try:
        # Part A: supported keys
        config_keys = ["username", "stream_logs", "log_verbosity"]
        get_msg_after_unset = {
            "username": "Username not set",
            "stream_logs": "True",
            "log_verbosity": "info",
        }
        for key in config_keys:
            unset_result = runner.invoke(app, ["config", "unset", key], color=False)
            assert unset_result.exit_code == 0
            assert f"{key.capitalize()} unset" in unset_result.stdout
            get_result = runner.invoke(app, ["config", "get", key], color=False)
            assert get_result.exit_code == 0
            assert get_msg_after_unset.get(key) in get_result.stdout
        # Part B: unsupported key
        result = runner.invoke(
            app, ["config", "unset", "NoSuchKey"], color=False, env={"COLUMNS": "200"}
        )
        assert result.exit_code == 2
        result_output = result.stdout or result.stderr
        assert "Error" in result_output
        assert (
            "Invalid value for '[KEY]': Valid keys are: api_url, cluster_config, install_namespace, install_url, license_key, log_verbosity, namespace, queue, stream_logs, tracing_enabled, username"
            in result_output
        )

    finally:
        # set the config keys (also locally) to their original values, even if the test fails
        kt.config.username = original_username
        kt.config.stream_logs = original_stream_logs
        kt.config.log_verbosity = original_log_verbosity
        kt.config.write()


@pytest.mark.level("unit")
def test_cli_kt_config_unknown_action():
    unsupported_action = "NoSuchAction"
    result = runner.invoke(
        app, ["config", unsupported_action], color=False, env={"COLUMNS": "200"}
    )
    assert result.exit_code == 1
    result_output = result.stdout or result.stderr
    assert f"Unknown action: {unsupported_action}" in result_output
    assert "Valid actions are: set, get, list" in result_output


####################################
########## kt check tests ##########
####################################
@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_check_basic(remote_logs_fn):
    service_name = remote_logs_fn.service_name

    original_log_streaming = kt.config.stream_logs

    if not original_log_streaming:
        kt.config.stream_logs = True

    result = runner.invoke(app, ["check", service_name], color=False)
    assert result.exit_code == 0
    result_output = result.stdout
    expected_checks = [
        "deployment service",
        "deployment pod",
        "rsync",
        "service call",
        "log streaming",
    ]
    for check in expected_checks:
        assert f"Checking {check}..." in result_output

    assert "All service checks passed" in result_output
    kt.config.stream_logs = (
        original_log_streaming  # set stream_logs to its original value
    )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_check_basic_multipod(remote_logs_fn_autoscaled):
    service_name = remote_logs_fn_autoscaled.service_name

    original_log_streaming = kt.config.stream_logs
    if not original_log_streaming:
        kt.config.stream_logs = True

    result = runner.invoke(app, ["check", service_name], color=False)
    assert result.exit_code == 0
    result_output = result.stdout
    expected_checks = [
        "knative service",
        "deployment pod",
        "rsync",
        "service call",
        "log streaming",
    ]
    for check in expected_checks:
        assert f"Checking {check}..." in result_output

    assert "All service checks passed" in result_output
    kt.config.stream_logs = (
        original_log_streaming  # set stream_logs to its original value
    )


####################################
######## kt describe tests #########
####################################


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_cli_kt_describe_basic(remote_logs_fn):
    service_name, service_namespace = (
        remote_logs_fn.service_name,
        remote_logs_fn.namespace,
    )
    has_ingress = kt.config.api_url is not None
    args_placeholder = []

    if has_ingress:
        service_path = has_ingress.replace("wss://", "https://").replace(
            "ws://", "http://"
        )
        service_path = (
            f"{service_path}/{service_namespace}/{service_name}/METHOD_OR_CLS_NAME"
        )
    else:
        service_path = f"http://{service_name}.{service_namespace}.svc.cluster.local/METHOD_OR_CLS_NAME"

    result = runner.invoke(
        app,
        ["describe", service_name],
        color=False,
        env={"COLUMNS": "200"},
    )
    assert result.exit_code == 0
    output = result.output

    assert f"Found deployment service {service_name}" in output

    expected_python_code = textwrap.dedent(
        f"""\
        import requests

        url = "{service_path}"
        headers = {{
            "Content-Type": "application/json"
        }}
        data = {{
            "args": {args_placeholder},
            "kwargs": {{}}
        }}

        response = requests.post(url, headers=headers, json=data)
        print(response.json())
    """
    )

    expected_curl_code = textwrap.dedent(
        f"""\
        curl -X POST \\
          -H "Content-Type: application/json" \\
          -d '{{"args": {args_placeholder}, "kwargs": {{}}}}' \\
          {service_path}
    """
    )

    if not has_ingress:
        assert "No ingress found" in output
        assert "Calling the service from inside the cluster:\n" in output

    else:
        ingress = load_ingress()
        host = get_ingress_host(ingress)

        expected_python_code = expected_python_code.replace(
            '"Content-Type": "application/json"',
            f'"Host": "{host}",\n    "Content-Type": "application/json"',
        )

        expected_curl_code = expected_curl_code.replace(
            '-H "Content-Type: application/json"',
            f'-H "Host: {host}" \\\n  -H "Content-Type: application/json"',
        )

    expected_python_code = expected_python_code.split("\n")
    expected_curl_code = expected_curl_code.split("\\")

    for code_line in expected_curl_code:
        assert code_line.strip() in output

    for code_line in expected_python_code:
        assert code_line in output


####################################
######## kt teardown tests #########
####################################


@pytest.mark.level("minimal")
def test_cli_kt_teardown_long_confirmation():
    remote_fn = remote_fn_for_teardown()
    service_name = remote_fn.service_name
    teardown_result = runner.invoke(
        app, ["teardown", service_name, "--yes"], color=False, env={"COLUMNS": "200"}
    )

    assert teardown_result.exit_code == 0
    result_output = teardown_result.output
    validate_teardown_output(teardown_output=result_output, service_name=service_name)


@pytest.mark.level("minimal")
def test_cli_kt_teardown_prefix():
    remote_fn = remote_fn_for_teardown()
    service_name = remote_fn.service_name
    service_name_prefix = f"{get_test_hash()}-td"
    teardown_result = runner.invoke(
        app,
        ["teardown", service_name, "-y", "--prefix", service_name_prefix],
        color=False,
        env={"COLUMNS": "200"},
    )

    assert teardown_result.exit_code == 0
    result_output = teardown_result.output
    validate_teardown_output(teardown_output=result_output, service_name=service_name)


@pytest.mark.level("minimal")
def test_cli_kt_teardown_namespace():
    remote_fn = remote_fn_for_teardown()
    service_name = remote_fn.service_name
    teardown_result = runner.invoke(
        app,
        ["teardown", service_name, "-y", "--namespace", "default"],
        color=False,
        env={"COLUMNS": "200"},
    )

    assert teardown_result.exit_code == 0
    result_output = teardown_result.output
    validate_teardown_output(teardown_output=result_output, service_name=service_name)


@pytest.mark.level("minimal")
def test_cli_kt_teardown_force():
    remote_fn = remote_fn_for_teardown()
    service_name = remote_fn.service_name
    teardown_result = runner.invoke(
        app,
        ["teardown", service_name, "--force"],
        color=False,
        env={"COLUMNS": "200"},
    )

    assert teardown_result.exit_code == 0
    result_output = teardown_result.stdout
    validate_teardown_output(
        teardown_output=result_output, service_name=service_name, force_delete=True
    )


@pytest.mark.level("minimal")
def test_cli_kt_teardown_multiple_flags():
    flag_combinations = [
        ["-p", f"{get_test_hash()}-td-", "-n", "default", "-y"],
        ["-p", f"{get_test_hash()}-td-", "-f"],
        ["-f", "-n", "default"],
        ["-p", f"{get_test_hash()}-td-", "-n", "default", "-f"],
    ]
    for flag_combination in flag_combinations:
        remote_fn = remote_fn_for_teardown()
        service_name = remote_fn.service_name
        teardown_result = runner.invoke(
            app,
            ["teardown", service_name] + flag_combination,
            color=False,
            env={"COLUMNS": "200"},
        )

        assert teardown_result.exit_code == 0
        result_output = teardown_result.output
        force_delete: bool = "-f" in flag_combination
        validate_teardown_output(
            teardown_output=result_output,
            service_name=service_name,
            force_delete=force_delete,
        )


@pytest.mark.level("minimal")
def test_cli_kt_teardown_wrong_usage():
    # Part A: no name provided
    teardown_result = runner.invoke(
        app, ["teardown"], color=False, env={"COLUMNS": "200"}
    )

    assert teardown_result.exit_code == 1
    assert (
        "Please provide a service name or use the --all or --prefix flags"
        in teardown_result.output
    )

    # Part B: teardown non-existing service
    service_name = "noSuchService"
    teardown_result = runner.invoke(
        app, ["teardown", service_name, "-y"], color=False, env={"COLUMNS": "200"}
    )

    assert teardown_result.exit_code == 1
    output = teardown_result.output
    assert (
        f"Finding resources for service {service_name.lower()} in default namespace..."
        in output
    )
    assert f"Service {service_name.lower()} not found" in output

    # Part C: teardown service with non-existing prefix
    teardown_result = runner.invoke(
        app, ["teardown", "-p", service_name, "-y"], color=False, env={"COLUMNS": "200"}
    )

    assert teardown_result.exit_code == 0
    output = teardown_result.output
    assert (
        f"Deleting all services with prefix {service_name} in default namespace"
        in output
    )
    assert "No services are found" in output

    # Part D: teardown service but provide wrong namespace
    remote_fn = remote_fn_for_teardown()
    service_name = remote_fn.service_name
    service_namespace = remote_fn.namespace
    teardown_result = runner.invoke(
        app,
        ["teardown", service_name, "-y", "-n", f"{service_namespace}1"],
        color=False,
        env={"COLUMNS": "200"},
    )

    assert teardown_result.exit_code == 1
    output = teardown_result.output
    assert (
        f"Finding resources for service {service_name.lower()} in {service_namespace}1 namespace..."
        in output
    )
    assert f"Service {service_name.lower()} not found"


#########################################
# ------------ secrets tests ------------
#########################################


@pytest.mark.skip("Skipping since CI is running on a GKE cluster")
@pytest.mark.level("unit")
def test_cli_secrets_create_aws():
    cmd = ["secrets", "create", "--provider", "aws"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    assert "Successfully created secret aws in namespace default" in result.output


@pytest.mark.skip("Skipping since CI is running on a GKE cluster")
@pytest.mark.level("unit")
def test_cli_secrets_create_aws_with_name():
    secret_name = f"{get_test_hash()}-aws"
    cmd = ["secrets", "create", secret_name, "--provider", "aws"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    assert (
        f"Successfully created secret {secret_name} in namespace default"
        in result.output
    )


@pytest.mark.skip("Skipping since CI is running on a GKE cluster")
@pytest.mark.level("unit")
def test_cli_secrets_create_aws_test_ns():
    tests_ns = get_tests_namespace()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    cmd = ["secrets", "create", "--provider", "aws", "--namespace", tests_ns]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    assert f"Successfully created secret aws in namespace {tests_ns}" in result.output


@pytest.mark.level("unit")
def test_cli_secrets_create_gcp():
    cmd = ["secrets", "create", "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    output = result.output
    assert "✔ Secret created successfully\n" in output
    assert f"Name: {kt.config.username}-gcp\n" in output
    assert "Namespace: default\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_create_gcp_with_name():
    secret_name = f"{get_test_hash()}1-gcp"
    cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    output = result.output
    assert "✔ Secret created successfully\n" in output
    assert f"Name: {secret_name}\n" in output
    assert "Namespace: default\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_create_gcp_test_ns():
    tests_ns = get_tests_namespace()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    cmd = ["secrets", "create", "--provider", "gcp", "--namespace", tests_ns]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    output = result.output
    assert "✔ Secret created successfully\n" in output
    assert f"Name: {kt.config.username}-gcp\n" in output
    assert f"Namespace: {tests_ns}\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_create_huggingface():
    cmd = ["secrets", "create", "--provider", "huggingface"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0
    output = result.output
    assert "✔ Secret created successfully\n" in output
    assert f"Name: {kt.config.username}-huggingface\n" in output
    assert "Namespace: default\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_create_huggingface_with_name():
    secret_name = f"{get_test_hash()}1-huggingface"
    cmd = ["secrets", "create", secret_name, "--provider", "huggingface"]
    result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
    assert result.exit_code == 0
    output = result.output
    assert "✔ Secret created successfully\n" in output
    assert f"Name: {secret_name}\n" in output
    assert "Namespace: default\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_create_not_supported_provider():
    cmd = ["secrets", "create", "--provider", "NoSuchProvider"]
    result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert (
        f"Failed to create the secret: NoSuchProvider is not a supported provider: {kt.Secret.builtin_providers(as_str=True)}"
        in result.output.replace("\n", "")
    )


@pytest.mark.skip("Skipping since CI is running on a GKE cluster")
@pytest.mark.level("unit")
def test_cli_secrets_create_same_secret_twice():
    secret_name = f"{get_test_hash()}-aws-twice"
    cmd = ["secrets", "create", secret_name, "--provider", "aws"]

    # create the secret for the first time
    result_first_creation = runner.invoke(app, cmd, color=False)
    assert result_first_creation.exit_code == 0
    assert (
        f"Successfully created secret {secret_name} in namespace default"
        in result_first_creation.output
    )

    # create the secret for the second time
    result_second_creation = runner.invoke(
        app, cmd, color=False, env={"COLUMNS": "200"}
    )
    assert result_second_creation.exit_code == 0
    assert (
        f"Secret '{secret_name}' already exists in namespace default for username {kt.globals.config.username}, skipping creation"
        in result_second_creation.output
    )


@pytest.mark.level("unit")
def test_cli_secrets_create_same_secret_twice_gcp():
    secret_name = f"{get_test_hash()}-gcp-twice"
    cmd = ["secrets", "create", secret_name, "--provider", "gcp"]

    # create the secret for the first time
    result_first_creation = runner.invoke(app, cmd, color=False)
    assert result_first_creation.exit_code == 0
    first_output = result_first_creation.output
    assert "✔ Secret created successfully\n" in first_output
    assert f"Name: {secret_name}\n" in first_output
    assert "Namespace: default\n" in first_output

    # create the secret for the second time
    result_second_creation = runner.invoke(
        app, cmd, color=False, env={"COLUMNS": "200"}
    )
    assert result_second_creation.exit_code == 0
    assert (
        f"Secret '{secret_name}' already exists in namespace default, skipping creation"
        in result_second_creation.output
    )


@pytest.mark.level("unit")
def test_cli_secrets_create_using_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        secret_file_content = "key1=value1\nkey2=value2"
        temp_file = Path(tmpdir) / ".secrets" / "test_secret"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text(secret_file_content)

        secret_name = f"{get_test_hash()}-create-from-path-secret"
        cmd = ["secrets", "create", secret_name, "--path", str(temp_file.resolve())]
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        assert "✔ Secret created successfully\n" in output
        assert f"Name: {secret_name}\n" in output
        assert "Namespace: default\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_create_using_env_vars():
    os.environ["TEST_KEY_1"] = "val_1"
    os.environ["TEST_KEY_2"] = "val_2"
    secret_name = f"{get_test_hash()}-create-from-env-secret"
    cmd = ["secrets", "create", secret_name, "-v", "TEST_KEY_1", "-v", "TEST_KEY_2"]
    result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
    assert result.exit_code == 0
    output = result.output
    assert "✔ Secret created successfully\n" in output
    assert f"Name: {secret_name}\n" in output
    assert "Namespace: default\n" in output


@pytest.mark.level("unit")
def test_cli_secrets_list():
    secret_name = f"{get_test_hash()}-gcp-secret1"
    cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
    assert result.exit_code == 0

    list_cmds = [["secrets"], ["secrets", "list"]]
    expected_pattern = rf".*{secret_name}.*{kt.globals.config.username}.*default.*"
    for cmd in list_cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        assert re.search(expected_pattern, output, re.DOTALL)


@pytest.mark.level("unit")
def test_cli_secrets_list_in_test_ns():
    tests_ns = get_tests_namespace()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    secret_name = f"{get_test_hash()}-gcp-{tests_ns}"
    cmd = [
        "secrets",
        "create",
        secret_name,
        "--provider",
        "gcp",
        "--namespace",
        tests_ns,
    ]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    list_cmds = [
        ["secrets", "-n", tests_ns],
        ["secrets", "--namespace", tests_ns],
        ["secrets", "list", "-n", tests_ns],
        ["secrets", "list", "--namespace", tests_ns],
    ]
    expected_pattern = rf".*{secret_name}.*{kt.globals.config.username}.*{tests_ns}.*"
    for cmd in list_cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        assert re.search(expected_pattern, output, re.DOTALL)


@pytest.mark.level("unit")
def test_cli_secrets_list_all():
    tests_ns = get_tests_namespace()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    secret_names = [f"{get_test_hash()}-gcp-{tests_ns}-{i}" for i in range(2)]
    for secret_name in secret_names:
        cmd = [
            "secrets",
            "create",
            secret_name,
            "--provider",
            "gcp",
            "--namespace",
            tests_ns,
        ]
        result = runner.invoke(app, cmd, color=False)
        assert result.exit_code == 0
    secret_name_default = f"{get_test_hash()}-gcp-default"
    cmd = ["secrets", "create", secret_name_default, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    list_cmds = [
        ["secrets", "-A"],
        ["secrets", "--all-namespaces"],
        ["secrets", "list", "-A"],
        ["secrets", "list", "--all-namespaces"],
    ]
    expected_patterns = [
        rf".*{secret_name}.*{kt.globals.config.username}.*{tests_ns}.*"
        for secret_name in secret_names
    ]
    expected_patterns.append(
        rf".*{secret_name_default}.*{kt.globals.config.username}.*default.*"
    )
    for cmd in list_cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        for expected_pattern in expected_patterns:
            assert re.search(expected_pattern, output, re.DOTALL)


@pytest.mark.level("unit")
def test_cli_secrets_list_prefix():
    tests_ns = get_tests_namespace()
    test_hash = get_test_hash()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    secret_names = [f"{test_hash}-gcp-{tests_ns}-{i + 2}" for i in range(2)]
    for secret_name in secret_names:
        cmd = [
            "secrets",
            "create",
            secret_name,
            "--provider",
            "gcp",
            "--namespace",
            tests_ns,
        ]
        result = runner.invoke(app, cmd, color=False)
        assert result.exit_code == 0
    secret_name_default = f"{test_hash}-gcp-default1"
    cmd = ["secrets", "create", secret_name_default, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    list_cmds = [
        ["secrets", "-x", test_hash],
        ["secrets", "--prefix", test_hash],
        ["secrets", "list", "-x", test_hash],
        ["secrets", "list", "--prefix", test_hash],
    ]
    expected_pattern = (
        rf".*{secret_name_default}.*{kt.globals.config.username}.*default.*"
    )
    for cmd in list_cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        assert re.search(expected_pattern, output, re.DOTALL)


@pytest.mark.level("unit")
def test_cli_secrets_list_prefix_and_namespace():
    tests_ns = get_tests_namespace()
    test_hash = get_test_hash()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    secret_names = [f"{test_hash}-gcp-{tests_ns}-{i + 4}" for i in range(2)]
    for secret_name in secret_names:
        cmd = [
            "secrets",
            "create",
            secret_name,
            "--provider",
            "gcp",
            "--namespace",
            tests_ns,
        ]
        result = runner.invoke(app, cmd, color=False)
        assert result.exit_code == 0
    secret_name_default = f"{test_hash}-gcp-default2"
    cmd = ["secrets", "create", secret_name_default, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    list_cmds = [
        ["secrets", "-x", test_hash, "-n", tests_ns],
        ["secrets", "--prefix", test_hash, "--namespace", tests_ns],
        ["secrets", "list", "-x", test_hash, "-n", tests_ns],
        ["secrets", "list", "--prefix", test_hash, "--namespace", tests_ns],
    ]
    expected_patterns = [
        rf".*{secret_name}.*{kt.globals.config.username}.*{tests_ns}.*"
        for secret_name in secret_names
    ]
    for cmd in list_cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        for expected_pattern in expected_patterns:
            assert re.search(expected_pattern, output, re.DOTALL)


@pytest.mark.level("unit")
def test_cli_secrets_list_prefix_and_all():
    tests_ns = get_tests_namespace()
    test_hash = get_test_hash()
    create_test_ns_cmd = (
        f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    secret_names = [f"{test_hash}-gcp-{tests_ns}-{i + 4}" for i in range(2)]
    for secret_name in secret_names:
        cmd = [
            "secrets",
            "create",
            secret_name,
            "--provider",
            "gcp",
            "--namespace",
            tests_ns,
        ]
        result = runner.invoke(app, cmd, color=False)
        assert result.exit_code == 0
    secret_name_default = f"{test_hash}-gcp-default3"
    cmd = ["secrets", "create", secret_name_default, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    list_cmds = [
        ["secrets", "-x", test_hash, "-A"],
        ["secrets", "--prefix", test_hash, "--all-namespaces"],
        ["secrets", "list", "-x", test_hash, "-A"],
        ["secrets", "list", "--prefix", test_hash, "--all-namespaces"],
    ]
    expected_patterns = [
        rf".*{secret_name}.*{kt.globals.config.username}.*{tests_ns}.*"
        for secret_name in secret_names
    ]
    expected_patterns.append(
        rf".*{secret_name_default}.*{kt.globals.config.username}.*default.*"
    )
    for cmd in list_cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        for expected_pattern in expected_patterns:
            assert re.search(expected_pattern, output, re.DOTALL)


@pytest.mark.level("unit")
def test_cli_secrets_basic_delete():
    yes_flags = ["-y", "--yes"]
    for flag in yes_flags:
        secret_name = f"{get_test_hash()}-gcp-{create_random_name_prefix()}"
        create_cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
        result = runner.invoke(app, create_cmd, color=False)
        assert result.exit_code == 0
        delete_cmd = ["secrets", "delete", secret_name, flag]
        delete_result = runner.invoke(app, delete_cmd, color=False)
        assert delete_result.exit_code == 0
        delete_output = delete_result.output
        assert "Deleting 1 secret..." in delete_output
        assert f"- {secret_name}" in delete_output
        assert f"Deleted secret {secret_name}" in delete_output
        list_cmd = ["secrets"]
        list_result = runner.invoke(app, list_cmd, color=False)
        assert list_result.exit_code == 0
        assert secret_name not in list_result.output


@pytest.mark.level("unit")
def test_cli_secrets_delete_in_namespace():
    namespace_flags = ["-n", "--namespace"]
    test_ns = get_tests_namespace()
    create_test_ns_cmd = (
        f"kubectl get namespace {test_ns} || kubectl create namespace {test_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    for flag in namespace_flags:
        secret_name = f"{get_test_hash()}-gcp-{create_random_name_prefix()}"
        create_cmd = [
            "secrets",
            "create",
            secret_name,
            "--provider",
            "gcp",
            flag,
            test_ns,
        ]
        result = runner.invoke(app, create_cmd, color=False)
        assert result.exit_code == 0
        delete_cmd = ["secrets", "delete", secret_name, flag, test_ns, "-y"]
        delete_result = runner.invoke(app, delete_cmd, color=False)
        assert delete_result.exit_code == 0
        delete_output = delete_result.output
        assert "Deleting 1 secret..." in delete_output
        assert f"- {secret_name}" in delete_output
        assert f"Deleted secret {secret_name}" in delete_output
        list_cmd = ["secrets", flag, test_ns]
        list_result = runner.invoke(app, list_cmd, color=False)
        assert list_result.exit_code == 0
        assert secret_name not in list_result.output


@pytest.mark.level("unit")
def test_cli_secrets_delete_prefix():
    prefix_flags = ["-x", "--prefix"]
    for flag in prefix_flags:
        test_hash = get_test_hash()
        secret_name = f"{test_hash}-gcp-{create_random_name_prefix()}"
        create_cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
        result = runner.invoke(app, create_cmd, color=False)
        assert result.exit_code == 0
        delete_cmd = ["secrets", "delete", secret_name, flag, test_hash, "-y"]
        delete_result = runner.invoke(app, delete_cmd, color=False)
        assert delete_result.exit_code == 0
        delete_output = delete_result.output
        assert "Deleting 1 secret..." in delete_output
        assert f"- {secret_name}" in delete_output
        assert f"Deleted secret {secret_name}" in delete_output
        list_cmd = ["secrets", flag, test_hash]
        list_result = runner.invoke(app, list_cmd, color=False)
        assert list_result.exit_code == 0
        assert secret_name not in list_result.output


@pytest.mark.level("unit")
def test_cli_secrets_delete_prefix_and_namespace():
    test_ns = get_tests_namespace()
    create_test_ns_cmd = (
        f"kubectl get namespace {test_ns} || kubectl create namespace {test_ns}"
    )
    subprocess.run(create_test_ns_cmd, shell=True, check=True)
    test_hash = get_test_hash()
    secret_name = f"{test_hash}-gcp-{create_random_name_prefix()}"
    create_cmd = ["secrets", "create", secret_name, "--provider", "gcp", "-n", test_ns]
    result = runner.invoke(app, create_cmd, color=False)
    assert result.exit_code == 0
    delete_cmd = [
        "secrets",
        "delete",
        secret_name,
        "-p",
        test_hash,
        "-n",
        test_ns,
        "-y",
    ]
    delete_result = runner.invoke(app, delete_cmd, color=False)
    assert delete_result.exit_code == 0
    delete_output = delete_result.output
    assert "Deleting 1 secret..." in delete_output
    assert f"- {secret_name}" in delete_output
    assert f"Deleted secret {secret_name}" in delete_output
    list_cmd = ["secrets", "-A"]
    list_result = runner.invoke(app, list_cmd, color=False)
    assert list_result.exit_code == 0
    assert secret_name not in list_result.output


@pytest.mark.level("unit")
def test_cli_secrets_delete_few():
    test_hash = get_test_hash()
    secret_names = [f"{test_hash}-prefix-gcp-{i + 6}" for i in range(2)]
    for secret_name in secret_names:
        cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
        result = runner.invoke(app, cmd, color=False)
        assert result.exit_code == 0
    delete_cmd = ["secrets", "delete", "--prefix", f"{test_hash}-prefix", "--yes"]
    delete_result = runner.invoke(app, delete_cmd, color=False)
    assert delete_result.exit_code == 0
    delete_output = delete_result.output
    assert "Deleting 2 secrets..." in delete_output

    for secret_name in secret_names:
        assert f"- {secret_name}" in delete_output
        assert f"Deleted secret {secret_name}" in delete_output

    list_cmd = ["secrets", "--all-namespaces"]
    list_result = runner.invoke(app, list_cmd, color=False)
    assert list_result.exit_code == 0

    for secret_name in secret_names:
        assert secret_name not in list_result.output


@pytest.mark.level("unit")
def test_cli_secrets_describe_no_show():
    test_hash = get_test_hash()
    secret_name = f"{test_hash}-prefix-gcp-no-show"
    cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    cmds = [
        ["secrets", "describe", secret_name],
        ["secrets", "describe", secret_name, "-n", "default"],
    ]
    for cmd in cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "200"})
        assert result.exit_code == 0
        output = result.output
        assert f"K8 Name: {secret_name}" in output
        assert "Namespace: default" in output
        assert (
            f"Labels: {{'kubetorch.com/mount-type': 'volume', 'kubetorch.com/provider': 'gcp', "
            f"'kubetorch.com/secret-name': '{secret_name}', 'kubetorch.com/username': '{kt.config.username}'}}"
            in output
        )
        assert "Type: Opaque" in output


@pytest.mark.level("unit")
def test_cli_secrets_describe_show():
    test_hash = get_test_hash()
    secret_name = f"{test_hash}-prefix-gcp-show"
    cmd = ["secrets", "create", secret_name, "--provider", "gcp"]
    result = runner.invoke(app, cmd, color=False)
    assert result.exit_code == 0

    cmds = [
        ["secrets", "describe", secret_name, "--show"],
        ["secrets", "describe", secret_name, "-n", "default", "--show"],
    ]
    for cmd in cmds:
        result = runner.invoke(app, cmd, color=False, env={"COLUMNS": "300"})
        assert result.exit_code == 0
        output = result.output
        assert f"K8 Name: {secret_name}" in output
        assert "Namespace: default" in output
        assert (
            f"Labels: {{'kubetorch.com/mount-type': 'volume', 'kubetorch.com/provider': 'gcp', "
            f"'kubetorch.com/secret-name': '{secret_name}', 'kubetorch.com/username': '{kt.config.username}'}}"
            in output
        )
        assert "Type: Opaque" in output
