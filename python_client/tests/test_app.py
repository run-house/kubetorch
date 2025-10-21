import os
import subprocess

import pytest

from kubetorch.globals import config

from tests.assets.app import summer_app
from tests.conftest import get_test_hash


@pytest.mark.level("minimal")
def test_run_app_base():
    username = get_test_hash()
    os.environ["KT_USERNAME"] = username
    service_name = f"{username}-app"

    a, b = 2, 3
    cmd = f"kt run python {summer_app.__file__} {a} {b} --name {service_name}"
    org_stream_logs = config.stream_logs

    try:
        config.set("stream_logs", True)
        result = subprocess.run(
            cmd, shell=True, text=True, check=True, capture_output=True
        ).stdout

        assert f"Hello from the cluster stdout! {a} {b}" in result
        assert f"Hello from the cluster logs! {a} {b}" in result
    except Exception as e:
        raise e
    finally:
        config.set("stream_logs", org_stream_logs)


@pytest.mark.level("minimal")
def test_run_app_nested():
    username = get_test_hash()
    os.environ["KT_USERNAME"] = username
    service_name = f"{username}-app-nested"

    a, b = 2, 3
    cmd = f"kt run python {summer_app.__file__} --nested {a} {b} --name {service_name}"
    org_stream_logs = config.stream_logs

    try:
        config.set("stream_logs", True)
        result = subprocess.run(
            cmd, shell=True, text=True, check=True, capture_output=True
        ).stdout

        assert f"result: {a+b}" in result
        assert f"Hello from the cluster stdout! {a} {b}" in result
        assert f"Hello from the cluster logs! {a} {b}" in result
    except Exception as e:
        raise e
    finally:
        config.set("stream_logs", org_stream_logs)
