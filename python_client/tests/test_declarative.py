import asyncio
import os
import subprocess
import time

import kubetorch as kt

import pytest
from kubetorch.globals import config


@kt.compute(cpus=".1")
def single_test_function():
    return os.environ["KT_SERVICE_NAME"]


@pytest.mark.level("minimal")
@pytest.mark.asyncio
async def test_declarative_fn():
    # Test declarative deployment with multiple @kt.compute decorators. Test that:
    # 1. Multiple modules are deployed asynchronously (faster than sequential)
    # 2. All deployed functions/classes work correctly
    # 3. The deployments teardown correctly

    # For debugging, uncomment this:
    # from kubetorch.cli import deploy
    # deploy(tests.assets.decorated_modules.decorated_modules.__file__)

    # Need to set the KT_USERNAME environment variable prior to importing deployed modules.
    # This ensures prefixing at import time and also flows through into the deploy call
    os.environ["KT_USERNAME"] = config.username

    import tests.assets.decorated_modules.decorated_modules
    from tests.assets.decorated_modules.decorated_modules import get_pod_id_1, get_pod_id_async, RemoteArray

    # Single module deployment time
    single_module_file = __file__
    start_time = time.time()
    subprocess.run(
        f"kt deploy {single_module_file}",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    single_time = time.time() - start_time

    # Multiple module deployment time
    start_time = time.time()
    subprocess.run(
        f"kt deploy {tests.assets.decorated_modules.decorated_modules.__file__}",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    deployment_time = time.time() - start_time

    # Async deployment should be much faster than 3x sequential
    expected_sequential_time = single_time * 3
    assert deployment_time < expected_sequential_time / 2

    assert get_pod_id_1() == get_pod_id_1.service_name

    assert RemoteArray.get_data() == [0] * 5

    RemoteArray.set_len(20)
    assert RemoteArray.get_data() == [0] * 20

    RemoteArray.set_len(5)
    assert RemoteArray.get_data() == [0] * 5

    # Test async function
    coroutine = get_pod_id_async()
    assert asyncio.iscoroutine(coroutine)
    result = await coroutine
    assert result == get_pod_id_async.service_name

    # Uncomment to debug
    # from kubetorch.cli import kt_teardown
    # kt_teardown(tests.assets.decorated_modules.decorated_modules.__file__, yes=True)

    teardown_cmd = f"kt teardown {tests.assets.decorated_modules.decorated_modules.__file__} --yes"
    subprocess.run(teardown_cmd, shell=True, check=True)
    teardown_single = f"kt teardown {single_module_file} --yes"
    subprocess.run(teardown_single, shell=True, check=True)

    time.sleep(3)

    # Confirm that the service is no longer up
    with pytest.raises(Exception):
        get_pod_id_1()
    with pytest.raises(Exception):
        RemoteArray.get_data()
    with pytest.raises(Exception):
        await get_pod_id_async()
