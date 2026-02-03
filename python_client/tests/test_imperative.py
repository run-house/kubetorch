import os

# Mimic CI for this test suite even locally, to ensure that
# resources are created with the branch name prefix
os.environ["CI"] = "true"

import os
import subprocess
import sys
import tempfile

import pytest

from .utils import (
    get_env_var,
    get_sys_module,
    get_test_fn_name,
    get_tests_namespace,
    OP_MUL,
    OP_SUM,
    ResourceHungryService,
    SlowNumpyArray,
    summer,
    write_temp_file_fn,
)


@pytest.fixture(autouse=True)
def setup_test_env(request):
    """Only set env vars for minimal-level tests that actually deploy to cluster."""
    marker = request.node.get_closest_marker("level")
    if marker and marker.args[0] == "minimal":
        old_launch_timeout = os.environ.get("KT_LAUNCH_TIMEOUT")
        os.environ["KT_LAUNCH_TIMEOUT"] = "150"
        yield
        # Restore original value
        if old_launch_timeout is None:
            os.environ.pop("KT_LAUNCH_TIMEOUT", None)
        else:
            os.environ["KT_LAUNCH_TIMEOUT"] = old_launch_timeout
    else:
        yield


@pytest.mark.level("unit")
def test_custom_template_dryrun():
    import kubetorch as kt

    custom_template = {
        "spec": {"template": {"spec": {"nodeSelector": {"node.kubernetes.io/instance-type": "g4dn.xlarge"}}}}
    }
    compute = kt.Compute(cpus=".1", service_template=custom_template)
    remote_fn = kt.fn(summer).to(compute, dryrun=True)

    assert remote_fn.compute == compute
    # In dryrun mode, verify the custom template was applied to the manifest
    manifest_node_selector = remote_fn.compute._manifest["spec"]["template"]["spec"].get("nodeSelector", {})
    assert manifest_node_selector.get("node.kubernetes.io/instance-type") == "g4dn.xlarge"


@pytest.mark.level("unit")
def test_default_images():
    import kubetorch as kt

    debian_img = kt.images.Debian()
    assert "ghcr.io/run-house/server" in debian_img.image_id
    assert debian_img.name == "debian"

    py_img = kt.images.Python311()
    assert py_img.image_id == "python:3.11-slim"
    assert py_img.name == "python311"

    custom_py_img = kt.images.python("3.9")
    assert custom_py_img.image_id == "python:3.9-slim"
    assert custom_py_img.name == "python39"

    ray_img = kt.images.Ray()
    assert ray_img.image_id == "rayproject/ray:latest"
    assert ray_img.name == "ray"

    custom_ray_img = kt.images.ray("2.32.0-py311")
    assert custom_ray_img.image_id == "rayproject/ray:2.32.0-py311"
    assert custom_ray_img.name == "ray2.32.0-py311"

    torch_img = kt.images.pytorch()
    assert torch_img.image_id == "nvcr.io/nvidia/pytorch:23.12-py3"
    assert torch_img.name == "pytorch2312py3"

    torch_alias = kt.images.Pytorch2312()
    assert torch_alias.image_id == "nvcr.io/nvidia/pytorch:23.12-py3"
    assert torch_alias.name == "pytorch2312py3"


@pytest.mark.level("minimal")
def test_working_dir_for_custom_image():
    import kubetorch as kt
    from kubetorch.cli_utils import get_pods_for_service_cli

    # Note: working dir will be: /usr/src/app
    remote_fn = kt.fn(summer, name="summer-working-dir").to(
        kt.Compute(
            image=kt.images.Python311(),
            working_dir="/usr/src/app",
        )
    )
    namespace = remote_fn.compute.namespace
    deploy_pods = get_pods_for_service_cli(remote_fn.service_name, namespace=namespace).get("items", [])
    deploy_pod = next(
        (
            p
            for p in deploy_pods
            if p.get("status", {}).get("phase") == "Running" and not p.get("metadata", {}).get("deletion_timestamp")
        ),
        None,
    )
    assert deploy_pod
    pod_name = deploy_pod.get("metadata", {}).get("name")
    working_dir = "."  # current working dir (set in the compute)
    check_cmd = [
        "kubectl",
        "exec",
        pod_name,
        "-n",
        namespace,
        "--",
        "ls",
        "-l",
        working_dir,
    ]
    result = subprocess.run(check_cmd, capture_output=True, text=True, check=True)
    assert "python_client" in result.stdout


@pytest.mark.level("minimal")
def test_image_copy():
    """Test various copy scenarios with different path types and options."""
    import shutil
    import tempfile
    from pathlib import Path

    import kubetorch as kt

    from tests.assets.rsync_testing.rsync_verifier import RsyncVerifier

    # Get the path to our test files
    test_assets_dir = Path(__file__).parent / "assets" / "rsync_testing" / "test_files"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Copy test files to temp directory for testing
        test_single = tmpdir / "single_file.txt"
        shutil.copy(test_assets_dir / "single_file.txt", test_single)

        test_dir = tmpdir / "test_dir"
        shutil.copytree(test_assets_dir / "test_dir", test_dir)

        # No longer creating files in home directory - use assets instead

        # Build image with all copy operations chained
        image = (
            kt.images.Debian()
            # Test 1: Single file with relative destination
            .copy(source=str(test_single), dest="relative_dest/single_file.txt")
            # Test 2: Single file with absolute destination
            .copy(source=str(test_single), dest="/data/absolute_dest/single_file.txt")
            # Test 3: Directory with contents=False (default)
            .copy(source=str(test_dir), dest="copied_dir")
            # Test 4: Directory with contents=True
            .copy(source=str(test_dir), dest="contents_only", contents=True)
            # Test 5: Absolute source and destination
            .copy(source=str(tmpdir.absolute()), dest="/data/copy_test", contents=True)
            # Test 6: Tilde in destination (should be treated as relative)
            .copy(
                source=str(test_assets_dir / "home_test.txt"),
                dest="~/from_home/home_test.txt",
            )
            # Test 7: No dest specified for a file (should use basename)
            .copy(source=str(test_assets_dir / "single_file.txt"))
            # Test 8: No dest specified for a directory (should use directory name)
            .copy(source=str(test_dir))
            # Test 9: No dest specified with absolute path source
            .copy(source=str(tmpdir / "single_file.txt"))
        )

        # Deploy verifier class to test all scenarios
        remote_verifier = kt.cls(RsyncVerifier, name=f"{get_test_fn_name()}_verifier").to(
            kt.Compute(
                cpus=".01",
                gpu_anti_affinity=True,
                image=image,
            )
        )

        # Run all verification tests
        results = remote_verifier.test_all_scenarios()

        # Verify test 1: Single file with relative destination
        assert results["single_relative"]["exists"], "Single file with relative path not found"
        assert results["single_relative"]["content_match"], "Single file content doesn't match"

        # Verify test 2: Single file with absolute destination
        assert results["single_absolute"]["exists"], "Single file with absolute path not found"
        assert results["single_absolute"]["content_match"], "Absolute file content doesn't match"

        # Verify test 3: Directory with contents=False
        dir_result = results["dir_no_contents"]
        assert dir_result["base_exists"], "Copied directory not found"
        assert all(
            f["exists"] for f in dir_result["files"].values()
        ), f"Missing files in copied dir: {dir_result['files']}"

        # Verify test 4: Directory with contents=True
        contents_result = results["dir_with_contents"]
        assert contents_result["base_exists"], "Contents directory not found"
        assert all(
            f["exists"] for f in contents_result["files"].values()
        ), f"Missing files in contents dir: {contents_result['files']}"

        # Verify test 5: Absolute paths
        abs_result = results["absolute_both"]
        assert abs_result["base_exists"], "Absolute destination directory not found"
        expected_files = [
            "single_file.txt",
            "test_dir/file1.txt",
            "test_dir/nested/deep.txt",
        ]
        for expected in expected_files:
            assert abs_result["files"][expected]["exists"], f"Missing {expected} in absolute path copy"

        # Verify test 6: Tilde expansion
        assert results["tilde_home"]["exists"], "File from home directory not found"
        assert results["tilde_home"]["content_match"], "Home file content doesn't match"

        # Verify test 7: No dest for file (should use basename)
        assert results["no_dest_file"]["exists"], "File without dest not found at basename location"
        assert results["no_dest_file"]["content_match"], "No-dest file content doesn't match"

        # Verify test 8: No dest for directory (should use directory name)
        no_dest_dir = results["no_dest_dir"]
        assert no_dest_dir["base_exists"], "Directory without dest not found"
        assert all(
            f["exists"] for f in no_dest_dir["files"].values()
        ), f"Missing files in no-dest dir: {no_dest_dir['files']}"

        # Verify test 9: No dest with absolute path (should use basename)
        assert results["no_dest_absolute"]["exists"], "Absolute source without dest not found"
        assert results["no_dest_absolute"]["content_match"], "Absolute no-dest file content doesn't match"


@pytest.mark.level("minimal")
def test_autoscaling_fn():
    import kubetorch as kt

    replicas = 2

    compute = kt.Compute(cpus=".01", gpu_anti_affinity=True,).autoscale(
        min_scale=replicas,
        max_scale=replicas,
        concurrency=1,  # Set concurrency to 1 to ensure each pod handles one request at a time
    )

    remote_fn = kt.fn(summer, name=get_test_fn_name()).to(compute)

    num_requests = 4

    # Make calls simultaneously to override queuing
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    with ThreadPoolExecutor(max_workers=replicas) as executor:
        results = list(
            executor.map(
                partial(remote_fn, 1, return_pod_name=True, sleep_time=0.25),
                [2] * num_requests,
            )
        )
        sums, pod_names = zip(*results)

    assert sum(sums) == 3 * num_requests
    assert len(pod_names) >= replicas


@pytest.mark.level("minimal")
def test_declarative_fn_freeze():
    import kubetorch as kt
    from kubetorch.utils import string_to_dict

    from .conftest import KUBETORCH_IMAGE

    # Note: use a function that will be baked into the image (here an arbitrary kubetorch utils helper)
    remote_fn = kt.fn(string_to_dict, name=get_test_fn_name()).to(
        kt.Compute(
            cpus=".1",
            image=kt.Image(image_id=KUBETORCH_IMAGE),
            freeze=True,
            gpu_anti_affinity=True,
        )
    )
    assert remote_fn('{"a": 1}') == {"a": 1}


@pytest.mark.level("minimal")
def test_refresh_fn_cache():
    import kubetorch as kt

    in_a, in_b = 2, 3
    name = f"{get_test_fn_name()}-op"
    compute = kt.Compute(
        cpus=".01",
        env_vars={"OMP_NUM_THREADS": 1},
        gpu_anti_affinity=True,
    )

    try:
        # create temp file
        test_dir = os.path.dirname(os.path.abspath(__file__))
        temp_fd, temp_path = tempfile.mkstemp(dir=test_dir, suffix=".py")
        module_name = os.path.basename(temp_path)[:-3]

        # write original function
        write_temp_file_fn(temp_fd, temp_path, fn_contents=OP_SUM)
        temp_module = get_sys_module(module_name, temp_path)

        remote_fn = kt.fn(temp_module.operation, name=name).to(compute)
        assert remote_fn(in_a, in_b) == in_a + in_b

        # update function contents
        temp_fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        write_temp_file_fn(temp_fd, temp_path, fn_contents=OP_MUL)
        temp_module = get_sys_module(module_name, temp_path)

        remote_fn = kt.fn(temp_module.operation, name=name).to(compute)
        assert remote_fn(in_a, in_b) == in_a * in_b
    finally:
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            print(f"Deleting temp fn file {temp_path}")
            os.unlink(temp_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {temp_path}: {e}")


@pytest.mark.level("minimal")
def test_image_update():
    import kubetorch as kt

    image = kt.images.Debian().set_env_vars({"empty": "", "val": "old_val", "val2": "some_other_val"})
    compute = kt.Compute(
        cpus=".01",
        image=image,
        env_vars={"OMP_NUM_THREADS": 1},
        gpu_anti_affinity=True,
    )

    remote_fn = kt.fn(get_env_var, name=get_test_fn_name()).to(compute)
    assert remote_fn("val") == "old_val"
    assert remote_fn("val2") == "some_other_val"
    assert remote_fn("empty") == ""

    compute.image = kt.images.Debian().set_env_vars({"val": "new_val"})
    remote_fn = kt.fn(get_env_var, name=get_test_fn_name()).to(compute)
    assert remote_fn("val") == "new_val"
    assert remote_fn("val2") == "some_other_val"


@pytest.mark.level("minimal")
def test_env_var_expansion():
    """Test that environment variables are properly expanded in Image.set_env_vars()."""
    import kubetorch as kt

    # Set up an image with env vars that reference other env vars
    image = kt.images.Debian().set_env_vars(
        {
            "BASE_PATH": "/usr/local",
            "EXTENDED_PATH": "$BASE_PATH/bin:$PATH",  # Should expand BASE_PATH and PATH
            "LD_LIBRARY_PATH": "/opt/lib:${LD_LIBRARY_PATH}",  # Should expand or be empty if not set
            "CUSTOM_VAR": "prefix_${BASE_PATH}_suffix",  # Should expand BASE_PATH with braces
        }
    )

    compute = kt.Compute(
        cpus=".01",
        image=image,
        gpu_anti_affinity=True,
    )

    remote_fn = kt.fn(get_env_var, name=get_test_fn_name()).to(compute)

    # Test that BASE_PATH is set correctly
    assert remote_fn("BASE_PATH") == "/usr/local"

    # Test that EXTENDED_PATH expands $BASE_PATH and includes existing PATH
    extended_path = remote_fn("EXTENDED_PATH")
    assert extended_path.startswith("/usr/local/bin:")
    assert "/usr/bin" in extended_path  # PATH should contain standard directories

    # Test that LD_LIBRARY_PATH handles expansion (even if original is empty)
    ld_library_path = remote_fn("LD_LIBRARY_PATH")
    assert ld_library_path.startswith("/opt/lib")

    # Test that CUSTOM_VAR expands ${BASE_PATH} with braces
    custom_var = remote_fn("CUSTOM_VAR")
    assert custom_var == "prefix_/usr/local_suffix"


@pytest.mark.level("minimal")
def test_global_kt_cache():
    import kubetorch as kt
    from kubetorch.constants import KT_MOUNT_FOLDER

    compute = kt.Compute(
        cpus=".01",
        env_vars={
            "OMP_NUM_THREADS": 1,
            "UV_CACHE_DIR": f"/{KT_MOUNT_FOLDER}/kt-global-cache/uv_cache",
            "HF_HOME": f"/{KT_MOUNT_FOLDER}/kt-global-cache/hf_cache",
        },
        gpu_anti_affinity=True,
        volumes=[
            kt.Volume(
                "kt-global-cache",
                size="10Gi",
                access_mode="ReadWriteOnce",
                mount_path=f"/{KT_MOUNT_FOLDER}/kt-global-cache",
            )
        ],
    )

    remote_fn = kt.fn(get_env_var, name=get_test_fn_name()).to(compute)
    result1 = remote_fn.compute.run_bash(
        f"uv pip install pandas --system --cache-dir /{KT_MOUNT_FOLDER}/kt-global-cache/uv_cache -v"
    )
    stdout = "".join(line[1] for line in result1 if len(line) > 1)
    # Check that pandas and numpy were installed (other deps may already exist in base image)
    assert "Installed" in stdout and "pandas" in stdout

    result2 = remote_fn.compute.run_bash(
        f"uv pip install pandas --system --cache-dir /{KT_MOUNT_FOLDER}/kt-global-cache/uv_cache -v"
    )
    stdout = "".join(line[1] for line in result2 if len(line) > 1)
    assert "Requirement satisfied" in stdout

    result3 = remote_fn.compute.run_bash(
        f"ls -l /{KT_MOUNT_FOLDER}/kt-global-cache/uv_cache/wheels-v5/pypi | grep pandas"
    )
    stdout = "".join(line[1] for line in result3 if len(line) > 1)
    assert "pandas" in stdout


@pytest.mark.level("minimal")
def test_notebook_fn():
    from pathlib import Path
    from unittest.mock import patch

    import kubetorch as kt

    def notebook_fn(a, b):
        return a + b

    fn_file = Path.cwd() / f"{notebook_fn.__name__}_fn.py"
    try:
        # Patch _extract_module_path to return None, simulating notebook/ipython (no __file__)
        with patch("kubetorch.resources.callables.utils._extract_module_path", return_value=None):
            compute = kt.Compute(cpus=".01", gpu_anti_affinity=True)
            remote_fn = kt.fn(notebook_fn, name=get_test_fn_name()).to(compute)

            assert fn_file.exists()
            assert remote_fn(1, 2) == 3
    finally:
        if fn_file.exists():
            fn_file.unlink()


# --------- Error handling tests ---------
@pytest.mark.skip("Too slow for CI but useful for manual testing")
@pytest.mark.level("minimal")
def test_callable_service_timeout_error():
    import kubetorch as kt

    with pytest.raises(kt.ServiceTimeoutError):
        remote_cls = kt.cls(SlowNumpyArray, name=get_test_fn_name()).to(
            kt.Compute(
                cpus=".1",
                env_vars={"OMP_NUM_THREADS": 1},
                gpu_anti_affinity=True,
                launch_timeout=60,  # default 120
            ),
            init_args={"size": 10, "sleep_time": 120},
        )
        remote_cls.print_and_log(1)


@pytest.mark.level("minimal")
def test_callable_launch_timeout():
    import kubetorch as kt

    with pytest.raises(kt.ServiceTimeoutError):
        remote_cls = kt.cls(SlowNumpyArray, name=get_test_fn_name()).to(
            kt.Compute(
                cpus=".1", launch_timeout=5, gpu_anti_affinity=True, image=kt.images.Python312().pip_install(["numpy"])
            ),
            init_args={"size": 10},
        )
        remote_cls.print_and_log(1)


@pytest.mark.level("minimal")
def test_image_pull_error_deployment():
    import kubetorch as kt

    with pytest.raises(kt.ImagePullError) as exc_info:
        remote_cls = kt.cls(SlowNumpyArray, name=get_test_fn_name()).to(
            kt.Compute(
                cpus=".1",
                image=kt.Image(image_id="nonexistent/image:latest"),
                gpu_anti_affinity=True,
                launch_timeout=60,
            ),
            init_args={"size": 10},
        )
        remote_cls.print_and_log(1)

    assert "failed to pull" in str(exc_info.value)


@pytest.mark.level("minimal")
def test_image_pull_error_knative():
    import kubetorch as kt

    with pytest.raises(kt.ImagePullError) as exc_info:
        remote_cls = kt.cls(SlowNumpyArray, name=get_test_fn_name()).to(
            kt.Compute(
                cpus=".1",
                image=kt.Image(image_id="nonexistent/image:latest"),
                gpu_anti_affinity=True,
                launch_timeout=60,
            ).autoscale(min_replicas=1),
            init_args={"size": 10},
        )
        remote_cls.print_and_log(1)

    assert "failed to pull" in str(exc_info.value)


@pytest.mark.skip("Skipping in CI - keeping for manual verification of expected K8s behavior")
@pytest.mark.level("minimal")
def test_unschedulable_pod_knative():
    import kubetorch as kt

    # Requesting 1000 CPUs and 0.1 memory should be unschedulable
    name = get_test_fn_name()
    namespace = kt.globals.config.namespace
    service_name = f"{kt.config.username}-{name}"

    with pytest.raises(kt.ServiceTimeoutError):
        remote_cls = kt.cls(ResourceHungryService, name=name).to(
            kt.Compute(cpus="1000", memory="0.1", gpu_anti_affinity=True, launch_timeout=40).autoscale(min_replicas=1)
        )
        remote_cls.some_method()

    # Verify pod is actually Unschedulable (not some other issue)
    controller = kt.globals.controller_client()
    pods_result = controller.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={service_name}",
    )
    pods = pods_result.get("items", [])

    assert len(pods) > 0, "Expected at least one pod to be created"
    pod = pods[0]

    # Check for Unschedulable condition
    conditions = pod.get("status", {}).get("conditions", [])
    unschedulable = any(
        c.get("type") == "PodScheduled" and c.get("status") == "False" and c.get("reason") == "Unschedulable"
        for c in conditions
    )
    assert unschedulable, f"Expected pod to be Unschedulable, got conditions: {conditions}"


@pytest.mark.skip("Skipping in CI - keeping for manual verification of expected K8s behavior")
@pytest.mark.level("minimal")
def test_unschedulable_pod_deployment():
    import kubetorch as kt

    # Requesting 1000 CPUs and 0.1 memory should be unschedulable
    name = get_test_fn_name()
    namespace = kt.globals.config.namespace
    service_name = f"{kt.config.username}-{name}"

    with pytest.raises(kt.ServiceTimeoutError):
        remote_cls = kt.cls(ResourceHungryService, name=name).to(
            kt.Compute(cpus="1000", memory="0.1", gpu_anti_affinity=True, launch_timeout=40)
        )
        remote_cls.some_method()

    # Verify pod is actually Unschedulable (not some other issue)
    controller = kt.globals.controller_client()
    pods_result = controller.list_pods(
        namespace=namespace,
        label_selector=f"kubetorch.com/service={service_name}",
    )
    pods = pods_result.get("items", [])

    assert len(pods) > 0, "Expected at least one pod to be created"
    pod = pods[0]

    # Check for Unschedulable condition
    conditions = pod.get("status", {}).get("conditions", [])
    unschedulable = any(
        c.get("type") == "PodScheduled" and c.get("status") == "False" and c.get("reason") == "Unschedulable"
        for c in conditions
    )
    assert unschedulable, f"Expected pod to be Unschedulable, got conditions: {conditions}"


@pytest.mark.skip("Skipping in CI - keeping for manual verification of expected K8s behavior")
@pytest.mark.level("minimal")
def test_pod_terminated_error():
    import time

    import kubetorch as kt

    # Test that pod gets terminated when it runs out of disk space
    # Pod starts successfully, error happens during consume_disk()
    name = get_test_fn_name()
    namespace = kt.globals.config.namespace
    service_name = f"{kt.config.username}-{name}"

    # Use 500Mi limit so the service can start, then consume_disk writes 50MB files which will exceed the limit and
    # trigger eviction
    disk_size = "500Mi"
    with pytest.raises(Exception):
        remote_cls = kt.cls(ResourceHungryService, name=name).to(
            kt.Compute(cpus="0.1", disk_size=disk_size, disk_limit=disk_size, gpu_anti_affinity=True, launch_timeout=90)
        )
        remote_cls.consume_disk()

    # Poll for eviction
    controller = kt.globals.controller_client()
    any_evicted = False

    for i in range(18):
        pods_result = controller.list_pods(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )
        pods = pods_result.get("items", [])

        if not pods:
            break  # All pods deleted

        # Check if ANY pod shows eviction/termination
        any_evicted = any(
            pod.get("status", {}).get("phase") == "Failed"
            or pod.get("status", {}).get("reason") == "Evicted"
            or any(
                cs.get("state", {}).get("terminated") is not None
                for cs in pod.get("status", {}).get("containerStatuses", [])
            )
            for pod in pods
        )
        if any_evicted:
            break
        time.sleep(5)

    assert (
        any_evicted
    ), f"Expected at least one pod to be evicted, got pods: {[p.get('status', {}).get('phase') for p in pods]}"


@pytest.mark.skip("Skipping in CI - keeping for manual verification of expected K8s behavior")
@pytest.mark.level("minimal")
def test_pod_oom_error_after_startup():
    import time

    import kubetorch as kt

    # Test that pod gets OOM killed when it runs out of memory
    # Pod starts successfully, OOM happens during consume_memory()
    name = get_test_fn_name()
    namespace = kt.globals.config.namespace
    service_name = f"{kt.config.username}-{name}"

    # Use 256Mi so the service can start (Python + server needs ~100-150MB),
    # then consume_memory allocates 50MB per iteration to quickly exceed limit
    memory = "256Mi"
    with pytest.raises(Exception):
        remote_cls = kt.cls(ResourceHungryService, name=name).to(
            kt.Compute(
                cpus="0.1",
                memory=memory,
                memory_limit=memory,
                gpu_anti_affinity=True,
                launch_timeout=90,
            )
        )
        # Service should start fine; OOM happens only when calling consume_memory
        remote_cls.consume_memory()

    # Poll for OOM killed status
    controller = kt.globals.controller_client()
    any_oom = False

    for i in range(18):  # Wait up to 90 seconds
        pods_result = controller.list_pods(
            namespace=namespace,
            label_selector=f"kubetorch.com/service={service_name}",
        )
        pods = pods_result.get("items", [])

        if not pods:
            break

        # Check if any pod shows OOMKilled
        any_oom = any(
            any(
                cs.get("state", {}).get("terminated", {}).get("reason") == "OOMKilled"
                or cs.get("lastState", {}).get("terminated", {}).get("reason") == "OOMKilled"
                for cs in pod.get("status", {}).get("containerStatuses", [])
            )
            for pod in pods
        )
        if any_oom:
            break
        time.sleep(5)

    assert any_oom, f"Expected at least one pod to be OOMKilled, got pods: {[p.get('status', {}) for p in pods]}"


@pytest.mark.level("minimal")
def test_default_allowed_serialization(remote_logs_fn):
    msg = "Default serialization test"
    n = 5
    expected_result = f"{msg} was logged {n} times"
    result_valid_serialization = remote_logs_fn(msg=msg, n=n, serialization="json")
    assert result_valid_serialization == expected_result

    with pytest.raises(Exception, match="Serialization format 'pickle' not allowed"):
        remote_logs_fn(msg=msg, n=n, serialization="pickle")


@pytest.mark.level("minimal")
def test_allowed_serialization_as_env_env_var(monkeypatch):
    monkeypatch.setenv("KT_ALLOWED_SERIALIZATION", "json,pickle")

    import kubetorch as kt

    from .utils import log_n_messages

    compute = kt.Compute(cpus=".01", gpu_anti_affinity=True, launch_timeout=300)
    remote_logs_fn = kt.fn(log_n_messages, name="multiple_serialization_fn").to(compute)

    n = 5

    msg_json = "JSON serialization test"
    expected_result_json = f"{msg_json} was logged {n} times"
    result_json_serialization = remote_logs_fn(msg=msg_json, n=n, serialization="json")
    assert result_json_serialization == expected_result_json

    msg_pickle = "Pickle serialization test"
    expected_result_pickle = f"{msg_pickle} was logged {n} times"
    result_pickle_serialization = remote_logs_fn(msg=msg_pickle, n=n, serialization="pickle")
    assert result_pickle_serialization == expected_result_pickle


@pytest.mark.level("minimal")
def test_module_teardown():
    from tests.test_cli import validate_service_not_in_kt_list
    from .utils import remote_cls_for_teardown, remote_fn_for_teardown

    fn_to_delete = remote_fn_for_teardown()
    fn_to_delete.teardown()
    validate_service_not_in_kt_list(fn_to_delete.service_name)

    cls_to_delete = remote_cls_for_teardown()
    cls_to_delete.teardown()
    validate_service_not_in_kt_list(cls_to_delete.service_name)


@pytest.mark.level("unit")
def test_compute_factory_cpus():
    import kubetorch as kt

    compute_str_as_int_cpus_core = kt.Compute(cpus="1", launch_timeout=300)
    assert compute_str_as_int_cpus_core
    assert compute_str_as_int_cpus_core.cpus == "1"
    compute_str_as_float_cpus_core = kt.Compute(cpus="0.5", launch_timeout=300)
    assert compute_str_as_float_cpus_core
    assert compute_str_as_float_cpus_core.cpus == "0.5"
    compute_str_cpus_millicores = kt.Compute(cpus="2000m", launch_timeout=300)
    assert compute_str_cpus_millicores
    assert compute_str_cpus_millicores.cpus == "2000m"
    compute_int_cpus = kt.Compute(cpus=3, launch_timeout=300)
    assert compute_int_cpus
    assert compute_int_cpus.cpus == "3"


@pytest.mark.level("unit")
def test_compute_factory_memory():
    import kubetorch as kt

    compute_str_as_int_mem = kt.Compute(memory="1000000")
    assert compute_str_as_int_mem
    assert compute_str_as_int_mem.memory == "1000000"

    binary_units = ["Ki", "Mi", "Gi", "Ti"]
    for unit in binary_units:
        compute_binary_units = kt.Compute(memory=f"2{unit}")
        assert compute_binary_units
        assert compute_binary_units.memory == f"2{unit}"

    decimal_units = ["K", "M", "G", "T"]
    for unit in decimal_units:
        compute_binary_units = kt.Compute(memory=f"3{unit}")
        assert compute_binary_units
        assert compute_binary_units.memory == f"3{unit}"


@pytest.mark.level("unit")
def test_compute_factory_disk_size():
    import kubetorch as kt

    compute_str_as_int_disk_size = kt.Compute(disk_size="1000000")
    assert compute_str_as_int_disk_size
    assert compute_str_as_int_disk_size.disk_size == "1000000"

    binary_units = ["Ki", "Mi", "Gi", "Ti"]
    for unit in binary_units:
        compute_binary_units = kt.Compute(disk_size=f"2{unit}")
        assert compute_binary_units
        assert compute_binary_units.disk_size == f"2{unit}"

    decimal_units = ["K", "M", "G", "T"]
    for unit in decimal_units:
        compute_binary_units = kt.Compute(disk_size=f"3{unit}")
        assert compute_binary_units
        assert compute_binary_units.disk_size == f"3{unit}"


@pytest.mark.level("unit")
def test_compute_factory_gpus():
    import kubetorch as kt

    compute_str_as_int_gpus_core = kt.Compute(gpus="1")
    assert compute_str_as_int_gpus_core
    assert compute_str_as_int_gpus_core.gpus == "1"
    compute_int_gpus = kt.Compute(gpus=3)
    assert compute_int_gpus
    assert compute_int_gpus.gpus == "3"


@pytest.mark.level("unit")
def test_compute_factory_gpu_type():
    import kubetorch as kt

    compute_gpus_full = kt.Compute(gpu_type="nvidia.com/gpu.product: L4")
    assert compute_gpus_full
    assert compute_gpus_full.gpu_type == "L4"

    compute_gpus_short = kt.Compute(gpu_type="NVIDIA-L4")
    assert compute_gpus_short
    assert compute_gpus_short.gpu_type == "NVIDIA-L4"


@pytest.mark.level("unit")
def test_compute_factory_gpu_memory():
    import kubetorch as kt

    binary_units = ["Gi", "Mi", "Ti"]

    for unit, value in binary_units:
        compute_binary_units = kt.Compute(gpus=1, gpu_memory=f"2{unit}")
        assert compute_binary_units.gpu_memory == f"2{unit}"
        assert compute_binary_units.gpu_annotations.get("gpu-memory") == f"2{unit}"


@pytest.mark.level("unit")
def test_compute_factory_namespace():
    import kubetorch as kt

    default_ns_compute = kt.Compute(cpus="1")
    assert default_ns_compute
    assert default_ns_compute.namespace == kt.config.namespace

    tests_ns = get_tests_namespace()
    create_test_ns_cmd = f"kubectl get namespace {tests_ns} || kubectl create namespace {tests_ns}"
    subprocess.run(create_test_ns_cmd, shell=True, check=True)

    provided_ns_compute = kt.Compute(cpus="1", namespace=tests_ns)
    assert provided_ns_compute
    assert provided_ns_compute.namespace == tests_ns


@pytest.mark.level("unit")
def test_compute_factory_image():
    import kubetorch as kt

    image = kt.images.Debian()
    compute_valid_image = kt.Compute(image=image)
    assert compute_valid_image.image.image_id == image.image_id
    assert compute_valid_image.image.name == image.name
    with pytest.raises(AttributeError) as error:
        kt.Compute(image="ghcr.io/run-house/server:v3")
    assert error.value.args[0] == "'str' object has no attribute 'image_id'"


@pytest.mark.level("unit")
def test_compute_factory_kubeconfig_path():
    from pathlib import Path

    import kubetorch as kt
    import yaml

    compute_default_path = kt.Compute(cpus="1")
    assert compute_default_path
    assert compute_default_path.kubeconfig_path == str(Path("~/.kube/config").expanduser())

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_file = Path(tmpdir) / ".kt" / "config"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with temp_file.expanduser().open("w") as stream:
            config_vals = {k: str(v) if isinstance(v, dict) else v for k, v in dict(kt.config).items() if v is not None}
            yaml.safe_dump(config_vals, stream)
        compute_kubeconfig_path = kt.Compute(cpus="1", kubeconfig_path=str(temp_file.expanduser()))
        assert compute_kubeconfig_path
        assert compute_kubeconfig_path.kubeconfig_path == str(temp_file.expanduser())


@pytest.mark.level("unit")
def test_compute_factory_service_account():
    import kubetorch as kt

    compute_sa = kt.Compute(cpus="1", service_account_name="test_service_account")
    assert compute_sa
    assert compute_sa.service_account_name == "test_service_account"


@pytest.mark.level("unit")
def test_compute_factory_image_pull_policy():
    import kubetorch as kt

    image_pull_policies = ["IfNotPresent", "Always", "Never"]

    for policy in image_pull_policies:
        compute = kt.Compute(cpus="0.1", image_pull_policy=policy)
        assert compute.image_pull_policy == policy


@pytest.mark.level("unit")
def test_compute_factory_shared_memory_limit():
    import kubetorch as kt

    binary_units = ["Gi", "Mi", "Ti"]
    for unit in binary_units:
        compute_binary_units = kt.Compute(gpus="1", shared_memory_limit=f"2{unit}")
        assert compute_binary_units
        assert compute_binary_units.shared_memory_limit == f"2{unit}"


@pytest.mark.level("minimal")
def test_compute_nonexisting_priority_class():
    import kubetorch as kt

    from .utils import summer

    priority_class_name = "random-priority-class"
    with pytest.raises(kt.ResourceNotAvailableError):
        my_compute = kt.Compute(cpus="0.1", priority_class_name=priority_class_name)
        remote_fn = kt.fn(summer, name="random-priority-class-summer").to(my_compute)

        assert remote_fn(4, 5) == 9


@pytest.mark.level("minimal")
def test_single_file_sync_no_project_markers():
    """Test that deploying a function from a standalone file (no project markers)
    only syncs that single file, not the entire directory.

    This mimics: python ~/Downloads/helloworld.py
    Where ~/Downloads has no .git, pyproject.toml, etc.
    """
    import importlib.util
    from pathlib import Path

    import kubetorch as kt

    # Create isolated temp directory with ONLY a single .py file (no project markers)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a standalone Python file
        script_path = Path(tmpdir) / "standalone_script.py"
        script_path.write_text(
            '''
def standalone_fn(x, y):
    """A function in a standalone file with no project markers."""
    return x + y
'''
        )

        # Also create a decoy file that should NOT be synced
        decoy_file = Path(tmpdir) / "decoy.txt"
        decoy_file.write_text("This file should NOT be synced")

        # Import the function from the standalone file
        spec = importlib.util.spec_from_file_location("standalone_script", script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["standalone_script"] = module
        spec.loader.exec_module(module)

        try:
            compute = kt.Compute(cpus=".01", gpu_anti_affinity=True)
            remote_fn = kt.fn(module.standalone_fn, name="standalone-file-test").to(compute)

            # Verify function works
            result = remote_fn(3, 4)
            assert result == 7, f"Function should return 7, got {result}"

            # Verify only the .py file was synced, not the decoy
            # Check via bash command on the remote
            # run_bash returns [[exit_code, stdout, stderr]] structure
            ls_result = remote_fn.compute.run_bash("ls -la")
            ls_output = str(ls_result)  # Convert to string for simple containment check
            assert "standalone_script.py" in ls_output, "Script file should be synced"
            assert "decoy.txt" not in ls_output, "Decoy file should NOT be synced (single-file mode)"

        finally:
            sys.modules.pop("standalone_script", None)
