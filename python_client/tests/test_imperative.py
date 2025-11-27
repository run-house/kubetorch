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
    OP_MUL,
    OP_SUM,
    ResourceHungryService,
    SlowNumpyArray,
    summer,
    write_temp_file_fn,
)


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Keep the launch timeout low for this test suite, unless overridden (ex: for GPU tests)
    os.environ["KT_LAUNCH_TIMEOUT"] = "120"
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
    assert remote_fn.service_config["spec"]["template"].spec.node_selector == {
        "node.kubernetes.io/instance-type": "g4dn.xlarge"
    }


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
    from kubernetes.client import CoreV1Api
    from kubernetes.config import load_kube_config
    from kubetorch.cli_utils import get_pods_for_service_cli

    load_kube_config()
    core_api = CoreV1Api()

    # Note: working dir will be: /usr/src/app
    remote_fn = kt.fn(summer, name="summer-working-dir").to(
        kt.Compute(
            image=kt.images.Python311(),
            working_dir="/usr/src/app",
        )
    )
    namespace = remote_fn.compute.namespace
    deploy_pods = get_pods_for_service_cli(remote_fn.service_name, namespace=namespace, core_api=core_api).items
    deploy_pod = next(
        (p for p in deploy_pods if p.status.phase == "Running" and not p.metadata.deletion_timestamp),
        None,
    )
    assert deploy_pod
    pod_name = deploy_pod.metadata.name
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
def test_image_rsync():
    """Test various rsync scenarios with different path types and options."""
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

        # Build image with all rsync operations chained
        image = (
            kt.images.Debian()
            # Test 1: Single file with relative destination
            .rsync(source=str(test_single), dest="relative_dest/single_file.txt")
            # Test 2: Single file with absolute destination
            .rsync(source=str(test_single), dest="/data/absolute_dest/single_file.txt")
            # Test 3: Directory with contents=False (default)
            .rsync(source=str(test_dir), dest="copied_dir")
            # Test 4: Directory with contents=True
            .rsync(source=str(test_dir), dest="contents_only", contents=True)
            # Test 5: Absolute source and destination
            .rsync(source=str(tmpdir.absolute()), dest="/data/rsync_test", contents=True)
            # Test 6: Tilde in destination (should be treated as relative)
            .rsync(
                source=str(test_assets_dir / "home_test.txt"),
                dest="~/from_home/home_test.txt",
            )
            # Test 7: No dest specified for a file (should use basename)
            .rsync(source=str(test_assets_dir / "single_file.txt"))
            # Test 8: No dest specified for a directory (should use directory name)
            .rsync(source=str(test_dir))
            # Test 9: No dest specified with absolute path source
            .rsync(source=str(tmpdir / "single_file.txt"))
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
            assert abs_result["files"][expected]["exists"], f"Missing {expected} in absolute rsync"

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
    # if you run the test on a EKS cluster, use the following image_id:
    # image_id="your-account.dkr.ecr.us-east-1.amazonaws.com/kubetorch-client:main"

    import kubetorch as kt

    # Note: provided image has kubetorch already installed, and service account updated with permissions to pull from ECR
    remote_fn = kt.fn(summer, name=get_test_fn_name()).to(
        kt.Compute(
            cpus=".1",
            image=kt.Image(image_id="us-east1-docker.pkg.dev/runhouse-test/kubetorch-images/kubetorch-client:main"),
            freeze=True,
            gpu_anti_affinity=True,
        )
    )
    assert remote_fn(1, 2) == 3


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

    # Longer timeout to account for initial JuiceFS mount to finish (CSI provisioner needs to allocate the volume)
    launch_timeout = 300

    compute = kt.Compute(
        cpus=".01",
        env_vars={
            "OMP_NUM_THREADS": 1,
            "UV_CACHE_DIR": f"/{KT_MOUNT_FOLDER}/kt-global-cache/uv_cache",
            "HF_HOME": f"/{KT_MOUNT_FOLDER}/kt-global-cache/hf_cache",
        },
        launch_timeout=launch_timeout,
        gpu_anti_affinity=True,
        volumes=[kt.Volume("kt-global-cache", size="10Gi", access_mode="ReadWriteOnce")],
    )

    remote_fn = kt.fn(get_env_var, name=get_test_fn_name()).to(compute)
    result1 = remote_fn.compute.run_bash(
        f"uv pip install pandas --system --cache-dir /{KT_MOUNT_FOLDER}/kt-global-cache/uv_cache -v"
    )
    stdout = "".join(line[1] for line in result1 if len(line) > 1)
    assert "Installed 4 packages" in stdout

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
                cpus=".1",
                launch_timeout=10,
                gpu_anti_affinity=True,
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
            ).autoscale(min_replicas=1),
            init_args={"size": 10},
        )
        remote_cls.print_and_log(1)

    assert "failed to pull" in str(exc_info.value)


@pytest.mark.skip("Removed specific error message, times out instead")
@pytest.mark.level("minimal")
def test_unschedulable_pod_knative():
    import kubetorch as kt

    # Requesting 1000 CPUs and 0.1 memory should be unschedulable
    with pytest.raises(kt.ResourceNotAvailableError):
        remote_cls = kt.cls(ResourceHungryService, name=get_test_fn_name()).to(
            kt.Compute(cpus="1000", memory="0.1", gpu_anti_affinity=True, launch_timeout=40).autoscale(min_replicas=1)
        )
        remote_cls.some_method()


@pytest.mark.level("minimal")
def test_unschedulable_pod_deployment():
    import kubetorch as kt

    # Requesting 1000 CPUs and 0.1 memory should be unschedulable
    with pytest.raises(kt.ResourceNotAvailableError):
        remote_cls = kt.cls(ResourceHungryService, name=get_test_fn_name()).to(
            kt.Compute(cpus="1000", memory="0.1", gpu_anti_affinity=True, launch_timeout=40)
        )
        remote_cls.some_method()


@pytest.mark.level("minimal")
def test_pod_terminated_error():
    import kubetorch as kt

    disk_size = "50Mi"
    with pytest.raises(Exception):
        remote_cls = kt.cls(ResourceHungryService).to(kt.Compute(cpus="0.1", disk_size=disk_size))
        remote_cls.consume_disk()


@pytest.mark.level("minimal")
def test_pod_oom_error_after_startup():
    import kubetorch as kt

    with pytest.raises(Exception):
        remote_cls = kt.cls(ResourceHungryService).to(
            kt.Compute(
                cpus="0.1",
                gpu_anti_affinity=True,
            )
        )

        # Service should start fine; OOM happens only when calling consume_memory
        remote_cls.consume_memory()
