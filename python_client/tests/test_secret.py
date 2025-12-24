import os
import tempfile
from pathlib import Path

import pytest

from kubetorch.resources.secrets.kubernetes_secrets_client import KubernetesSecretsClient
from kubetorch.resources.secrets.provider_secrets.aws_secret import AWSSecret
from kubetorch.resources.secrets.provider_secrets.gcp_secret import GCPSecret
from kubetorch.resources.secrets.provider_secrets.huggingface_secret import HuggingFaceSecret
from kubetorch.resources.secrets.utils import check_env_vars_on_kubernetes_pods, check_path_on_kubernetes_pods

from .conftest import get_test_hash
from .utils import create_random_name_prefix, get_env_var, summer


@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    os.environ["KT_GPU_ANTI_AFFINITY"] = "True"
    yield


@pytest.mark.skip("Skipping since CI is running on a GKE cluster")
@pytest.mark.level("unit")
def test_aws_secret_init():
    """Test that the AWS secret class is properly initialized."""
    import kubetorch as kt

    secret = kt.secret(provider="aws")
    assert secret.provider == AWSSecret._PROVIDER
    assert set(secret.values.keys()) == set(AWSSecret._DEFAULT_FILENAMES)
    assert secret.path == AWSSecret._DEFAULT_PATH


@pytest.mark.level("unit")
def test_gcp_secret_init():
    """Test that the GCP secret class is properly initialized."""
    import kubetorch as kt

    secret = kt.secret(provider="gcp")
    assert secret.provider == GCPSecret._PROVIDER
    if secret.path:
        assert secret.path == GCPSecret._DEFAULT_PATH
        assert set(secret.values.keys()) == set(GCPSecret._DEFAULT_FILENAMES)
    else:
        assert secret.env_vars == GCPSecret._DEFAULT_ENV_VARS
        assert set(secret.values.keys()) == set(GCPSecret._DEFAULT_ENV_VARS)


@pytest.mark.level("unit")
def test_huggingface_secret_init():
    """Test that the HuggingFace secret class is properly initialized."""
    import kubetorch as kt

    secret = kt.secret(provider="huggingface")
    assert secret.provider == HuggingFaceSecret._PROVIDER
    if secret.path:
        assert secret.path == HuggingFaceSecret._DEFAULT_PATH
        assert set(secret.values.keys()) == set(HuggingFaceSecret._DEFAULT_FILENAMES)
    else:
        assert secret.env_vars == HuggingFaceSecret._DEFAULT_ENV_VARS
        assert set(secret.values.keys()) == set(HuggingFaceSecret._DEFAULT_ENV_VARS)


@pytest.mark.level("unit")
def test_custom_secret_init_fails_without_values():
    """Test that a secret is none, if the secret is not found in kubernetes."""
    import kubetorch as kt

    secret = kt.secret(name="custom-secret")
    assert secret is None


@pytest.mark.level("unit")
def test_secret_create_using_path_and_reload():
    import kubetorch as kt

    dir_suffix = "-secret-test"
    with tempfile.TemporaryDirectory(suffix=dir_suffix) as tmpdir:
        secret_file_content = "key1=value1\nkey2=value2"
        temp_file = Path(tmpdir) / ".secrets" / "test_secret"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text(secret_file_content)

        secret_name = f"{get_test_hash()}-create-from-path-secret"
        full_secret_path = str(temp_file.resolve())
        secret_filenames = str(temp_file.name)

        secret = kt.secret(name=secret_name, path=full_secret_path)
        client = KubernetesSecretsClient()

        # Delete if exists from previous run
        try:
            client.delete_secret(secret_name)
        except Exception:
            pass

        success = client.create_secret(secret)
        assert success

        reloaded_secret = client.load_secret(name=secret_name)
        assert reloaded_secret
        assert reloaded_secret.path.endswith(".secrets")
        assert reloaded_secret.filenames[0] == secret_filenames


@pytest.mark.level("unit")
def test_secret_create_provider_and_path_and_reload():
    import kubetorch as kt

    with tempfile.TemporaryDirectory() as tmpdir:
        secret_file_content = "key1=value1\nkey2=value2\nkey3=value3"
        temp_file = Path(tmpdir) / ".my_provider" / "credentials"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_text(secret_file_content)

        secret_name = f"{get_test_hash()}-provider-from-path-secret"
        full_secret_path = str(temp_file.resolve())
        secret_path_parent, secret_filenames = str(temp_file.parent.resolve()), str(temp_file.name)
        provider = "aws"

        secret = kt.secret(name=secret_name, provider=provider, path=full_secret_path)
        client = KubernetesSecretsClient()

        # Delete if exists from previous run
        try:
            client.delete_secret(secret_name)
        except Exception:
            pass

        success = client.create_secret(secret)
        assert success

        reloaded_secret = kt.secret(name=secret_name)
        assert reloaded_secret
        assert reloaded_secret.path == secret_path_parent
        assert reloaded_secret.filenames[0] == secret_filenames
        assert reloaded_secret.values == {secret_filenames: secret_file_content}


@pytest.mark.level("minimal")
def test_kubernetes_secret_create_update_delete():
    """Test that a secret can be created, updated, and deleted."""
    import kubetorch as kt

    name = f"{create_random_name_prefix()}-secret"  # prevent collisions
    secret = kt.secret(name=name, provider="gcp")  # comment if you are running on eks cluster
    #  secret = kt.secret(name=name, provider="aws")  # un-comment if you are running on eks cluster
    assert secret.provider == "gcp"  # switch to aws if you are running on an eks cluster
    client = KubernetesSecretsClient()

    success = client.create_secret(secret)
    assert success
    loaded_secret = client.load_secret(name)
    assert loaded_secret
    # Check that the secret values match
    for key in secret.values.keys():
        assert loaded_secret.values[key] == secret.values[key]

    client.delete_secret(name)
    secret = client.load_secret(name)
    assert not secret


@pytest.mark.level("minimal")
def test_kubernetes_secret_create_update_delete_custom_env_vars():
    """Test that a custom secret with env vars can be created, updated, and deleted."""
    import kubetorch as kt

    name = f"{create_random_name_prefix()}-custom-secret"

    os.environ["CUSTOM_ENV_VAR"] = "1234567890"
    secret = kt.secret(
        name=name,
        env_vars=["CUSTOM_ENV_VAR"],
    )
    assert secret.env_vars == ["CUSTOM_ENV_VAR"]

    client = KubernetesSecretsClient()
    success = client.create_secret(secret)
    assert success

    loaded_secret = client.load_secret(name)
    assert loaded_secret
    assert loaded_secret.values["CUSTOM_ENV_VAR"] == "1234567890"

    client.delete_secret(name)
    secret = client.load_secret(name)
    assert not secret


@pytest.mark.skip("Skipping since CI is running on a GKE cluster")
@pytest.mark.level("minimal")
def test_secret_aws_propagated_to_pod_as_string():
    import kubetorch as kt

    provider = "aws"
    name = f"{create_random_name_prefix()}-custom-secret-{provider}"
    remote_fn = kt.fn(summer, name=name).to(kt.Compute(cpus=".01", secrets=[provider]))
    # Check that the secret is propagated to Kubernetes
    client = KubernetesSecretsClient()
    secret = client.load_secret(provider)
    assert secret

    # Check that the secret is propagated to the pod
    for filename in AWSSecret._DEFAULT_FILENAMES:
        assert check_path_on_kubernetes_pods(f"{AWSSecret._DEFAULT_PATH}/{filename}", remote_fn.service_name)


@pytest.mark.level("minimal")
def test_secret_gcp_propagated_to_pod_as_string():
    import kubetorch as kt

    provider = "gcp"
    name = f"{create_random_name_prefix()}-custom-secret-{provider}"
    remote_fn = kt.fn(summer, name=name).to(kt.Compute(cpus=".01", secrets=[provider]))
    # Check that the secret is propagated to Kubernetes
    client = KubernetesSecretsClient()
    secret = client.load_secret(f"{kt.config.username}-{provider}")
    assert secret

    # Check that the secret is propagated to the pod
    is_ci_env = os.getenv("CI", False)
    if is_ci_env:
        assert check_env_vars_on_kubernetes_pods(GCPSecret._DEFAULT_ENV_VARS, remote_fn.service_name)
    else:
        for filename in GCPSecret._DEFAULT_FILENAMES:
            assert check_path_on_kubernetes_pods(f"{GCPSecret._DEFAULT_PATH}/{filename}", remote_fn.service_name)


@pytest.mark.level("minimal")
def test_secret_gcp_propagated_to_pod_as_object():
    import kubetorch as kt

    provider = "gcp"  # switch to aws if running on a gke cluster
    name = f"{create_random_name_prefix()}-custom-secret-{provider}"

    gcp_secret = kt.secret(provider=provider, override=True)
    remote_fn = kt.fn(summer, name=name).to(kt.Compute(cpus=".1", secrets=[gcp_secret]))

    # Check that the secret is propagated to Kubernetes
    client = KubernetesSecretsClient()
    secret = client.load_secret(f"{kt.config.username}-{provider}")
    assert secret

    # Check that the secret is propagated to the pod
    is_ci_env = os.getenv("CI", False)
    if is_ci_env:
        assert check_env_vars_on_kubernetes_pods(GCPSecret._DEFAULT_ENV_VARS, remote_fn.service_name)
    else:
        for filename in GCPSecret._DEFAULT_FILENAMES:
            assert check_path_on_kubernetes_pods(f"{GCPSecret._DEFAULT_PATH}/{filename}", remote_fn.service_name)


@pytest.mark.level("minimal")
def test_secret_huggingface_propagated_to_pod():
    import kubetorch as kt

    provider = "huggingface"
    name = f"{create_random_name_prefix()}-custom-secret-hf"

    remote_fn = kt.fn(get_env_var, name=name).to(kt.Compute(cpus=".1", secrets=[provider]))

    assert remote_fn("HF_TOKEN")

    # Check that the secret is propagated to Kubernetes
    client = KubernetesSecretsClient()
    loaded_secret = client.load_secret(f"{kt.config.username}-{provider}")
    assert loaded_secret

    # Check that the secret is propagated to the pod (check env_vars)
    env_vars = check_env_vars_on_kubernetes_pods(HuggingFaceSecret._DEFAULT_ENV_VARS, remote_fn.service_name)
    if env_vars:
        assert env_vars
        for k in HuggingFaceSecret._DEFAULT_ENV_VARS:
            assert env_vars[k] == loaded_secret.values[k]
    else:
        for filename in HuggingFaceSecret._DEFAULT_FILENAMES:
            assert check_path_on_kubernetes_pods(
                f"{HuggingFaceSecret._DEFAULT_PATH}/{filename}", remote_fn.service_name
            )


@pytest.mark.level("minimal")
def test_secret_custom_env_vars_propagated_to_pod():
    import kubetorch as kt

    CUSTOM_ENV_VAR = "CUSTOM_ENV_VAR"
    CUSTOM_ENV_VAR_VALUE = "1234567890"
    os.environ[CUSTOM_ENV_VAR] = CUSTOM_ENV_VAR_VALUE

    secret_name = "custom-secret"
    name = f"{create_random_name_prefix()}-{secret_name}"

    secret = kt.secret(
        name=secret_name,
        env_vars=[CUSTOM_ENV_VAR],
    )
    assert secret.values == {CUSTOM_ENV_VAR: CUSTOM_ENV_VAR_VALUE}
    assert secret.env_vars == [CUSTOM_ENV_VAR]

    remote_fn = kt.fn(get_env_var, name=name).to(kt.Compute(cpus=".1", secrets=[secret]).autoscale(min_replicas=1))

    assert remote_fn(CUSTOM_ENV_VAR) == CUSTOM_ENV_VAR_VALUE

    # Check that the secret is propagated to Kubernetes
    client = KubernetesSecretsClient()
    loaded_secret = client.load_secret(secret_name)
    assert loaded_secret

    # Make another call to ensure pod is running before checking env vars
    assert remote_fn(CUSTOM_ENV_VAR) == CUSTOM_ENV_VAR_VALUE

    # Check that the secret is propagated to the pod (check env_vars)
    env_var_names = [CUSTOM_ENV_VAR]
    env_vars = check_env_vars_on_kubernetes_pods(env_var_names, remote_fn.service_name)
    assert env_vars
    for k in secret.env_vars:
        assert env_vars[k] == loaded_secret.values[k]
