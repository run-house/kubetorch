import contextlib
import enum
import os
import pkgutil
import shlex
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

import dotenv

import numpy as np
import pandas as pd
import pytest

import runhouse as rh
from runhouse.globals import configs

SSH_USER = "rh-docker-user"


class TestLevels(str, enum.Enum):
    UNIT = "unit"
    LOCAL = "local"
    MINIMAL = "minimal"
    THOROUGH = "thorough"
    MAXIMAL = "maximal"


DEFAULT_LEVEL = TestLevels.UNIT


def pytest_addoption(parser):
    parser.addoption(
        "--level",
        action="store",
        default=DEFAULT_LEVEL,
        help="Fixture set to spin up: unit, local, minimal, thorough, or maximal",
    )


def pytest_generate_tests(metafunc):
    level = metafunc.config.getoption("level")
    level_fixtures = getattr(metafunc.module, level.upper(), default_fixtures[level])
    for fixture_name, fixture_list in level_fixtures.items():
        if fixture_name in metafunc.fixturenames:
            metafunc.parametrize(fixture_name, fixture_list, indirect=True)


############## FIXTURES ##############


# https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files

############## HELPERS ##############


def build_and_run_image(
    image_name: str,
    container_name: str,
    detached: bool,
    dir_name: str,
    keypath=None,
    pwd_file=None,
):
    import subprocess

    import docker

    local_rh_package_path = Path(pkgutil.get_loader("runhouse").path).parent
    dockerfile_path = local_rh_package_path / f"docker/slim/{dir_name}/Dockerfile"
    rh_parent_path = local_rh_package_path.parent
    rh_path = "runhouse" if (rh_parent_path / "setup.py").exists() else None
    rh_version = rh.__version__ if not rh_path else None

    # Check if the container is already running, and if so, skip build and run
    client = docker.from_env()
    containers = client.containers.list(
        all=True,
        filters={
            "ancestor": f"runhouse:{image_name}",
            "status": "running",
            "name": container_name,
        },
    )
    if len(containers) > 0 and detached:
        print(f"Container {container_name} already running, skipping build and run")
    else:
        # Build the Docker image, but need to cd into base runhouse directory first
        if keypath:
            build_cmd = [
                "docker",
                "build",
                "--pull",
                "--rm",
                "-f",
                str(dockerfile_path),
                "--build-arg",
                f"RUNHOUSE_PATH={rh_path}"
                if rh_path
                else f"RUNHOUSE_VERSION={rh_version}",
                "--secret",
                f"id=ssh_key,src={keypath}.pub",
                "-t",
                f"runhouse:{image_name}",
                ".",
            ]
        elif pwd_file:
            # Build a password container
            build_cmd = [
                "docker",
                "build",
                "--pull",
                "--rm",
                "-f",
                str(dockerfile_path),
                "--build-arg",
                f"DOCKER_USER_PASSWORD_FILE={pwd_file}",
                "--build-arg",
                f"RUNHOUSE_PATH={rh_path}"
                if rh_path
                else f"RUNHOUSE_VERSION={rh_version}",
                "-t",
                f"runhouse:{image_name}",
                ".",
            ]
        else:
            raise ValueError("No keypath or password file path provided")

        print(shlex.join(build_cmd))
        run_shell_command(subprocess, build_cmd, cwd=str(rh_parent_path.parent))

        # Run the Docker image
        run_cmd = [
            "docker",
            "run",
            "--name",
            container_name,
            "-d",
            "--rm",
            "--shm-size=3gb",
            "-p",
            "32300:32300",
            "-p",
            "6379:6379",
            "-p",
            "52365:52365",
            "-p",
            "443:443",
            "-p",
            "80:80",
            "-p",
            "22:22",
            f"runhouse:{image_name}",
        ]
        print(shlex.join(run_cmd))
        res = popen_shell_command(subprocess, run_cmd, cwd=str(rh_parent_path.parent))
        stdout, stderr = res.communicate()
        if res.returncode != 0:
            raise RuntimeError(f"Failed to run docker image {image_name}: {stderr}")

    return client, rh_parent_path


@contextlib.contextmanager
def test_account():
    """Used for the purposes of testing resource sharing among different accounts.
    When inside the context manager, use the test account credentials before reverting back to the original
    account when exiting."""
    dotenv.load_dotenv()

    test_token = os.getenv("TEST_TOKEN")
    test_username = os.getenv("TEST_USERNAME")
    assert test_token and test_username

    current_token = configs.get("token")
    current_username = configs.get("username")

    try:
        # Assume the role of the test account when inside the context manager
        test_account_token = test_token
        test_account_username = test_username
        test_account_folder = f"/{test_account_username}"

        configs.set("token", test_account_token)
        configs.set("username", test_account_username)
        configs.set("default_folder", test_account_folder)

        yield {
            "test_token": test_account_token,
            "test_username": test_account_username,
            "test_folder": test_account_folder,
        }

    finally:
        # Reset configs back to original account
        configs.set("token", current_token)
        configs.set("username", current_username)
        configs.set("default_folder", f"/{current_username}")


def load_and_share_resources(username_to_share, test_level):
    # Create the shared cluster using the test account
    if test_level in [TestLevels.UNIT, TestLevels.LOCAL]:
        keypath = str(
            Path(
                rh.configs.get("default_keypair", "~/.ssh/runhouse/docker/id_rsa")
            ).expanduser()
        )

        build_and_run_image(
            image_name="keypair",
            container_name="rh-slim-server-public-key-auth",
            detached=True,
            dir_name="public-key-auth",
            keypath=keypath,
        )
        # TODO turn into fixture, and clean up image and container at the end

        c = rh.cluster(
            name="local-docker-slim-public-key-auth",
            host="localhost",
            den_auth=True,
            server_host="0.0.0.0",
            ssh_creds={
                "ssh_user": SSH_USER,
                "ssh_private_key": keypath,
            },
        )

        # Save the test account config to ~/.rh directory in the container
        rh_config = rh.configs.load_defaults_from_file()

        rh.env(
            reqs=["pytest"],
            working_dir=None,
            setup_cmds=[
                f"mkdir -p ~/.rh; touch ~/.rh/config.yaml; "
                f"echo '{rh_config}' > ~/.rh/config.yaml"
            ],
            name="base_env",
        ).to(c)

    else:
        dotenv.load_dotenv()
        test_username = os.getenv("TEST_USERNAME")
        assert test_username

        c = rh.ondemand_cluster(
            name=f"/{test_username}/rh-cpu-shared",
            instance_type="CPU:2+",
            den_auth=True,
            server_connection_type="tls",
            open_ports=[443],
        )
        c.up_if_not()

    c.install_packages(["pytest"])

    # Create function on shared cluster with the same test account
    f = rh.function(summer).to(c, env=["pytest"]).save()

    # Share the cluster & function with the current account
    c.share(username_to_share, access_type="read")
    f.share(username_to_share, access_type="read")

    return c, f


############## FIXTURES ##############


@pytest.fixture(scope="session")
def blob_data():
    return [np.arange(50), "test", {"a": 1, "b": 2}]


@pytest.fixture
def local_file(blob_data, tmp_path):
    return rh.blob(
        data=blob_data,
        system="file",
        path=tmp_path / "test_blob.pickle",
    )


@pytest.fixture
def local_blob(blob_data):
    return rh.blob(
        data=blob_data,
    )


@pytest.fixture
def s3_blob(blob_data, blob_s3_bucket):
    return rh.blob(
        data=blob_data,
        system="s3",
        path=f"/{blob_s3_bucket}/test_blob.pickle",
    )


@pytest.fixture
def gcs_blob(blob_data, blob_gcs_bucket):
    return rh.blob(
        data=blob_data,
        system="gs",
        path=f"/{blob_gcs_bucket}/test_blob.pickle",
    )


@pytest.fixture
def cluster_blob(blob_data, ondemand_cpu_cluster):
    return rh.blob(
        data=blob_data,
        system=ondemand_cpu_cluster,
    )


@pytest.fixture
def cluster_file(blob_data, ondemand_cpu_cluster):
    return rh.blob(
        data=blob_data,
        system=ondemand_cpu_cluster,
        path="test_blob.pickle",
    )


@pytest.fixture
def blob(request):
    """Parametrize over multiple blobs - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def file(request):
    """Parametrize over multiple files - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


# ----------------- Folders -----------------


@pytest.fixture
def local_folder(tmp_path):
    local_folder = rh.folder(path=tmp_path / "tests_tmp")
    local_folder.put({f"sample_file_{i}.txt": f"file{i}".encode() for i in range(3)})
    return local_folder


@pytest.fixture
def cluster_folder(ondemand_cpu_cluster, local_folder):
    return local_folder.to(system=ondemand_cpu_cluster)


@pytest.fixture
def s3_folder(local_folder):
    s3_folder = local_folder.to(system="s3")
    yield s3_folder

    # Delete files from S3
    s3_folder.rm()


@pytest.fixture
def gcs_folder(local_folder):
    gcs_folder = local_folder.to(system="gs")
    yield gcs_folder

    # Delete files from GCS
    gcs_folder.rm()


@pytest.fixture
def folder(request):
    """Parametrize over multiple folders - useful for running the same test on multiple storage types."""
    return request.getfixturevalue(request.param)


# ----------------- Tables -----------------
@pytest.fixture
def huggingface_table():
    from datasets import load_dataset

    dataset = load_dataset("yelp_review_full", split="train[:1%]")
    return dataset


@pytest.fixture
def arrow_table():
    import pyarrow as pa

    df = pd.DataFrame(
        {
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "str": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )
    arrow_table = pa.Table.from_pandas(df)
    return arrow_table


@pytest.fixture
def cudf_table():
    import cudf

    gdf = cudf.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    return gdf


@pytest.fixture
def pandas_table():
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4, 5, 6], "grade": ["a", "b", "b", "a", "a", "e"]}
    )
    return df


@pytest.fixture
def dask_table():
    import dask.dataframe as dd

    index = pd.date_range("2021-09-01", periods=2400, freq="1H")
    df = pd.DataFrame({"a": range(2400), "b": list("abcaddbe" * 300)}, index=index)
    ddf = dd.from_pandas(df, npartitions=10)
    return ddf


@pytest.fixture
def ray_table():
    import ray

    ds = ray.data.range(10000)
    return ds


# ------------------ SageMaker -----------------


@pytest.fixture
def sm_entry_point():
    return "pytorch_train.py"


@pytest.fixture
def pytorch_training_script():
    training_script = textwrap.dedent(
        """\
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data as data
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms


    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            # Define your model architecture here
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x


    # Define your dataset class
    class MyDataset(data.Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            return x, y

        def __len__(self):
            return len(self.data)

    if __name__ == "__main__":
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Define hyperparameters
        learning_rate = 0.001
        batch_size = 64
        num_epochs = 10

        # Load the MNIST dataset and apply transformations
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )

        # Extract the training data and targets
        train_data = (
                train_dataset.data.view(-1, 28 * 28).float() / 255.0
        )  # Flatten and normalize the input
        train_targets = train_dataset.targets

        dataset = MyDataset(train_data, train_targets)
        dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize your model
        model = MyModel()

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            """
    )
    return training_script


@pytest.fixture
def sm_source_dir(sm_entry_point, pytorch_training_script):
    """Create the source directory and entry point needed for creating a SageMaker estimator"""

    # create tmp directory and file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, sm_entry_point)
    with open(file_path, "w") as file:
        file.write(pytorch_training_script)

    yield temp_dir

    # delete tmp directory and file
    os.remove(file_path)
    shutil.rmtree(temp_dir)


# ----------------- Clusters -----------------


@pytest.fixture(scope="session")
def cluster(request):
    """Parametrize over multiple fixtures - useful for running the same test on multiple hardware types."""
    # Example: @pytest.mark.parametrize("cluster", [v100_gpu_cluster, k80_gpu_cluster], indirect=True)"""
    return request.getfixturevalue(request.param.__name__)


@pytest.fixture(scope="session")
def ondemand_cpu_cluster():
    c = rh.ondemand_cluster("^rh-cpu")
    c.up_if_not()

    # Save to RNS - to be loaded in other tests (ex: Runs)
    c.save()

    # Call save before installing in the event we want to use TLS / den auth
    c.install_packages(["pytest"])
    return c


@pytest.fixture(scope="session")
def ondemand_https_cluster_with_auth():
    c = rh.ondemand_cluster(
        name="rh-cpu-https",
        instance_type="CPU:2+",
        den_auth=True,
        server_connection_type="tls",
        open_ports=[443],
    )
    c.up_if_not()

    c.install_packages(["pytest"])
    return c


@pytest.fixture(scope="session")
def sm_cluster():
    c = (
        rh.sagemaker_cluster(
            name="rh-sagemaker",
            profile="sagemaker",
        )
        .up_if_not()
        .save()
    )

    c.install_packages(["pytest"])

    return c


@pytest.fixture(scope="session")
def sm_cluster_with_auth():
    c = (
        rh.sagemaker_cluster(
            name="rh-sagemaker-den-auth",
            profile="sagemaker",
            den_auth=True,
        )
        .up_if_not()
        .save()
    )

    c.install_packages(["pytest"])

    return c


@pytest.fixture(scope="session")
def other_sm_cluster():
    c = (
        rh.sagemaker_cluster(name="rh-sagemaker-2", profile="sagemaker")
        .up_if_not()
        .save()
    )
    c.install_packages(["pytest"])
    return c


@pytest.fixture(scope="session")
def sm_gpu_cluster():
    c = (
        rh.sagemaker_cluster(
            name="rh-sagemaker-gpu", instance_type="ml.g5.2xlarge", profile="sagemaker"
        )
        .up_if_not()
        .save()
    )
    c.install_packages(["pytest"])

    return c


@pytest.fixture(scope="session")
def byo_cpu():
    # Spin up a new basic m5.xlarge EC2 instance
    c = (
        rh.ondemand_cluster(
            instance_type="m5.xlarge",
            provider="aws",
            region="us-east-1",
            # image_id="ami-0a313d6098716f372",  # Upgraded to python 3.11.4 which is not compatible with ray 2.4.0
            name="test-byo-cluster",
        )
        .up_if_not()
        .save()
    )

    c = rh.cluster(
        name="different-cluster", ips=[c.address], ssh_creds=c.ssh_creds()
    ).save()

    c.install_packages(["pytest"])
    c.sync_secrets(["ssh"])

    return c


@pytest.fixture(scope="session")
def v100_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-v100", instance_type="V100:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def k80_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-k80", instance_type="K80:1", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def a10g_gpu_cluster():
    return rh.ondemand_cluster(
        name="rh-a10x", instance_type="g5.2xlarge", provider="aws"
    ).up_if_not()


@pytest.fixture(scope="session")
def password_cluster():
    sky_cluster = rh.cluster("temp-rh-password", instance_type="CPU:4").save()
    if not sky_cluster.is_up():
        sky_cluster.up()

        # set up password on remote
        sky_cluster.run(
            [
                [
                    'sudo sed -i "/^[^#]*PasswordAuthentication[[:space:]]no/c\PasswordAuthentication yes" '
                    "/etc/ssh/sshd_config"
                ]
            ]
        )
        sky_cluster.run(["sudo /etc/init.d/ssh force-reload"])
        sky_cluster.run(["sudo /etc/init.d/ssh restart"])
        sky_cluster.run(
            ["(echo 'cluster-pass' && echo 'cluster-pass') | sudo passwd ubuntu"]
        )
        sky_cluster.run(["pip uninstall skypilot runhouse -y", "pip install pytest"])
        sky_cluster.run(["rm -rf runhouse/"])

    # instantiate byo cluster with password
    ssh_creds = {"ssh_user": "ubuntu", "password": "cluster-pass"}
    cluster = rh.cluster(
        name="rh-password", ips=[sky_cluster.address], ssh_creds=ssh_creds
    ).save()

    return cluster


cpu_clusters = pytest.mark.parametrize(
    "cluster",
    [
        "ondemand_cpu_cluster",
        "ondemand_https_cluster_with_auth",
        "password_cluster",
        "byo_cpu",
    ],
    indirect=True,
)

sagemaker_clusters = pytest.mark.parametrize(
    "cluster", ["sm_cluster", "other_sm_cluster"], indirect=True
)


# ----------------- Envs -----------------


@pytest.fixture
def test_env():
    return rh.env(["pytest"])


# ----------------- Packages -----------------


@pytest.fixture
def local_package(local_folder):
    return rh.package(path=local_folder.path, install_method="local")


@pytest.fixture
def s3_package(s3_folder):
    return rh.package(
        path=s3_folder.path, system=s3_folder.system, install_method="local"
    )


# ----------------- Functions -----------------
def summer(a: int, b: int):
    print("Running summer function")
    return a + b


def save_and_load_artifacts():
    cpu = rh.ondemand_cluster("^rh-cpu").save()
    loaded_cluster = rh.load(name=cpu.name)
    return loaded_cluster.name


def slow_running_func(a, b):
    import time

    time.sleep(20)
    return a + b


@pytest.fixture(scope="session")
def summer_func(ondemand_cpu_cluster):
    return rh.function(summer, name="summer_func").to(
        ondemand_cpu_cluster, env=["pytest"]
    )


@pytest.fixture(scope="session")
def summer_func_with_auth(ondemand_https_cluster_with_auth):
    return rh.function(summer, name="summer_func").to(
        ondemand_https_cluster_with_auth, env=["pytest"]
    )


@pytest.fixture(scope="session")
def summer_func_shared(shared_cluster):
    return rh.function(summer, name="summer_func").to(shared_cluster, env=["pytest"])


@pytest.fixture(scope="session")
def summer_func_sm_auth(sm_cluster_with_auth):
    return rh.function(summer, name="summer_func").to(
        sm_cluster_with_auth, env=["pytest"]
    )


@pytest.fixture(scope="session")
def func_with_artifacts(ondemand_cpu_cluster):
    return rh.function(save_and_load_artifacts, name="artifacts_func").to(
        ondemand_cpu_cluster, env=["pytest"]
    )


@pytest.fixture(scope="session")
def slow_func(ondemand_cpu_cluster):
    return rh.function(slow_running_func, name="slow_func").to(
        ondemand_cpu_cluster, env=["pytest"]
    )


# ----------------- S3 -----------------
@pytest.fixture(scope="session")
def runs_s3_bucket():
    runs_bucket = create_s3_bucket("runhouse-runs")
    return runs_bucket.name


@pytest.fixture(scope="session")
def blob_s3_bucket():
    blob_bucket = create_s3_bucket("runhouse-blob")
    return blob_bucket.name


@pytest.fixture(scope="session")
def table_s3_bucket():
    table_bucket = create_s3_bucket("runhouse-table")
    return table_bucket.name


# ----------------- GCP -----------------


@pytest.fixture(scope="session")
def blob_gcs_bucket():
    blob_bucket = create_gcs_bucket("runhouse-blob")
    return blob_bucket.name


@pytest.fixture(scope="session")
def table_gcs_bucket():
    table_bucket = create_gcs_bucket("runhouse-table")
    return table_bucket.name


# ----------------- Runs -----------------


@pytest.fixture(scope="session")
def submitted_run(summer_func):
    """Initializes a Run, which will run synchronously on the cluster. Returns the function's result."""
    run_name = "synchronous_run"
    res = summer_func(1, 2, run_name=run_name)
    assert res == 3
    return run_name


@pytest.fixture(scope="session")
def submitted_async_run(summer_func):
    """Execute function async on the cluster. If a run already exists, do not re-run. Returns a Run object."""
    run_name = "async_run"
    async_run = summer_func.run(run_name=run_name, a=1, b=2)

    assert isinstance(async_run, rh.Run)
    return run_name


def create_s3_bucket(bucket_name: str):
    """Create bucket in S3 if it does not already exist."""
    from sky.data.storage import S3Store

    s3_store = S3Store(name=bucket_name, source="")
    return s3_store


def create_gcs_bucket(bucket_name: str):
    """Create bucket in GS if it does not already exist."""
    from sky.data.storage import GcsStore

    gcs_store = GcsStore(name=bucket_name, source="")
    return gcs_store


# ----------------- Docker -----------------


def run_shell_command_direct(subprocess, cmd: str):
    # Run the command and wait for it to complete
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("subprocess failed, stdout: " + result.stdout)
        print("subprocess failed, stderr: " + result.stderr)

    # Check for success
    assert result.returncode == 0


def run_shell_command(subprocess, cmd: list[str], cwd: str = None):
    # Run the command and wait for it to complete
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or Path.cwd())

    if result.returncode != 0:
        print("subprocess failed, stdout: " + result.stdout)
        print("subprocess failed, stderr: " + result.stderr)

    # Check for success
    assert result.returncode == 0


def popen_shell_command(subprocess, command: list[str], cwd: str = None):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd or Path.cwd(),
    )
    # Wait for 10 seconds before resuming execution
    time.sleep(10)
    return process


@pytest.fixture(scope="session")
def local_docker_cluster_passwd(detached=True):
    image_name = "pwd"
    container_name = "rh-slim-server-password-auth"
    dir_name = "password-file-auth"
    pwd_file = "docker_user_passwd"

    client, rh_parent_path = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        detached=detached,
        pwd_file=pwd_file,
    )

    # Runhouse commands can now be run locally
    pwd = (rh_parent_path.parent / pwd_file).read_text().strip()
    c = rh.cluster(
        name="local-docker-slim-password-file-auth",
        host="localhost",
        server_host="0.0.0.0",
        ssh_creds={"ssh_user": SSH_USER, "password": pwd},
    )
    rh.env(
        reqs=["pytest"],
        working_dir=None,
        setup_cmds=[
            f'mkdir -p ~/.rh; echo "token: {rh.configs.get("token")}" > ~/.rh/config.yaml'
        ],
        name="base_env",
    ).to(c)
    c.save()

    # Yield the cluster
    yield c

    # Stop the Docker container
    if not detached:
        client.containers.get("rh-slim-server-password-auth").stop()
        client.containers.prune()
        client.images.prune()


@pytest.fixture(scope="session")
def local_docker_cluster_public_key(detached=True):
    image_name = "keypair"
    container_name = "rh-slim-server-public-key-auth"
    dir_name = "public-key-auth"
    keypath = str(
        Path(
            rh.configs.get("default_keypair", "~/.ssh/runhouse/docker/id_rsa")
        ).expanduser()
    )

    client, rh_parent_path = build_and_run_image(
        image_name=image_name,
        container_name=container_name,
        dir_name=dir_name,
        detached=detached,
        keypath=keypath,
    )

    # Runhouse commands can now be run locally
    c = rh.cluster(
        name="local-docker-slim-public-key-auth",
        host="localhost",
        server_host="0.0.0.0",
        ssh_creds={
            "ssh_user": SSH_USER,
            "ssh_private_key": keypath,
        },
    )
    rh.env(
        reqs=["pytest"],
        working_dir=None,
        setup_cmds=[
            f'mkdir -p ~/.rh; echo "token: {rh.configs.get("token")}" > ~/.rh/config.yaml'
        ],
        name="base_env",
    ).to(c)
    c.save()

    # Yield the cluster
    yield c

    # Stop the Docker container
    if not detached:
        client.containers.get(container_name).stop()
        client.containers.prune()
        client.images.prune()


########## DEFAULT LEVELS ##########

default_fixtures = {}
default_fixtures[TestLevels.UNIT] = {"cluster": [local_docker_cluster_public_key]}
default_fixtures[TestLevels.LOCAL] = {
    "cluster": [local_docker_cluster_passwd, local_docker_cluster_public_key]
}
default_fixtures[TestLevels.MINIMAL] = {"cluster": [ondemand_cpu_cluster]}
default_fixtures[TestLevels.THOROUGH] = {
    "cluster": [
        local_docker_cluster_passwd,
        local_docker_cluster_public_key,
        ondemand_cpu_cluster,
        ondemand_https_cluster_with_auth,
        password_cluster,
        byo_cpu,
    ]
}
default_fixtures[TestLevels.MAXIMAL] = {
    "cluster": [
        local_docker_cluster_passwd,
        local_docker_cluster_public_key,
        ondemand_cpu_cluster,
        ondemand_https_cluster_with_auth,
        password_cluster,
        byo_cpu,
    ]
}
