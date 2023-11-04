import os
import shutil
import unittest

from pathlib import Path

import pytest

import runhouse as rh
import yaml
from runhouse.resources.folders.folder import Folder
from runhouse.resources.packages import Package

pip_reqs = ["torch", "numpy"]

# -------- BASE ENV TESTS ----------- #


@pytest.mark.localtest
def test_create_env_from_name_local():
    env_name = "~/local_env"
    local_env = rh.env(name=env_name, reqs=pip_reqs).save()
    del local_env

    remote_env = rh.env(name=env_name)
    assert set(pip_reqs).issubset(set(remote_env.reqs))


@pytest.mark.rnstest
def test_create_env_from_name_rns():
    env_name = "rns_env"
    env = rh.env(name=env_name, reqs=pip_reqs).save()
    del env

    remote_env = rh.env(name=env_name)
    assert set(pip_reqs).issubset(set(remote_env.reqs))


@pytest.mark.localtest
def test_create_env():
    test_env = rh.env(name="test_env", reqs=pip_reqs)
    assert len(test_env.reqs) >= 2
    assert set(pip_reqs).issubset(set(test_env.reqs))


@pytest.mark.clustertest
@pytest.mark.parametrize("env_name", ["base", "test_env"])
def test_to_cluster(cluster, env_name):
    test_env = rh.env(name=env_name, reqs=["transformers"])

    test_env.to(cluster, force_install=True)
    res = cluster.run_python(["import transformers"])
    assert res[0][0] == 0  # import was successful

    cluster.run(["pip uninstall transformers -y"])


@pytest.mark.awstest
@pytest.mark.clustertest
def test_to_fs_to_cluster(cluster, s3_package):
    cluster.install_packages(["s3fs"])

    test_env_s3 = rh.env(name="test_env_s3", reqs=["s3fs", "scipy", s3_package]).to(
        "s3"
    )
    for req in test_env_s3.reqs:
        if isinstance(req, Package) and isinstance(req.install_target, Folder):
            assert req.install_target.system == "s3"
            assert req.install_target.exists_in_system()

    folder_name = "test_package"
    test_env_cluster = test_env_s3.to(
        system=cluster, path=folder_name, mount=True, force_install=True
    )
    for req in test_env_cluster.reqs:
        if isinstance(req, Package) and isinstance(req.install_target, Folder):
            assert req.install_target.system == cluster

    assert "sample_file_0.txt" in cluster.run([f"ls {folder_name}"])[0][1]
    cluster.run([f"rm -r {folder_name}"])


@pytest.mark.clustertest
def test_function_to_env(cluster):
    test_env = rh.env(name="test-env", reqs=["parameterized"]).save()

    def summer(a, b):
        return a + b

    rh.function(summer).to(cluster, test_env, force_install=True)
    res = cluster.run_python(["import parameterized"])
    assert res[0][0] == 0

    cluster.run(["pip uninstall parameterized -y"])
    res = cluster.run_python(["import parameterized"])
    assert res[0][0] == 1

    rh.function(summer).to(system=cluster, env="test-env", force_install=True)
    res = cluster.run_python(["import parameterized"])
    assert res[0][0] == 0

    cluster.run(["pip uninstall parameterized -y"])


def _get_env_var_value(env_var):
    import os

    return os.environ[env_var]


@pytest.mark.clustertest
def test_function_env_vars(cluster):
    test_env_var = "TEST_ENV_VAR"
    test_value = "value"
    test_env = rh.env(env_vars={test_env_var: test_value})

    get_env_var_cpu = rh.function(_get_env_var_value).to(cluster, test_env)
    res = get_env_var_cpu(test_env_var)

    assert res == test_value


@pytest.mark.clustertest
def test_function_env_vars_file(cluster):
    env_file = ".env"
    contents = ["# comment", "", "ENV_VAR1=value", "# comment with =", "ENV_VAR2 =val2"]
    with open(env_file, "w") as f:
        for line in contents:
            f.write(line + "\n")

    test_env = rh.env(name="test-env", env_vars=env_file)

    get_env_var_cpu = rh.function(_get_env_var_value).to(cluster, test_env)
    assert get_env_var_cpu("ENV_VAR1") == "value"
    assert get_env_var_cpu("ENV_VAR2") == "val2"

    os.remove(env_file)
    assert not Path(env_file).exists()


@pytest.mark.clustertest
def test_env_vars_to_cluster(cluster):
    env_vars = {"TEST_ENV_VAR": "value"}
    env = rh.env(name="base", reqs=["parameterized"], env_vars=env_vars)
    env.to(cluster, force_install=True)


@pytest.mark.clustertest
def test_env_vars_conda_env(cluster):
    test_env_var = "TEST_ENV_VAR_CONDA"
    test_value = "conda"
    conda_dict = _get_conda_env(name="conda_env_vars")
    test_env = rh.env(conda_env=conda_dict, env_vars={test_env_var: test_value})

    get_env_var_cpu = rh.function(_get_env_var_value).to(cluster, test_env)
    res = get_env_var_cpu(test_env_var)

    assert res == test_value


@pytest.mark.clustertest
def test_env_git_reqs(cluster):
    git_package = rh.GitPackage(
        git_url="https://github.com/huggingface/diffusers.git",
        install_method="pip",
        revision="v0.11.1",
    )
    env = rh.env(reqs=[git_package])
    env.to(cluster)
    res = cluster.run(["pip freeze | grep diffusers"])
    assert "diffusers" in res[0][1]
    cluster.run(["pip uninstall diffusers -y"])


@pytest.mark.clustertest
def test_working_dir(cluster, tmp_path):
    working_dir = tmp_path / "test_working_dir"
    working_dir.mkdir(exist_ok=True)

    env = rh.env(working_dir=str(working_dir))
    assert str(working_dir) in env.reqs

    env.to(cluster)
    assert working_dir.name in cluster.run(["ls"])[0][1]

    cluster.run([f"rm -r {working_dir.name}"])
    assert working_dir.name not in cluster.run(["ls"])[0][1]


# -------- CONDA ENV TESTS ----------- #


def _get_conda_env(name="rh-test", python_version="3.10.9"):
    conda_env = {
        "name": name,
        "channels": ["defaults"],
        "dependencies": [
            f"python={python_version}",
        ],
    }
    return conda_env


def _get_conda_python_version(env, system):
    env.to(system)
    py_version = system.run([f"{env._run_cmd} python --version"])
    return py_version[0][1]


@pytest.mark.localtest
def test_conda_env_from_name_local():
    env_name = "~/local_conda_env"
    conda_env = _get_conda_env()
    local_env = rh.env(name=env_name, conda_env=conda_env).save()
    del local_env

    remote_env = rh.env(name=env_name, dryrun=True)
    assert remote_env.conda_yaml == conda_env
    assert remote_env.env_name == conda_env["name"]


@pytest.mark.rnstest
def test_conda_env_from_name_rns():
    env_name = "rns_conda_env"
    conda_env = _get_conda_env()
    env = rh.env(name=env_name, conda_env=conda_env).save()
    del env

    remote_env = rh.env(name=env_name)
    assert remote_env.conda_yaml == conda_env
    assert remote_env.env_name == conda_env["name"]


@pytest.mark.clustertest
def test_conda_env_path_to_system(cluster, tmp_path):
    env_name = "from_path"
    python_version = "3.9.16"
    tmp_path = tmp_path / "test-env"
    file_path = f"{tmp_path}/{env_name}.yml"
    tmp_path.mkdir(exist_ok=True)
    yaml.dump(
        _get_conda_env(name=env_name, python_version=python_version),
        open(file_path, "w"),
    )
    conda_env = rh.env(conda_env=file_path)
    shutil.rmtree(tmp_path)

    assert python_version in _get_conda_python_version(conda_env, cluster)


@pytest.mark.clustertest
def test_conda_env_local_to_system(cluster):
    env_name = "local-env"
    python_version = "3.9.16"
    os.system(f"conda create -n {env_name} -y python=={python_version}")

    conda_env = rh.env(conda_env=env_name)
    assert python_version in _get_conda_python_version(conda_env, cluster)


@pytest.mark.clustertest
def test_conda_env_dict_to_system(cluster):
    conda_env = _get_conda_env(python_version="3.9.16")
    test_env = rh.env(name="test_env", conda_env=conda_env)

    assert "3.9.16" in _get_conda_python_version(test_env, cluster)

    # ray installed successfully
    res = cluster.run([f'{test_env._run_cmd} python -c "import ray"'])
    assert res[0][0] == 0


@pytest.mark.clustertest
def test_function_to_conda_env(cluster):
    conda_env = _get_conda_env(python_version="3.9.16")
    test_env = rh.env(name="test_env", conda_env=conda_env)

    def summer(a, b):
        return a + b

    rh.function(summer).to(cluster, test_env)
    assert "3.9.16" in _get_conda_python_version(test_env, cluster)


@pytest.mark.clustertest
def test_conda_additional_reqs(cluster):
    new_conda_env = _get_conda_env(name="test-add-reqs")
    new_conda_env = rh.env(name="conda_env", reqs=["scipy"], conda_env=new_conda_env)
    new_conda_env.to(cluster)
    res = cluster.run([f'{new_conda_env._run_cmd} python -c "import scipy"'])
    assert res[0][0] == 0  # reqs successfully installed
    cluster.run([f"{new_conda_env._run_cmd} pip uninstall scipy"])


@pytest.mark.clustertest
def test_conda_git_reqs(cluster):
    conda_env = _get_conda_env(name="test-add-reqs")
    git_package = rh.GitPackage(
        git_url="https://github.com/huggingface/diffusers.git",
        install_method="pip",
        revision="v0.11.1",
    )
    conda_env = rh.env(conda_env=conda_env, reqs=[git_package])
    conda_env.to(cluster)
    res = cluster.run([f"{conda_env._run_cmd} pip freeze | grep diffusers"])
    assert "diffusers" in res[0][1]
    cluster.run([f"{conda_env._run_cmd} pip uninstall diffusers"])


@pytest.mark.clustertest
def test_conda_env_to_fs_to_cluster(cluster, s3_package):
    conda_env = _get_conda_env(name="s3-env")
    conda_env_s3 = rh.env(reqs=["s3fs", "scipy", s3_package], conda_env=conda_env).to(
        "s3"
    )
    count = 0
    for req in conda_env_s3.reqs:
        if isinstance(req, Package) and isinstance(req.install_target, Folder):
            assert req.install_target.system == "s3"
            assert req.install_target.exists_in_system()
            count += 1
    assert count >= 1

    folder_name = "test_package"
    count = 0
    conda_env_cluster = conda_env_s3.to(system=cluster, path=folder_name, mount=True)
    for req in conda_env_cluster.reqs:
        if isinstance(req, Package) and isinstance(req.install_target, Folder):
            assert req.install_target.system == cluster
            count += 1
    assert count >= 1

    assert "sample_file_0.txt" in cluster.run([f"ls {folder_name}"])[0][1]
    assert "s3-env" in cluster.run(["conda info --envs"])[0][1]

    cluster.run([f"rm -r {folder_name}"])
    cluster.run(["conda env remove -n s3-env"])


# -------- CONDA ENV + FUNCTION TESTS ----------- #


def np_summer(a, b):
    import numpy as np

    return int(np.sum([a, b]))


@pytest.mark.clustertest
def test_conda_call_fn(cluster):
    conda_dict = _get_conda_env(name="c")
    conda_env = rh.env(conda_env=conda_dict, reqs=["pytest", "numpy"])
    fn = rh.function(np_summer).to(system=cluster, env=conda_env)
    result = fn(1, 4)
    assert result == 5


@unittest.skip("Map does not work properly following Module refactor.")
@pytest.mark.clustertest
def test_conda_map_fn(cluster):
    conda_dict = _get_conda_env(name="test-map-fn")
    conda_env = rh.env(conda_env=conda_dict, reqs=["pytest", "numpy"])
    map_fn = rh.function(np_summer, system=cluster, env=conda_env)
    inputs = list(zip(range(5), range(4, 9)))
    results = map_fn.starmap(inputs)
    assert results == [4, 6, 8, 10, 12]


if __name__ == "__main__":
    unittest.main()
