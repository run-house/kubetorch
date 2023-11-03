import inspect
import logging
import os
import time
import unittest

import numpy as np
import pandas as pd

import pytest

import runhouse as rh
from runhouse import Package


logger = logging.getLogger(__name__)


""" Tests for runhouse.Module. Structure:
    - Test call_module_method rpc, with various envs
    - Test creating module from class
    - Test creating module as rh.Module subclass
    - Test calling Module methods async
"""


def resolve_test_helper(obj):
    return obj


@pytest.mark.clustertest
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
def test_call_module_method(cluster, env):
    cluster.put("numpy_pkg", Package.from_string("numpy"), env=env)

    # Test for method
    res = cluster.call("numpy_pkg", "_detect_cuda_version_or_cpu", stream_logs=True)
    assert res == "cpu"

    # Test for property
    res = cluster.call("numpy_pkg", "config_for_rns", stream_logs=True)
    numpy_config = Package.from_string("numpy").config_for_rns
    assert res
    assert isinstance(res, dict)
    assert res == numpy_config

    # Test iterator
    cluster.put("config_dict", list(numpy_config.keys()), env=env)
    res = cluster.call("config_dict", "__iter__", stream_logs=True)
    # Checks that all the keys in numpy_config were returned
    inspect.isgenerator(res)
    for key in res:
        assert key
        numpy_config.pop(key)
    assert not numpy_config


class SlowNumpyArray:
    def __init__(self, size=5):
        self.size = size
        self.arr = np.zeros(self.size)
        self._hidden_1 = "hidden"

    def slow_iter(self):
        self._hidden_2 = "hidden"
        if not self._hidden_1 and self._hidden_2:
            raise ValueError("Hidden attributes not set")
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.arr[i] = i
            yield f"Hello from the cluster! {self.arr}"

    @classmethod
    def cpu_count(cls, local=True):
        return os.cpu_count()

    def size_minus_cpus(self):
        return self.size - self.cpu_count()

    @classmethod
    def factory_constructor(cls, size=5):
        return cls(size=size)


@pytest.mark.clustertest
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
def test_module_from_factory(cluster, env):
    size = 3
    RemoteClass = rh.module(SlowNumpyArray).to(cluster)
    remote_array = RemoteClass(size=size, name="remote_array1")

    # Test that naming works properly, and "class" module was unaffacted
    assert remote_array.name == "remote_array1"
    assert RemoteClass.name == "SlowNumpyArray"

    # Test that module was initialized correctly on the cluster
    assert remote_array.system == cluster
    assert remote_array.remote.size == size
    assert all(remote_array.remote.arr == np.zeros(size))
    assert remote_array.remote._hidden_1 == "hidden"

    results = []
    out = ""
    with rh.capture_stdout() as stdout:
        for i, val in enumerate(remote_array.slow_iter()):
            assert val
            print(val)
            results += [val]
            out = out + str(stdout)
    assert len(results) == 3

    # Check that stdout was captured. Skip the last result because sometimes we
    # don't catch it and it makes the test flaky.
    for i in range(size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    cluster_cpus = int(cluster.run_python(["import os; print(os.cpu_count())"])[0][1])
    # Test classmethod on remote class
    assert RemoteClass.cpu_count() == os.cpu_count()
    assert RemoteClass.cpu_count(local=False) == cluster_cpus

    # Test classmethod on remote instance
    assert remote_array.cpu_count() == os.cpu_count()
    assert remote_array.cpu_count(local=False) == cluster_cpus

    # Test instance method
    assert remote_array.size_minus_cpus() == size - cluster_cpus

    # Test remote getter
    arr = remote_array.remote.arr
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (size,)
    assert arr[0] == 0
    assert arr[2] == 2

    # Test remote setter
    remote_array.remote.size = 20
    assert remote_array.remote.size == 20

    # Test creating a second instance of the same class
    remote_array2 = RemoteClass(size=30, name="remote_array2")
    assert remote_array2.system == cluster
    assert remote_array2.remote.size == 30

    # Test creating a third instance with the factory method
    remote_array3 = RemoteClass.factory_constructor.remote(
        size=40, run_name="remote_array3"
    )
    assert remote_array3.system.config_for_rns == cluster.config_for_rns
    assert remote_array3.remote.size == 40
    assert remote_array3.cpu_count(local=False) == cluster_cpus

    # Make sure first array and class are unaffected by this change
    assert remote_array.remote.size == 20
    assert RemoteClass.cpu_count(local=False) == cluster_cpus

    # Test resolve()
    helper = rh.function(resolve_test_helper).to(cluster, env=rh.Env())
    resolved_obj = helper(remote_array.resolve())
    assert resolved_obj.__class__.__name__ == "SlowNumpyArray"
    assert not hasattr(resolved_obj, "config_for_rns")
    assert resolved_obj.size == 20
    assert list(resolved_obj.arr) == [0, 1, 2]


class SlowPandas(rh.Module):
    def __init__(self, size=5):
        super().__init__()
        self.size = size
        self.df = pd.DataFrame(np.zeros((self.size, self.size)))
        self._hidden_1 = "hidden"

    def slow_iter(self):
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.df[i] = i
            yield f"Hello from the cluster! {self.df.loc[[i]]}"

    async def slow_iter_async(self):
        for i in range(self.size):
            time.sleep(1)
            print(f"Hello from the cluster stdout! {i}")
            logger.info(f"Hello from the cluster logs! {i}")
            self.df[i] = i
            yield f"Hello from the cluster! {self.df.loc[[i]]}"

    def cpu_count(self, local=True):
        return os.cpu_count()

    async def cpu_count_async(self, local=True):
        return os.cpu_count()


@pytest.mark.clustertest
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
def test_module_from_subclass(cluster, env):
    size = 3
    remote_df = SlowPandas(size=size).to(cluster, env)
    assert remote_df.system == cluster

    # Test that module was initialized correctly on the cluster
    assert remote_df.remote.size == size
    assert len(remote_df.remote.df) == size
    assert remote_df.remote._hidden_1 == "hidden"

    results = []
    # Capture stdout to check that it's working
    out = ""
    with rh.capture_stdout() as stdout:
        for i, val in enumerate(remote_df.slow_iter()):
            assert val
            print(val)
            results += [val]
            out = out + str(stdout)
    assert len(results) == 3

    # Check that stdout was captured. Skip the last result because sometimes we
    # don't catch it and it makes the test flaky.
    for i in range(remote_df.size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    cpu_count = int(cluster.run_python(["import os; print(os.cpu_count())"])[0][1])
    print(remote_df.cpu_count())
    assert remote_df.cpu_count() == os.cpu_count()
    print(remote_df.cpu_count(local=False))
    assert remote_df.cpu_count(local=False) == cpu_count

    # Test setting and getting properties
    df = remote_df.remote.df
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.loc[0, 0] == 0
    assert df.loc[2, 2] == 2

    remote_df.size = 20
    assert remote_df.remote.size == 20

    del remote_df

    # Test get_or_to
    remote_df = SlowPandas(size=3).get_or_to(cluster, env=env, name="SlowPandas")
    assert remote_df.system.config_for_rns == cluster.config_for_rns
    assert remote_df.cpu_count(local=False, stream_logs=False) == cpu_count
    # Check that size is unchanged from when we set it to 20 above
    assert remote_df.remote.size == 20

    # Test that resolve() has no effect
    helper = rh.function(resolve_test_helper).to(cluster, env=rh.Env())
    resolved_obj = helper(remote_df.resolve())
    assert resolved_obj.__class__.__name__ == "SlowPandas"
    assert resolved_obj.remote.size == 20
    assert resolved_obj.config_for_rns == remote_df.config_for_rns


@pytest.mark.clustertest
@pytest.mark.asyncio
# @pytest.mark.parametrize("env", [None, "base", "pytorch"])
@pytest.mark.parametrize("env", [None])
async def test_module_from_subclass_async(cluster, env):
    remote_df = SlowPandas(size=3).to(cluster, env)
    assert remote_df.system == cluster
    results = []
    # Capture stdout to check that it's working
    out = ""
    with rh.capture_stdout() as stdout:
        async for val in remote_df.slow_iter_async():
            assert val
            print(val)
            results += [val]
            out = out + str(stdout)
    assert len(results) == 3

    # Check that stdout was captured. Skip the last result because sometimes we
    # don't catch it and it makes the test flaky.
    for i in range(remote_df.size - 1):
        assert f"Hello from the cluster stdout! {i}" in out
        assert f"Hello from the cluster logs! {i}" in out

    cpu_count = int(cluster.run_python(["import os; print(os.cpu_count())"])[0][1])
    print(await remote_df.cpu_count_async())
    assert await remote_df.cpu_count_async() == os.cpu_count()
    print(await remote_df.cpu_count_async(local=False))
    assert await remote_df.cpu_count_async(local=False) == cpu_count

    # Properties
    df = await remote_df.fetch_async("df")
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df.loc[0, 0] == 0
    assert df.loc[2, 2] == 2

    await remote_df.set_async("size", 20)
    assert remote_df.remote.size == 20


@unittest.skip("Not working yet")
@pytest.mark.clustertest
def test_hf_autotokenizer(cluster):
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained("bert-base-uncased")
    RemoteAutoTokenizer = rh.module(AutoTokenizer).to(cluster, env=["transformers"])
    tok = RemoteAutoTokenizer.from_pretrained.remote(
        "bert-base-uncased", run_name="bert-tok"
    )
    # assert tok.remote.pad_token == "<pad>"
    prompt = "Tell me about unified development interfaces into compute and data infrastructure."
    assert tok(prompt, return_tensors="pt").shape == (1, 18)


if __name__ == "__main__":
    unittest.main()
