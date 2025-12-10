import logging
import tempfile
import uuid
from pathlib import Path

import pytest
from kubetorch.globals import ProfilerConfig

logging.getLogger("kubetorch").propagate = True
dir_suffix = "-profiler-output"  # for creating a tempdir where the profile outputs will be saved


def profiler_test_helper(profiler_output_path: Path, file_suffix: str = "svg"):
    assert profiler_output_path
    assert profiler_output_path.exists()
    assert file_suffix == str(profiler_output_path).split(".")[-1]  # check that the file suffix is correct
    assert profiler_output_path.stat().st_size > 0  # check that the output file is not empty


@pytest.mark.level("minimal")
@pytest.mark.asyncio
def test_profiling_pyspy_default_behavior(remote_profiling_pyspy_fn, caplog):
    caplog.set_level("INFO", logger="kubetorch")

    res = remote_profiling_pyspy_fn(num_iterations=6, profiler=ProfilerConfig(profiler_type="pyspy"))

    assert res == "matrix_dot_np ran successfully!"

    profiler_output_path = ""

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "profiler output can be found in" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="svg")
    profiler_output_path.unlink()
    assert not profiler_output_path.exists()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
def test_profiling_pyspy_output_paths(remote_profiling_pyspy_fn, caplog):
    caplog.set_level("INFO", logger="kubetorch")

    with tempfile.TemporaryDirectory(suffix=dir_suffix) as output_tmpdir:

        # case 1: pass both directory output path and output filename
        random_file_name = f"{uuid.uuid4().hex[:8]}.svg"
        pyspy_config = ProfilerConfig(
            profiler_type="pyspy", output_path=output_tmpdir, output_filename=random_file_name
        )
        res = remote_profiling_pyspy_fn(profiler=pyspy_config)

        assert res == "matrix_dot_np ran successfully!"

        expected_output_file_path = Path(output_tmpdir) / Path(random_file_name)
        profiler_test_helper(profiler_output_path=expected_output_file_path)

        # case 2: pass both directory output path and output filename without suffix
        random_file_name = f"{uuid.uuid4().hex[:8]}"
        pyspy_config = ProfilerConfig(
            profiler_type="pyspy", output_path=output_tmpdir, output_filename=random_file_name
        )
        res = remote_profiling_pyspy_fn(profiler=pyspy_config)

        assert res == "matrix_dot_np ran successfully!"

        expected_output_file_path0 = Path(output_tmpdir) / Path(f"{random_file_name}.svg")
        profiler_test_helper(profiler_output_path=expected_output_file_path0)

        # case 3: pass directory output path but NOT output filename
        pyspy_config_only_dir = ProfilerConfig(profiler_type="pyspy", output_path=output_tmpdir)
        res1 = remote_profiling_pyspy_fn(profiler=pyspy_config_only_dir)

        assert res1 == "matrix_dot_np ran successfully!"

        profiler_output_path1 = ""
        http_client_logs = caplog.text.split("\n")
        for log in http_client_logs:
            if "profiler output can be found in" in log:
                profiler_output_path1 = Path(log.split(" ")[-1])

        profiler_test_helper(profiler_output_path=profiler_output_path1)
        assert (
            str(profiler_output_path1.parent) == output_tmpdir
        )  # check that the file was created in the passed directory

        # case 4: DO NOT pass directory output path but pass output filename
        random_file_name2 = f"{uuid.uuid4().hex[:8]}.svg"
        pyspy_config_output_file = ProfilerConfig(profiler_type="pyspy", output_filename=random_file_name2)
        res2 = remote_profiling_pyspy_fn(profiler=pyspy_config_output_file)

        assert res2 == "matrix_dot_np ran successfully!"

        profiler_output_path2 = ""
        http_client_logs = caplog.text.split("\n")
        for log in http_client_logs:
            if "profiler output can be found in" in log:
                profiler_output_path2 = Path(log.split(" ")[-1])

        profiler_test_helper(profiler_output_path=profiler_output_path2)
        assert str(profiler_output_path2.parent) == str(Path.cwd())  # check that the file was created in the cwd
        assert random_file_name2 == str(
            profiler_output_path2.name
        )  # check that the file was created with the user-given name
        profiler_output_path2.unlink()
        assert not profiler_output_path2.exists()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
def test_profiling_pyspy_output_types_fn(remote_profiling_pyspy_fn):
    valid_pyspy_outputs = ["raw", "speedscope", "chrometrace"]
    file_type = {"raw": "txt", "speedscope": "json", "chrometrace": "json"}

    with tempfile.TemporaryDirectory(suffix=dir_suffix) as output_tmpdir:

        for pyspy_output in valid_pyspy_outputs:

            random_file_name = f"{uuid.uuid4().hex[:8]}"
            pyspy_config = ProfilerConfig(
                profiler_type="pyspy",
                output_path=output_tmpdir,
                output_filename=random_file_name,
                output_format=pyspy_output,
            )

            res = remote_profiling_pyspy_fn(profiler=pyspy_config)
            assert res == "matrix_dot_np ran successfully!"

            expected_output_file_path = Path(output_tmpdir) / Path(f"{random_file_name}.{file_type.get(pyspy_output)}")
            profiler_test_helper(
                profiler_output_path=expected_output_file_path, file_suffix=file_type.get(pyspy_output)
            )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
def test_profiling_pyspy_output_types_cls(remote_profiling_pyspy_cls):
    valid_pyspy_outputs = ["flamegraph", "raw", "speedscope", "chrometrace"]

    file_type = {"flamegraph": "svg", "raw": "txt", "speedscope": "json", "chrometrace": "json"}

    with tempfile.TemporaryDirectory(suffix=dir_suffix) as output_tmpdir:

        for pyspy_output in valid_pyspy_outputs:

            random_file_name = f"{uuid.uuid4().hex[:8]}"
            pyspy_config = ProfilerConfig(
                profiler_type="pyspy",
                output_path=output_tmpdir,
                output_filename=random_file_name,
                output_format=pyspy_output,
            )

            res = remote_profiling_pyspy_cls.dot_np(profiler=pyspy_config)
            assert res == "dot_np in Matrix class instance ran successfully!"

            expected_output_file_path = Path(output_tmpdir) / Path(f"{random_file_name}.{file_type.get(pyspy_output)}")
            profiler_test_helper(
                profiler_output_path=expected_output_file_path, file_suffix=file_type.get(pyspy_output)
            )


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.gpu_test
def test_profiling_pytorch_default_behavior_fn(remote_profiling_torch_fn, caplog):
    caplog.set_level("INFO", logger="kubetorch")

    res = remote_profiling_torch_fn(num_iterations=6, profiler=ProfilerConfig(profiler_type="pytorch"))

    assert res == "matrix_dot_torch ran successfully!"

    profiler_output_path = ""

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "profiler output can be found in" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="json")
    profiler_output_path.unlink()
    assert not profiler_output_path.exists()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.gpu_test
def test_profiling_pytorch_default_behavior_cls(remote_profiling_torch_cls, caplog):
    caplog.set_level("INFO", logger="kubetorch")

    res = remote_profiling_torch_cls.dot_torch(profiler=ProfilerConfig(profiler_type="pytorch"))

    assert res == "dot_torch in Matrix_CPU class ran successfully!"

    profiler_output_path = ""

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "profiler output can be found in" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="json")
    profiler_output_path.unlink()
    assert not profiler_output_path.exists()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.gpu_test
def test_profiling_pytorch_output_paths(remote_profiling_torch_fn, caplog):
    caplog.set_level("INFO", logger="kubetorch")

    with tempfile.TemporaryDirectory(suffix=dir_suffix) as output_tmpdir:

        # case 1: pass both directory output path and output filename
        random_file_name = f"{uuid.uuid4().hex[:8]}.json"
        torch_config = ProfilerConfig(
            profiler_type="pytorch", output_path=output_tmpdir, output_filename=random_file_name
        )
        res = remote_profiling_torch_fn(profiler=torch_config)

        assert res == "matrix_dot_torch ran successfully!"

        expected_output_file_path = Path(output_tmpdir) / Path(random_file_name)
        profiler_test_helper(profiler_output_path=expected_output_file_path, file_suffix="json")

        # case 2: pass both directory output path and output filename without suffix
        random_file_name = f"{uuid.uuid4().hex[:8]}"
        torch_config = ProfilerConfig(
            profiler_type="pytorch", output_path=output_tmpdir, output_filename=random_file_name
        )
        res = remote_profiling_torch_fn(profiler=torch_config)

        assert res == "matrix_dot_torch ran successfully!"

        expected_output_file_path0 = Path(output_tmpdir) / Path(f"{random_file_name}.json")
        profiler_test_helper(profiler_output_path=expected_output_file_path0, file_suffix="json")

        # case 3: pass directory output path but NOT output filename
        torch_config_only_dir = ProfilerConfig(profiler_type="pytorch", output_path=output_tmpdir)
        res1 = remote_profiling_torch_fn(profiler=torch_config_only_dir)

        assert res1 == "matrix_dot_torch ran successfully!"

        profiler_output_path1 = ""
        http_client_logs = caplog.text.split("\n")
        for log in http_client_logs:
            if "profiler output can be found in" in log:
                profiler_output_path1 = Path(log.split(" ")[-1])

        profiler_test_helper(profiler_output_path=profiler_output_path1, file_suffix="json")
        assert (
            str(profiler_output_path1.parent) == output_tmpdir
        )  # check that the file was created in the passed directory

        # case 4: DO NOT pass directory output path but pass output filename
        random_file_name2 = f"{uuid.uuid4().hex[:8]}.json"
        torch_config_output_file = ProfilerConfig(profiler_type="pytorch", output_filename=random_file_name2)
        res2 = remote_profiling_torch_fn(profiler=torch_config_output_file)

        assert res2 == "matrix_dot_torch ran successfully!"

        profiler_output_path2 = ""
        http_client_logs = caplog.text.split("\n")
        for log in http_client_logs:
            if "profiler output can be found in" in log:
                profiler_output_path2 = Path(log.split(" ")[-1])

        profiler_test_helper(profiler_output_path=profiler_output_path2, file_suffix="json")
        assert str(profiler_output_path2.parent) == str(Path.cwd())  # check that the file was created in the cwd
        assert random_file_name2 == str(
            profiler_output_path2.name
        )  # check that the file was created with the user-given name
        profiler_output_path2.unlink()
        assert not profiler_output_path2.exists()


@pytest.mark.level("unit")
@pytest.mark.asyncio
def test_profiling_invalid_config_setup(caplog):
    from kubetorch.servers.http.constants import SUPPORTED_PYSPY_OUTPUTS, SUPPORTED_PYTORCH_OUTPUTS

    invalid_profiler_type = "unknown_profiler"
    invalid_output_format = "invalid_output"

    # case 1: invalid profiler
    with pytest.raises(ValueError) as raised_exception:
        ProfilerConfig(profiler_type=invalid_profiler_type)

    assert f"profiler_type must be 'pyspy' or 'pytorch', got '{invalid_profiler_type}'" in str(raised_exception.value)

    # case 2: invalid pyspy profiler output
    with pytest.raises(ValueError) as raised_exception:
        ProfilerConfig(profiler_type="pyspy", output_format=invalid_output_format)

    assert f"Invalid profiler_output for Pyspy profiler: '{invalid_output_format}'" in str(raised_exception.value)
    assert f"Must be one of {SUPPORTED_PYSPY_OUTPUTS}" in str(raised_exception.value)

    # case 3: invalid pytorch profiler output
    with pytest.raises(ValueError) as raised_exception:
        ProfilerConfig(profiler_type="pytorch", output_format=invalid_output_format)

    assert f"Invalid profiler_output for Pytorch profiler: '{invalid_output_format}'" in str(raised_exception.value)
    assert f"Must be one of {SUPPORTED_PYTORCH_OUTPUTS}" in str(raised_exception.value)
