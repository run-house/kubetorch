import logging
import random
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

    res = remote_profiling_pyspy_fn(num_iterations=15, profiler=ProfilerConfig(profiler_type="pyspy"))

    assert res == "matrix_dot_np ran successfully!"

    profiler_output_path = ""

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "Profiler output saved to:" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="svg")
    profiler_output_path.unlink()
    assert not profiler_output_path.exists()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
def test_profiling_pyspy_default_behavior_cls(remote_profiling_pyspy_cls, caplog):
    # this test makes sure that we profile the methods correctly.

    caplog.set_level("INFO", logger="kubetorch")

    # case 1: execute a method that does not accept any argument
    res = remote_profiling_pyspy_cls.dot_np(profiler=ProfilerConfig(profiler_type="pyspy"))

    assert res == "dot_np in Matrix class instance ran successfully!"

    profiler_output_path = ""

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "Profiler output saved to:" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="svg")

    with open(profiler_output_path, "r", encoding="utf-8") as f:
        svg_content = str(f.read())
        # test that the svg file contains profiling info of the method we called
        assert "dot_np" in svg_content
        # test that the svg file DO NOT contain profiling info of the method we DID NOT call
        assert "add_np" not in svg_content

    profiler_output_path.unlink()
    assert not profiler_output_path.exists()

    # reset the profiler profiler_output_path for the next run
    profiler_output_path = ""

    # case 2: execute a method that accept a python object (list of dicts)
    inputs = [{"size": random.randint(500, 700), "iterations": random.randint(10, 20)} for i in range(10)]
    res = remote_profiling_pyspy_cls.add_np(inputs=inputs, profiler=ProfilerConfig(profiler_type="pyspy"))

    assert res == "add_np in Matrix class instance ran successfully!"

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "Profiler output saved to:" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="svg")

    with open(profiler_output_path, "r", encoding="utf-8") as f:
        svg_content = str(f.read())
        # test that the svg file contains profiling info of the method we called
        assert "add_np" in svg_content
        # test that the svg file DO NOT contain profiling info of the method we DID NOT call
        assert "dot_np" not in svg_content

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
            if "Profiler output saved to:" in log:
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
            if "Profiler output saved to:" in log:
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

    res = remote_profiling_torch_fn(profiler=ProfilerConfig(profiler_type="pytorch"))

    assert res == "matrix_dot_torch ran successfully!"

    profiler_output_path = ""

    # get the output path, we need to get it from the logs because the filename contains the request_id
    http_client_logs = caplog.text.split("\n")
    for log in http_client_logs:
        if "Profiler output saved to:" in log:
            profiler_output_path = Path(log.split(" ")[-1])

    profiler_test_helper(profiler_output_path=profiler_output_path, file_suffix="json")
    profiler_output_path.unlink()
    assert not profiler_output_path.exists()


@pytest.mark.level("minimal")
@pytest.mark.asyncio
@pytest.mark.gpu_test
def test_profiling_pytorch_table_output_fn(remote_profiling_torch_fn, caplog):
    caplog.set_level("INFO", logger="kubetorch")

    res, profiler_output = remote_profiling_torch_fn(
        num_iterations=6, profiler=ProfilerConfig(profiler_type="pytorch", output_format="table")
    )

    assert res == "matrix_dot_torch ran successfully!"
    assert profiler_output


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
        if "Profiler output saved to:" in log:
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
            if "Profiler output saved to:" in log:
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
            if "Profiler output saved to:" in log:
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
    caplog.set_level("WARNING", logger="kubetorch")

    invalid_profiler_type = "unknown_profiler"
    invalid_output_format = "invalid_output"

    # case 1: invalid profiler - should warn and disable, not raise
    config = ProfilerConfig(profiler_type=invalid_profiler_type)
    assert config._disabled is True
    assert f"Invalid profiler_type '{invalid_profiler_type}'" in caplog.text
    assert "Profiling will be skipped" in caplog.text

    # case 2: invalid pyspy profiler output - should warn and disable, not raise
    caplog.clear()
    config2 = ProfilerConfig(profiler_type="pyspy", output_format=invalid_output_format)
    assert config2._disabled is True
    assert f"Invalid output_format '{invalid_output_format}' for pyspy profiler" in caplog.text
    assert "Profiling will be skipped" in caplog.text

    # case 3: invalid pytorch profiler output - should warn and disable, not raise
    caplog.clear()
    config3 = ProfilerConfig(profiler_type="pytorch", output_format=invalid_output_format)
    assert config3._disabled is True
    assert f"Invalid output_format '{invalid_output_format}' for pytorch profiler" in caplog.text
    assert "Profiling will be skipped" in caplog.text


@pytest.mark.level("unit")
def test_profiling_invalid_table_sort_by(caplog):
    caplog.set_level("WARNING", logger="kubetorch")

    invalid_sort_key = "invalid_sort_key"

    # table_sort_by is only validated when output_format is "table" - should warn and disable
    config = ProfilerConfig(profiler_type="pytorch", output_format="table", table_sort_by=invalid_sort_key)
    assert config._disabled is True
    assert f"Invalid table_sort_by '{invalid_sort_key}' for pytorch profiler" in caplog.text
    assert "Profiling will be skipped" in caplog.text


@pytest.mark.level("unit")
def test_profiling_table_sort_by_ignored_for_non_table_format():
    # table_sort_by should NOT raise an error when output_format is not "table"
    # (the sort key is simply ignored)
    config = ProfilerConfig(profiler_type="pytorch", output_format="chrometrace", table_sort_by="cpu_time")
    assert config.table_sort_by == "cpu_time"
    assert config.output_format == "chrometrace"


@pytest.mark.level("unit")
def test_profiling_pyspy_output_format_on_pytorch_disables(caplog):
    # pyspy-specific output formats should disable profiling for pytorch profiler
    caplog.set_level("WARNING", logger="kubetorch")
    pyspy_only_formats = ["flamegraph", "raw", "speedscope"]

    for fmt in pyspy_only_formats:
        caplog.clear()
        config = ProfilerConfig(profiler_type="pytorch", output_format=fmt)
        assert config._disabled is True
        assert f"Invalid output_format '{fmt}' for pytorch profiler" in caplog.text


@pytest.mark.level("unit")
def test_profiling_pytorch_output_format_on_pyspy_disables(caplog):
    # pytorch-specific output formats should disable profiling for pyspy profiler
    caplog.set_level("WARNING", logger="kubetorch")
    pytorch_only_formats = ["table", "memory_timeline", "stacks"]

    for fmt in pytorch_only_formats:
        caplog.clear()
        config = ProfilerConfig(profiler_type="pyspy", output_format=fmt)
        assert config._disabled is True
        assert f"Invalid output_format '{fmt}' for pyspy profiler" in caplog.text


@pytest.mark.level("unit")
def test_profiling_default_output_formats():
    # Test that defaults are set correctly
    pyspy_config = ProfilerConfig(profiler_type="pyspy")
    assert pyspy_config.output_format == "flamegraph"

    pytorch_config = ProfilerConfig(profiler_type="pytorch")
    assert pytorch_config.output_format == "chrometrace"


@pytest.mark.level("unit")
def test_profiling_output_file_suffix():
    # Test output_file_suffix method returns correct extensions
    assert ProfilerConfig(profiler_type="pyspy", output_format="flamegraph").output_file_suffix() == "svg"
    assert ProfilerConfig(profiler_type="pyspy", output_format="raw").output_file_suffix() == "txt"
    assert ProfilerConfig(profiler_type="pyspy", output_format="speedscope").output_file_suffix() == "json"
    assert ProfilerConfig(profiler_type="pyspy", output_format="chrometrace").output_file_suffix() == "json"

    assert ProfilerConfig(profiler_type="pytorch", output_format="chrometrace").output_file_suffix() == "json"
    assert ProfilerConfig(profiler_type="pytorch", output_format="table").output_file_suffix() == "json"
    assert ProfilerConfig(profiler_type="pytorch", output_format="stacks").output_file_suffix() == "json"

    # memory_timeline has different suffixes based on memory_timeline_output_type
    assert (
        ProfilerConfig(
            profiler_type="pytorch", output_format="memory_timeline", memory_timeline_output_type="html"
        ).output_file_suffix()
        == "html"
    )
    assert (
        ProfilerConfig(
            profiler_type="pytorch", output_format="memory_timeline", memory_timeline_output_type="json"
        ).output_file_suffix()
        == "json"
    )
    assert (
        ProfilerConfig(
            profiler_type="pytorch", output_format="memory_timeline", memory_timeline_output_type="raw"
        ).output_file_suffix()
        == "raw.json.gz"
    )
    assert (
        ProfilerConfig(
            profiler_type="pytorch", output_format="memory_timeline", memory_timeline_output_type="json_zip"
        ).output_file_suffix()
        == "json.gz"
    )


@pytest.mark.level("unit")
def test_profiling_valid_table_sort_keys():
    from kubetorch.constants import SUPPORTED_PYTORCH_TABLE_SORT_KEYS

    # All valid sort keys should work without raising
    for sort_key in SUPPORTED_PYTORCH_TABLE_SORT_KEYS:
        config = ProfilerConfig(profiler_type="pytorch", output_format="table", table_sort_by=sort_key)
        assert config.table_sort_by == sort_key


@pytest.mark.level("unit")
def test_profiling_config_to_dict():
    config = ProfilerConfig(
        profiler_type="pytorch",
        output_format="chrometrace",
        output_path="/tmp/output",
        output_filename="test_profile",
        analyze_stack_traces=False,
        memory_timeline_output_type="json",
        table_sort_by="cpu_time",
        consolidate_table=True,
        group_by_input_shape=True,
        group_by_stack_n=5,
    )

    config_dict = config.to_dict()

    assert config_dict["profiler_type"] == "pytorch"
    assert config_dict["output_format"] == "chrometrace"
    assert config_dict["output_path"] == "/tmp/output"
    assert config_dict["output_filename"] == "test_profile"
    assert config_dict["analyze_stack_traces"] is False
    assert config_dict["memory_timeline_output_type"] == "json"
    assert config_dict["table_sort_by"] == "cpu_time"
    assert config_dict["consolidate_table"] is True
    assert config_dict["group_by_input_shape"] is True
    assert config_dict["group_by_stack_n"] == 5
