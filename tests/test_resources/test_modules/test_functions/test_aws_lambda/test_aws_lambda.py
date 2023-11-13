import logging
import os
import shutil
import time
import unittest
from pathlib import Path

import boto3
import pytest
import runhouse as rh

logger = logging.getLogger(__name__)
CUR_WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES = f"{CUR_WORK_DIR}/test_helpers/lambda_tests"
CRED_PATH_MAC = f"{Path.home()}/.aws/credentials"
CRED_PATH_WIN = f"{Path.home()}\.aws\credentials"
DEFAULT_REGION = "us-east-1"

if Path(CRED_PATH_MAC).is_file() or Path(CRED_PATH_WIN).is_file():
    LAMBDA_CLIENT = boto3.client("lambda")
else:
    LAMBDA_CLIENT = boto3.client("lambda", region_name=DEFAULT_REGION)
IAM_CLIENT = boto3.client("iam")
LAMBDAS_NAMES = set()


@pytest.fixture(scope="session", autouse=True)
def download_resources():
    curr_folder = os.getcwd()
    s3_resource = boto3.resource("s3")
    bucket = s3_resource.Bucket("runhouse-lambda-resources")
    remoteDirectoryName = "test_helpers/lambda_tests"
    objs = bucket.objects.filter(Prefix=remoteDirectoryName)
    for obj in objs:
        dir_name = "/".join(obj.key.split("/")[:-1])
        if not os.path.exists(f"{curr_folder}/{dir_name}"):
            os.makedirs(f"{curr_folder}/{dir_name}")
        bucket.download_file(obj.key, f"{curr_folder}/{obj.key}")


def test_create_and_run_no_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_create_and_run"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
    )

    time.sleep(5)  # letting the lambda be updated in AWS.
    my_lambda.save()
    res = my_lambda(3, 4)
    assert res == "7"
    reload_func = rh.aws_lambda_function(name=name)
    res2 = reload_func(12, 7)
    assert res2 == "19"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_load_not_existing_lambda():
    name = "test_lambda_create_and_run1"
    my_lambda = rh.aws_lambda_function(name=name)
    assert my_lambda == "LambdaNotFoundInAWS"


def test_crate_no_arguments():
    try:
        rh.aws_lambda_function()
    except Exception:
        assert True is True


def test_bad_handler_path_to_factory(caplog):
    name = "test_lambda_create_and_run"
    caplog.set_level(logging.ERROR)
    try:
        rh.aws_lambda_function(
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert "Please provide a path to the lambda handler file." in caplog.text

    try:
        rh.aws_lambda_function(
            paths_to_code=None,
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert "Please provide a path to the lambda handler file." in caplog.text

    try:
        rh.aws_lambda_function(
            paths_to_code=[],
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert "Please provide a path to the lambda handler file." in caplog.text


def test_bad_handler_func_name_to_factory(caplog):
    name = "test_lambda_create_and_run"
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    caplog.set_level(logging.ERROR)
    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the name of the function that should be executed by the lambda."
            in caplog.text
        )

    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name=None,
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the name of the function that should be executed by the lambda."
            in caplog.text
        )

    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="",
            runtime="python3.9",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the name of the function that should be executed by the lambda."
            in caplog.text
        )


def test_bad_runtime_to_factory(caplog):
    name = "test_lambda_create_and_run"
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    SUPPORTED_RUNTIMES = [
        "python3.7",
        "python3.8",
        "python3.9",
        "python3.10",
        "python 3.11",
    ]
    caplog.set_level(logging.ERROR)
    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="lambda_sum",
            runtime="python3.91",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert (
            f"Please provide a supported lambda runtime, should be one of the following: {SUPPORTED_RUNTIMES}"
            in caplog.text
        )

    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="lambda_sum",
            runtime=None,
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert (
            f"Please provide a supported lambda runtime, should be one of the following: {SUPPORTED_RUNTIMES}"
            in caplog.text
        )

    try:
        rh.aws_lambda_function(
            paths_to_code=None,
            handler_function_name="lambda_sum",
            args_names=["arg1", "arg2"],
            name=name,
        )
    except RuntimeError:
        assert (
            f"Please provide a supported lambda runtime, should be one of the following: {SUPPORTED_RUNTIMES}"
            in caplog.text
        )


def test_bad_args_names_to_factory(caplog):
    name = "test_lambda_create_and_run"
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    caplog.set_level(logging.ERROR)
    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="lambda_sum",
            runtime="python3.9",
            args_names=None,
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the names of the arguments provided to handler function, in the order they are"
            + " passed to the lambda function."
            in caplog.text
        )

    try:
        rh.aws_lambda_function(
            paths_to_code=handler_path,
            handler_function_name="lambda_sum",
            runtime="python3.9",
            name=name,
        )
    except RuntimeError:
        assert (
            "Please provide the names of the arguments provided to handler function, in the order they are"
            + " passed to the lambda function."
            in caplog.text
        )


def test_func_no_args(capsys):
    handler_path = [f"{TEST_RESOURCES}/basic_handler_no_args.py"]
    name = "test_lambda_no_args"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="basic_handler",
        runtime="python3.9",
        args_names=[],
        name=name,
    )
    time.sleep(5)
    assert my_lambda() == "-1"
    assert "This a func with not args" in capsys.readouterr().out


def test_create_and_run_generate_name():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
    )
    time.sleep(5)  # letting the lambda be updated in AWS.
    res = my_lambda(3, 4)
    assert res == "7"
    my_lambda.save()
    reload_func = rh.aws_lambda_function(name="lambda_sum")
    time.sleep(1)
    res2 = reload_func(12, 7)
    assert res2 == "19"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_create_and_run_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == "12"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_different_runtimes_and_layers():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy"
    my_lambda_37 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.7",
        args_names=["arr1", "arr2"],
        name=name + "_37",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res37 = my_lambda_37([1, 2, 3], [2, 5, 6])
    assert res37 == "19"
    LAMBDAS_NAMES.add(my_lambda_37.name)

    my_lambda_38 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.8",
        args_names=["arr1", "arr2"],
        name=name + "_38",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res38 = my_lambda_38([1, 2, 3], [12, 5, 9])
    assert res38 == "32"
    LAMBDAS_NAMES.add(my_lambda_38.name)

    my_lambda_310 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.10",
        args_names=["arr1", "arr2"],
        name=name + "_310",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res310 = my_lambda_310([-2, 5, 1], [12, 5, 9])
    assert res310 == "30"
    LAMBDAS_NAMES.add(my_lambda_310.name)

    my_lambda_311 = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.11",
        args_names=["arr1", "arr2"],
        name=name + "_311",
        env=["numpy", "pandas"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res311 = my_lambda_311([-2, 5, 1], [8, 7, 6])
    assert res311 == "25"
    LAMBDAS_NAMES.add(my_lambda_311.name)


def test_create_and_run_layers_txt():
    handler_path = [f"{TEST_RESOURCES}/basic_handler_layer.py"]
    name = "test_lambda_numpy_txt"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_handler",
        runtime="python3.9",
        args_names=["arr1", "arr2"],
        name=name,
        env=f"{os.getcwd()}/test_helpers/lambda_tests/requirements.txt",
    )
    time.sleep(5)  # letting the lambda be updated in AWS.
    res = my_lambda([1, 2, 3], [1, 2, 3])
    assert res == "12"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_update_lambda_one_file():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_create_and_run"
    my_lambda = rh.aws_lambda_function(
        paths_to_code=handler_path,
        handler_function_name="lambda_sum",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
    )
    time.sleep(5)  # letting the lambda be updated in AWS.
    res = my_lambda(6, 4)
    assert res == "10"
    reload_func = rh.aws_lambda_function(name=name)
    time.sleep(1)
    res2 = reload_func(12, 13)
    assert res2 == "25"
    LAMBDAS_NAMES.add(my_lambda.name)


def test_mult_files_each():
    """The handler function calls function from each file separately.
    For example, there are a.py, b.py and c.py. Each file has the following funcs, respectively: func_a, func_b and
    func_c. So in the test, the main function (handler) will look something like this:
    import a.py, b.py and c.py
    def handler_func:
        func_a()
        func_b()
        func_c()
    """
    prefix = "call_files_separately"
    folder_path = f"{TEST_RESOURCES}/{prefix}"
    handler_paths = os.listdir(folder_path)
    handler_paths = [p for p in handler_paths if ".py" in p]
    handler_paths.sort()
    handler_paths = [f"{folder_path}/{p}" for p in handler_paths]
    name = "test_lambda_multiple_files_s"
    my_lambda_calc_1 = rh.aws_lambda_function(
        paths_to_code=handler_paths,
        handler_function_name="my_calc",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
        env=["numpy"],
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res1 = my_lambda_calc_1(2, 3)
    res2 = my_lambda_calc_1(5, 3)
    res3 = my_lambda_calc_1(2, 7)
    res4 = my_lambda_calc_1(10, 5)
    assert res1 == "2.5"
    assert res2 == "3.2"
    assert res3 == "22.5"
    assert res4 == "7.5"
    LAMBDAS_NAMES.add(my_lambda_calc_1.name)


def test_few_python_files_chain():
    """The handler function calls functions from different files in chain.
    For example, there are a.py, b.py and c.py. Each file has the following funcs, respectively: func_a, func_b and
    func_c. So in the test, the main function (handler) will look something like this:
    import c.py
    def handler_func:
        func_c()
    where func_c() calls func_b() which calls func_a().
    """
    prefix = "call_files_chain"
    folder_path = f"{TEST_RESOURCES}/{prefix}"
    handler_paths = os.listdir(folder_path)
    handler_paths = [p for p in handler_paths if ".py" in p]
    handler_paths.sort()
    handler_paths = [f"{folder_path}/{p}" for p in handler_paths]
    name = "test_lambda_multiple_files_c"
    my_lambda_calc_2 = rh.aws_lambda_function(
        paths_to_code=handler_paths,
        handler_function_name="special_calc",
        runtime="python3.9",
        args_names=["arg1", "arg2"],
        name=name,
    )
    time.sleep(4)  # letting the lambda be updated in AWS.
    res1 = my_lambda_calc_2(2, 3)
    res2 = my_lambda_calc_2(5, 3)
    res3 = my_lambda_calc_2(2, 7)
    res4 = my_lambda_calc_2(10, 5)
    assert res1 == "16"
    assert res2 == "17"
    assert res3 == "20"
    assert res4 == "20"
    LAMBDAS_NAMES.add(my_lambda_calc_2.name)


def test_args():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    time.sleep(1)
    res1 = basic_func(2, 3)
    res2 = basic_func(5, arg2=3)
    res3 = basic_func(arg1=2, arg2=7)
    assert res1 == "5"
    assert res2 == "8"
    assert res3 == "9"


def test_map_starmap():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    time.sleep(1)
    res_map1 = basic_func.map([1, 2, 3], [4, 5, 6])
    res_map2 = basic_func.map([6, 2, 3], [15, 52, 61])
    res_map3 = basic_func.starmap([(1, 2), (3, 4), (5, 6)])
    res_map4 = basic_func.starmap([(12, 5), (44, 32), (8, 3)])
    assert res_map1 == ["5", "7", "9"]
    assert res_map2 == ["21", "54", "64"]
    assert res_map3 == ["3", "7", "11"]
    assert res_map4 == ["17", "76", "11"]


def test_create_from_config():
    handler_path = [f"{TEST_RESOURCES}/basic_test_handler.py"]
    name = "test_lambda_from_config"
    config = {
        "paths_to_code": handler_path,
        "handler_function_name": "lambda_sum",
        "runtime": "python3.9",
        "args_names": ["arg1", "arg2"],
        "name": name,
    }
    config_lambda = rh.AWSLambdaFunction.from_config(config)
    time.sleep(4)  # letting the lambda be updated in AWS.
    res1 = config_lambda(1, 2)
    res2 = config_lambda(8, 12)
    res3 = config_lambda(14, 17)

    assert res1 == "3"
    assert res2 == "20"
    assert res3 == "31"
    LAMBDAS_NAMES.add(config_lambda.name)


def test_share_lambda():
    basic_func = rh.aws_lambda_function(name="test_lambda_create_and_run")
    time.sleep(1)
    users = ["josh@run.house"]
    added_users, new_users = basic_func.share(
        users=users, notify_users=True, access_type="write"
    )
    assert added_users == {}
    assert new_users == {}


def test_remove_resources():
    curr_folder = os.getcwd()
    remoteDirectoryName = "test_helpers"
    shutil.rmtree(f"{curr_folder}/{remoteDirectoryName}")
    for lambda_name in LAMBDAS_NAMES:
        policy_name = f"{lambda_name}_Policy"
        role_name = f"{lambda_name}_Role"
        del_policy = IAM_CLIENT.delete_role_policy(
            RoleName=role_name, PolicyName=policy_name
        )
        del_role = IAM_CLIENT.delete_role(RoleName=role_name)
        del_lambda = LAMBDA_CLIENT.delete_function(FunctionName=lambda_name)
        assert del_policy is not None
        assert del_role is not None
        assert del_lambda is not None


if __name__ == "__main__":
    download_resources()
    unittest.main()
