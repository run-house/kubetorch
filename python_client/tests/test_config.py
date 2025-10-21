import os

import pytest

from kubetorch.globals import config

from .utils import summer


@pytest.mark.level("unit")
def test_config_load_username_from_env_var():
    """Test helper function that loads username from the environment variable"""
    os.environ["KT_USERNAME"] = "test-user"
    assert config._get_env_var("username") == "test-user"


@pytest.mark.level("unit")
def test_config_set_and_get_username():
    """Test that the username is loaded from the config file"""
    org_username = config.username

    try:
        # Set username
        config.set("username", "test-user")
        assert config.username == "test-user"

        config.set("username", "test-user-2")
        assert config.username == "test-user-2"
    except Exception as e:
        raise e
    finally:
        # Restore original config
        config.set("username", org_username)

    assert config.username == org_username


@pytest.mark.level("unit")
def test_config_set_invalid_username():
    """Test that the username is invalid if it is too long or contains invalid characters"""
    config.set("username", "test-user-2" * 10)
    assert config.username == "test-user-2test"

    config.set("username", "01-test-user")
    assert config.username == "test-user"

    with pytest.raises(ValueError):
        config.set("username", "test.user-2")

    with pytest.raises(ValueError):
        config.set("username", "usern@me")


@pytest.mark.level("unit")
def test_config_username_set_on_module():
    """Test that the username is set on the module"""
    import kubetorch as kt

    org_username = config.username

    try:
        config.set("username", "test-user")
        assert config.username == "test-user"

        fn = kt.fn(summer, name="summer")
        assert fn.service_name == "test-user-summer"
    except Exception as e:
        raise e
    finally:
        config.set("username", org_username)

    assert config.username == org_username
