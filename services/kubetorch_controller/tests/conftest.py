import os
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

TEST_DB = "/tmp/kubetorch_test.db"


# Set DB path for the whole test session
@pytest.fixture(scope="session", autouse=True)
def setup_test_db_path():
    os.environ["KUBETORCH_DB_PATH"] = TEST_DB


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    # Create a clean DB file before and after the test session
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    yield

    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


@pytest.fixture(autouse=True)
def clean_db():
    # Reset DB tables BEFORE EACH TEST so counts are deterministic
    from core.database import Base, get_engine

    engine = get_engine()

    # Drop & recreate all tables → clean slate for each test
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


# FastAPI TestClient
@pytest.fixture
def client():
    from server import app

    return TestClient(app)


@pytest.fixture
def mock_k8s_clients():
    from core import k8s

    apps = MagicMock()
    core = MagicMock()
    custom = MagicMock()
    dyn = MagicMock()

    k8s.init(apps, core, custom, dyn)
    return apps, core, custom, dyn
