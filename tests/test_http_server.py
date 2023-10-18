import unittest

import pytest


@pytest.mark.dockertest
@pytest.mark.httpservertest
def test_cluster_is_up(local_docker_slim):
    cluster = local_docker_slim
    cluster.check_server()
    assert cluster.is_up()  # Should be true for a Cluster object


if __name__ == "__main__":
    unittest.main()
