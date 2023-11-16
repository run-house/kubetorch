import logging
import time
import unittest

import pytest
import requests

from runhouse.globals import rns_client

logger = logging.getLogger(__name__)


def profile(func, reps=10):
    times = []
    for _ in range(reps):
        start = time.time()
        assert func()
        times.append(round((time.time() - start) * 1000, 2))
    return times, sum(times) / len(times)


def run_performance_tests(summer_func):
    cluster = summer_func.system
    times_list, avg_time = profile(lambda: summer_func.system.keys() is not None)
    print(f"Listing keys took {round(avg_time, 2)} ms: {times_list}")

    times_list, avg_time = profile(lambda: summer_func(1, 5) == 6)
    print(f"Call with logs took {round(avg_time, 2)} ms: {times_list}")

    times_list, avg_time = profile(lambda: summer_func(1, 5, stream_logs=False) == 6)
    print(f"Call without logs took {round(avg_time, 2)} ms: {times_list}")

    port = cluster.client.port
    suffix = "https" if cluster._use_https else "http"
    address = cluster.address

    call_url = f"{suffix}://{address}:{port}/call/summer_func/call/?serialization=None"
    logger.info(f"Call url: {call_url}")
    times_list, avg_time = profile(
        lambda: requests.post(
            call_url,
            json={"args": [1, 2]},
            headers=rns_client.request_headers if cluster.den_auth else None,
            verify=cluster.client.verify,
        ).json()
        == 3
    )
    print(f"{suffix} call took {round(avg_time, 2)} ms: {times_list}")


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_roundtrip_performance(summer_func):
    run_performance_tests(summer_func)


@pytest.mark.clustertest
@pytest.mark.rnstest
def test_https_roundtrip_performance(summer_func_with_auth):
    run_performance_tests(summer_func_with_auth)


if __name__ == "__main__":
    unittest.main()
