import os

import kubetorch as kt


@kt.compute(cpus=".1")
def get_pod_id_1():
    return os.environ["KT_SERVICE_NAME"]


@kt.compute(cpus=".1")
class RemoteArray:
    def __init__(self, length=5):
        self.data = [0] * length

    def set_len(self, length):
        self.data = [0] * length

    def get_data(self):
        return self.data


@kt.compute(cpus=".1")
@kt.async_
def get_pod_id_async():
    return os.environ["KT_SERVICE_NAME"]
