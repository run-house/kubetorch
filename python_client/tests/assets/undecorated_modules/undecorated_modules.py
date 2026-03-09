"""Test module with undecorated functions and classes for CLI deploy tests."""


def plain_function(x):
    return x + 1


class PlainClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value


def check_numpy_version():
    import numpy as np

    return np.__version__


not_a_callable = "just a string"
