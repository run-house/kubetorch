"""Test module with undecorated functions and classes for CLI deploy tests."""


def plain_function(x):
    return x + 1


class PlainClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value


not_a_callable = "just a string"
