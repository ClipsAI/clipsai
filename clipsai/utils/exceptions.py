"""
Exceptions that can be raised by the utils package.
"""


class InvalidComputeDeviceError(Exception):
    pass


class TimerError(Exception):
    pass


class EnvironmentVariableNotSetError(Exception):
    pass
