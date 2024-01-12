"""
Exceptions that can be raised by the utils package.
"""


class ConfigError(Exception):
    pass


class EnvironmentVariableNotSetError(Exception):
    pass


class InvalidComputeDeviceError(Exception):
    pass


class InvalidInputDataError(Exception):
    pass


class TimerError(Exception):
    pass
