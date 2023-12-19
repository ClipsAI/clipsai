"""
Exceptions that can be raised by the api package.
"""


class InvalidRequestError(Exception):
    pass


class AuthenticationError(InvalidRequestError):
    pass


class InvalidRequestDataError(InvalidRequestError):
    pass


class ServerError(Exception):
    pass
