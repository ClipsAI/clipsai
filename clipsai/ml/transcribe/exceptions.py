"""
Exceptions that can be raised by the transcribe package.
"""
# local package imports
from ..exceptions import MLConfigError


class WhisperXTranscriberConfigError(MLConfigError):
    pass


class NoSpeechError(Exception):
    pass
