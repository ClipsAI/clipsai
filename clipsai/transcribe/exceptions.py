"""
Exceptions that can be raised by the transcribe package.
"""
# local package imports
from ..exceptions import InvalidInputDataError


class WhisperXTranscriberConfigError(InvalidInputDataError):
    pass


class NoSpeechError(WhisperXTranscriberConfigError):
    pass


class WhisperXTranscriptionError(NoSpeechError):
    pass
