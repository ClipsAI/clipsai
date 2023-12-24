"""
Exceptions that can be raised by the transcribe package.
"""


class WhisperXTranscriberConfigError(Exception):
    pass


class NoSpeechError(WhisperXTranscriberConfigError):
    pass


class WhisperXTranscriptionError(NoSpeechError):
    pass
