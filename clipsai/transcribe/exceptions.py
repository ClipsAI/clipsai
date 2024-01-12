"""
Exceptions that can be raised by the transcribe package.
"""


class TranscriberConfigError(Exception):
    pass


class NoSpeechError(TranscriberConfigError):
    pass


class TranscriptionError(NoSpeechError):
    pass
