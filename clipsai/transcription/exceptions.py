"""
Exceptions that can be raised by the transcription package.
"""


class TranscriptionError(Exception):
    pass


class WhisperXTranscriptionError(TranscriptionError):
    pass
