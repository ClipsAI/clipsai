"""
Handles and extracts useful information from errors (Exceptions)
"""
# standard library imports
import sys
import traceback

# local package imports
from api.exceptions import InvalidRequestError
from api.internal.assets.upload.exceptions import MaxContentUploadDurationLimitError
from media.exceptions import NoAudioStreamError
from models.asset.exceptions import AssetNotResizedError
from models.user.exceptions import (
    ContentSecondsUploadedLimitError,
    StorageLimitError,
    ClipsExportedLimitError,
)
from ml.transcribe.exceptions import NoSpeechError
from webscrape.exceptions import InvalidYoutubeURLError, YoutubeDownloadError


class ExceptionHandler:
    """
    A class for handling and extracting useful information from errors (Exceptions)
    """

    # Status Codes
    SUCCESS = 0

    # Invalid Client Requests
    INVALID_REQUEST = SUCCESS + 1

    # Invalid User Inputs
    INVALID_YOUTUBE_URL_ERROR = INVALID_REQUEST + 1
    MAX_CONTENT_UPLOAD_DURATION_LIMIT_ERROR = INVALID_YOUTUBE_URL_ERROR + 1

    # Invalid User Limits
    CONTENT_SECONDS_UPLOADED_LIMIT_ERROR = MAX_CONTENT_UPLOAD_DURATION_LIMIT_ERROR + 1
    STORAGE_LIMIT_ERROR = CONTENT_SECONDS_UPLOADED_LIMIT_ERROR + 1
    CLIPS_EXPORTED_LIMIT_ERROR = STORAGE_LIMIT_ERROR + 1

    # Files must have an audio stream to transcribed
    NO_SPEECH_ERROR = CLIPS_EXPORTED_LIMIT_ERROR + 1

    YOUTUBE_DOWNLOAD_ERROR = NO_SPEECH_ERROR + 1

    ASSET_NOT_RESIZED_ERROR = YOUTUBE_DOWNLOAD_ERROR + 1

    # Other
    OTHER = ASSET_NOT_RESIZED_ERROR + 1

    def get_status_code(self, e: Exception) -> int:
        """
        Returns the CAI status code for a given Exception

        Parameters
        ----------
        error: Exception
            the error thrown by some code

        Returns
        -------
        status_code: int
            CAI status code representing that particular error
        """
        if isinstance(e, InvalidRequestError):
            return self.INVALID_REQUEST

        elif isinstance(e, InvalidYoutubeURLError):
            return self.INVALID_YOUTUBE_URL_ERROR

        elif isinstance(e, MaxContentUploadDurationLimitError):
            return self.MAX_CONTENT_UPLOAD_DURATION_LIMIT_ERROR

        elif isinstance(e, ContentSecondsUploadedLimitError):
            return self.CONTENT_SECONDS_UPLOADED_LIMIT_ERROR

        elif isinstance(e, StorageLimitError):
            return self.STORAGE_LIMIT_ERROR

        elif isinstance(e, ClipsExportedLimitError):
            return self.CLIPS_EXPORTED_LIMIT_ERROR

        elif isinstance(e, NoAudioStreamError):
            return self.NO_SPEECH_ERROR

        elif isinstance(e, NoSpeechError):
            return self.NO_SPEECH_ERROR

        elif isinstance(e, YoutubeDownloadError):
            return self.YOUTUBE_DOWNLOAD_ERROR

        elif isinstance(e, AssetNotResizedError):
            return self.ASSET_NOT_RESIZED_ERROR

        return self.OTHER

    def get_stack_trace_info(self) -> dict:
        """
        Returns the stack trace information for a given error

        Parameters
        ----------
        None

        Returns
        -------
        stack_trace_info: dict
            Stack trace information of the error
        """
        stack_trace_info = []
        exc_type, exc_value, exc_tb = sys.exc_info()
        stack_summary = traceback.extract_tb(exc_tb)
        for stack in stack_summary:
            err_type = type(exc_value).__name__
            err_msg = str(exc_value)
            message = (
                "Error Type: {} | ".format(err_type) +
                "Filename: {} | ".format(stack.filename) +
                "Stack Name: {} | ".format(stack.name) +
                "Line Number: {} | ".format(stack.lineno) +
                "Line of Code Responsible: {!r} | ".format(stack.line) +
                "Error Message: {} | ".format(err_msg)
            )
            stack_trace_info.append(message + " " + "#" * 10 + " ")
        return stack_trace_info
