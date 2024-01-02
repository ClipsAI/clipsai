"""
Handles and extracts useful information from errors (Exceptions)
"""
# standard library imports
import sys
import traceback

# local package imports
from ..exceptions import InvalidInputDataError
from ..transcribe.exceptions import NoSpeechError


class ExceptionHandler:
    """
    A class for handling and extracting useful information from errors (Exceptions)
    """

    # Status Codes
    SUCCESS = 0

    # Invalid Input Data
    INVALID_INPUT_DATA = SUCCESS + 1

    # Files must have an audio stream to transcribed
    NO_SPEECH_ERROR = INVALID_INPUT_DATA + 1

    # Other
    OTHER = NO_SPEECH_ERROR + 1

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

        if isinstance(e, InvalidInputDataError):
            return self.INVALID_INPUT_DATA

        elif isinstance(e, NoSpeechError):
            return self.NO_SPEECH_ERROR

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
                "Error Type: {} | ".format(err_type)
                + "Filename: {} | ".format(stack.filename)
                + "Stack Name: {} | ".format(stack.name)
                + "Line Number: {} | ".format(stack.lineno)
                + "Line of Code Responsible: {!r} | ".format(stack.line)
                + "Error Message: {} | ".format(err_msg)
            )
            stack_trace_info.append(message + " " + "#" * 10 + " ")
        return stack_trace_info
