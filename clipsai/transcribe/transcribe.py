"""
Processes a request to transcribe a media file. Trancribing a media file is the
prerequisite to clipping it.
"""
# standard library imports
import logging

# current package imports
from .transcriber import WhisperXTranscriber
from .transcription import Transcription
from .transcribe_input_validator import TranscribeInputValidator

# local package imports
from utils.exception_handler import ExceptionHandler
from media.editor import MediaEditor


def transcribe(
    media_file_path: str, language_code: str = "auto", device: str = "auto"
) -> Transcription:
    """
    Takes in a file in the form of mp3 or mp4 and transcribes it using whisper.

    Parameters
    ----------
    media_file_path: str
        The path to the media mp3 or mp4 file to transcribe.
    language_code: str
        The language code of the media file. Ex: 'en'
    device: str
        The device to use when transcribing. Ex: 'cpu', 'cuda'

    Returns
    -------
    Transcription
        The Transcription object containing transcription information.
    """
    # validate the input request data
    try:
        transcribe_input_validator = TranscribeInputValidator()
        exception_handler = ExceptionHandler()
        editor = MediaEditor()
        temp_data: dict = {
            "mediaFilePath": media_file_path,
            "computeDevice": device,
            "precision": None,
            "languageCode": language_code,
            "whisperModelSize": None,
        }
        input_data = transcribe_input_validator.impute_input_data_defaults(
            input_data=temp_data
        )
        transcribe_input_validator.assert_valid_input_data(input_data)
    except Exception as e:
        status_code = exception_handler.get_status_code(e)
        err_msg = str(e)
        stack_trace = exception_handler.get_stack_trace_info()

        error_info = {
            "success": False,
            "status": status_code,
            "message": err_msg,
            "stackTraceInfo": stack_trace,
        }
        logging.error(error_info)

        return {"state": "failed"}

    # now we can transcribe the media file
    try:
        media_file = editor.instantiate_as_temporal_media_file(media_file_path)
        logging.debug("TRANSCRIBING MEDIA FILE")
        transcriber = WhisperXTranscriber(
            input_data["whisperModelSize"],
            input_data["computeDevice"],
            input_data["precision"],
        )
        transcription = transcriber.transcribe(
            media_file,
            input_data["languageCode"],
        )
        logging.debug("TRANSCRIPTION STAGE COMPLETE")
        return transcription

    except Exception as e:
        status_code = exception_handler.get_status_code(e)
        err_msg = str(e)
        stack_trace = exception_handler.get_stack_trace_info()

        # define failure information
        error_info = {
            "success": False,
            "status": status_code,
            "message": err_msg,
            "stackTraceInfo": stack_trace,
        }
        logging.error("ERROR INFO FOR FAILED REQUESTR: {}".format(error_info))
        logging.error("DATA FOR FAILED REQUEST: {}".format(input_data))

        return {"state": "failed"}
