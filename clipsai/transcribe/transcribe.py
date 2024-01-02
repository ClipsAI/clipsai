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
from ..utils.exception_handler import ExceptionHandler
from ..media.editor import MediaEditor


def transcribe(
    media_file_path: str,
    language_code: str = "auto",
    whisper_model_size: str = None,
    precision: str = None,
    device: str = "auto"
) -> Transcription:
    """
    Takes in a file in the form of mp3 or mp4 and transcribes it using whisper.

    Parameters
    ----------
    media_file_path: str
        The path to the media mp3 or mp4 file to transcribe.
    language_code: str
        The ISO 639-1 language code of the media file. Default is "auto".
    whisper_model_size: str
        The size of the whisper model to use
        Options: tiny, base, small, medium, large, large-v1, large-v2, large-v3
    precision: 'float16' | 'int8'
        Precision to perform prediction with
    device: str
        The device to use when transcribing. Ex: 'cpu', 'cuda'

    Returns
    -------
    Transcription
        The Transcription object containing transcription information.
    """
    # validate the input request data
    exception_handler = ExceptionHandler()
    input_data = valid_input_data(
        media_file_path,
        language_code,
        whisper_model_size,
        precision,
        device,
        exception_handler
    )

    # now we can transcribe the media file
    try:
        editor = MediaEditor()
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

def valid_input_data(
    media_file_path: str,
    language_code: str,
    whisper_model_size: str,
    precision: str,
    device: str,
    exception_handler: ExceptionHandler
) -> dict:
    """
    Validates the paramters for the transcribe function.

    Parameters
    ----------
    language_code: str
        The ISO 639-1 language code of the media file. Default is "auto".
    whisper_model_size: str 
        The size of the whisper model to use
        Options: tiny, base, small, medium, large, large-v1, large-v2, large-v3
    precision: 'float16' | 'int8'
        Precision to perform prediction with
    device: str
        The device to use when transcribing. Ex: 'cpu', 'cuda'

    Returns
    -------
    dict
        The input data dictionary.
    """
    try:
        transcribe_input_validator = TranscribeInputValidator()
        temp_data: dict = {
            "mediaFilePath": media_file_path,
            "computeDevice": device,
            "precision": precision,
            "languageCode": language_code,
            "whisperModelSize": whisper_model_size,
        }
        input_data = transcribe_input_validator.impute_input_data_defaults(
            input_data=temp_data
        )
        transcribe_input_validator.assert_valid_input_data(input_data)
        return input_data
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
