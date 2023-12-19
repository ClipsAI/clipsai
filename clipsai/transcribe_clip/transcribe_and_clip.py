"""
Processes a request to transcribe and clip a media file.
"""
# standard library imports
import logging

# current package imports
from .clip_request_validator import TranscribeAndClipRequestValidator

# local package imports
from ..utils.exception_handler import ExceptionHandler
from .clipfinder import ClipFinder
from .transcriber import Transcriber
from ..media.audio_file import AudioFile


class TranscribeAndClip:
    """
    Class that transcribes and clips an asset.
    """

    def __init__(self) -> None:
        """
        Initialize the TranscribeAssetWorker class
        """
        self._request_validator = TranscribeAndClipRequestValidator()
        self._exception_handler = ExceptionHandler()

    def process(self, request_data: dict) -> dict:
        """
        Kicks off the transcribe and clip request.

        Parameters
        ----------
        request_data: dict
            The request data to be handled.

        Returns
        -------
        dict
            The request results.
        """
        # authenticate the request
        try:
            request_data = self._request_validator.impute_request_data_defaults(
                request_data
            )
            self._request_validator.assert_valid_request_data(request_data)
        # authentication failure
        except Exception as e:
            status_code = self._exception_handler.get_status_code(e)
            err_msg = str(e)
            stack_trace = self._exception_handler.get_stack_trace_info()

            error_info = {
                "success": False,
                "status": status_code,
                "message": err_msg,
                "stackTraceInfo": stack_trace,
            }
            logging.error(error_info)

            return {"state": "failed"}

        # run the worker process
        try:
            return self._run(request_data)
        # failure
        except Exception as e:
            self._handle_exception(e, request_data)
            return {"state": "failed"}

    def _run(self, request_data: dict) -> list:
        """
        Transcribes and finds clips in the media file.

        Parameters
        ----------
        request_data: dict
            The request data to be handled.

        Returns
        -------
        list
            The list of clip info, each containing a start_char, end_char,
            start_time, and end_time.

        Should we get rid of "request_data" and just use this?
        optional_fields_default_values = {
            "computeDevice": None,
            # transcription
            "precision": None,
            "languageCode": "en",
            "whisperModelSize": None,
            # clip
            "cutoffPolicy": "high",
            "embeddingAggregationPoolMethod": "max",
            "minClipTime": 15,
            "maxClipTime": 900,
            "smoothingWidth": 3,
            "windowComparePoolMethod": "mean",
        }
        """
        # we already validates that the request data is valid, including the file path
        media_file = AudioFile(request_data["mediaFilePath"])
        # transcribe the media file
        logging.info("TRANSCRIBING ASSET")
        transcriber = Transcriber()
        transcribe_config = {
            "model_size": request_data["whisperModelSize"],
            "language": request_data["languageCode"],
            "precision": request_data["precision"],
            "device": request_data["computeDevice"],
        }
        transcript = transcriber.build(media_file, transcribe_config)

        # find clips in the media file
        logging.info("FINDING ASSET CLIPS")
        clip_finder = ClipFinder()
        texttile_config = {
            "cutoff_policy": request_data["cutoffPolicy"],
            "device": request_data["computeDevice"],
            "embedding_aggregation_pool_method": request_data[
                "embeddingAggregationPoolMethod"
            ],
            "min_clip_duration_secs": request_data["minClipTime"],
            "max_clip_duration_secs": request_data["maxClipTime"],
            "smoothing_width": request_data["smoothingWidth"],
            "window_compare_pool_method": request_data["windowComparePoolMethod"],
        }
        clip_infos = clip_finder.build(media_file, transcript, texttile_config)

        logging.info("POPULATING CLIPS DICT")
        clips = []
        for clip_info in clip_infos:
            clip = {}
            clip["startChar"] = clip_info["startChar"]
            clip["endChar"] = clip_info["endChar"]
            clip["startTime"] = clip_info["startTime"]
            clip["endTime"] = clip_info["endTime"]
            clip["norm"] = clip_info["norm"]  # what is this?
            clips.append(clip)

        # success
        logging.info("SUCCESSFULLY TRANSCRIBED AND CLIPPED")

        return clips

    def _handle_exception(self, e: Exception, request_data: dict) -> dict:
        """
        Handles an exception that occurred during transcribing or clipping.

        Parameters
        ----------
        e: Exception
            The exception that occurred.
        request_data: dict
            The request data needed to run the Worker.

        Returns
        -------
        dict
            The request results failure information.
        """
        status_code = self._exception_handler.get_status_code(e)
        err_msg = str(e)
        stack_trace = self._exception_handler.get_stack_trace_info()

        # define failure information
        error_info = {
            "success": False,
            "status": status_code,
            "message": err_msg,
            "stackTraceInfo": stack_trace,
        }
        logging.error("ERROR INFO FOR FAILED REQUESTR: {}".format(error_info))
        logging.error("DATA FOR FAILED REQUEST: {}".format(request_data))

        return {"state": "failed", "error": error_info}
