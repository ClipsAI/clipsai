"""
Processes a request to transcribe and clip a media file.
"""
# standard library imports
import logging

# current package imports
from .clip_input_validator import ClipInputValidator
from .texttile import TextTileClipFinder
from .clip_model import Clip

# local package imports
from utils.exception_handler import ExceptionHandler
from transcribe.transcription import Transcription


def clip(
    transcription: Transcription,
    min_clip_duration: float = 15,
    max_clip_duration: float = 900,
    cutoff_policy: str = "high",
    smoothing_width: int = 3,
    window_compare_pool_method: str = "mean",
    embedding_aggregation_pool_method: str = "mean",
    device: str = "auto",
) -> list[Clip]:
    """
    Takes in the transcript of an mp4 or mp3 file and finds engaging audio or
    video clips based on the passed in transcript.

    Parameters
    ----------
    transcription: Transcription
        The transcription of the media file.
    min_clip_duration: float
        The minimum clip duration in seconds.
    max_clip_duration: float
        The maximum clip duration in seconds.
    cutoff_policy: str
        The policy used to determine how dissimilar adjacent embedding windows must
        be to consider them to be from different segments (a boundary).
        Possible values: 'average', 'high', or 'low'
    smoothing_width: int
        The width of the window used by the smoothing method
    window_compare_pool_method: str
        the method used to pool embeddings within windows (of size k) for comparison
        to adjacent windows.
        Possible values: 'mean', 'max'
    embedding_aggregation_pool_method: str
        the method used to pool embeddings within a segment to create a single
        embedding for the segment.
        Possible values: 'mean', 'max'
    device: str
        The device to use when clipping on. Ex: 'cpu', 'cuda'

    Returns
    -------
    list[Clip]
        A list containing all of the clips found in the media file. Each clip
        contains a start_time, end_time, start_char, and end_char,
        corresponding to the transcript.
    """
    exception_handler = ExceptionHandler()
    input_data = validate_input_data(
        min_clip_duration,
        max_clip_duration,
        cutoff_policy,
        smoothing_width,
        window_compare_pool_method,
        embedding_aggregation_pool_method,
        device,
        exception_handler,
    )

    # run the clip process
    try:
        logging.debug("FINDING ASSET CLIPS")
        clip_finder = TextTileClipFinder(
            device=input_data["computeDevice"],
            min_clip_duration_secs=input_data["minClipTime"],
            max_clip_duration_secs=input_data["maxClipTime"],
            cutoff_policy=input_data["cutoffPolicy"],
            embedding_aggregation_pool_method=input_data[
                "embeddingAggregationPoolMethod"
            ],
            smoothing_width=input_data["smoothingWidth"],
            window_compare_pool_method=input_data["windowComparePoolMethod"],
            save_results=False,
        )
        clip_infos = clip_finder.find_clips(transcription)
        logging.debug("POPULATING LIST OF CLIPS")
        clips = []
        for clip_info in clip_infos:
            clip = Clip(
                clip_info["startTime"],
                clip_info["endTime"],
                clip_info["startChar"],
                clip_info["endChar"]
            )
            clips.append(clip)

        logging.debug("FINISHED CLIPPING MEDIA FILE")
        return clips

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
            "data": input_data,
        }
        logging.error("ERROR INFO FOR FAILED REQUESTR: {}".format(error_info))
        logging.error("DATA FOR FAILED REQUEST: {}".format(input_data))

        return error_info
    
def validate_input_data(
    min_clip_duration: float,
    max_clip_duration: float,
    cutoff_policy: str,
    smoothing_width: int,
    window_compare_pool_method: str,
    embedding_aggregation_pool_method: str,
    device: str,
    exception_handler: ExceptionHandler,
) -> dict:
    """
    Validates the paramters for the clip function.

    Parameters
    ----------
    min_clip_duration: float
        The minimum clip duration in seconds.
    max_clip_duration: float
        The maximum clip duration in seconds.
    cutoff_policy: str
        The policy used to determine how dissimilar adjacent embedding windows must
        be to consider them to be from different segments (a boundary).
        Possible values: 'average', 'high', or 'low'
    smoothing_width: int
        The width of the window used by the smoothing method
    window_compare_pool_method: str
        the method used to pool embeddings within windows (of size k) for comparison
        to adjacent windows.
        Possible values: 'mean', 'max'
    embedding_aggregation_pool_method: str
        embedding_aggregation_pool_method: str
        the method used to pool embeddings within a segment to create a single
        embedding for the segment.
        Possible values: 'mean', 'max'
    device: str = "auto"
        The device to use when clipping on. Ex: 'cpu', 'cuda'

    """
    try:
        clip_input_validator = ClipInputValidator()
        temp_data = {
            "computeDevice": device,
            "cutoffPolicy": cutoff_policy,
            "embeddingAggregationPoolMethod": embedding_aggregation_pool_method,
            "minClipTime": min_clip_duration,
            "maxClipTime": max_clip_duration,
            "smoothingWidth": smoothing_width,
            "windowComparePoolMethod": window_compare_pool_method,
        }
        input_data = clip_input_validator.impute_input_data_defaults(temp_data)
        clip_input_validator.assert_valid_input_data(input_data)
        return input_data, 
    except Exception as e:
        status_code = exception_handler.get_status_code(e)
        err_msg = str(e)
        stack_trace = exception_handler.get_stack_trace_info()

        error_info = {
            "success": False,
            "status": status_code,
            "message": err_msg,
            "stackTraceInfo": stack_trace,
            "data": input_data,
        }
        logging.error(error_info)
        return error_info