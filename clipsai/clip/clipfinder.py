"""
Finding clips in a asset's media.
"""
# standard library imports
import logging

# local package imports
from ..transcribe.whisperx_transcription import WhisperXTranscription

# machine learning imports
from .texttile import TextTileClipFinder


class ClipFinder:
    """
    A class for finding clips in a asset's media.
    """

    def build(
        self,
        transcription: WhisperXTranscription,
        texttile_config: dict,
    ) -> list[dict]:
        """
        Finds clips using the transcription.

        Parameters
        ----------
        transcription: WhisperXTranscription
            the transcription of the asset media
        texttile_config: dict
            dictionary containing the configuration settings for the clip finder

        Returns
        -------
        list[dict]
            list of clips found in the asset's media
        """
        logging.info("FINDING CLIPS IN MEDIA FILE")

        # find clips
        clip_finder = TextTileClipFinder(
            device=texttile_config["device"],
            min_clip_duration_secs=texttile_config["min_clip_duration_secs"],
            max_clip_duration_secs=texttile_config["max_clip_duration_secs"],
            cutoff_policy=texttile_config["cutoff_policy"],
            embedding_aggregation_pool_method=texttile_config[
                "embedding_aggregation_pool_method"
            ],
            smoothing_width=texttile_config["smoothing_width"],
            window_compare_pool_method=texttile_config["window_compare_pool_method"],
            save_results=False,
        )
        clips = clip_finder.find_clips(transcription)

        return clips
