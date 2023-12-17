"""
Finding clips in a asset's media.
"""
# standard library imports
import logging

# local package imports
from transcription.whisperx import WhisperXTranscription
from media.temporal_media_file import TemporalMediaFile

# machine learning imports
from ml.clipfind.texttile import TextTileClipFinder


class ClipFinder():
    """
    A class for finding clips in a asset's media.
    """

    def build(
        self,
        temporal_media_file: TemporalMediaFile,
        transcription: WhisperXTranscription,
        texttile_config: dict,
    ) -> list[dict]:
        """
        Finds clips from the temporal_media_file.

        Parameters
        ----------
        temporal_media_file: TemporalMediaFile
            the media file to find clips from
        transcription: WhisperXTranscription
            the transcription of the asset media
        texttile_config: dict
            dictionary containing the configuration settings for the clip finder

        Returns
        -------
        list[dict]
            list of clips found in the asset's media
        """
        logging.info("FINDING CLIPS IN MEDIA")

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
