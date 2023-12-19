"""
Finding clips with AudioFiles using the TextTiling algorithm.
"""
# standard library imports
import logging

# current package imports
from .clipfinder import ClipFinder
from .exceptions import TextTileClipFinderError
from .texttile_config_manager import TextTileClipFinderConfigManager

# local package imports
from ..embed.roberta import RobertaTextEmbedder
from ..texttile.texttiler import TextTiler
from ...transcription.whisperx import WhisperXTranscription

# 3rd party imports
import torch

BOUNDARY = 1


class TextTileClipFinder(ClipFinder):
    """
    A class for finding clips within some audio file using the TextTiling Algorithm.
    """

    def __init__(
        self,
        device: str,
        min_clip_duration_secs: int = 15,
        max_clip_duration_secs: int = 900,
        cutoff_policy: str = "high",
        embedding_aggregation_pool_method: str = "mean",
        smoothing_width: int = 3,
        window_compare_pool_method: str = "mean",
        save_results: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        device: str
            PyTorch device to performm computations on
        min_clip_duration_secs: int
            minimum clip length for a clip to be created
        max_clip_duration_secs: int
            max clip length for a clip to be created
        cutoff_policy: str
            The policy used to determine how dissimilar adjacent embedding windows must
            be to consider them to be from different segments (a boundary).
            Possible values: 'average', 'high', or 'low'
        embedding_aggregation_pool_method: str
            the method used to pool embeddings within a segment to create a single
            embedding for the segment.
            Possible values: 'mean', 'max'
        smoothing_width: int
            The width of the window used by the smoothing method
        window_compare_pool_method: str
            the method used to pool embeddings within windows (of size k) for comparison
            to adjacent windows.
            Possible values: 'mean', 'max'
        save_results: bool
            if True, saves the results of the TextTiling algorithm plots
        """
        super().__init__(min_clip_duration_secs, max_clip_duration_secs, device)
        # configuration check
        config_manager = TextTileClipFinderConfigManager()
        config_manager.assert_valid_config(
            {
                "cutoff_policy": cutoff_policy,
                "embedding_aggregation_pool_method": embedding_aggregation_pool_method,
                "max_clip_duration_secs": max_clip_duration_secs,
                "min_clip_duration_secs": min_clip_duration_secs,
                "smoothing_width": smoothing_width,
                "window_compare_pool_method": window_compare_pool_method,
            }
        )
        self._cutoff_policy = cutoff_policy
        self._embedding_aggregation_pool_method = embedding_aggregation_pool_method
        self._min_clip_duration_secs = min_clip_duration_secs
        self._max_clip_duration_secs = max_clip_duration_secs
        self._smoothing_width = smoothing_width
        self._window_compare_pool_method = window_compare_pool_method
        # save results for analysis purposes or not
        self._save_results = save_results

    def find_clips(
        self,
        transcription: WhisperXTranscription,
    ) -> list[dict]:
        """
        Finds clips within some audio transcription using the TextTiling Algorithm

        Parameters
        ----------
        transcription: TranscriptionWhisperX
            the transcription of the source media to find clips within
        save_results: bool
            if True, saves the results of the TextTiling algorithm plots

        Returns
        -------
        list[dict]
            list of tuples containing data about clips
        """
        # get the transcription as a list of sentences
        sentences = []
        sentences_info = transcription.get_sentence_info(False)
        for sentence_info in sentences_info:
            sentences.append(sentence_info["sentence"])

        # embed sentences
        text_embedder = RobertaTextEmbedder()
        sentence_embeddings = text_embedder.embed_sentences(sentences)

        # add full media as clip
        clips = []
        if transcription.get_end_time(False) <= 900:
            full_media_clip = {}
            full_media_clip["startChar"] = 0
            full_media_clip["endChar"] = len(transcription.get_char_info(False))
            full_media_clip["startTime"] = 0
            full_media_clip["endTime"] = transcription.get_end_time(False)
            full_media_clip["norm"] = 1.0
            clips.append(full_media_clip)

        # <3 min clips
        k_vals = [5, 7]
        for k in k_vals:
            clips = self._text_tile_multiple_rounds(
                sentences_info,
                sentence_embeddings,
                k,
                self._min_clip_duration_secs,
                self._max_clip_duration_secs,
                clips,
            )

        # 3+ min clips
        k_vals = [11, 17]
        min_duration_secs = 180  # 3 minutes
        for k in k_vals:
            clips = self._text_tile_multiple_rounds(
                sentences_info,
                sentence_embeddings,
                k,
                min_duration_secs,
                self._max_clip_duration_secs,
                clips,
            )

        # 10+ min clips
        k_vals = [37, 53, 73, 97]
        min_duration_secs = 600  # 10 minutes
        for k in k_vals:
            clips = self._text_tile_multiple_rounds(
                sentences_info,
                sentence_embeddings,
                k,
                min_duration_secs,
                self._max_clip_duration_secs,
                clips,
            )

        return clips

    def _text_tile_multiple_rounds(
        self,
        clips: list[dict],
        clip_embeddings: torch.tensor,
        k: int,
        min_clip_duration_secs: int,
        max_clip_duration_secs: int,
        final_clips: list[dict] = [],
    ) -> tuple[list, torch.Tensor]:
        """
        Segments the embeddings multiple rounds using the TextTiling algorithm.

        Parameters
        ----------
        clips: list[dict]
            list of dictionaries containing information about clips' transcript
        clip_embeddings: torch.tensor
            clip embeddings used to segment the clips into larger clips
        k: int
            text tiling window size
        min_duration_secs: int
            minimum clip length for a clip to be created
        max_duration_secs: int
            max clip length for a clip to be created
        final_clips: list[dict]
            list of dictionaries containing information about already chosen clips

        Returns
        -------
        list[dict]
            list of dictionaries containing information about the chosen clips
        """
        self._text_tile_round = 0
        while len(clip_embeddings) > 8:
            self._text_tile_round += 1
            # segment the embeddings using the TextTiling algorithm
            super_clips, super_clip_embeddings = self._text_tile(
                clips, clip_embeddings, k
            )
            # filter clips based on length
            new_clips = self._remove_duplicates(
                super_clips,
                final_clips,
                min_clip_duration_secs,
                max_clip_duration_secs,
            )
            final_clips += new_clips
            clips = super_clips
            clip_embeddings = super_clip_embeddings

        return final_clips

    def _text_tile(
        self,
        clips: list[dict],
        clip_embeddings: torch.tensor,
        k: int,
    ) -> tuple[list, torch.Tensor]:
        """
        Segments the embeddings using the TextTiling algorithm.

        Parameters
        ----------
        clips: list[dict]
            list of dictionaries containing information about clips' transcript
        clip_embeddings: torch.tensor
            clip embeddings used to segment the clips into larger clips

        Returns
        -------
        tuple[list, torch.Tensor]
            list of dictionaries containing information about clips and the embeddings
            of the super clips
        """
        # check that the number of embeddings matches the number of clips
        if len(clip_embeddings) != len(clips):
            err = (
                "Length of embeddings ({}) does not match length of clip ({})"
                "".format(len(clip_embeddings), len(clips))
            )
            logging.error(err)
            raise TextTileClipFinderError(err)

        # execute text tiling
        texttiler = TextTiler(
            self._device,
            "kval:{}-round{}".format(k, self._text_tile_round),
        )

        # use smaller k value if number of clips is small
        if k >= len(clip_embeddings):
            k = 3

        boundaries, super_clip_embeddings = texttiler.text_tile(
            clip_embeddings,
            k,
            self._window_compare_pool_method,
            self._embedding_aggregation_pool_method,
            self._smoothing_width,
            self._cutoff_policy,
            self._save_results,
        )

        # combine clips into super clips (larger clips composed of smaller clips)
        num_clips = len(clips)
        super_clips = []
        clip_start_idx = 0
        clip_end_idx = None
        super_clip_num = 0

        for i in range(num_clips):
            if boundaries[i] == BOUNDARY:
                clip_end_idx = i
                super_clip = {}
                super_clip["startChar"] = clips[clip_start_idx]["startChar"]
                super_clip["endChar"] = clips[clip_end_idx]["endChar"]
                super_clip["startTime"] = clips[clip_start_idx]["startTime"]
                super_clip["endTime"] = clips[clip_end_idx]["endTime"]
                super_clip["norm"] = torch.linalg.norm(
                    super_clip_embeddings[super_clip_num], dim=0, ord=2
                ).item()

                super_clips.append(super_clip)
                clip_start_idx = clip_end_idx

                super_clip_num += 1

        return super_clips, super_clip_embeddings

    def _remove_duplicates(
        self,
        potential_clips: dict,
        clips_to_check_against: list[dict],
        min_duration_secs: int,
        max_duration_secs: int,
    ) -> tuple:
        """
        Removes duplicate clips from 'potential_clips' that are within the
        'clips_to_check_against' list.

        Parameters
        ----------
        potential_clips: dict
            list of potential clips
        clips_to_check_against: list[dict]
            list of clips to check against
        min_duration_secs: int
            minimum clip length for a clip to be created
        max_duration_secs: int
            max clip length for a clip to be created

        Returns
        -------
        list[dict]
            list of potential clips with duplicates removed
        """
        filtered_clips = []

        # create clip objects
        for clip in potential_clips:
            clip_duration = clip["endTime"] - clip["startTime"]
            if clip_duration < min_duration_secs:
                continue
            if clip_duration > max_duration_secs:
                continue

            if self._is_duplicate(clip, clips_to_check_against):
                continue

            filtered_clips.append(clip)

        return filtered_clips

    def _is_duplicate(
        self, potential_clip: dict, clips_to_check_against: list[dict]
    ) -> bool:
        """
        Checks if 'potential_clip' is a duplicate of any clip in clips.

        Parameters
        ----------
        potential_clip: dict
            a potential clip
        clips_to_check_against: list[dict]
            list of clips to check against

        Returns
        -------
        bool
            True if 'potential_clip' is a duplicate, False otherwise.
        """
        for clip in clips_to_check_against:
            start_time_diff = abs(potential_clip["startTime"] - clip["startTime"])
            end_time_diff = abs(potential_clip["endTime"] - clip["endTime"])

            if (start_time_diff + end_time_diff) < 15:
                return True

        return False
