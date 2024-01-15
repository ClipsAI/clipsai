"""
Finding clips with AudioFiles using the TextTiling algorithm.
"""
# standard library imports
import logging

# current package imports
from .clip import Clip
from .exceptions import ClipFinderError
from .text_embedder import TextEmbedder
from .texttiler import TextTiler
from .texttiler import TextTilerConfigManager

# local package imports
from clipsai.transcribe.transcription import Transcription
from clipsai.utils.pytorch import get_compute_device, assert_compute_device_available
from clipsai.utils.utils import find_missing_dict_keys

# 3rd party imports
import torch

BOUNDARY = 1


class ClipFinder:
    """
    A class for finding clips within some audio file using the TextTiling Algorithm.
    """

    def __init__(
        self,
        device: str = None,
        min_clip_duration: int = 15,
        max_clip_duration: int = 900,
        cutoff_policy: str = "high",
        embedding_aggregation_pool_method: str = "max",
        smoothing_width: int = 3,
        window_compare_pool_method: str = "mean",
    ) -> None:
        """
        Parameters
        ----------
        device: str
            PyTorch device to perform computations on. Ex: 'cpu', 'cuda'. Default is
            None (auto detects the correct device)
        min_clip_duration: int
            Minimum duration in seconds for a clip
        max_clip_duration: int
            Maximum duration in seconds for a clip
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
        """
        # configuration check
        config_manager = ClipFinderConfigManager()
        config_manager.assert_valid_config(
            {
                "cutoff_policy": cutoff_policy,
                "embedding_aggregation_pool_method": embedding_aggregation_pool_method,
                "max_clip_duration": max_clip_duration,
                "min_clip_duration": min_clip_duration,
                "smoothing_width": smoothing_width,
                "window_compare_pool_method": window_compare_pool_method,
            }
        )
        if device is None:
            device = get_compute_device()
        assert_compute_device_available(device)
        self._device = device
        self._cutoff_policy = cutoff_policy
        self._embedding_aggregation_pool_method = embedding_aggregation_pool_method
        self._min_clip_duration = min_clip_duration
        self._max_clip_duration = max_clip_duration
        self._smoothing_width = smoothing_width
        self._window_compare_pool_method = window_compare_pool_method

    def find_clips(
        self,
        transcription: Transcription,
    ) -> list[Clip]:
        """
        Finds clips in an audio file's transcription using the TextTiling Algorithm.

        Parameters
        ----------
        transcription: Transcription
            the transcription of the source media to find clips within

        Returns
        -------
        list[dict]
            list of tuples containing data about clips
        """
        # get the transcription as a list of sentences
        sentences = []
        sentences_info = transcription.get_sentence_info()
        for sentence_info in sentences_info:
            sentences.append(sentence_info["sentence"])

        # embed sentences
        text_embedder = TextEmbedder()
        sentence_embeddings = text_embedder.embed_sentences(sentences)

        # add full media as clip
        clips = []
        if transcription.end_time <= self._max_clip_duration:
            full_media_clip = {}
            full_media_clip["start_char"] = 0
            full_media_clip["end_char"] = len(transcription.get_char_info())
            full_media_clip["start_time"] = 0
            full_media_clip["end_time"] = transcription.end_time
            full_media_clip["norm"] = 1.0
            clips.append(full_media_clip)

        # <3 min clips
        k_vals = [5, 7]
        for k in k_vals:
            clips = self._text_tile_multiple_rounds(
                sentences_info,
                sentence_embeddings,
                k,
                self._min_clip_duration,
                self._max_clip_duration,
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
                self._max_clip_duration,
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
                self._max_clip_duration,
                clips,
            )

        clip_objects = []
        for clip_info in clips:
            clip_objects.append(
                Clip(
                    clip_info["start_time"],
                    clip_info["end_time"],
                    clip_info["start_char"],
                    clip_info["end_char"],
                )
            )

        return clip_objects

    def _text_tile_multiple_rounds(
        self,
        clips: list[dict],
        clip_embeddings: torch.tensor,
        k: int,
        min_clip_duration: int,
        max_clip_duration: int,
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
                min_clip_duration,
                max_clip_duration,
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
            raise ClipFinderError(err)

        # execute text tiling
        texttiler = TextTiler(self._device)

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
                super_clip["start_char"] = clips[clip_start_idx]["start_char"]
                super_clip["end_char"] = clips[clip_end_idx]["end_char"]
                super_clip["start_time"] = clips[clip_start_idx]["start_time"]
                super_clip["end_time"] = clips[clip_end_idx]["end_time"]
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
            clip_duration = clip["end_time"] - clip["start_time"]
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
            start_time_diff = abs(potential_clip["start_time"] - clip["start_time"])
            end_time_diff = abs(potential_clip["end_time"] - clip["end_time"])

            if (start_time_diff + end_time_diff) < 15:
                return True

        return False


class ClipFinderConfigManager(TextTilerConfigManager):
    """
    A class for getting information about and validating TextTiler configuration
    settings.
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        None
        """
        super().__init__()

    def impute_default_config(self, config: dict) -> dict:
        """
        Populates input data with default values if they are not provided.

        Parameters
        ----------
        config: dict
            The input data to be imputed.

        Returns
        -------
        dict
            The imputed input data.
        """
        default_values = {
            "compute_device": "cpu",
            "cutoff_policy": "high",
            "embedding_aggregation_pool_method": "max",
            "min_clip_time": 15,
            "max_clip_time": 900,
            "smoothing_width": 3,
            "window_compare_pool_method": "mean",
        }

        for key in default_values.keys():
            if key not in config:
                config[key] = default_values[key]

        return config

    def check_valid_config(
        self,
        texttile_config: dict,
    ) -> str or None:
        """
        Checks that 'texttile_config' contains valid configuration settings. Returns
        None if valid, a descriptive error message if invalid.

        Parameters
        ----------
        texttile_config: dict
            A dictionary containing the configuration settings for TextTiler.

        Returns
        -------
        str or None
            None if the inputs are valid, otherwise an error message.
        """
        # existence check
        required_keys = [
            "cutoff_policy",
            "embedding_aggregation_pool_method",
            "max_clip_duration",
            "min_clip_duration",
            "smoothing_width",
            "window_compare_pool_method",
        ]
        missing_keys = find_missing_dict_keys(texttile_config, required_keys)
        if len(missing_keys) != 0:
            return "TextTiler missing configuration settings: {}".format(missing_keys)

        # value checks
        err = self.check_valid_clip_times(
            texttile_config["min_clip_duration"],
            texttile_config["max_clip_duration"],
        )
        if err is not None:
            return err

        setting_checkers = {
            "cutoff_policy": self.check_valid_cutoff_policy,
            "embedding_aggregation_pool_method": self.check_valid_embedding_aggregation_pool_method,
            "smoothing_width": self.check_valid_smoothing_width,
            "window_compare_pool_method": self.check_valid_window_compare_pool_method,
        }
        for setting, checker in setting_checkers.items():
            err = checker(texttile_config[setting])
            if err is not None:
                return err

        return None

    def check_valid_clip_times(
        self, min_clip_duration: float, max_clip_duration: float
    ) -> str or None:
        """
        Checks the clip times are valid. Returns None if the clip times are valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        min_clip_duration: float
            The minimum clip time in seconds
        max_clip_duration: float
            The maximum clip time in seconds

        Returns
        -------
        str or None
            None if the clip times are valid, otherwise an error message.
        """
        # type check
        self._type_checker.check_type(
            min_clip_duration, "min_clip_duration", (float, int)
        )
        self._type_checker.check_type(
            max_clip_duration, "max_clip_duration", (float, int)
        )

        # minimum clip time
        if min_clip_duration < 0:
            error = "min_clip_duration must be 0 or greater, not {}" "".format(
                min_clip_duration
            )
            return error

        # maximum clip time
        if max_clip_duration <= min_clip_duration:
            error = (
                "max_clip_duration of {} must be greater than "
                "min_clip_duration of {}"
                "".format(max_clip_duration, min_clip_duration)
            )
            return error

        return None
