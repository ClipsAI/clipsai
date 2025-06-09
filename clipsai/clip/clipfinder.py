"""
Finds thematic clips within a transcribed media file using Google Cloud Natural
Language API for content classification.
"""

from __future__ import annotations
# standard library imports
import logging
import os # Added for NLP API, potentially remove if not used in final find_clips
import uuid # Added for NLP API, potentially remove if not used in final find_clips


# current package imports
from .clip import Clip
from .exceptions import ClipFinderError
from clipsai.gcloud.config import GCloudConfig
from clipsai.utils.config_manager import ConfigManager
from clipsai.utils.type_checker import TypeChecker

# local package imports
from clipsai.transcribe.transcription import Transcription
# Google Cloud NLP imports will be inside find_clips method
# Removed: from clipsai.utils.pytorch import get_compute_device, assert_compute_device_available
from clipsai.utils.utils import find_missing_dict_keys # Keep if ClipFinderConfigManager uses it

# 3rd party imports
# Removed: import torch

# BOUNDARY = 1 # No longer needed if TextTiler logic is removed


class ClipFinder:
    """
    Identifies relevant clips within a video based on thematic segmentation of its
    transcription using the Google Cloud Natural Language API.
    """

    def __init__(
        self,
        gcloud_config: GCloudConfig,
        min_clip_duration: int = 15,
        max_clip_duration: int = 900,
        nlp_text_chunk_size_words: int = 150,
    ) -> None:
        """
        Initializes the ClipFinder.

        Parameters
        ----------
        gcloud_config : GCloudConfig
            Configuration object for Google Cloud services, used for initializing
            the Natural Language API client.
        min_clip_duration : int, optional
            Minimum duration in seconds for a generated clip. Default is 15.
        max_clip_duration : int, optional
            Maximum duration in seconds for a generated clip. Default is 900.
        nlp_text_chunk_size_words : int, optional
            The approximate number of words to group into each text chunk before
            sending to the Natural Language API for content classification.
            Default is 150.
        """
        self._config_manager = ClipFinderConfigManager()
        self._config_manager.assert_valid_config({
            "min_clip_duration": min_clip_duration,
            "max_clip_duration": max_clip_duration,
            "nlp_text_chunk_size_words": nlp_text_chunk_size_words, # ADDED
        })

        self._gcloud_config = gcloud_config
        self._min_clip_duration = min_clip_duration
        self._max_clip_duration = max_clip_duration
        self.nlp_text_chunk_size_words = nlp_text_chunk_size_words # ADDED

    def find_clips(
        self,
        transcription: Transcription,
    ) -> list[Clip]:
        """
        Identifies clips by segmenting the transcription based on content categories
        determined by the Google Cloud Natural Language API.

        The process involves:
        1. Retrieving word-level information (including timings and character indices)
           from the input `Transcription` object.
        2. Grouping these words into text chunks of approximately
           `self.nlp_text_chunk_size_words`.
        3. Sending each chunk to the `classify_text` method of the Natural Language API.
        4. Defining potential clip boundaries where the primary content category changes
           between consecutive chunks.
        5. Creating `Clip` objects from these thematic segments.
        6. Filtering the generated clips by `self._min_clip_duration` and
           `self._max_clip_duration`.
        7. Including the full media as a potential clip if it meets duration criteria.

        Parameters
        ----------
        transcription : Transcription
            The `Transcription` object containing the text, word timings, and
            character indices of the media file.

        Returns
        -------
        list[Clip]
            A list of `Clip` objects representing the identified thematic segments,
            filtered by duration.

        Raises
        ------
        ClipFinderError
            If the Google Cloud Language library is not installed or if API calls fail.
        """
        try:
            from google.cloud import language_v1
            from google.api_core import exceptions as google_exceptions
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[-2]
            logging.error(f"Google Cloud Language library '{missing_module}' not found.")
            raise ClipFinderError(f"Google Cloud Language library '{missing_module}' not found. Please install google-cloud-language.") from e

        word_infos = transcription.get_word_info()
        if not word_infos:
            logging.info("No word information in transcription, cannot find clips based on text chunks.")
            return []

        # Initialize client (could be done once in __init__ if client is thread-safe and config doesn't change per call)
        # Using self.gcloud_config.project_id if needed by client, though often not for Language API.
        nlp_client = language_v1.LanguageServiceClient(
             client_options={"project_id": self._gcloud_config.project_id} if self._gcloud_config.project_id else None
        )

        text_chunks = []
        current_chunk_words_list = []
        current_chunk_text_content = ""

        for i, word_info in enumerate(word_infos):
            current_chunk_words_list.append(word_info)
            current_chunk_text_content += word_info["word"] + " "

            if len(current_chunk_words_list) >= self.nlp_text_chunk_size_words or i == len(word_infos) - 1:
                if not current_chunk_words_list: continue

                chunk_start_time = current_chunk_words_list[0]["start_time"]
                chunk_end_time = current_chunk_words_list[-1]["end_time"]
                chunk_start_char = current_chunk_words_list[0]["start_char"]
                chunk_end_char = current_chunk_words_list[-1]["end_char"]

                category_name = None
                trimmed_text_content = current_chunk_text_content.strip()
                if trimmed_text_content: # Only classify if there's actual text
                    try:
                        document = language_v1.Document(content=trimmed_text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
                        # language can be specified if known, e.g., document.language = transcription.language
                        # This might improve classification accuracy.
                        if transcription.language:
                             document.language = transcription.language

                        response = nlp_client.classify_text(document=document)
                        if response.categories:
                            category_name = response.categories[0].name
                            logging.debug(f"Chunk from {chunk_start_time:.2f}-{chunk_end_time:.2f} classified as: {category_name} (Confidence: {response.categories[0].confidence:.2f})")
                        else:
                            logging.debug(f"Chunk from {chunk_start_time:.2f}-{chunk_end_time:.2f} yielded no categories.")
                    except google_exceptions.GoogleAPICallError as e:
                        logging.error(f"NLP API classify_text failed for chunk ({chunk_start_time:.2f}-{chunk_end_time:.2f}): {e}")
                    except Exception as e:
                        logging.error(f"Error during NLP classification for chunk ({chunk_start_time:.2f}-{chunk_end_time:.2f}): {e}")
                else:
                    logging.debug(f"Skipping empty text chunk for classification ({chunk_start_time:.2f}-{chunk_end_time:.2f}).")


                text_chunks.append({
                    "text": trimmed_text_content,
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time,
                    "start_char": chunk_start_char,
                    "end_char": chunk_end_char,
                    "category": category_name
                })
                current_chunk_words_list = []
                current_chunk_text_content = ""

        if not text_chunks:
            logging.info("No text chunks were processed or classified.")
            return []

        potential_clips_info = []
        if len(text_chunks) > 0:
            current_clip_start_chunk_idx = 0
            # Use first chunk's category, even if None, to start comparison
            current_clip_category = text_chunks[0]["category"]

            for i in range(1, len(text_chunks)):
                # Segment break if category changes OR if one of them is None (meaning unclassified/error)
                # and we don't want to merge unclassified with classified, or two different unclassified chunks.
                # A stricter rule: break if text_chunks[i]["category"] != current_clip_category (None != None is False)
                # A looser rule: merge if both are None. Let's be strict: different category or change to/from None is a break.
                is_break = False
                if text_chunks[i]["category"] is None and current_clip_category is None:
                    is_break = False # Both unclassified, treat as same "segment" of unclassified text
                elif text_chunks[i]["category"] != current_clip_category:
                    is_break = True

                if is_break:
                    clip_start_time = text_chunks[current_clip_start_chunk_idx]["start_time"]
                    clip_end_time = text_chunks[i-1]["end_time"]
                    clip_start_char = text_chunks[current_clip_start_chunk_idx]["start_char"]
                    clip_end_char = text_chunks[i-1]["end_char"]
                    if clip_end_time > clip_start_time : # Ensure valid duration
                        potential_clips_info.append({"start_time": clip_start_time, "end_time": clip_end_time,
                                                   "start_char": clip_start_char, "end_char": clip_end_char,
                                                   "category": current_clip_category}) # Store category for potential scoring
                    current_clip_start_chunk_idx = i
                    current_clip_category = text_chunks[i]["category"]

            # Add the last ongoing clip
            clip_start_time = text_chunks[current_clip_start_chunk_idx]["start_time"]
            clip_end_time = text_chunks[-1]["end_time"]
            clip_start_char = text_chunks[current_clip_start_chunk_idx]["start_char"]
            clip_end_char = text_chunks[-1]["end_char"]
            if clip_end_time > clip_start_time: # Ensure valid duration
                potential_clips_info.append({"start_time": clip_start_time, "end_time": clip_end_time,
                                           "start_char": clip_start_char, "end_char": clip_end_char,
                                           "category": current_clip_category})

        final_clips = []
        for p_clip in potential_clips_info:
            duration = p_clip["end_time"] - p_clip["start_time"]
            if self._min_clip_duration <= duration <= self._max_clip_duration:
                # Ensure start_char and end_char are valid if coming from word_infos
                # Clip constructor expects integers for char indices.
                final_clips.append(Clip(start_time=p_clip["start_time"],
                                       end_time=p_clip["end_time"],
                                       start_char=int(p_clip["start_char"]),
                                       end_char=int(p_clip["end_char"])))

        # Add full media as clip if within duration
        if transcription.end_time is not None and \
           self._min_clip_duration <= transcription.end_time <= self._max_clip_duration:
            full_char_info = transcription.get_char_info()
            if full_char_info:
                total_chars = len(full_char_info)
                full_media_clip_obj = Clip(start_time=0.0,
                                           end_time=transcription.end_time,
                                           start_char=0,
                                           end_char=total_chars)
                # Avoid adding duplicate full clip
                is_duplicate = any(
                    fc.start_time == full_media_clip_obj.start_time and \
                    fc.end_time == full_media_clip_obj.end_time and \
                    fc.start_char == full_media_clip_obj.start_char and \
                    fc.end_char == full_media_clip_obj.end_char
                    for fc in final_clips
                )
                if not is_duplicate:
                    final_clips.append(full_media_clip_obj)

        logging.info(f"Found {len(final_clips)} clips after NLP processing and duration filtering.")
        return final_clips


class ClipFinderConfigManager(ConfigManager):
    """
    Manages configuration settings for the ClipFinder, focusing on parameters
    like clip duration and text chunk size for NLP processing.
    """

    def __init__(self) -> None:
        super().__init__()
        if not hasattr(self, '_type_checker'):
            self._type_checker = TypeChecker()

    def impute_default_config(self, config: dict) -> dict:
        """
        Populates the configuration dictionary with default values for
        `min_clip_duration`, `max_clip_duration`, and
        `nlp_text_chunk_size_words` if they are not already provided.

        Parameters
        ----------
        config : dict
            The configuration dictionary to impute.

        Returns
        -------
        dict
            The configuration dictionary with defaults applied.
        """
        default_values = {
            "min_clip_duration": 15,
            "max_clip_duration": 900,
            "nlp_text_chunk_size_words": 150,
        }
        for key, value in default_values.items():
            if key not in config:
                config[key] = value
        return config

    def check_valid_config(
        self,
        clip_finder_config: dict,
    ) -> str or None:
        """
        Validates the provided ClipFinder configuration.

        Checks for the presence and validity of:
        - `min_clip_duration`
        - `max_clip_duration`
        - `nlp_text_chunk_size_words`

        Parameters
        ----------
        clip_finder_config : dict
            The configuration dictionary to validate.

        Returns
        -------
        str or None
            None if the configuration is valid, otherwise an error message
            describing the first validation failure.
        """
        required_keys = ["min_clip_duration", "max_clip_duration", "nlp_text_chunk_size_words"]
        missing_keys = find_missing_dict_keys(clip_finder_config, required_keys)
        if len(missing_keys) != 0:
            return f"ClipFinder missing configuration settings: {missing_keys}"

        err = self.check_valid_clip_times(
            clip_finder_config["min_clip_duration"],
            clip_finder_config["max_clip_duration"],
        )
        if err is not None:
            return err

        chunk_size = clip_finder_config["nlp_text_chunk_size_words"]
        type_err_chunk = self._type_checker.check_type(chunk_size, "nlp_text_chunk_size_words", int)
        if type_err_chunk:
            return type_err_chunk
        if chunk_size <= 0:
            return f"nlp_text_chunk_size_words must be a positive integer, not {chunk_size}"

        return None

    def check_valid_clip_times(
        self, min_clip_duration: float, max_clip_duration: float
    ) -> str or None:
        """
        Checks the clip times are valid. Returns None if the clip times are valid, a
        descriptive error message if invalid.
        """
        # Type check for min_clip_duration
        type_err = self._type_checker.check_type(min_clip_duration, "min_clip_duration", (float, int))
        if type_err:
            return type_err
        # Type check for max_clip_duration
        type_err = self._type_checker.check_type(max_clip_duration, "max_clip_duration", (float, int))
        if type_err:
            return type_err

        if min_clip_duration < 0:
            return f"min_clip_duration must be 0 or greater, not {min_clip_duration}"
        if max_clip_duration <= min_clip_duration:
            return f"max_clip_duration of {max_clip_duration} must be greater than min_clip_duration of {min_clip_duration}"
        return None

    # Removed all TextTiler specific check_valid_... and get_valid_... methods
    # e.g., check_valid_cutoff_policy, get_valid_embedding_aggregation_pool_methods, etc.
