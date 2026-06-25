"""
Finding clips using TwelveLabs Pegasus video understanding.

Notes
-----
- Pegasus is a video-language model that analyzes the actual video (visuals, audio,
  and pacing) rather than the transcript alone. This makes it an alternative to the
  transcript-only TextTiling heuristic used by ``ClipFinder``, useful for videos where
  highlights are not purely verbal.
- Requires a TwelveLabs API key (set ``TWELVELABS_API_KEY`` or pass ``api_key``). Get a
  free key with a generous free tier at https://twelvelabs.io.
- TwelveLabs Python SDK: https://github.com/twelvelabs-io/twelvelabs-python
"""
# standard library imports
from __future__ import annotations
import json
import logging
import os
import re

# current package imports
from .clip import Clip
from .exceptions import ClipFinderError

# local package imports
from clipsai.transcribe.transcription import Transcription
from clipsai.utils.config_manager import ConfigManager
from clipsai.utils.utils import find_missing_dict_keys

DEFAULT_PROMPT = (
    "You are a video editor selecting standalone short-form clips from a longer "
    "video. Identify the {max_clips} most engaging, self-contained highlights. Each "
    "highlight must be between {min_clip_duration} and {max_clip_duration} seconds "
    "long. Respond with ONLY a JSON array of objects, each with numeric "
    '"start_time" and "end_time" fields in seconds, e.g. '
    '[{{"start_time": 12.0, "end_time": 48.5}}]. Do not include any other text.'
)


class PegasusClipFinder:
    """
    Finds clips within a video using TwelveLabs' Pegasus video understanding model.

    This is an opt-in alternative to :class:`~clipsai.clip.clipfinder.ClipFinder`,
    which relies solely on the transcript. Pegasus analyzes the video itself to
    propose highlight time ranges, which are then aligned to the transcript's
    character offsets so the returned :class:`~clipsai.clip.clip.Clip` objects are
    interchangeable with those produced by ``ClipFinder``.
    """

    def __init__(
        self,
        api_key: str = None,
        model_name: str = "pegasus1.5",
        min_clip_duration: int = 15,
        max_clip_duration: int = 900,
        max_clips: int = 10,
        max_tokens: int = 2048,
    ) -> None:
        """
        Parameters
        ----------
        api_key: str
            TwelveLabs API key. If None, read from the ``TWELVELABS_API_KEY``
            environment variable.
        model_name: str
            Pegasus model to use. Possible values: 'pegasus1.5', 'pegasus1.2'
        min_clip_duration: int
            Minimum duration in seconds for a clip
        max_clip_duration: int
            Maximum duration in seconds for a clip
        max_clips: int
            Maximum number of clips to request from Pegasus
        max_tokens: int
            Maximum number of tokens Pegasus may generate
        """
        config_manager = PegasusClipFinderConfigManager()
        config_manager.assert_valid_config(
            {
                "model_name": model_name,
                "min_clip_duration": min_clip_duration,
                "max_clip_duration": max_clip_duration,
                "max_clips": max_clips,
            }
        )

        api_key = api_key or os.environ.get("TWELVELABS_API_KEY")
        if not api_key:
            raise ClipFinderError(
                "A TwelveLabs API key is required. Pass api_key or set the "
                "TWELVELABS_API_KEY environment variable. Get a free key at "
                "https://twelvelabs.io."
            )

        self._api_key = api_key
        self._model_name = model_name
        self._min_clip_duration = min_clip_duration
        self._max_clip_duration = max_clip_duration
        self._max_clips = max_clips
        self._max_tokens = max_tokens

    def find_clips(
        self,
        transcription: Transcription,
        video_url: str = None,
        video_id: str = None,
    ) -> list[Clip]:
        """
        Finds clips in a video using TwelveLabs Pegasus.

        Provide exactly one of ``video_url`` or ``video_id``. The transcription is
        used to map Pegasus' proposed time ranges to character offsets so the returned
        clips match the format produced by ``ClipFinder``.

        Parameters
        ----------
        transcription: Transcription
            the transcription of the source media, used to map times to char offsets
        video_url: str
            a publicly accessible URL to the video to analyze
        video_id: str
            the id of a video already indexed in a TwelveLabs index

        Returns
        -------
        list[Clip]
            list of clips found in the video
        """
        if (video_url is None) == (video_id is None):
            raise ClipFinderError("Provide exactly one of 'video_url' or 'video_id'.")

        prompt = DEFAULT_PROMPT.format(
            max_clips=self._max_clips,
            min_clip_duration=self._min_clip_duration,
            max_clip_duration=self._max_clip_duration,
        )

        # imported lazily so the optional dependency is only required when used
        from twelvelabs import TwelveLabs

        client = TwelveLabs(api_key=self._api_key)
        if video_id is not None:
            response = client.analyze(
                model_name=self._model_name,
                video_id=video_id,
                prompt=prompt,
                max_tokens=self._max_tokens,
            )
        else:
            response = client.analyze(
                model_name=self._model_name,
                video={"type": "url", "url": video_url},
                prompt=prompt,
                max_tokens=self._max_tokens,
            )

        time_ranges = self._parse_response(response.data)
        return self._ranges_to_clips(time_ranges, transcription)

    def _parse_response(self, text: str) -> list[dict]:
        """
        Parses the JSON array of time ranges out of Pegasus' text response.

        Parameters
        ----------
        text: str
            the raw text returned by Pegasus

        Returns
        -------
        list[dict]
            list of dicts with 'start_time' and 'end_time' keys
        """
        if not text:
            raise ClipFinderError("Pegasus returned an empty response.")

        # Pegasus may wrap the JSON in prose or markdown fences; extract the array.
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match is None:
            raise ClipFinderError(
                "Could not find a JSON array in the Pegasus response: {}".format(text)
            )
        try:
            ranges = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ClipFinderError(
                "Failed to parse JSON from the Pegasus response: {}".format(exc)
            )
        if not isinstance(ranges, list):
            raise ClipFinderError("Expected a JSON array from Pegasus.")
        return ranges

    def _ranges_to_clips(
        self,
        time_ranges: list[dict],
        transcription: Transcription,
    ) -> list[Clip]:
        """
        Converts Pegasus time ranges into Clip objects, dropping ranges that fall
        outside the configured duration bounds or the transcript's bounds.

        Parameters
        ----------
        time_ranges: list[dict]
            list of dicts with 'start_time' and 'end_time' keys (seconds)
        transcription: Transcription
            transcription used to map times to character offsets

        Returns
        -------
        list[Clip]
            list of clips
        """
        media_end = transcription.end_time
        clips = []
        for time_range in time_ranges:
            if "start_time" not in time_range or "end_time" not in time_range:
                logging.warning("Skipping malformed Pegasus range: %s", time_range)
                continue

            start_time = max(0.0, float(time_range["start_time"]))
            end_time = min(float(time_range["end_time"]), media_end)
            duration = end_time - start_time
            if duration < self._min_clip_duration:
                continue
            if duration > self._max_clip_duration:
                continue

            start_char = transcription.find_char_index(start_time, "start")
            end_char = transcription.find_char_index(end_time, "end")
            clips.append(Clip(start_time, end_time, start_char, end_char))

        clips.sort(key=lambda clip: clip.start_time)
        return clips


class PegasusClipFinderConfigManager(ConfigManager):
    """
    A class for validating PegasusClipFinder configuration settings.
    """

    VALID_MODEL_NAMES = ("pegasus1.5", "pegasus1.2")

    def check_valid_config(self, config: dict) -> str or None:
        """
        Checks that 'config' contains valid configuration settings. Returns None if
        valid, a descriptive error message if invalid.

        Parameters
        ----------
        config: dict
            A dictionary containing the configuration settings for PegasusClipFinder.

        Returns
        -------
        str or None
            None if the inputs are valid, otherwise an error message.
        """
        required_keys = [
            "model_name",
            "min_clip_duration",
            "max_clip_duration",
            "max_clips",
        ]
        missing_keys = find_missing_dict_keys(config, required_keys)
        if len(missing_keys) != 0:
            return "PegasusClipFinder missing configuration settings: {}".format(
                missing_keys
            )

        if config["model_name"] not in self.VALID_MODEL_NAMES:
            return "model_name must be one of {}, not {}".format(
                self.VALID_MODEL_NAMES, config["model_name"]
            )

        self._type_checker.check_type(
            config["min_clip_duration"], "min_clip_duration", (float, int)
        )
        self._type_checker.check_type(
            config["max_clip_duration"], "max_clip_duration", (float, int)
        )
        self._type_checker.check_type(config["max_clips"], "max_clips", int)

        if config["min_clip_duration"] < 0:
            return "min_clip_duration must be 0 or greater, not {}".format(
                config["min_clip_duration"]
            )
        if config["max_clip_duration"] <= config["min_clip_duration"]:
            return (
                "max_clip_duration of {} must be greater than min_clip_duration of {}"
                "".format(config["max_clip_duration"], config["min_clip_duration"])
            )
        if config["max_clips"] < 1:
            return "max_clips must be 1 or greater, not {}".format(config["max_clips"])

        return None
