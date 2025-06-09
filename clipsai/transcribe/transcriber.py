"""
Handles audio transcription using Google Cloud Speech-to-Text API.
Provides functionality to transcribe audio content from various media files,
including audio extraction from video files, and processes the results into a
standardized `Transcription` object. It supports speaker diarization and
returns word-level timestamps with interpolated character-level timings.
"""

from __future__ import annotations
# standard library imports
from datetime import datetime
import logging

# current package imports
from .exceptions import NoSpeechError
from .exceptions import TranscriberConfigError
from .transcription import Transcription

# local imports
from clipsai.gcloud.config import GCloudConfig # Added import
from clipsai.media.audio_file import AudioFile
from clipsai.media.editor import MediaEditor
from clipsai.utils.config_manager import ConfigManager
from clipsai.utils.type_checker import TypeChecker
from clipsai.utils.utils import find_missing_dict_keys

# third party imports are removed as whisperx and torch are no longer directly used here.


class Transcriber:
    """
    A class to transcribe audio using Google Cloud Speech-to-Text.
    """

    def __init__(
        self,
        gcloud_config: GCloudConfig,
        default_language_code: str = "en-US",
    ) -> None:
        """
        Initializes the Transcriber.

        Parameters
        ----------
        gcloud_config : GCloudConfig
            Configuration object for Google Cloud services. This must include
            `project_id` and `temp_gcs_bucket_name` if local files need to be
            processed, and `location` for some specific STT features if used.
        default_language_code : str, optional
            Default BCP-47 language code (e.g., "en-US") to use for transcription
            if no specific language is provided to the `transcribe` method, or to
            aid auto-detection by Google STT. Default is "en-US".
        """
        self._config_manager = TranscriberConfigManager()
        self._type_checker = TypeChecker()
        self.gcloud_config = gcloud_config
        self.default_language_code = default_language_code

        # Example: validate default_language_code
        config_to_check = {"language": default_language_code}
        err = self._config_manager.check_valid_config(config_to_check)
        if err:
            # Ensure TranscriberConfigError is correctly imported and used
            from .exceptions import TranscriberConfigError
            raise TranscriberConfigError(err)

    def transcribe(
        self,
        audio_file_path: str,
        iso6391_lang_code: str | None = None,
        # batch_size parameter is removed as it's WhisperX specific
    ) -> Transcription:
        """
        Transcribes the given media file using Google Cloud Speech-to-Text API.

        If the input `audio_file_path` points to a video file, audio is first
        extracted into a temporary WAV file. This audio data (either the original
        audio file or the extracted WAV) is then uploaded to a temporary GCS bucket
        if it's not already a GCS URI. The `long_running_recognize` method of the
        Speech-to-Text API is used for transcription, with speaker diarization,
        automatic punctuation, and word time offsets enabled.

        Character-level timings in the returned `Transcription` object are interpolated
        from the word-level timings provided by the API. Speaker tags are also derived
        from the API's diarization output.

        Parameters
        ----------
        audio_file_path : str
            Absolute path to the audio or video file to be transcribed.
            If a video file is provided, its audio stream will be extracted.
        iso6391_lang_code : str or None, optional
            The BCP-47 language code of the language to be transcribed (e.g., "en-US",
            "es-ES"). If None, the `default_language_code` specified during
            `Transcriber` initialization will be used, or Google STT will attempt
            auto-detection if the `default_language_code` also permits this (e.g. by being a base lang like "en").
            Default is None.
        Returns
        -------
        Transcription
            The media file transcription.
        Raises
        ------
        ConfigError, VideoProcessingError, NoSpeechError, google.api_core.exceptions.GoogleAPICallError, ModuleNotFoundError
        """
        # Dynamic imports for Google Cloud libraries
        try:
            from google.cloud import speech
            from google.cloud import storage
            from google.api_core import exceptions as google_exceptions
            import os # for path operations
            import uuid # for unique naming
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[-2]
            raise ModuleNotFoundError(
                f"Google Cloud library '{missing_module}' is not installed. "
                "Please install it (e.g., google-cloud-speech, google-cloud-storage) "
                "to use transcription with Google Cloud."
            )

        editor = MediaEditor()
        input_media_file = editor.instantiate_as_temporal_media_file(audio_file_path)
        input_media_file.assert_exists()
        input_media_file.assert_has_audio_stream()

        temp_local_audio_path = None
        actual_audio_file_to_upload = input_media_file

        if not isinstance(input_media_file, AudioFile):
            logging.info(f"Input '{audio_file_path}' is not solely audio. Extracting audio stream.")
            # Create a unique name for the temporary local audio file
            # Place temp files in a subdirectory of where this script is, e.g. ../temp_audio_files
            # This path construction might need to be more robust depending on execution context
            script_dir = os.path.dirname(os.path.abspath(__file__))
            temp_audio_dir = os.path.join(script_dir, "..", "temp_audio_files")
            os.makedirs(temp_audio_dir, exist_ok=True) # Ensure temp directory exists

            temp_local_audio_path = os.path.join(
                temp_audio_dir,
                f"{uuid.uuid4()}_{input_media_file.get_filename_without_extension()}.wav"
            )

            extracted_audio = input_media_file.extract_audio(
                extracted_audio_file_path=temp_local_audio_path,
                audio_codec="pcm_s16le", # WAV format
                overwrite=True
            )
            if not extracted_audio:
                # Attempt to clean up if extraction fails but file was created
                if temp_local_audio_path and os.path.exists(temp_local_audio_path):
                    try:
                        os.remove(temp_local_audio_path)
                    except Exception as e_clean:
                        logging.error(f"Failed to clean up temp audio file {temp_local_audio_path} after extraction failure: {e_clean}")
                raise VideoProcessingError(f"Failed to extract audio from {audio_file_path}")
            actual_audio_file_to_upload = extracted_audio
            logging.info(f"Audio extracted to temporary file: {temp_local_audio_path}")

        storage_client = None
        gcs_uri = None
        gcs_object_name = None
        perform_gcs_upload = not actual_audio_file_to_upload.path.startswith("gs://")

        if perform_gcs_upload:
            if not self.gcloud_config.temp_gcs_bucket_name:
                if temp_local_audio_path and os.path.exists(temp_local_audio_path):
                    os.remove(temp_local_audio_path)
                raise ConfigError(
                    "Temporary GCS bucket ('temp_gcs_bucket_name') not configured in "
                    "GCloudConfig, required for uploading local audio files."
                )

            storage_client = storage.Client(project=self.gcloud_config.project_id)
            bucket = storage_client.bucket(self.gcloud_config.temp_gcs_bucket_name)

            safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in actual_audio_file_to_upload.get_filename())
            gcs_object_name = f"clipsai_temp_audio/{uuid.uuid4()}/{safe_filename}"
            gcs_uri = f"gs://{self.gcloud_config.temp_gcs_bucket_name}/{gcs_object_name}"

            logging.info(f"Uploading {actual_audio_file_to_upload.path} to {gcs_uri}")
            try:
                blob = bucket.blob(gcs_object_name)
                blob.upload_from_filename(actual_audio_file_to_upload.path, timeout=300)
                logging.info(f"Successfully uploaded to {gcs_uri}")
            except Exception as e:
                if temp_local_audio_path and os.path.exists(temp_local_audio_path):
                    os.remove(temp_local_audio_path)
                logging.error(f"GCS upload failed for {actual_audio_file_to_upload.path}: {e}")
                raise VideoProcessingError(f"GCS upload to {gcs_uri} failed: {e}")
        else:
            gcs_uri = actual_audio_file_to_upload.path
            logging.info(f"Using existing GCS URI for audio: {gcs_uri}")

        try:
            speech_client = speech.SpeechClient(client_options={"project_id": self.gcloud_config.project_id} if self.gcloud_config.project_id else None)

            recognition_config_args = {
                "enable_automatic_punctuation": True,
                "enable_word_time_offsets": True,
                "diarization_config": speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=1,
                    max_speaker_count=6,
                ),
            }

            if iso6391_lang_code:
                self._config_manager.assert_valid_language(iso6391_lang_code) # Validate provided lang
                recognition_config_args["language_code"] = iso6391_lang_code
            else:
                recognition_config_args["language_code"] = self.default_language_code
                logging.info(f"No specific language code provided, using default: {self.default_language_code}")

            # If audio was extracted to local WAV or original is local WAV, get its sample rate.
            # Otherwise, for GCS URIs of unknown format, let STT auto-detect sample rate and encoding.
            if perform_gcs_upload or actual_audio_file_to_upload.path.lower().endswith(".wav"):
                try:
                    # Assuming AudioFile or TemporalMediaFile has a method to get sample rate
                    sample_rate = actual_audio_file_to_upload.get_sample_rate()
                    recognition_config_args["sample_rate_hertz"] = sample_rate
                    recognition_config_args["encoding"] = speech.RecognitionConfig.AudioEncoding.LINEAR16
                except Exception as e:
                    logging.warning(f"Could not determine sample rate or encoding for {actual_audio_file_to_upload.path}, letting STT auto-detect: {e}")


            config = speech.RecognitionConfig(**recognition_config_args)
            audio_input = speech.RecognitionAudio(uri=gcs_uri)

            logging.info(f"Sending transcription request for {gcs_uri} to Google Speech-to-Text API.")
            operation = speech_client.long_running_recognize(config=config, audio=audio_input)
            response = operation.result(timeout=1800)

            char_info_list = []
            # Use language_code from the first result if available, otherwise stick to what was requested/default
            detected_language = response.results[0].language_code if response.results and response.results[0].language_code else recognition_config_args["language_code"]
            all_speakers = set()

            if not response.results or not any(alt.transcript for res in response.results for alt in res.alternatives if alt.transcript):
                raise NoSpeechError(f"Media file '{audio_file_path}' contains no active speech according to Google STT.")

            for result_idx, result in enumerate(response.results):
                if not result.alternatives: continue
                alternative = result.alternatives[0]

                for word_info_gcp in alternative.words:
                    word_text = word_info_gcp.word
                    start_time_s = word_info_gcp.start_time.total_seconds()
                    end_time_s = word_info_gcp.end_time.total_seconds()
                    speaker_tag = word_info_gcp.speaker_tag
                    if speaker_tag != 0: all_speakers.add(speaker_tag)

                    num_chars = len(word_text)
                    if num_chars == 0: continue

                    duration_per_char = (end_time_s - start_time_s) / num_chars

                    for i, char_val in enumerate(word_text):
                        char_start_time = start_time_s + (i * duration_per_char)
                        char_end_time = start_time_s + ((i + 1) * duration_per_char)
                        char_info_list.append({
                            "char": char_val,
                            "start_time": round(char_start_time, 6),
                            "end_time": round(char_end_time, 6),
                            "speaker": speaker_tag if speaker_tag != 0 else None,
                        })

                    # Add space after the word, unless it's the last word of the entire transcript
                    is_last_word_overall = (result_idx == len(response.results) - 1 and \
                                            word_info_gcp == alternative.words[-1])

                    if not is_last_word_overall:
                        # Heuristic: assume space follows, with minimal duration
                        char_info_list.append({
                            "char": " ",
                            "start_time": round(end_time_s, 6),
                            "end_time": round(end_time_s + 0.01, 6), # Brief space
                            "speaker": speaker_tag if speaker_tag != 0 else None,
                        })

            num_unique_speakers = len(all_speakers) if all_speakers else (1 if char_info_list else 0)


            transcription_data = {
                "source_software": "GoogleCloudSpeechToText",
                "time_created": datetime.now(),
                "language": detected_language,
                "num_speakers": num_unique_speakers if num_unique_speakers > 0 else None,
                "char_info": char_info_list,
            }
            return Transcription(transcription_data)

        finally:
            if perform_gcs_upload and gcs_object_name and storage_client:
                logging.info(f"Deleting temporary GCS audio object {gcs_uri}")
                try:
                    bucket = storage_client.bucket(self.gcloud_config.temp_gcs_bucket_name)
                    blob = bucket.blob(gcs_object_name)
                    blob.delete(timeout=60)
                    logging.info(f"Successfully deleted {gcs_uri}")
                except Exception as e:
                    logging.error(f"Failed to delete temporary GCS object {gcs_uri}: {e}")

            if temp_local_audio_path and os.path.exists(temp_local_audio_path):
                logging.info(f"Deleting temporary local audio file {temp_local_audio_path}")
                try:
                    os.remove(temp_local_audio_path)
                except Exception as e:
                    logging.error(f"Failed to delete temporary local audio file {temp_local_audio_path}: {e}")

class TranscriberConfigManager(ConfigManager):
    """
    Manages configuration settings for the Transcriber, specifically for use with
    Google Cloud Speech-to-Text. Validates language codes.
    """

    def __init__(self):
        super().__init__()
        # Ensure _type_checker is available, initializing if not provided by ConfigManager
        if not hasattr(self, '_type_checker'):
            self._type_checker = TypeChecker()

    def check_valid_config(self, config: dict) -> str or None:
        """
        Checks that the provided 'config' dictionary contains valid settings for
        the Transcriber, primarily focusing on 'language'.

        Parameters
        ----------
        config : dict
            A dictionary containing configuration settings. Expected to have a
            'language' key for language code validation.

        Returns
        -------
        str or None
            None if the inputs are valid, otherwise an error message.
        """
        # type check inputs
        setting_checkers = {
            "language": self.check_valid_language,
            # Add new Google STT specific configs here later
        }

        # existence check
        # Ensure all keys in config are known, or specific required keys are present.
        # For now, only 'language' is checked if present in config.
        # This logic might need refinement based on how config is used with Google STT.
        required_keys = [] # Example: if some Google STT settings were mandatory from config dict
        missing_keys = find_missing_dict_keys(config, required_keys)
        if len(missing_keys) != 0:
            return "WhisperXTranscriber missing configuration settings: {}".format(
                missing_keys
            )

        # value checks
        for setting_key, value_to_check in config.items():
            if setting_key in setting_checkers:
                checker_func = setting_checkers[setting_key]
                # None values might mean 'use default' or 'not set',
                # depending on the setting. For language, it's usually required.
                if value_to_check is None and setting_key == "language": # Example: disallow None for language
                     return f"'{setting_key}' cannot be None."
                if value_to_check is not None: # Only check non-None values, or handle None explicitly
                    err = checker_func(value_to_check)
                    if err is not None:
                        return err
            # else: # Optional: warn or error on unknown config keys
            #     logging.warning(f"Unknown configuration setting: {setting_key}")


        return None

    # Removed get_valid_model_sizes, check_valid_model_size, is_valid_model_size, assert_valid_model_size

    def get_valid_languages(self) -> list[str]:
        """
        Returns a list of example BCP-47 language codes that can be used with
        Google Cloud Speech-to-Text. This list is not exhaustive.

        For a comprehensive list, refer to the Google Cloud documentation:
        https://cloud.google.com/speech-to-text/docs/languages

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            A list of example BCP-47 language codes.
        """
        # This is a sample list. Users should refer to Google Cloud documentation
        # for the full list of supported BCP-47 codes.
        valid_languages = [
            "en-US",  # English, United States
            "en-GB",  # English, United Kingdom
            "es-ES",  # Spanish, Spain
            "fr-FR",  # French, France
            "de-DE",  # German, Germany
            "it-IT",  # Italian, Italy
            "ja-JP",  # Japanese, Japan
            "zh-CN",  # Mandarin Chinese, Simplified
            "nl-NL",  # Dutch, Netherlands
            "pt-BR",  # Portuguese, Brazil
            "ru-RU",  # Russian, Russia
            # Add more common examples or direct users to documentation.
        ]
        return valid_languages

    def check_valid_language(self, language_code: str) -> str or None:
        """
        Checks if 'language_code' is plausible as a BCP-47 language code.
        This is a basic check and does not guarantee the language is supported by Google STT.

        Parameters
        ----------
        language_code : str
            The BCP-47 language code to check (e.g., "en-US").

        Returns
        -------
        str or None
            None if 'language_code' seems valid, otherwise a descriptive error message.
        """
        # Basic BCP-47 format check (e.g., lang-REGION, lang)
        # This is not a strict validation against all possible BCP-47 tags.
        if not isinstance(language_code, str) or not (2 <= len(language_code) <= 10) or not "-" in language_code:
            if len(language_code) == 2: # Allow simple 2-letter codes like "en", "es"
                 pass # Valid simple form
            else:
                msg = (
                    f"Invalid language code format '{language_code}'. Expected BCP-47 format "
                    "(e.g., 'en-US', 'es', 'fr-CA'). Please refer to Google Cloud STT documentation "
                    "for supported language codes."
                )
                return msg
        # For simplicity, we're not checking against self.get_valid_languages() here,
        # as that list is just a sample. The API will ultimately validate the language code.
        return None

    def is_valid_language(self, language_code: str) -> bool:
        """
        Returns True if 'language_code' has a plausible BCP-47 format, False otherwise.
        Does not guarantee the language is supported by Google STT.

        Parameters
        ----------
        language_code : str
            The BCP-47 language code to check.

        Returns
        -------
        bool
            True if the format seems valid, False otherwise.
        """
        msg = self.check_valid_language(language_code)
        if msg is None:
            return True
        else:
            return False

    def assert_valid_language(self, iso6391_lang_code: str) -> None:
        """
        Raises TranscriptionError if 'iso6391_lang_code' is not a valid ISO 639-1
        language code for whisperx to transcribe in

        Parameters
        ----------
        iso6391_lang_code: str
            The language code to check

        Raises
        ------
        WhisperXTranscriberConfigError: if 'iso6391_lang_code' is not a valid
        ISO 639-1 language code for whisperx to transcribe in
        """
        msg = self.check_valid_language(iso6391_lang_code)
        if msg is not None:
            raise TranscriberConfigError(msg)

    # Removed get_valid_precisions, check_valid_precision, is_valid_precision, assert_valid_precision
