"""
Transcribe audio files using whisperx.

Notes
-----
- WhisperX GitHub: https://github.com/m-bain/whisperX
"""
# standard library imports
from datetime import datetime
import logging

# current package imports
from .exceptions import NoSpeechError
from .exceptions import TranscriberConfigError
from .transcription import Transcription

# local imports
from clipsai.media.audio_file import AudioFile
from clipsai.media.editor import MediaEditor
from clipsai.utils.config_manager import ConfigManager
from clipsai.utils.pytorch import assert_valid_torch_device, get_compute_device
from clipsai.utils.type_checker import TypeChecker
from clipsai.utils.utils import find_missing_dict_keys

# third party imports
import torch
import whisperx


class Transcriber:
    """
    A class to transcribe using whisperx.
    """

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        precision: str = None,
    ) -> None:
        """
        Parameters
        ----------
        model_size: str
            One of the model sizes implemented by whisper/whisperx. Default is None,
            which selects large-v2 if cuda is available and tiny if not (cpu).
        device: str
            PyTorch device to perform computations on. Default is None, which auto
            detects the correct device.
        precision: 'float16' | 'int8'
            Precision to perform prediction with. Default is None, which selects
            float16 if cuda is available and int8 if not (cpu).
        """
        self._config_manager = TranscriberConfigManager()
        self._type_checker = TypeChecker()

        if device is None:
            device = get_compute_device()
        if precision is None:
            precision = "float16" if torch.cuda.is_available() else "int8"
        if model_size is None:
            model_size = "large-v2" if torch.cuda.is_available() else "tiny"

        # check valid inputs
        assert_valid_torch_device(device)
        self._config_manager.assert_valid_model_size(model_size)
        self._config_manager.assert_valid_precision(precision)

        self._precision = precision
        self._device = device
        self._model_size = model_size
        self._model = whisperx.load_model(
            whisper_arch=self._model_size,
            device=self._device,
            compute_type=self._precision,
        )

    def transcribe(
        self,
        audio_file_path: str,
        iso6391_lang_code: str or None = None,
        batch_size: int = 16,
    ) -> Transcription:
        """
        Transcribes the media file

        Parameters
        ----------
        audio_file_path: str
            Absolute path to the audio or video file to transcribe.
        iso6391_lang_code: str or None
            ISO 639-1 language code to transcribe the media in. Default is None, which
            autodetects the media's language.
        batch_size: int = 16
            reduce if low in GPU memory (not actually sure what it does though -Ben)
        Returns
        -------
        Transcription
            the media file transcription
        """
        editor = MediaEditor()
        media_file = editor.instantiate_as_temporal_media_file(audio_file_path)
        media_file.assert_exists()
        media_file.assert_has_audio_stream()

        if iso6391_lang_code is not None:
            self._config_manager.assert_valid_language(iso6391_lang_code)

        # if iso6391_lang_code is None, whisperx will try to detect the language
        transcription = self._model.transcribe(
            media_file.path, language=iso6391_lang_code, batch_size=batch_size
        )

        # align whisper output to get word level times
        model_a, metadata = whisperx.load_align_model(
            language_code=transcription["language"],
            device=self._device,
        )
        aligned_transcription = whisperx.align(
            transcription["segments"],
            model_a,
            metadata,
            media_file.path,
            self._device,
            return_char_alignments=True,
        )

        """
        ALIGNED_TRANSCRIPTION DATA STRUCTURE
        ------------------------------------
        s = number of segments in the transcription
        w = number of words in the transcription
        n_s = number of chars in the transcription of segment s
        m_s = number of words in the transcription of segment s
        aligned_transcription = {
            "segments":
            [
                {<segment-0>},
                {
                    "start": float (start time in seconds)
                    "end": float (start time in seconds)
                    "text": str (text transcription for that segment)
                    "words":
                    [
                        {<word-0>},
                        {
                            "word": str (word transcription)
                            "start": float (start time in seconds)
                            "end": float (start time in seconds)
                            "score": float (score for that word)
                        },
                        {<word-m_s>},
                    ]
                    "chars":
                    [
                        {<char-0>},
                        {
                            "char": str (char transcription)
                            "start": float (start time in seconds)
                            "end": float (start time in seconds)
                            "score": float (score for that char)
                        }
                        {<char-n_s>},
                    ]
                },
                {segment_n},
            ]
            "word_segments":
            [
                {<word-segment-0>},
                {
                    "word":
                    "start":
                    "end":
                    "score":
                },
                {word-segment-w},
            ]
        }
        """
        if len(aligned_transcription["segments"]) == 0:
            err = "Media file '{}' contains no active speech.".format(media_file.path)
            logging.error(err)
            raise NoSpeechError(err)

        # final destination for transcript information
        char_info = []

        # remove global first character -> always a space
        try:
            del aligned_transcription["segments"][0]["chars"][0]
        except Exception as e:
            print("Error:", str(e))
            print("Aligned Transcription:", aligned_transcription)
            raise Exception(str(e))

        for i, segment in enumerate(aligned_transcription["segments"]):
            segment_chars = segment["chars"]

            # iterate through each char in the segment
            for j, char in enumerate(segment_chars):
                char_start_time = (
                    float(char["start"]) if "start" in char.keys() else None
                )
                char_end_time = float(char["end"]) if "end" in char.keys() else None

                # character information
                new_char_dic = {
                    "char": char["char"],
                    "start_time": char_start_time,
                    "end_time": char_end_time,
                    "speaker": None,
                }
                char_info.append(new_char_dic)

        transcription_dict = {
            "source_software": "whisperx-v3",
            "time_created": datetime.now(),
            "language": transcription["language"],
            "num_speakers": None,
            "char_info": char_info,
        }
        return Transcription(transcription_dict)

    def detect_language(self, media_file: AudioFile) -> str:
        """
        Detects the language of the media file

        Parameters
        ----------
        media_file: AudioFile
            the media file to detect the language of

        Returns
        -------
        str
            the ISO 639-1 language code of the media file
        """
        self._type_checker.assert_type(media_file, "media_file", (AudioFile))
        media_file.assert_exists()
        media_file.assert_has_audio_stream()

        audio = whisperx.load_audio(media_file.path)
        language = self._model.detect_language(audio)
        return language


class TranscriberConfigManager(ConfigManager):
    """
    A class for getting information about and validating Transcriber
    configuration settings.
    """

    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        super().__init__()

    def check_valid_config(self, config: dict) -> str or None:
        """
        Checks that 'config' contains valid configuration settings. Returns None if
        valid, a descriptive error message if invalid.

        Parameters
        ----------
        config: dict
            A dictionary containing the configuration settings for WhisperXTranscriber.

        Returns
        -------
        str or None
            None if the inputs are valid, otherwise an error message.
        """
        # type check inputs
        setting_checkers = {
            "language": self.check_valid_language,
            "model_size": self.check_valid_model_size,
            "precision": self.check_valid_precision,
        }

        # existence check
        missing_keys = find_missing_dict_keys(config, setting_checkers.keys())
        if len(missing_keys) != 0:
            return "WhisperXTranscriber missing configuration settings: {}".format(
                missing_keys
            )

        # value checks
        for setting, checker in setting_checkers.items():
            # None values = default values (depends on the compute device)
            if config[setting] is None:
                continue
            err = checker(config[setting])
            if err is not None:
                return err

        return None

    def get_valid_model_sizes(self) -> list[str]:
        """
        Returns the valid model sizes to transcribe with whisperx

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            list of valid model sizes to transcribe with whisperx
        """
        valid_model_sizes = [
            "tiny",
            "base",
            "small",
            "medium",
            "large-v1",
            "large-v2",
        ]
        return valid_model_sizes

    def check_valid_model_size(self, model_size: str) -> str or None:
        """
        Checks if 'model_size' is valid

        Parameters
        ----------
        model_size: str
            The transcription model size

        Returns
        -------
        str or None
            None if 'model_size' is valid. A descriptive error message if 'model_size'
            is invalid
        """
        if model_size not in self.get_valid_model_sizes():
            msg = "Invalid whisper model size '{}'. Must be one of: {}." "".format(
                model_size, self.get_valid_model_sizes()
            )
            return msg

        return None

    def is_valid_model_size(self, model_size: str) -> bool:
        """
        Returns True is 'model_size' is valid, False if not

        Parameters
        ----------
        model_size: str
            The transcription model size

        Returns
        -------
        bool
            True is 'model_size' is valid, False if not
        """
        msg = self.check_valid_model_size(model_size)
        if msg is None:
            return True
        else:
            return False

    def assert_valid_model_size(self, model_size: str) -> None:
        """
        Raises an Error if 'model_size' is invalid

        Parameters
        ----------
        model_size: str
            The transcription model size

        Raises
        ------
        WhisperXTranscriberConfigError: 'model_size' is invalid
        """
        msg = self.check_valid_model_size(model_size)
        if msg is not None:
            raise TranscriberConfigError(msg)

    def get_valid_languages(self) -> list[str]:
        """
        Returns the valid languages to transcribe with whisperx

        - See https://github.com/m-bain/whisperX#other-languages for updated lang info

        Parameters
        ----------
        None

        Returns
        -------
        list[str]:
            list of ISO 639-1 language codes of languages that can be transcribed
        """
        valid_languages = [
            "en",  # english
            "fr",  # french
            "de",  # german
            "es",  # spanish
            "it",  # italian
            "ja",  # japanese
            "zh",  # chinese
            "nl",  # dutch
            "uk",  # ukrainian
            "pt",  # portuguese
        ]
        return valid_languages

    def check_valid_language(self, iso6391_lang_code: str) -> str or None:
        """
        Checks if 'iso6391_lang_code' is a valid ISO 639-1 language code for whisperx to
        transcribe

        Parameters
        ----------
        iso6391_lang_code: str
            The language code to check

        Returns
        -------
        str or None
            None if 'iso6391_lang_code' is a valid ISO 639-1 language code for whisperx
            to transcribe. A descriptive error message if 'iso6391_lang_code' is invalid
        """
        if iso6391_lang_code not in self.get_valid_languages():
            msg = "Invalid ISO 639-1 language '{}'. Must be one of: {}." "".format(
                iso6391_lang_code, self.get_valid_languages()
            )
            return msg

        return None

    def is_valid_language(self, iso6391_lang_code: str) -> bool:
        """
        Returns True if 'iso6391_lang_code' is a valid ISO 639-1 language code for
        whisperx to transcribe, False if not

        Parameters
        ----------
        iso6391_lang_code: str
            The language code to check

        Returns
        -------
        bool
            True if 'iso6391_lang_code' is a valid ISO 639-1 language code for whisperx
            to transcribe, False if not
        """
        msg = self.check_valid_language(iso6391_lang_code)
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

    def get_valid_precisions(self) -> list[str]:
        """
        Returns the valid precisions to transcribe with whisperx

        Parameters
        ----------
        None

        Returns
        -------
        list[str]:
            list of compute types that can be used to transcribe
        """
        valid_precisions = [
            "float32",
            "float16",
            "int8",
        ]
        return valid_precisions

    def check_valid_precision(self, precision: str) -> str or None:
        """
        Checks if 'precision' is valid to transcribe with whisperx

        Parameters
        ----------
        precision: str
            The precision to check

        Returns
        -------
        str or None
            None if 'precision' is valid. A descriptive error message if invalid
        """
        if precision not in self.get_valid_precisions():
            msg = "Invalid compute type '{}'. Must be one of: {}." "".format(
                precision, self.get_valid_precisions()
            )
            return msg

        return None

    def is_valid_precision(self, precision: str) -> bool:
        """
        Returns True if 'precision' is valid to transcribe with whisperx, False if not

        Parameters
        ----------
        precision: str
            The precision to check

        Returns
        -------
        bool
            True if 'precision' is valid to transcribe with whisperx, False if not
        """
        msg = self.check_valid_precision(precision)
        if msg is None:
            return True
        else:
            return False

    def assert_valid_precision(self, precision: str) -> None:
        """
        Raises TranscriptionError if 'precision' is invalid to transcribe with whisperx

        Parameters
        ----------
        precision: str
            The precision to check

        Returns
        -------
        None

        Raises
        ------
        WhisperXTranscriberConfigError: if 'precision' is invalid to transcribe with
        whisperx
        """
        msg = self.check_valid_precision(precision)
        if msg is not None:
            raise TranscriberConfigError(msg)
