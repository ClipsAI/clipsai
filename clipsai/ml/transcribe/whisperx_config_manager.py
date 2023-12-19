"""
Config manager for WhisperXTranscriber.
"""
# current package imports
from .exceptions import WhisperXTranscriberConfigError

# local package imports
from ..config_manager import ConfigManager
from ...utils.utils import find_missing_dict_keys


class WhisperXTranscriberConfigManager(ConfigManager):
    """
    A class for getting information about and validating WhisperXTranscriber
    configuration settings.
    """

    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        super().__init__()

    def check_valid_config(self, whisperx_config: dict) -> str or None:
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
        missing_keys = find_missing_dict_keys(whisperx_config, setting_checkers.keys())
        if len(missing_keys) != 0:
            return "WhisperXTranscriber missing configuration settings: {}".format(
                missing_keys
            )

        # value checks
        for setting, checker in setting_checkers.items():
            # None values = default values (depends on the compute device)
            if whisperx_config[setting] is None:
                continue
            err = checker(whisperx_config[setting])
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
            raise WhisperXTranscriberConfigError(msg)

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
            raise WhisperXTranscriberConfigError(msg)

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
            raise WhisperXTranscriberConfigError(msg)
