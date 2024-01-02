"""
Parameter Input Validator for Transcribe.
"""
# current package imports
from .whisperx_config_manager import WhisperXTranscriberConfigManager

# local package imports
from utils.pytorch import check_valid_torch_device
from input_validator import InputValidator


class TranscribeInputValidator(InputValidator):
    """
    InputValidator Class for Transcribe.
    """

    def __init__(self) -> None:
        """
        Initialize the TranscribeInputValidator class.
        """
        super().__init__()

    def check_valid_input_data(self, input_data: dict) -> str or None:
        """
        Checks if the input data is valid. Returns None if so, a descriptive error
        message if not.

        Parameters
        ----------
        input_data: dict
            The input data to be validated.

        Returns
        -------
        str or None
            None if the input data is valid, a descriptive error message if not.
        """
        correct_types = {
            "mediaFilePath": (str),
            "computeDevice": (str, type(None)),
            "precision": (str, type(None)),
            "languageCode": (str, type(None)),
            "whisperModelSize": (str, type(None)),
        }

        # existence and type check
        error = self.check_input_data_existence_and_types(input_data, correct_types)
        if error is not None:
            return error

        # mediaFilePath
        media_file_path: str = input_data["mediaFilePath"]
        if not media_file_path.endswith((".mp3", ".mp4")):
            error = "mediaFilePath must be of type mp3 or mp4. Received: {}".format(
                media_file_path
            )
            return error

        # computeDevice
        if input_data["computeDevice"] is not None:
            error = check_valid_torch_device(input_data["computeDevice"])
            if error is not None:
                return error

        # WhisperXTranscriber configuration
        whisperx_config = {
            "language": input_data["languageCode"],
            "model_size": input_data["whisperModelSize"],
            "precision": input_data["precision"],
        }
        whisperx_config_manager = WhisperXTranscriberConfigManager()
        error = whisperx_config_manager.check_valid_config(whisperx_config)
        if error is not None:
            return error

        return None

    def impute_input_data_defaults(self, input_data: dict) -> dict:
        """
        Populates input data with default values if they are not provided.

        Parameters
        ----------
        input_data: dict
            The input data to be imputed.

        Returns
        -------
        dict
            The imputed input data.
        """
        # setting to none allows whisper to choose the default
        if input_data["computeDevice"] == "auto":
            input_data["computeDevice"] = None
        if input_data["languageCode"] == "auto":
            input_data["languageCode"] = None

        optional_fields_default_values = {
            "computeDevice": None,
            "precision": None,
            "languageCode": "en",
            "whisperModelSize": None,
        }

        for key in optional_fields_default_values.keys():
            if key not in input_data:
                input_data[key] = optional_fields_default_values[key]

        return input_data
