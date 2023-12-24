"""
Parameter Input Validator for TranscribeAndClip.
"""
# current package imports
from .texttile_config_manager import TextTileClipFinderConfigManager

# local package imports
from utils.pytorch import check_valid_torch_device
from input_validator import InputValidator


class ClipInputValidator(InputValidator):
    """
    InputValidator Class for TranscribeAndClip.
    """

    def __init__(self) -> None:
        """
        Initialize the ClipInputValidator class.
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
            "computeDevice": (str, type(None)),
            "cutoffPolicy": (str),
            "embeddingAggregationPoolMethod": (str),
            "minClipTime": (float, int),
            "maxClipTime": (float, int),
            "smoothingWidth": (int),
            "windowComparePoolMethod": (str),
        }

        # existence and type check
        error = self.check_input_data_existence_and_types(input_data, correct_types)
        if error is not None:
            return error

        # computeDevice
        if input_data["computeDevice"] is not None:
            error = check_valid_torch_device(input_data["computeDevice"])
            if error is not None:
                return error

        # TextTiler Configuration
        texttile_config = {
            "cutoff_policy": input_data["cutoffPolicy"],
            "embedding_aggregation_pool_method": input_data[
                "embeddingAggregationPoolMethod"
            ],
            "max_clip_duration_secs": input_data["maxClipTime"],
            "min_clip_duration_secs": input_data["minClipTime"],
            "smoothing_width": input_data["smoothingWidth"],
            "window_compare_pool_method": input_data["windowComparePoolMethod"],
        }
        texttile_config_manager = TextTileClipFinderConfigManager()
        error = texttile_config_manager.check_valid_config(texttile_config)
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
        if input_data["computeDevice"] == "auto":
            input_data["computeDevice"] = "cpu"
        optional_fields_default_values = {
            "computeDevice": "cpu",
            "cutoffPolicy": "high",
            "embeddingAggregationPoolMethod": "max",
            "minClipTime": 15,
            "maxClipTime": 900,
            "smoothingWidth": 3,
            "windowComparePoolMethod": "mean",
        }

        for key in optional_fields_default_values.keys():
            if key not in input_data:
                input_data[key] = optional_fields_default_values[key]

        return input_data
