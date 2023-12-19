"""
Config manager for TextTileClipFinder.
"""
# local package imports
from ...ml.texttile.config_manager import TextTilerConfigManager
from ...utils.utils import find_missing_dict_keys


class TextTileClipFinderConfigManager(TextTilerConfigManager):
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
            "max_clip_duration_secs",
            "min_clip_duration_secs",
            "smoothing_width",
            "window_compare_pool_method",
        ]
        missing_keys = find_missing_dict_keys(texttile_config, required_keys)
        if len(missing_keys) != 0:
            return "TextTiler missing configuration settings: {}".format(missing_keys)

        # value checks
        err = self.check_valid_clip_times(
            texttile_config["min_clip_duration_secs"],
            texttile_config["max_clip_duration_secs"],
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
        self, min_clip_duration_secs: float, max_clip_duration_secs: float
    ) -> str or None:
        """
        Checks the clip times are valid. Returns None if the clip times are valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        min_clip_duration_secs: float
            The minimum clip time in seconds
        max_clip_duration_secs: float
            The maximum clip time in seconds

        Returns
        -------
        str or None
            None if the clip times are valid, otherwise an error message.
        """
        # type check
        self._type_checker.check_type(
            min_clip_duration_secs, "min_clip_duration_secs", (float, int)
        )
        self._type_checker.check_type(
            max_clip_duration_secs, "max_clip_duration_secs", (float, int)
        )

        # minimum clip time
        if min_clip_duration_secs < 0:
            error = "min_clip_duration_secs must be 0 or greater, not {}" "".format(
                min_clip_duration_secs
            )
            return error

        # maximum clip time
        if max_clip_duration_secs <= min_clip_duration_secs:
            error = (
                "max_clip_duration_secs of {} must be greater than "
                "min_clip_duration_secs of {}"
                "".format(max_clip_duration_secs, min_clip_duration_secs)
            )
            return error

        return None
