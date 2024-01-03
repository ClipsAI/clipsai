"""
Configuration manager for TextTile algorithm.
"""
# local package imports
from clipsai.utils.utils import find_missing_dict_keys
from clipsai.utils.config_manager import ConfigManager


class TextTilerConfigManager(ConfigManager):
    """
    A class for getting information about and validating
    TextTiler configuration settings.
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
        Checks that 'texttile_config' contains valid configuration settings.
        Returns None if valid, a descriptive error message if invalid.

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
            "k",
            "smoothing_width",
            "window_compare_pool_method",
        ]
        missing_keys = find_missing_dict_keys(texttile_config, required_keys)
        if len(missing_keys) != 0:
            return "TextTiler missing configuration settings: {}".format(missing_keys)

        setting_checkers = {
            "cutoff_policy": self.check_valid_cutoff_policy,
            "embedding_aggregation_pool_method": self.check_valid_embedding_aggregation_pool_method,
            "k": self.check_valid_k,
            "smoothing_width": self.check_valid_smoothing_width,
            "window_compare_pool_method": self.check_valid_window_compare_pool_method,
        }
        for setting, checker in setting_checkers.items():
            err = checker(texttile_config[setting])
            if err is not None:
                return err

        return None

    def check_valid_k(self, k: int) -> str or None:
        """
        Checks the window size is valid. Returns None if the window size is valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        k: int
            The window size used by TextTiling algorithm

        Returns
        -------
        str or None
            None if the window size is valid, otherwise an error message.
        """
        err = self._type_checker.check_type(k, "k", int)
        if err is not None:
            return err

        if k < 2:
            return "k value must be 2 or greater, not '{}'".format(k)

        return None

    def check_valid_pool_method(self, pool_method: str) -> str or None:
        """
        Checks the pool method is valid. Returns None if the pool method is valid, a
        descriptive error message if invalid.

        Parameters
        ----------
        pool_method: str
            the method used to pool embeddings within windows (of size k) to compare
            windows to each other.
            Possible values: 'mean', 'max'

        Returns
        -------
        str or None
            None if the pool method is valid, otherwise an error message.
        """
        pool_methods = ["mean", "max"]
        if pool_method not in pool_methods:
            return "pool_method must be one of {} not '{}'" "".format(
                pool_methods, pool_method
            )

        return None

    def check_valid_window_compare_pool_method(self, pool_method: str) -> str or None:
        """
        Checks the pool method used to compare adjacent windows is valid. Returns None
        if the pool method is valid, a descriptive error message if invalid.

        Parameters
        ----------
        pool_method: str
            the method used to pool embeddings within windows (of size k) for comparison
            to adjacent windows.
            Possible values: 'mean', 'max'

        Returns
        -------
        str or None
            None if the pool method is valid, otherwise an error message.
        """
        return self.check_valid_pool_method(pool_method)

    def check_valid_embedding_aggregation_pool_method(
        self, pool_method: str
    ) -> str or None:
        """
        Checks the pool method used to aggregate embeddings within a segment is valid.

        Parameters
        ----------
        pool_method: str
            the method used to pool embeddings within a segment to create a single
            embedding for the segment.
            Possible values: 'mean', 'max'

        Returns
        -------
        str or None
            None if the pool method is valid, otherwise an error message.
        """
        return self.check_valid_pool_method(pool_method)

    def check_valid_smoothing_width(self, smoothing_width: int) -> str or None:
        """
        Checks the smoothing width is valid. Returns None if the smoothing width is
        valid, a descriptive error message if invalid.

        Parameters
        ----------
        smoothing_width: int
            The width of the window used by the smoothing method

        Returns
        -------
        str or None
            None if the smoothing width is valid, otherwise an error message.
        """
        err = self._type_checker.check_type(smoothing_width, "smoothing_width", int)
        if err is not None:
            return err

        if smoothing_width < 3:
            return "smoothing_width must be greater than 2, not '{}'" "".format(
                smoothing_width
            )

        return None

    def check_valid_cutoff_policy(self, cutoff_policy: str) -> str or None:
        """
        Checks the cutoff policy is valid. Returns None if the cutoff policy is valid,
        a descriptive error message if invalid.

        Parameters
        ----------
        cutoff_policy: str
            The policy used to determine how dissimilar adjacent embedding windows must
            be to consider them to be from different segments (a boundary).
            Possible values: 'average', 'high', or 'low'

        Returns
        -------
        str or None
            None if the cutoff policy is valid, otherwise an error message.
        """
        cutoff_policies = ["average", "low", "high"]
        if cutoff_policy not in cutoff_policies:
            return "cutoff_policy must be one of {} not '{}'" "".format(
                cutoff_policies, cutoff_policy
            )

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
