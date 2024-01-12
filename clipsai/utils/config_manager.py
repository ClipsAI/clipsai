"""
Defines an abstract class for getting information about and validating configuration
settings for machine learning model classes.
"""
# standard library imports
import abc

# current package imports
from .exceptions import ConfigError
from .type_checker import TypeChecker


class ConfigManager(abc.ABC):
    """
    Abstract class for getting information about and validating configuration
    settings for machine learning model classes.
    """

    def __init__(self) -> None:
        """
        Parameters
        ----------
        None
        """
        self._type_checker = TypeChecker()

    def impute_default_config(self, config: dict) -> dict:
        """
        Populates missing config fields with default values.

        Parameters
        ----------
        config: dict
            The configuration to impute.

        Returns
        -------
        dict
            The config imputed with default values.
        """
        return config

    @abc.abstractmethod
    def check_valid_config(self, config: dict) -> str or None:
        """
        Checks that 'config' contains valid configuration settings. Returns None if
        valid, a descriptive error message if invalid.

        Parameters
        ----------
        config: dict
            A dictionary containing the configuration settings for the machine learning
            model.

        Returns
        -------
        str or None
            None if the inputs are valid, otherwise an error message.
        """
        pass

    def is_valid_config(self, config: dict) -> bool:
        """
        Returns True if 'config' contains valid configuration settings, False if
        invalid.

        Parameters
        ----------
        config: dict
            A dictionary containing the configuration settings for the machine learning
            model.

        Returns
        -------
        bool
            True if the inputs are valid, False otherwise.
        """
        return self.check_valid_config(config) is None

    def assert_valid_config(self, config: dict) -> None:
        """
        Raises an Error if 'config' contains invalid configuration settings.

        Parameters
        ----------
        config: dict
            A dictionary containing the configuration settings for the machine learning
            model.

        Returns
        -------
        None
        """
        error = self.check_valid_config(config)
        if error is not None:
            raise ConfigError(error)
