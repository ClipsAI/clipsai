"""
Defines an abstract class for all validating incoming inputs.
"""
# standard library imports
import abc

# current package imports
from exceptions import InvalidInputDataError

# local package imports
from utils.utils import find_missing_dict_keys
from utils.type_checker import TypeChecker


class InputValidator:
    """
    Abstract class defining input validators for verifying calls to our package.
    """

    def __init__(self) -> None:
        """
        Initialize the InputValidator class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._type_checker = TypeChecker()

    @abc.abstractmethod
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
        pass

    def is_valid_input_data(self, input_data: dict) -> str or None:
        """
        Returns True if the input data is valid, False if not.

        Parameters
        ----------
        input_data: dict
            The input data to be validated.

        Returns
        -------
        bool
            True if the input data is valid, False if not.
        """
        return self.check_valid_input_data(input_data) is None

    def assert_valid_input_data(self, input_data: dict) -> None:
        """
        Raises an error if the input data is invalid.

        Parameters
        ----------
        input_data: dict
            The input data to be validated.

        Returns
        -------
        None

        Raises
        ------
        InvalidinputDataError
            The input data is not valid.
        """
        error = self.check_valid_input_data(input_data)
        if error is not None:
            raise InvalidInputDataError(error)

    def check_input_data_existence_and_types(
        self,
        input_data: dict,
        correct_types: dict,
    ) -> str or None:
        """
        Checks if the input data contains all the required keys and that all values
        are of the proper type. Returns None if so, a descriptive error message if not.

        Parameters
        ----------
        input_data: str
            input data with keys to verify
        correct_types: dict
            A dictionary with the necessary keys for input data where the value for
            each key is the correct type for that key in input data.

        Returns
        -------
        str or None
            Returns None if the input data contains all the required keys, a
            descriptive error message if not.
        """
        missing_keys = find_missing_dict_keys(input_data, correct_types.keys())
        if len(missing_keys) != 0:
            return "input data is missing the following keys: {}".format(missing_keys)

        error = self._type_checker.check_dict_types(input_data, correct_types)
        if error is not None:
            return error

        return None

    def impute_input_data_defaults(self, input_data: dict) -> dict:
        """
        Populates missing input data fields with default values.

        Parameters
        ----------
        input_data: dict
            The input data to impute.

        Returns
        -------
        dict
            The input data imputed with default values.
        """
        return input_data

    @abc.abstractmethod
    def validate(self, input_data: dict) -> dict:
        """
        Validates the input and returns useful information extracted during the
        validation process.

        Parameters
        ----------
        input_data: dict
            The input data to validate.

        Returns
        -------
        dict
            Useful information extracted during the validation process.
        """
        pass
