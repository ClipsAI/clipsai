"""
Defines an abstract class for all validating incoming requests.
"""
# standard library imports
import abc

# current package imports
from .exceptions import InvalidRequestDataError

# local package imports
from .utils.utils import find_missing_dict_keys
from .utils.type_checker import TypeChecker


class RequestValidator:
    """
    Abstract class defining API request validators.
    """

    def __init__(self) -> None:
        """
        Initialize the RequestValidator class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._type_checker = TypeChecker()

    @abc.abstractmethod
    def check_valid_request_data(self, request_data: dict) -> str or None:
        """
        Checks if the request data is valid. Returns None if so, a descriptive error
        message if not.

        Parameters
        ----------
        request_data: dict
            The request data to be validated.

        Returns
        -------
        str or None
            None if the request data is valid, a descriptive error message if not.
        """
        pass

    def is_valid_request_data(self, request_data: dict) -> str or None:
        """
        Returns True if the request data is valid, False if not.

        Parameters
        ----------
        request_data: dict
            The request data to be validated.

        Returns
        -------
        bool
            True if the request data is valid, False if not.
        """
        return self.check_valid_request_data(request_data) is None

    def assert_valid_request_data(self, request_data: dict) -> None:
        """
        Raises an error if the request data is invalid.

        Parameters
        ----------
        request_data: dict
            The request data to be validated.

        Returns
        -------
        None

        Raises
        ------
        InvalidRequestDataError
            The request data is not valid.
        """
        error = self.check_valid_request_data(request_data)
        if error is not None:
            raise InvalidRequestDataError(error)

    def check_request_data_existence_and_types(
        self,
        request_data: dict,
        correct_types: dict,
    ) -> str or None:
        """
        Checks if the request data contains all the required keys and that all values
        are of the proper type. Returns None if so, a descriptive error message if not.

        Parameters
        ----------
        request_data: str
            Request data with keys to verify
        correct_types: dict
            A dictionary with the necessary keys for request data where the value for
            each key is the correct type for that key in request data.

        Returns
        -------
        str or None
            Returns None if the request data contains all the required keys, a
            descriptive error message if not.
        """
        missing_keys = find_missing_dict_keys(request_data, correct_types.keys())
        if len(missing_keys) != 0:
            return "Request data is missing the following keys: {}".format(missing_keys)

        error = self._type_checker.check_dict_types(request_data, correct_types)
        if error is not None:
            return error

        return None

    def impute_request_data_defaults(self, request_data: dict) -> dict:
        """
        Populates missing request data fields with default values.

        Parameters
        ----------
        request_data: dict
            The request data to impute.

        Returns
        -------
        dict
            The request data imputed with default values.
        """
        return request_data

    @abc.abstractmethod
    def validate(self, request_data: dict) -> dict:
        """
        Validates the request and returns useful information extracted during the
        validation process.

        Parameters
        ----------
        request_data: dict
            The request data to validate.

        Returns
        -------
        dict
            Useful information extracted during the validation process.
        """
        pass
