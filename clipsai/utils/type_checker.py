"""
Checking the types of variables, lists, and dictionaries.
"""
# standard library imports
import logging


class TypeChecker:
    """
    A class to check the types of variables, lists, and dictionaries.
    """

    def check_type(self, data, data_label: str, correct_types: tuple) -> str or None:
        """
        Checks if 'data' is of any type in 'correct_types' and returns a descriptive
        message if it isn't

        Parameters
        ----------
        data:
            a variable
        data_label: str
            the name of 'data' used to give a descriptive error message
        correct_type: tuple
            the possible types 'data' could be

        Returns
        -------
        str or None
            None if 'data' is of any type in 'correct_types'. Returns a descriptive
            error message if 'data' is not of 'correct_type'
        """
        if isinstance(data, correct_types) is False:
            msg = (
                "Variable '{}' with value '{}' must be one of the following types: "
                "'{}', not type '{}'"
                "".format(data_label, data, [correct_types], type(data))
            )
            return msg

        return None

    def is_type(self, data, correct_types: tuple) -> bool:
        """
        Returns True if 'data' is of any type in 'correct_types', False if it isn't

        Parameters
        ----------
        data:
            a variable
        correct_type: tuple
            the possible types 'data' could be

        Returns
        -------
        bool
            True if 'data' is of any type in 'correct_types', False if it isn't
        """
        if self.check_type(data, correct_types) is None:
            return True
        else:
            return False

    def assert_type(self, data, data_label: str, correct_types: tuple) -> None:
        """
        Raises an error if 'data' is not of any type in 'correct_types'

        Parameters
        ----------
        data:
            a variable
        data_label: str
            the name of 'data' used to give a descriptive error message
        correct_type: tuple
            the possible types 'data' could be

        Returns
        -------
        None

        Raises
        ------
        TypeError: raised if 'data' is not of any type in 'correct_types'
        """
        msg = self.check_type(data, data_label, correct_types)
        if msg is not None:
            logging.error(msg)
            raise TypeError(msg)

    def check_list_types(
        self,
        data: list,
        data_labels: list,
        correct_types: tuple,
    ) -> str or None:
        """
        Checks if all variables in the list 'data' have type that is one of
        'correct_types'

        Parameters
        ----------
        data:
            list of data
        data_labels: str
            names of the variables in data used to give a descriptive error message.
            Index "i" of 'data_labels' is the label for data located at index "i" of
            'data'
        correct_types: tuple
            the possible types that objects in 'data' could be

        Returns
        -------
        str or None
            None if all variables in 'data' are one of any type in 'correct_types'.
            Returns a descriptive error message if any variable in 'data' is not of any
            type in 'correct_types'

        Raises
        -----
        ValueError: raised if len(data) != len(data_labels)
        """
        # ensure data and data_labels are lists
        self.assert_type(data, "data", list)
        self.assert_type(data_labels, "data_labels", list)

        # ensure data and data_labels are the same length
        data_len = len(data)
        data_labels_len = len(data_labels)
        if data_len != data_labels_len:
            err_msg = (
                "Length of list 'data' ({}) must be equal to the length of list "
                "'data_labels_len ({})".format(data_len, data_labels_len)
            )
            logging.error(err_msg)
            raise ValueError(err_msg)

        for i in range(data_len):
            msg = self.check_type(data[i], data_labels[i], correct_types)
            if msg is not None:
                return msg

        return None

    def are_list_elems_of_type(self, data: list, correct_types: tuple) -> bool:
        """
        Returns True if all variables in the list 'data' have type that is one of
        'correct_types', False if any variable in 'data' is not of any type in
        'correct_types'

        Parameters
        ----------
        data:
            list of data
        correct_types: tuple
            the possible types that objects in 'data' could be

        Returns
        -------
        bool
            True if all variables in 'data' are one of any type in 'correct_types'.
            Returns False if any variable in 'data' is not of any type in
            'correct_types'
        """
        msg = self.check_list_types(data, correct_types)
        if msg is None:
            return True
        else:
            return False

    def assert_list_elems_type(
        self,
        data: list,
        data_labels: list,
        correct_types: tuple,
    ) -> None:
        """
        Raises an error if any variable in the list 'data' is not of any type in
        'correct_types'

        Parameters
        ----------
        data:
            list of data
        data_labels: str
            names of the variables in data used to give a descriptive error message.
            Index "i" of 'data_labels' is the label for data located at index "i" of
            'data'
        correct_types: tuple
            the possible types that objects in 'data' could be

        Returns
        -------
        None

        Raises
        ------
        TypeError: raised if any variable in 'data' is not of any type in
        'correct_types'
        """
        msg = self.check_list_types(data, data_labels, correct_types)
        if msg is not None:
            logging.error(msg)
            raise TypeError(msg)

    def check_dict_types(self, data: dict, correct_types: dict) -> str or None:
        """
        Checks that elements in 'data' are of the type specified by the same key in
        'correct_types'.

        - If a key in 'data' is not in 'correct_data_types', that key is not checked

        Parameters
        ----------
        data: dict
            a dictionary of data
        correct_types: dict
            a dictionary containing a subset of keys from 'data' with values at each
            key being a tuple of possible data types for that key in 'data'

        Returns
        -------
        str or None
            None if all elements in 'data' are of the correct type. Returns a
            descriptive error message if any elements is not of the correct type.

        Raises
        ------
        KeyError: raised if 'correct_data_types' has keys that 'data' doesn't have
        """
        data_keys = data.keys()
        correct_types_keys = correct_types.keys()
        # check if 'data' has keys 'correct_data_types' doesn't have
        if len(data_keys - correct_types_keys) > 0:
            debug_msg = (
                "Keys '{}' are in 'data' but are not in 'correct_data_types'"
                "".format(
                    data_keys - correct_types_keys,
                )
            )
            logging.debug(debug_msg)
        # check if 'correct_data_types' has keys 'data' doesn't have
        if len(correct_types_keys - data_keys) > 0:
            err = (
                "Keys '{}' are in 'correct_data_types' but are not in 'data'"
                "".format(correct_types_keys - data_keys)
            )
            logging.error(err)
            raise KeyError(err)

        for key in correct_types.keys():
            msg = self.check_type(data[key], key, correct_types[key])
            if msg is not None:
                return msg

        return None

    def are_dict_elems_of_type(self, data: dict, correct_types: dict) -> bool:
        """
        Returns True if all elements in 'data' are of the type specified by the same
        key in 'correct_types', False if any element is not of the correct type.

        - If a key in 'data' is not in 'correct_data_types', that key is not checked

        Parameters
        ----------
        data: dict
            a dictionary of data
        correct_types: dict
            a dictionary containing a subset of keys from 'data' with values at each
            key being a tuple of possible data types for that key in 'data'

        Returns
        -------
        bool
            True if all elements in 'data' are of the correct type. Returns False if
            any elements is not of the correct type.

        Raises
        ------
        KeyError: raised if 'correct_data_types' has keys that 'data' doesn't have
        """
        msg = self.check_dict_types(data, correct_types)
        if msg is None:
            return True
        else:
            return False

    def assert_dict_elems_type(self, data: dict, correct_types: dict) -> None:
        """
        Raises an error if any elements in 'data' are not of the type specified by the
        same key in 'correct_types'.

        - If a key in 'data' is not in 'correct_data_types', that key is not checked

        Parameters
        ----------
        data: dict
            a dictionary of data
        correct_types: dict
            a dictionary containing a subset of keys from 'data' with values at each
            key being a tuple of possible data types for that key in 'data'

        Returns
        -------
        None

        Raises
        ------
        TypeError: raised if any elements in 'data' are not of the correct type
        """
        msg = self.check_dict_types(data, correct_types)
        if msg is not None:
            logging.error(msg)
            raise TypeError(msg)
