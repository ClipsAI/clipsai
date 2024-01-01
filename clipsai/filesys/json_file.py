"""
Working with json files in the local file system.
"""
# standard library imports
from __future__ import annotations
import json

# current package imports
from .file import File

# local imports
from utils.type_checker import TypeChecker


class JSONFile(File):
    """
    A class for working with json files in the local file system.
    """

    def __init__(self, json_file_path: str) -> None:
        """
        Initialize Json File

        Parameters
        ----------
        json_file_path: str
            absolute path of a json file to set JsonFile's path to

        Returns
        -------
        None
        """
        super().__init__(json_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'JsonFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'JsonFile' as a string.
        """
        return "JsonFile"

    def check_exists(self) -> str or None:
        """
        Checks that JsonFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if JsonFile exists in the file system, a descriptive error
            message if not
        """
        # check if the path is a valid File
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if the path is a valid JsonFile
        file_extension = self.get_file_extension()
        if file_extension != "json":
            return (
                "'{}' is a valid {} but is not a valid {} because it has file "
                "extension '{}' instead of 'json'.".format(
                    self._path, super().get_type(), self.get_type(), file_extension
                )
            )

    def create(self, data: dict) -> None:
        """
        Creates a new json file at 'file_path' with contents 'data'

        Parameters
        ----------
        file_path: str
            absolute path of json file to create
        data: dict
            data to write to json file

        Returns
        -------
        None
        """
        super().create(json.dumps(data))
        self.assert_exists()

    def read(self) -> dict:
        """
        Returns the json file data as a dictionary

        Parameters
        ----------
        None

        Returns
        -------
        dict:
            json file data as a dictionary
        """
        self.assert_exists()

        # read in data
        file = open(self._path)
        file_data = json.loads(file.read())
        file.close()

        return file_data

    def write(self, new_data: dict) -> None:
        """
        Writes a dictionary to a json file

        Parameters
        ----------
        new_data: dict
            data to replace the json file's contents with
        """
        self.assert_exists()

        type_checker = TypeChecker()
        type_checker.assert_type(new_data, "data", dict)

        with open(self._path, "w") as file_object:
            json.dump(new_data, file_object)
