"""
Managing the local file system.
"""
# standard library imports
import logging
import re

# current package imports
from .exceptions import FileSystemObjectError
from .dir import Dir
from .object import FileSystemObject

# local imports
from ..utils.type_checker import TypeChecker


class FileSystemManager:
    """
    A class for managing the local file system.
    """

    def __init__(self) -> None:
        """
        Initialize FileSystemManager

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._type_checker = TypeChecker()

    def assert_paths_not_equal(
        self,
        path1: str,
        path2: str,
        path1_name: str,
        path2_name: str,
    ) -> None:
        """
        Raises an error if 'path1' and 'path2' are equal

        Parameters
        ----------
        path1: str
            absolute path to a file system object
        path2: str
            absolute path to a file system object

        Returns
        -------
        None

        Raises
        ------
        PathError: 'path1' and 'path2' are equal
        """
        if path1 == path2:
            msg = (
                "{} with path '{}' is equal to {} with path '{}' but paths are "
                "supposed to be different.".format(path1_name, path1, path2_name, path2)
            )
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def check_valid_path_for_new_fs_object(self, path: str) -> str or None:
        """
        Checks that 'path' is a valid path for a new FileSystemObject. Returns None if
        so, a descriptive error message if not.

        Parameters
        ----------
        path: str
            Absolute path to place a new FileSystemObject.

        Returns
        -------
        None
        """
        # check the path doesn't exist
        fs_object = FileSystemObject(path)
        msg = fs_object.check_does_not_exist()
        if msg is not None:
            return msg

        # check the parent directory exists
        dir = Dir(fs_object.get_parent_dir_path())
        msg = dir.check_exists()
        if msg is not None:
            return msg

        return None

    def is_valid_path_for_new_fs_object(self, path: str) -> bool:
        """
        Returns True if 'path' is a valid path for a new FileSystemObject, False if not.

        Parameters
        ----------
        path: str
            Absolute path to place a new FileSystemObject.

        Returns
        -------
        bool
            True if 'path' is a valid path for a new FileSystemObject, False if not.
        """
        return self.check_valid_path_for_new_fs_object(path) is None

    def assert_valid_path_for_new_fs_object(self, path: str) -> None:
        """
        Raises an error if 'path' is not a valid path for a new FileSystemObject.

        Parameters
        ----------
        path: str
            Absolute path to place a new FileSystemObject.

        Returns
        -------
        None

        Raises
        ------
        FileSystemObjectError: 'path' is not a valid path for a new FileSystemObject.
        """
        msg = self.check_valid_path_for_new_fs_object(path)
        if msg is not None:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def check_parent_dir_exists(self, fs_object: FileSystemObject) -> str or None:
        """
        Checks that the parent directory of 'fs_object' exists. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        fs_object: FileSystemObject
            The file system object to check.

        Returns
        -------
        str or None
            None if the parent directory of 'fs_object' exists, a descriptive error
            message if not.
        """
        # check the parent directory exists
        dir = Dir(fs_object.get_parent_dir_path())
        return dir.check_exists()

    def parent_dir_exists(self, fs_object: FileSystemObject) -> bool:
        """
        Returns True if the parent directory of 'fs_object' exists, False if not.

        Parameters
        ----------
        fs_object: FileSystemObject
            The file system object to check.

        Returns
        -------
        bool
            True if the parent directory of 'fs_object' exists, False if not.
        """
        return self.check_parent_dir_exists(fs_object) is None

    def assert_parent_dir_exists(self, fs_object: FileSystemObject) -> None:
        """
        Raises an error if the parent directory of 'fs_object' does not exist.

        Parameters
        ----------
        fs_object: FileSystemObject
            The file system object to check.

        Returns
        -------
        None

        Raises
        ------
        FileSystemObjectError: The parent directory of 'fs_object' does not exist.
        """
        msg = self.check_parent_dir_exists(fs_object)
        if msg is not None:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def filter_filename(self, filename: str) -> str:
        """
        Filters out invalid characters from a filename to avoid file system issues

        - Invalid characters: \\/.,:*?"<>|

        Parameters
        ----------
        filename: str
            original filename

        Returns
        -------
        str
            sanitized filename
        """
        # Define a reg ex pattern for invalid characters in most file systems
        invalid_chars = r'[\\/.,:*?"<>|]'
        return re.sub(invalid_chars, "", filename)
