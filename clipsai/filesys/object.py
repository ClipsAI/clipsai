"""
Defines base class for working with objects (files, directories, etc.) in the local
file system.
"""
# standard library imports
import logging
import os

# current package imports
from .exceptions import FileSystemObjectError

# local imports
from clipsai.utils.type_checker import TypeChecker


class FileSystemObject:
    """
    A class for working with file system objects (files, directories, etc.) in the
    local file system.
    """

    ##################
    # Pubic Methods #
    ##################
    def __init__(self, path: str) -> None:
        """
        Initialize FileSystemObject.

        Parameters
        ----------
        path: str
            Absolute path of a file system object to set FileSystemObject's path to.

        Returns
        -------
        None
        """
        self._type_checker = TypeChecker()
        self._type_checker.assert_type(path, "path", str)
        self._path = path

    @property
    def path(self) -> str:
        """
        Returns the absolute path of the FileSystemObject.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Absolute path of FileSystemObject.
        """
        return self._path

    def get_path(self) -> str:
        """
        Returns the absolute path of the FileSystemObject.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Absolute path of FileSystemObject.
        """
        return self._path

    def get_type(self) -> str:
        """
        Returns the object type 'FileSystemObject' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'FileSystemObject' as a string.
        """
        return "FileSystemObject"

    def set_path(self, new_path: str) -> None:
        """
        Sets the absolute path of the FileSystemObject.

        Parameters
        ----------
        new_path: str
            Absolute path of a file sytem object to replace FileSystemObject's existing
            path with.

        Returns
        -------
        None
        """
        self._type_checker.assert_type(new_path, "new_path", str)
        self._path = new_path

    def check_exists(self) -> str or None:
        """
        Check if the FileSystemObject exists in the file system. Returns None if
        so, a descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the FileSystemObject exists in the file system, a descriptive
            error message if not.
        """
        if os.path.exists(self._path) is False:
            return "{} '{}' does not exist.".format(self.get_type(), self._path)

        return None

    def exists(self) -> bool:
        """
        Returns True if the FileSystemObject exists in the file system, False if not.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the FileSystemObject exists in the file system, False if not.
        """
        return self.check_exists() is None

    def assert_exists(self) -> None:
        """
        Raises an error if the FileSystemObject doesn't exist in the file system.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        FileSystemObjectError
            The FileSystemObject does not exist in the file system.
        """
        msg = self.check_exists()
        if msg is not None:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def check_does_not_exist(self) -> str or None:
        """
        Checks that the FileSystemObject does not exist in the file system. Returns
        None if it does not exist, a descriptive error message if it does exist.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the FileSystemObject does not exist, a descriptive error message if
            it does exist.
        """
        if self.exists() is True:
            return "{} '{}' already exists.".format(self.get_type(), self._path)

        return None

    def assert_does_not_exist(self) -> None:
        """
        Raises an error if the FileSystemObject exists in the file system.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        FileSystemObjectError
            The FileSystemObject exists in the file system.
        """
        msg = self.check_does_not_exist()
        if msg is not None:
            logging.error(msg)
            raise FileSystemObjectError(msg)

    def get_parent_dir_path(self) -> str:
        """
        Gets the absolute path of the FileSystemObject's parent directory.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Absolute path of the FileSystemObject's parent directory.
        """
        return os.path.dirname(self._path)
