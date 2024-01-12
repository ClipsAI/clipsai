"""
Working with files in the local file system.

Notes
-----
- Information on mimetypes:
https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types
"""
# standard library imports
from __future__ import annotations
import logging
import os

# current package imports
from .exceptions import FileError
from .object import FileSystemObject

# 3rd party imports
import magic


class File(FileSystemObject):
    """
    A class for working with files in the local file system
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize File

        Parameters
        ----------
        file_path: str
            absolute path of a file to set File's path to

        Returns
        -------
        None
        """
        super().__init__(file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'File' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'File' as a string.
        """
        return "File"

    def get_filename(self) -> str:
        """
        Returns the filename of the File.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The filename of the File.
        """
        return os.path.basename(self._path)

    def get_filename_without_extension(self) -> str:
        """
        Returns the filename of the File without the extension.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The filename of the File without the extension.
        """
        return os.path.splitext(self.get_filename())[0]

    def check_exists(self) -> str or None:
        """
        Checks that the File exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the File exists in the file system, a descriptive error message if
            not.
        """
        # check if the path is a valid FileSystemObject
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if the path is a valid File
        if os.path.isfile(self._path) is False:
            return "'{}' is a valid {} but not a valid {}." "".format(
                self._path, super().get_type(), self.get_type()
            )

        return None

    def create(self, data: str) -> None:
        """
        Creates a File at 'file_path' with 'data' as its contents.

        Parameters
        ----------
        data: str
            Data to be written to the new File.

        Returns
        -------
        None
        """
        self.assert_does_not_exist()

        # create file
        with open(self.path, "x") as f:
            f.write(data)

        self.assert_exists()

    def delete(self) -> None:
        """
        Deletes the File.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.exists() is False:
            logging.warning("File '{}' does not exist.".format(self.path))
            return None

        # delete file
        os.remove(self._path)
        logging.debug("File '{}' successfully deleted.".format(self.path))

    def move(self, new_path: str) -> None:
        """
        Moves the File.

        Parameters
        ----------
        new_path: str
            Absolute file path to be the new location of File

        Returns
        -------
        None
        """
        self.assert_exists()

        file = File(new_path)
        file.assert_does_not_exist()

        # move file
        os.rename(self._path, new_path)
        self._path = new_path

    def get_file_size(self) -> int:
        """
        Gets the file size in number of bytes.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The file size in number of bytes.
        """
        self.assert_exists()

        file_size = os.path.getsize(self._path)
        return file_size

    def get_mime_type(self) -> str:
        """
        Gets the mime type.

        - Information on mimetypes:
        https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types

        Parameters
        ----------
        None

        Returns
        -------
        str
            The mime type.
        """
        # File must exist to use magic to grab mimetype
        self.assert_exists()

        mime = magic.Magic(mime=True)
        mime_type = mime.from_file(self._path)
        return mime_type

    def get_mime_primary_type(self) -> str:
        """
        Gets the file type from the mime type.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The file type retrieved from the mime type.
        """
        file_type, _ = self.get_mime_type().split("/")
        return file_type

    def get_mime_secondary_type(self) -> str:
        """
        Gets the file subtype from the mime type.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            The file subtype retrieved from the mime type.
        """
        _, file_subtype = self.get_mime_type().split("/")
        return file_subtype

    def get_file_extension(self) -> str or None:
        """
        Gets the file extension. Returns None if the File doesn't have an extension.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            The File's extension. Returns None if the File doesn't have an extension.
        """
        _, file_extension = os.path.splitext(self._path)
        # file path doesn't have an extension
        if len(file_extension) == 0:
            return None
        # remove the "." from the file extension
        return "".join([char for char in file_extension if char != "."])

    def check_has_file_extension(self, extension: str) -> str or None:
        """
        Checks if the File has extension 'extension'. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        extension: str
            The file extension to check against.

        Returns
        -------
        str or None
            None if the File has extension 'extension', a descriptive error message if
            not.
        """
        if self.get_file_extension() != extension:
            return "{} '{}' should have extension '{}' not '{}'." "".format(
                self.get_type(), self._path, extension, self.get_file_extension()
            )

    def has_file_extension(self, extension: str) -> bool:
        """
        Returns True if the File has extension 'extension', False if not.

        Parameters
        ----------
        extension: str
            The file extension to check against.

        Returns
        -------
        bool
            True if the File has extension 'extension', False if not.
        """
        return self.check_has_file_extension(extension) is None

    def assert_has_file_extension(self, extension: str) -> None:
        """
        Raises an error if the File does not have extension 'extension'.

        Parameters
        ----------
        extension: str
            The file extension to check against.

        Returns
        -------
        None

        Raises
        ------
        FileError
            The File does not have extension 'extension'.
        """
        msg = self.check_has_file_extension(extension)
        if msg is not None:
            logging.error(msg)
            raise FileError(msg)
