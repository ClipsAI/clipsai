"""
Working with directories in the local file system.
"""
# standard library imports
from __future__ import annotations
import logging
import os
import shutil

# current package imports
from .file import File
from .object import FileSystemObject


class Dir(FileSystemObject):
    """
    A class for working with directories in the local file system.
    """

    def __init__(self, dir_path: str) -> None:
        """
        Initialize Dir

        Parameters
        ----------
        dir_path: str
            absolute path of a directory to set the Dir's path to
        """
        super().__init__(dir_path)

    def get_parent_dir(self) -> Dir:
        """
        Gets the parent directory of the File.

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            The parent directory of the File.
        """
        parent_dir = Dir(self.get_parent_dir_path())
        parent_dir.assert_exists()
        return parent_dir

    def create(self) -> None:
        """
        Creates a new Dir

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.assert_does_not_exist()
        parent_dir = self.get_parent_dir()
        parent_dir.assert_exists()

        # create directory
        os.mkdir(self._path)

    def delete(self) -> None:
        """
        Deletes Dir

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.assert_exists()

        # delete directory
        shutil.rmtree(self._path)
        logging.debug("Directory '{}' removed successfully.".format(self._path))

    def move(self, new_path: str) -> None:
        """
        Moves Dir to new_path

        Parameters
        ----------
        new_path: str
            new path to move Dir to

        Returns
        -------
        None
        """
        self.assert_exists()

        dir = Dir(new_path)
        dir.assert_does_not_exist()

        # move directory
        shutil.move(self._path, new_path)
        logging.debug("Directory '{}' moved successfully.".format(self._path))

    def get_type(self) -> str:
        """
        Returns the object type 'Dir' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'Dir' as a string.
        """
        return "Dir"

    def check_exists(self) -> str or None:
        """
        Checks that Dir exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if Dir exists in the file system, a descriptive error message if not.
        """
        # check if the path is a valid FileSystemObject
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if the path is a valid directory
        if os.path.isdir(self._path) is False:
            return (
                "'{}' is a valid {} but not a valid {}."
                "".format(self._path, super().get_type(), self.get_type())
            )

        return None

    def scan_dir(self) -> list[FileSystemObject]:
        """
        Scans the directory and returns a list of all files and subdirectories.

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            A list of all files and subdirectory paths
        """
        self.assert_exists()

        fs_objects = []
        with os.scandir(self._path) as paths:
            for path in paths:
                fs_object_path = os.path.join(self._path, path.name)
                if path.is_file():
                    fs_object = File(fs_object_path)
                elif path.is_dir():
                    fs_object = Dir(fs_object_path)
                else:
                    pass
                fs_object.assert_exists()
                fs_objects.append(fs_object)

        return fs_objects

    def get_files(self) -> list[File]:
        """
        Gets the files in Dir

        Parameters
        ----------
        None

        Returns
        -------
        list[File]
            all files in Dir
        """
        self.assert_exists()
        fs_objects = self.scan_dir()

        files = []
        for fs_object in fs_objects:
            if isinstance(fs_object, File):
                file = fs_object
                files.append(file)
        return files

    def get_subdirs(self) -> list[Dir]:
        """
        Gets the subdirectories in Dir

        Parameters
        ----------
        None

        Returns
        -------
        list[Dir]
            all subdirectories in Dir
        """
        self.assert_exists()
        fs_objects = self.scan_dir()

        subdirs = []
        for fs_object in fs_objects:
            if isinstance(fs_object, Dir):
                subdir = fs_object
                subdirs.append(subdir)
        return subdirs

    def get_files_with_extension(self, extension: str) -> list[File]:
        """
        Gets all files in Dir with a certain extension

        Parameters
        ----------
        extension: str
            file extension to filter by

        Returns
        -------
        list[File]
            all files in Dir with a certain extension
        """
        self.assert_exists()

        files = self.get_files()
        files_with_valid_ext = []
        for file in files:
            if file.get_file_extension() == extension:
                files_with_valid_ext.append(file)
        return files_with_valid_ext

    def get_file_paths_with_extension(self, extension: str) -> list[str]:
        """
        Gets all files paths in Dir with a certain extension

        Parameters
        ----------
        extension: str
            file extension to filter by

        Returns
        -------
        list[str]
            all files paths in Dir with a certain extension
        """
        self.assert_exists()

        files = self.get_files_with_extension(extension)
        file_paths = []
        for file in files:
            file_paths.append(file.path)
        return file_paths

    def zip(self, zip_file_name: str) -> File:
        """
        Zips the contents of Dir and places the zip file inside Dir's parent directory

        Parameters
        ----------
        zip_file_name: str
            name of the zip file

        Returns
        -------
        File
            the zip file as a file object
        """
        self.assert_exists()

        zip_file_path = shutil.make_archive(zip_file_name, "zip", self._path)
        desired_zip_file_path = os.path.join(
            self.get_parent_dir_path(), "{zip_file_name}.zip"
        )
        shutil.move(zip_file_path, desired_zip_file_path)

        return File(desired_zip_file_path)

    def delete_contents(self) -> None:
        """
        Deletes the contents of Dir but not Dir itself.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.assert_exists()

        # delete files media_file_to_transcode
        files = self.get_files()
        for file in files:
            file.delete()

        # delete subdirectories
        subdirs = self.get_subdirs()
        for dir in subdirs:
            dir.delete()

    def delete_contents_except_asset(self) -> None:
        """
        Deletes the contents of Dir but not Dir itself,
        except for media_file_to_transcode

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.assert_exists()

        files = self.get_files()
        # delete files that don't start with "media_file_to_transcode"
        for file in files:
            if file.get_filename().startswith("media_file_to_transcode"):
                logging.info(
                    "Skipping deletion of file '{}'".format(file.get_filename())
                )
                continue
            file.delete()

        # delete subdirectories
        subdirs = self.get_subdirs()
        for dir in subdirs:
            dir.delete()
