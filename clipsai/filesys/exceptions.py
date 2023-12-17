"""
Exceptions that can be raised by the filesys package.
"""


# Class errors
class FileSystemObjectError(OSError):
    pass


class FileError(FileSystemObjectError):
    pass


class JsonFileError(FileError):
    pass


class DirError(FileSystemObjectError):
    pass
