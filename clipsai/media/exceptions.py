"""
Exceptions that can be raised by the media package.
"""
# local imports
from filesys.exceptions import FileError


class MediaFileError(FileError):
    pass


class ImageFileError(MediaFileError):
    pass


class TemporalMediaFileError(MediaFileError):
    pass


class AudioFileError(TemporalMediaFileError):
    pass


class VideoFileError(TemporalMediaFileError):
    pass


class AudioVideoFileError(TemporalMediaFileError):
    pass


class MediaEditorError(Exception):
    pass


class NoAudioStreamError(AudioFileError):
    pass


class NoVideoStreamError(VideoFileError):
    pass
