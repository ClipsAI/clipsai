"""
Working with audio-video files.

Notes
-----
- AudioVideoFiles are defined to be files that contain both audio and video (cannot lack
either form).
"""
# standard library imports
import logging

# current package imports
from .audio_file import AudioFile
from .exceptions import AudioVideoFileError
from .video_file import VideoFile
from .temporal_media_file import TemporalMediaFile


class AudioVideoFile(AudioFile, VideoFile):
    """
    A class for working with audio-video files, files that contain both audio and video.
    """

    def __init__(self, audiovideo_file_path: str) -> None:
        """
        Initialize AudioVideoFile

        Parameters
        ----------
        audiovideo_file_path: str
            absolute path to an audio-video file

        Returns
        -------
        None
        """
        super().__init__(audiovideo_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'AudioVideoFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'AudioVideoFile' as a string.
        """
        return "AudioVideoFile"

    def check_exists(self) -> str or None:
        """
        Checks that the AudioVideoFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the AudioVideoFile exists in the file system, a descriptive error
            message if not.
        """
        # check if it's a temporal media file
        temporal_media_file = TemporalMediaFile(self._path)
        error = temporal_media_file.check_exists()
        if error:
            return error

        # check if it's an audio file
        if temporal_media_file.has_audio_stream() is False:
            return (
                "'{}' is a valid {} but has no audio stream so it is not a valid {} "
                "file."
                "".format(self._path, temporal_media_file.get_type(), self.get_type())
            )
        # check if it's a video file
        if temporal_media_file.has_video_stream() is False:
            return (
                "'{}' is a valid {} but has no video stream so it is not a valid {} "
                "file."
                "".format(self._path, temporal_media_file.get_type(), self.get_type())
            )

    def get_bitrate(self, stream) -> str or None:
        """
        Returns the bitrate of the audio stream.

        Parameters
        ----------
        stream: str
            The stream to get the bitrate of ("v:0" for video, "a:0" for audio)

        Returns
        -------
        str or None
            bitrate of the audio stream
        """
        if stream == "a:0":
            return AudioFile.get_bitrate(self)
        elif stream == "v:0":
            return VideoFile.get_bitrate(self)
        else:
            err = "Invalid stream '{}'. Must be 'a:0' (audio) or 'v:0' (video).".format(
                stream
            )
            logging.error(err)
            raise AudioVideoFileError(err)
