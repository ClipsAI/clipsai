"""
Working with temporal media files.

Notes
-----
- TemporalMediaFiles are defined to be files that contain audio or video or both.
"""
# standard library imports
import logging

# current package imports
from .media_file import MediaFile


class TemporalMediaFile(MediaFile):
    """
    A class for working with temporal media files that are time dependent (i.e. audio
    and video).
    """

    def __init__(self, media_file_path: str) -> None:
        """
        Initialize TemporalMediaFile

        Parameters
        ----------
        media_file_path: str
            absolute path to a temporal media file

        Returns
        -------
        None
        """
        super().__init__(media_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'TemporalMediaFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'TemporalMediaFile' as a string.
        """
        return "TemporalMediaFile"

    def check_exists(self) -> str or None:
        """
        Checks that the TemporalMediaFile still exists in the file system. Returns None
        if so, a descriptive error message if not

        Parameters
        ----------
        None

        Returns
        -------
        str or None
        """
        # check if it's a media file
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if it's a temporal media file
        media_file = MediaFile(self._path)
        if not media_file.has_audio_stream() and not media_file.has_video_stream():
            return (
                "'{}' is a valid {} but has neither audio nor video stream so it is "
                "not a valid {}.".format(
                    self._path, super().get_type(), self.get_type()
                )
            )
        return None

    def get_duration(self) -> float:
        """
        Gets the duration in number of seconds

        Parameters
        ----------
        None

        Returns
        -------
        float
            duration in seconds; -1 if duration can't be found
        """
        self.assert_exists()

        duration_str = self.get_format_info("duration")
        if duration_str is None:
            msg = "Retrieving duration of media file '{}' was unsuccessful.".format(
                self._path
            )
            logging.error(msg)
            return -1
        else:
            return float(duration_str)

    def get_bitrate(self, stream) -> int or None:
        """
        Returns the bitrate of the audio stream.

        Parameters
        ----------
        stream: str
            The stream to get the bitrate of ("v:0" for video, "a:0" for audio)

        Returns
        -------
        int or None
            bitrate of the selected stream
        """
        return int(self.get_stream_info(stream, "bit_rate"))
