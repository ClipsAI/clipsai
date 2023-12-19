"""
Working with video files

Notes
-----
- VideoFiles are defined to be files that contain only video and no other media.
"""
# standard library imports
from functools import lru_cache
import logging
from math import floor
from random import randint
import subprocess

# current package imports
from .exceptions import VideoFileError
from .image_file import ImageFile
from .temporal_media_file import TemporalMediaFile

# local imports
from ..utils.conversions import seconds_to_hms_time_format


SUCCESS = 0


class VideoFile(TemporalMediaFile):
    """
    A class for working with video files
    """

    def __init__(self, video_file_path: str) -> None:
        """
        Initialize VideoFile

        Parameters
        ----------
        video_file_path: str
            absolute path to a video file

        Returns
        -------
        None
        """
        super().__init__(video_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'VideoFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'VideoFile' as a string.
        """
        return "VideoFile"

    def check_exists(self) -> str or None:
        """
        Checks that the VideoFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the VideoFile still exists in the file system, a descriptive error
            message if not
        """
        # check if it's a temporal media file
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if it's a video file
        temporal_media_file = TemporalMediaFile(self._path)
        if temporal_media_file.has_video_stream() is False:
            return (
                "'{}' is a valid {} but has no video stream so it is not a valid video "
                "file.".format(self._path, super().get_type())
            )
        if temporal_media_file.is_video_only() is False:
            return (
                "'{}' is a valid {} but is not video only so it is not a valid video "
                "file. Use 'AudioVideoFile' class for files containing both audio and "
                "video.".format(self._path, super().get_type())
            )

    @lru_cache(maxsize=1)
    def get_frame_rate(self) -> float:
        """
        Returns the frame rate of the video file.

        Parameters
        ----------
        None

        Returns
        -------
        float
            The frame rate of the video file.
        """
        frame_rate: str = self.get_stream_info("v:0", "r_frame_rate")
        numerator, denominator = map(int, frame_rate.split("/"))
        frame_rate = numerator / denominator
        return frame_rate

    @lru_cache(maxsize=1)
    def get_height_pixels(self) -> int:
        """
        Returns the height in pixels of the video file.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The height in pixels of the video file.
        """
        return int(self.get_stream_info("v:0", "height"))

    @lru_cache(maxsize=1)
    def get_width_pixels(self) -> int:
        """
        Returns the width in pixels of the video file.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The width in pixels of the video file.
        """
        return int(self.get_stream_info("v:0", "width"))

    @lru_cache(maxsize=1)
    def get_bitrate(self) -> int or None:
        """
        Returns the bitrate in bits per second of the video file.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The bitrate in bits per second of the video file.
        """
        return int(self.get_stream_info("v:0", "bit_rate"))

    def extract_frame(
        self,
        extract_sec: float,
        dest_image_file_path: str,
        overwrite: bool = True,
    ) -> ImageFile or None:
        """
        Extracts a frame at 'extract_sec' to 'dest_image'.

        - The image type is inferred from the file extension of dest_image

        Parameters
        ----------
        extract_sec: str
            the time (in seconds) at which you would like to extract a frame
        dest_image_file_path: str
            the absolute file path to save the extracted frame to
        overwrite: bool
            Overwrites the file at dest_image_file_path if True; does not overwrite if
            False

        Returns
        -------
        ImageFile or None
            the extracted frame if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: extract_sec < 0
        MediaEditorError: extract_sec > video_file's duration
        """
        self.assert_exists()
        if overwrite is True:
            self._filesys_manager.assert_parent_dir_exists(
                ImageFile(dest_image_file_path)
            )
        else:
            self._filesys_manager.assert_valid_path_for_new_fs_object(
                dest_image_file_path
            )
        self._filesys_manager.assert_paths_not_equal(
            self.path,
            dest_image_file_path,
            "video_file path",
            "dest_image_file_path",
        )

        # ensure snapshot time isn't negative
        if extract_sec < 0:
            msg = "extract_sec ({} seconds) cannot be negative.".format(extract_sec)
            logging.error(msg)
            raise VideoFileError(msg)

        # ensure snapshot time doesn't exceed video length
        video_duration = self.get_duration()
        if video_duration == -1:
            msg = (
                "Duration of video file '{}' cannot be found to ensure extract_secs "
                "doesn't exceed video duration. Continuing with input of {} seconds "
                "regardless.".format(self.path, extract_sec)
            )
            logging.warn(msg)
        elif extract_sec > video_duration:
            msg = (
                "extract_sec ({} seconds) cannot exceed video duration ({} seconds)."
                "".format(extract_sec, video_duration)
            )
            logging.error(msg)
            raise VideoFileError(msg)

        # convert seconds to hours, minutes, seconds format '00:00:00.00'
        extract_hms = seconds_to_hms_time_format(extract_sec)
        logging.debug(extract_hms)
        # extract image
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                extract_hms,
                "-i",
                self.path,
                "-frames:v",
                "1",
                "-q:v",
                "0",
                dest_image_file_path,
            ],
            capture_output=True,
            text=True,
        )

        msg = (
            "Terminal return code: '{}'\n"
            "Output: '{}'\n"
            "Err Output: '{}'\n"
            "".format(result.returncode, result.stdout, result.stderr)
        )
        # failure
        if result.returncode != SUCCESS:
            err_msg = (
                "Extracting frame from video file '{}' to '{}' was unsuccessful. Here "
                "is some helpful troubleshooting information: {}"
                "".format(self.path, dest_image_file_path, msg)
            )
            logging.error(err_msg)
            return None
        # success
        else:
            image_file = ImageFile(dest_image_file_path)
            image_file.assert_exists()
            return image_file

    def extract_thumbnail(
        self,
        thumbnail_file_path: str,
        overwrite: bool = True,
    ) -> ImageFile:
        """
        Extracts a thumbnail (image) from a random time between the first 30 seconds
        and 2 minutes of the video

        - If the video is shorter than two minutes it will choose a random time from
        the video to extract the thumbnail

        Parameters
        ----------
        thumbnail_file_path: str
            the absolute path to which you would like to save the extracted image
        overwrite: bool
            Overwrites 'thumbnail_file_path' if True; does not overwrite if False

        Returns
        -------
        ImageFile or None
            the thumbnail if successful; None if unsuccessful
        """
        self.assert_exists()

        # check if able to get the video duration
        video_duration = self.get_duration()
        if video_duration == -1:
            msg = (
                "Can't retrieve video duration of from video file ({}). Attempting to "
                "extract a thumbnail from the first 30 seconds to 2 minutes of the "
                "video regardless.".format(video_duration)
            )
            logging.warn(msg)
            video_duration = 120

        max_time = min(120, floor(video_duration))
        min_time = max(min(30, floor(video_duration) - 30), 0)  # -30 is arbitrary
        extract_sec = randint(min_time, max_time)

        # snapshot video
        image_file = self.extract_frame(
            extract_sec=extract_sec,
            dest_image_file_path=thumbnail_file_path,
            overwrite=overwrite,
        )

        # failure
        if image_file is None:
            msg = (
                "Extracting a thumbnail from video file '{}' to '{}' was unsuccessful."
                "".format(self.path, thumbnail_file_path)
            )
            logging.error(msg)
            return None

        # sucess
        image_file.assert_exists()
        return image_file
