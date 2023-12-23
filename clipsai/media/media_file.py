"""
Working with media files (i.e. image, audio, video).
"""
# standard library imports
import json
import logging
import subprocess
import uuid
import os

# current package imports
from .exceptions import NoAudioStreamError, NoVideoStreamError

# local imports
from filesys.dir import Dir
from filesys.file import File
from filesys.manager import FileSystemManager
from utils.k8s import K8S_PVC_DIR_PATH


SUCCESS = 0
FALSE = 0


class MediaFile(File):
    """
    A class for working with media files (i.e. image, audio, video).
    """

    def __init__(
        self,
        media_file_path: str,
    ) -> None:
        """
        Initialize MediaFile

        Parameters
        ----------
        media_file_path: str
            absolute path to a media file
        """
        super().__init__(media_file_path)
        self._filesys_manager = FileSystemManager()

    def get_type(self) -> str:
        """
        Returns the object type 'MediaFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'MediaFile' as a string.
        """
        return "MediaFile"

    def check_exists(self) -> str or None:
        """
        Checks that the MediaFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the MediaFile exists in the file system, a descriptive error
            message if not.
        """
        # check if it's a file
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check filetype of file
        file = File(self._path)
        valid_media_file_types = ["audio", "video", "image"]
        if file.get_mime_primary_type() not in valid_media_file_types:
            return (
                "'{}' is a valid {} but is not a valid {} since it has file type '{}' "
                "which isn't one of: '{}'.".format(
                    self._path,
                    super().get_type(),
                    self.get_type(),
                    file.get_mime_primary_type(),
                    valid_media_file_types,
                )
            )

        return None

    def get_format_info(self, format_field: str) -> str or None:
        """
        Gets format information

        Parameters
        ----------
        format_info: str
            the information about the format you want to know

        Returns
        -------
        str
            formatting information
        """
        self.assert_exists()

        # get format info
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format={}".format(format_field),
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                self._path,
            ],
            capture_output=True,
            text=True,
        )
        format_info = str(result.stdout).strip("\n\t ")

        # logging message
        msg = (
            "\n{}\n".format("-" * 40)
            + "media_file_path: '{}'\n".format(self._path)
            + "format_field: '{}'\n".format(format_field)
            + "Terminal return code: '{}'\n".format(result.returncode)
            + "Output: '{}'\n".format(result.stdout)
            + "Err Output: '{}'\n".format(result.stderr)
            + "{}\n".format("-" * 40)
        )
        # failure
        if result.returncode != SUCCESS or format_info == "":
            logging.error(msg)
            return None
        # success
        return format_info

    def get_stream_info(self, stream: str, stream_field: str) -> str or None:
        """
        Gets stream information

        Parameters
        ----------
        stream: str
            the stream you want information about ('v:0' selects the video stream,
            'a:0' selects the audio stream)
        stream_field: str
            the information about the stream you want to know ('duration' for duration
            in seconds, 'r_frame_rate' for frame rate as a precise fraction, 'width' for
            number of horizontal pixels, 'height' for number of vertical pixels,
            'pix_fmt' for pixel format, 'bit_rate' for bit rate)

        Returns
        -------
        str
            stream information
        """
        self.assert_exists()

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-select_streams",
                stream,
                "-show_entries",
                "stream={}".format(stream_field),
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                self._path,
            ],
            capture_output=True,
            text=True,
        )
        stream_info = str(result.stdout).strip("\n\t ")

        # logging message
        msg = (
            "\n{}\n".format("-" * 40)
            + "media_file_path: '{}'\n".format(self._path)
            + "stream: '{}'\n".format(stream)
            + "stream_field: '{}'\n".format(stream_field)
            + "Terminal return code: '{}'\n".format(result.returncode)
            + "Output: '{}'\n".format(result.stdout)
            + "Err Output: '{}'\n".format(result.stderr)
            + "{}\n".format("-" * 40)
        )
        # failure
        if result.returncode != SUCCESS:
            logging.error(msg)
            return None
        # success
        return stream_info

    def get_path(self) -> str:
        """
        Returns the path of the media file.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The path of the media file.
        """
        self.assert_exists()

        return self._path

    def get_streams(self) -> list[dict]:
        """
        Gets the streams of the media file and their associated information

        Parameters
        ----------
        None

        Returns
        -------
        list[dict]
            list of dictionaries of stream information
        """
        self.assert_exists()

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                self._path,
            ],
            capture_output=True,
            text=True,
        )
        streams_info = json.loads(result.stdout)["streams"]

        # logging message
        msg = (
            "\n{}\n".format("-" * 40)
            + "media_file_path: '{}'\n".format(self._path)
            + "Terminal return code: '{}'\n".format(result.returncode)
            + "Output: '{}'\n".format(result.stdout)
            + "Err Output: '{}'\n".format(result.stderr)
            + "{}\n".format("-" * 40)
        )
        # failure
        if result.returncode != SUCCESS:
            logging.error(msg)
            return {}
        # succcess
        return streams_info

    def get_audio_streams(self) -> list[dict]:
        """
        Gets the audio streams of the media file and its associated information

        Parameters
        ----------
        None

        Returns
        -------
        dict
            dictionary of audio stream information
        """
        self.assert_exists()

        streams = self.get_streams()
        audio_streams = []
        for stream in streams:
            if stream["codec_type"] == "audio":
                audio_streams.append(stream)

        return audio_streams

    def get_video_streams(self) -> list[dict]:
        """
        Gets the video streams of the media file and its associated information

        Parameters
        ----------
        None

        Returns
        -------
        dict
            dictionary of video stream information
        """
        self.assert_exists()

        streams = self.get_streams()
        video_streams = []
        for stream in streams:
            if stream["codec_type"] == "video":
                video_streams.append(stream)

        return video_streams

    def check_has_audio_stream(self) -> str or None:
        """
        Checks that the media file has an audio stream. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the media file has an audio stream, a descriptive error
            message if not.
        """
        self.assert_exists()

        if len(self.get_audio_streams()) == 0:
            return "{} '{}' does not have an audio stream.".format(
                self.get_type(), self._path
            )

        return None

    def has_audio_stream(self) -> bool:
        """
        Returns True if media file has an audio stream, False otherwise

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if media file has an audio stream, False otherwise
        """
        return self.check_has_audio_stream() is None

    def assert_has_audio_stream(self) -> None:
        """
        Raises an error if media file does not have an audio stream

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        NoAudioStreamError: media file does not have an audio stream
        """
        err = self.check_has_audio_stream()
        if err is not None:
            logging.error(err)
            raise NoAudioStreamError(err)

    def has_video_stream(self) -> bool:
        """
        Returns True if media file has a video stream, False otherwise

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if media file has a video stream, False otherwise
        """
        self.assert_exists()

        video_streams = self.get_video_streams()
        for stream in video_streams:
            # check stream isn't an attached picture
            if stream["disposition"]["attached_pic"] != FALSE:
                continue
            return True
            # # check stream has a meaningful duration
            # if "duration" in stream.keys() and float(stream["duration"]) > 0.1:
            #     return True

        return False

    def check_has_video_stream(self) -> str or None:
        """
        Checks that the media file has a video stream. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the media file has a video stream, a descriptive error
            message if not.
        """
        if self.has_video_stream() is False:
            return "{} '{}' does not have a video stream.".format(
                self.get_type(), self._path
            )

    def assert_has_video_stream(self) -> None:
        """
        Raises an error if media file does not have a video stream

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        NoVideoStreamError: media file does not have a video stream
        """
        err = self.check_has_video_stream()
        if err is not None:
            logging.error(err)
            raise NoVideoStreamError(err)

    def is_audio_only(self) -> bool:
        """
        Returns True if media file has audio stream and no video stream, False otherwise

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if media file has audio stream and no video stream, False otherwise
        """
        self.assert_exists()
        return self.has_audio_stream() and not self.has_video_stream()

    def is_video_only(self) -> bool:
        """
        Returns True if media file has video stream and no audio stream, False otherwise

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if media file has video stream and no audio stream, False otherwise
        """
        self.assert_exists()
        return self.has_video_stream() and not self.has_audio_stream()

    def get_temp_frames_folder(
        start_time: float, video_path: str, min_frames: int, fps: int
    ):
        """
        Gets the temp frames folder which holds the desired min frames of the asset.

        Parameters
        ----------
        start_time: float
            the start time of the clip
        video_path: str
            the path to the video file
        min_frames: int
            the minimum number of frames to extract
        fps: int
            the frames per second of the video

        Returns
        -------
        str
            the path to the temp frames folder
        """
        # create the temp frames folder
        frames_dir = Dir(os.path.join(K8S_PVC_DIR_PATH, str(uuid.uuid4())))
        if frames_dir.exists():
            frames_dir.delete_contents()
        else:
            frames_dir.create()

        # Extract frames from video that we're interested in
        result = subprocess.run(
            [
                "ffmpeg",
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-vf",
                "fps={}".format(fps),
                "-frames:v",
                str(min_frames),
                "-pix_fmt",
                "yuv420p",
                "{}frame_%04d.jpg".format(frames_dir),
            ],
            check=True,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            logging.error(
                "FFmpeg failed for {} error: {}".format(video_path, result.stderr)
            )
            return None

        return frames_dir
