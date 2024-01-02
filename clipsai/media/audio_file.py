"""
Working with audio files.

Notes
-----
- AudioFiles are defined to be files that contain only audio and no other media.
"""
# standard library imports
from __future__ import annotations
import os
import logging
import subprocess

# current package imports
from .temporal_media_file import TemporalMediaFile

# local package imports
from ..filesys.file import File

SUCCESS = 0


class AudioFile(TemporalMediaFile):
    """
    A class for working with audio files.
    """

    def __init__(self, audio_file_path: str) -> None:
        """
        Initialize AudioFile

        Parameters
        ----------
        audio_file_path: str
            absolute path to an audio file

        Returns
        -------
        None
        """
        super().__init__(audio_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'AudioFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'AudioFile' as a string.
        """
        return "AudioFile"

    def check_exists(self) -> str or None:
        """
        Checks that the AudioFile still exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the AudioFile still exists in the file system, a descriptive error
            message if not
        """
        # check if it's a temporal media file
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if it's an audio file
        temporal_media_file = TemporalMediaFile(self._path)
        if temporal_media_file.has_audio_stream() is False:
            return (
                "'{}' is a valid {} but has no audio stream so it is not a valid {}."
                "".format(self._path, super().get_type(), self.get_type())
            )
        if temporal_media_file.is_audio_only() is False:
            return (
                "'{}' is a valid {} but is not audio only so it is not a valid {}. Use "
                "'AudioVideoFile' class for files containing both audio and video."
                "".format(self._path, super().get_type(), self.get_type())
            )

        return None

    def get_bitrate(self) -> int or None:
        """
        Gets the bitrate of the audio stream in the audio file.

        Parameters
        ----------
        None

        Returns
        -------
        int or None
            bitrate of the audio stream in the audio file
        """
        return int(self.get_stream_info("a:0", "bit_rate"))

    def extract_audio(
        self,
        extracted_audio_file_path: str,
        audio_codec: str,
        overwrite: bool = True,
    ) -> AudioFile or None:
        """
        Extracts the audio from a media file containing audio.

        - Specify the audio file format with the chosen extension in 'audio_file_path'

        Parameters
        ----------
        extracted_audio_file_path: str
            absolute path to store the audio file
        audio_codec: str
            audio codec to use for the extracted audio file
        overwrite: bool
            Overwrites 'audio_file_path' if True; does not overwrite if False

        Returns
        -------
        AudioFile
            the audio file as an AudioFile object if successful; None if unsuccessful
        """
        self.assert_exists()
        if overwrite is True:
            self._filesys_manager.assert_parent_dir_exists(
                File(extracted_audio_file_path)
            )
        else:
            self._filesys_manager.assert_valid_path_for_new_fs_object(
                extracted_audio_file_path
            )
        self._filesys_manager.assert_paths_not_equal(
            self.path,
            extracted_audio_file_path,
            "audio_file path",
            "extracted_audio_file_path",
        )

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                self.path,
                "-c:a",
                audio_codec,
                "-vn",  # disable video
                "-q:a",  # highest quality available
                "0",
                "-map",
                "a",
                extracted_audio_file_path,
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
            logging.error(msg)
            return None
        # success
        else:
            audio_file = AudioFile(extracted_audio_file_path)
            audio_file.assert_exists()
            return audio_file

    def convert_to_wav_path(self) -> str:
        """
        Converts an audio file path to a WAV file path.

        Returns
        -------
        str
            The modified path with a '.wav' extension.
        """
        # Split the path into root and extension
        root, _ = os.path.splitext(self.path)
        # Append the .wav extension to the root
        wav_path = root + ".wav"
        return wav_path
