"""
Editing media files with ffmpeg.
"""
# standard library imports
import logging
import subprocess
# import os
# import uuid

# current package imports
from .exceptions import MediaEditorError
from .audio_file import AudioFile
from .audiovideo_file import AudioVideoFile
from .image_file import ImageFile
from .media_file import MediaFile
from .temporal_media_file import TemporalMediaFile
from .video_file import VideoFile

# local imports
# from filesys.file import File
from filesys.manager import FileSystemManager
from utils.conversions import seconds_to_hms_time_format
from utils.type_checker import TypeChecker
# from utils.k8s import K8S_PVC_DIR_PATH


# ffmpeg return code of 0 means success; any other (positive) integer means failure
SUCCESS = 0


class MediaEditor:
    """
    A class to edit media files using ffmpeg.
    """

    def __init__(self) -> None:
        """
        Initialize FfmpegEditor

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._file_system_manager = FileSystemManager()
        self._type_checker = TypeChecker()

    def trim(
        self,
        media_file: TemporalMediaFile,
        start_sec: float,
        end_sec: float,
        trimmed_media_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_width: int = None,
        crop_height: int = None,
        crop_x: int = None,
    ) -> TemporalMediaFile or None:
        """
        Trims and potentially resizes a temporal media file (audio or video) into a
        new, trimmed media file

        - trimmed_media_file_path is overwritten if already exists

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to trim
        start_sec: float
            the time in seconds the trimmed media file begins
        end_sec: float
            the time in seconds the trimmed media file ends
        trimmed_media_file_path: str
            absolute path to store the trimmed media file
        overwrite: bool
            Overwrites 'trimmed_media_file_path' if True; does not overwrite if False
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding
        crop_x: int, optional
            x-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_y: int, optional
            y-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_width: int, optional
            Width of the crop area.
            none if no resizing
        crop_height: int, optional
            Height of the crop area.
            none if no resizing

        Returns
        -------
        MediaFile or None
            the trimmed media as a MediaFile object if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: start_sec < 0
        MediaEditorError: end_sec < 0
        MediaEditorError: start_sec > end_sec
        MediaEditorError: start_sec > media_file's duration
        MediaEditorError: end_sec > media_file's duration
        """
        self.assert_valid_media_file(media_file, TemporalMediaFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                MediaFile(trimmed_media_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                trimmed_media_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            media_file.path,
            trimmed_media_file_path,
            "media_file path",
            "trimmed_media_file_path",
        )
        self._assert_valid_trim_times(media_file, start_sec, end_sec)

        # convert seconds to '00:00:00.00' format for ffmpeg
        duration_secs = end_sec - start_sec
        start_time_hms_time_format = seconds_to_hms_time_format(start_sec)
        duration_hms_time_format = seconds_to_hms_time_format(duration_secs)

        # Initialize ffmpeg command with parameters that do not depend on conditional
        # logic
        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-ss",
            start_time_hms_time_format,
            "-t",
            duration_hms_time_format,
            "-i",
            media_file.path,
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-c:a",
            audio_codec,
            "-map",
            "0",  # include all streams from input file to output file
            "-crf",
            crf,
            "-threads",
            num_threads,
        ]

        # only add the crop filter if cropping parameters are provided
        if crop_height is not None and crop_width is not None and crop_x is not None:
            logging.debug("Trim with resizing.")
            original_height = int(media_file.get_stream_info("v", "height"))
            crop_y = max(original_height // 2 - crop_height // 2, 0)
            crop_vf = "crop={width}:{height}:{x}:{y}".format(
                width=crop_width, height=crop_height, x=crop_x, y=crop_y
            )
            ffmpeg_command.extend(["-vf", crop_vf])

        ffmpeg_command.append(trimmed_media_file_path)

        logging.debug("ffmpeg_command: %s", ffmpeg_command)
        result = subprocess.run(
            ffmpeg_command,
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
                "Trimming media file '{}' to '{}' was unsuccessful. Here is some "
                "helpful troubleshooting information:\n{}"
                "".format(media_file.path, trimmed_media_file_path, msg)
            )
            logging.error(err_msg)
            return None
        # success
        else:
            trimmed_media_file = self._create_media_file_of_same_type(
                trimmed_media_file_path, media_file
            )
            trimmed_media_file.assert_exists()
            return trimmed_media_file

    def copy_temporal_media_file(
        self,
        media_file: TemporalMediaFile,
        copied_media_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
    ) -> TemporalMediaFile or None:
        """
        Creates a copy of a temporal media file (audio or video)

        - 'copied_media_file_path' is overwritten if already exists

        Parameters
        ----------
        media_file: TemporalMediaFile
            absolute path to the media file to copy
        copied_media_file_path: str
            absolute path to copy the media file to
        overwrite: bool
            Overwrites 'copied_media_file_path' if True; does not overwrite if False
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding

        Returns
        -------
        MediaFile or None
            the copied media as a MediaFile object if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: media_file's duration could not be found
        """
        self.assert_valid_media_file(media_file, TemporalMediaFile)

        duration = media_file.get_duration()
        if duration == -1:
            msg = "Can't retrieve duration from media file '{}'".format(media_file.path)
            logging.error(msg)
            raise MediaEditorError(msg)

        copied_media_file = self.trim(
            media_file,
            0,
            duration,
            copied_media_file_path,
            overwrite,
            video_codec,
            audio_codec,
            crf,
            preset,
            num_threads,
        )
        if copied_media_file is None:
            msg = "Copying media file '{}' to '{}' was unsuccessful." "".format(
                media_file.path, copied_media_file_path
            )
            logging.error(msg)
            return None
        # success
        else:
            return copied_media_file

    def transcode(
        self,
        media_file: TemporalMediaFile,
        transcoded_media_file_path: str,
        video_codec: str,
        audio_codec: str,
        crf: str = "23",
        preset: str = "medium",
        overwrite: bool = True,
        num_threads: str = "0",
    ) -> TemporalMediaFile or None:
        """
        Transcodes media file (audio or video) to the specified codecs

        - 'transcoded_media_file_path' is overwritten if already exists

        Parameters
        ----------
        media_file: TemporalMediaFile
            absolute path to the media file to transcode
        transcoded_media_file_path: str
            absolute path to store the transcoded media file
        overwrite: bool
            Overwrites 'transcoded_media_file_path' if True; does not overwrite if False
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding

        Returns
        -------
        MediaFile or None
            the transcoded media as a MediaFile object if successful; None if
            unsuccessful
        """
        return self.copy_temporal_media_file(
            media_file,
            transcoded_media_file_path,
            overwrite,
            video_codec,
            audio_codec,
            crf,
            preset,
            num_threads,
        )

    def watermark_and_crop_video(
        self,
        video_file: VideoFile,
        watermark_file: ImageFile,
        watermarked_video_file_path: str,
        size_dim: str,
        watermark_to_video_ratio_size_dim: float,
        x: str,
        y: str,
        opacity: float,
        overwrite: bool = True,
        start_sec: float = None,
        end_sec: float = None,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_x: int = None,
        crop_y: int = None,
        crop_width: int = None,
        crop_height: int = None,
    ) -> VideoFile or None:
        """
        Watermark a video

        - 'watermarked_video_file_path' is overwritten if already exists
        - https://www.bannerbear.com/blog/how-to-add-watermark-to-videos-using-ffmpeg/
        #basic-command

        Parameters
        ----------
        video_file: VideoFile
            the video file to watermark
        watermark_file: ImageFile
            the image file to watermark the video with
        watermarked_video_file_path: str
            absolute path to store the watermarked video
        size_dim: str
            the dimension (height or width) to size the watermark with respect to the
            video. Needs to be 'h' (height) or 'w' (width)
        watermark_to_video_ratio_size_dim: float
            the size ratio of the watermark to the video along the chosen size
            dimension (width or height). Needs to be greater than zero
        x: str
            x position of watermark where the origin of both the video and watermark
            are the top left corner and x increases as you move right
                main_w: width of the video
                overlay_w: width of the watermark
        y: str
            y position of watermark where the origin of both the video and watermark
            are the top left corner and y increases as you move down
                main_h: height of the video
                overlay_h: height of the watermark
        opacity: float
            opacity of the watermark on the video; must be between 0 and 1
        overwrite: bool
            Overwrites 'watermarked_video_file_path' if True; does not overwrite if
            False
        start_sec: float
            the time in seconds the trimmed media file begins
        end_sec: float
            the time in seconds the trimmed media file ends
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use
        crop_x: int, optional
            x-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_y: int, optional
            y-coordinate of the top left corner of the crop area.
            none if no resizing
        crop_width: int, optional
            Width of the crop area.
            none if no resizing
        crop_height: int, optional
            Height of the crop area.
            none if no resizing

        Positioning Examples
        --------------------
        - top left corner: x=0 y=0
        - top right corner: x=main_w-overlay_w y=0
        - bottom left corner: x=0 y=main_h-overlay_h
        - bottom right corner: x=main_w-overlay_w y=main_h-overlay_h
        - middle of video: x=(main_w-overlay_w)/2 y=(main_h-overlay_h)/2

        Returns
        -------
        VideoFile
            the watermarked and possibly resized video as a VideoFile
            object if successful; None if unsuccessful

        Raises
        ------
        MediaEditorError: size_dim is not 'h' or 'w'
        MediaEditorError: watermark_to_video_ratio_size_dim <= 0
        MediaEditorError: opacity < 0 or opacity > 1
        """
        # check file inputs are valid
        self.assert_valid_media_file(video_file, VideoFile)
        self.assert_valid_media_file(watermark_file, ImageFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                MediaFile(watermarked_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                watermarked_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            video_file.path,
            watermark_file.path,
            "video_file path",
            "watermark_file path",
        )
        self._file_system_manager.assert_paths_not_equal(
            video_file.path,
            watermarked_video_file_path,
            "video_file path",
            "watermarked_video_file_path",
        )
        self._file_system_manager.assert_paths_not_equal(
            watermark_file.path,
            watermarked_video_file_path,
            "watermark_file path",
            "watermarked_video_file_path",
        )

        # check watermark specifications are valid
        if size_dim not in ["h", "w"]:
            msg = "size_dim must be one of '{0}', not '{1}'".format(
                ["h", "w"], size_dim
            )
            logging.error(msg)
            raise MediaEditorError(msg)
        if watermark_to_video_ratio_size_dim <= 0:
            msg = (
                "watermark_to_video_ratio_size_dim must be greater than zero, not "
                "'{0}'".format(watermark_to_video_ratio_size_dim)
            )
            logging.error(msg)
            raise MediaEditorError(msg)
        if opacity < 0 or opacity > 1:
            msg = "Opacity must be between 0 and 1, not '{0}'".format(opacity)
            logging.error(msg)
            raise MediaEditorError(msg)

        # check trim specifications are valid
        self._assert_valid_trim_times(video_file, start_sec, end_sec)

        duration_secs = end_sec - start_sec
        start_time_hms_time_format = seconds_to_hms_time_format(start_sec)
        duration_hms_time_format = seconds_to_hms_time_format(duration_secs)

        resize_tried = (
            crop_x is not None
            and crop_y is not None
            and crop_width is not None
            and crop_height is not None
        )

        filter_complex_parts = []
        if resize_tried:
            filter_complex_parts.append(
                "crop={width}:{height}:{x}:{y}[cropped]".format(
                    width=crop_width, height=crop_height, x=crop_x, y=crop_y
                )
            )
            # Uses the cropped video as input for the next filter stage
            input_video_label = "[cropped]"
        else:
            # Uses the original video as input for the next filter stage
            input_video_label = "[0]"

        filter_complex_parts.append(
            "[1]format=rgba,colorchannelmixer=aa={opacity}[logo]".format(
                opacity=opacity
            )
        )
        # Size of the watermark relative to the video
        filter_complex_parts.append(
            "[logo]{input_video_label}scale2ref=oh*mdar:i{size_dim}*{watermark_ratio}"
            "[logo][video]".format(
                input_video_label=input_video_label,
                size_dim=size_dim,
                watermark_ratio=watermark_to_video_ratio_size_dim,
            )
        )

        # Placement of watermark on video
        filter_complex_parts.append("[video][logo]overlay=({x}):({y})".format(x=x, y=y))

        # Join all filter parts
        filter_complex = ";".join(filter_complex_parts).strip(";")
        # logging.debug("filter_complex: %s", filter_complex)

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                start_time_hms_time_format,
                "-t",
                duration_hms_time_format,
                "-i",
                video_file.path,
                "-i",
                watermark_file.path,
                "-filter_complex",
                filter_complex,
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-c:a",
                audio_codec,
                "-crf",
                crf,
                "-threads",
                num_threads,
                watermarked_video_file_path,
            ],
            capture_output=True,
            text=True,
        )
        msg = (
            "\n{0}\n"
            "video_file path: '{1}'\n"
            "watermark_file path: '{2}'\n"
            "watermarked_video_file_path: '{3}'\n"
            "Resizing attempted: '{4}'\n"
            "Terminal return code: '{5}'\n"
            "Output: '{6}'\n"
            "Err Output: '{7}'\n"
            "\n{0}\n"
        ).format(
            "-" * 40,
            video_file.path,
            watermark_file.path,
            watermarked_video_file_path,
            resize_tried,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        # failure
        if result.returncode != SUCCESS:
            err_msg = (
                "Watermarking video file '{0}' with image file "
                "'{1}' to '{2}' "
                "was unsuccessful. Here is some helpful troubleshooting information:\n"
            ).format(
                video_file.path, watermark_file.path, watermarked_video_file_path
            ) + msg
            logging.error(err_msg)
            return None
        # success
        logging.debug("Watermarking video file successful")
        watermarked_video_file = self._create_media_file_of_same_type(
            watermarked_video_file_path, video_file
        )
        logging.debug("Watermarked video file created")
        return watermarked_video_file

    def watermark_corner_of_video(
        self,
        video_file: VideoFile,
        watermark_file: ImageFile,
        watermarked_video_file_path: str,
        watermark_to_video_ratio_along_smaller_video_dimension: float,
        corner: str,
        opacity: float,
        overwrite: bool = True,
        start_sec: float = None,
        end_sec: float = None,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: str = "23",
        preset: str = "medium",
        num_threads: str = "0",
        crop_x: int = None,
        crop_width: int = None,
        crop_height: int = None,
    ) -> VideoFile or None:
        """
        Watermark 'video_file' with 'watermark_file' in the chosen corner such that the
        watermark:video ratio is 0.25 along the shortest video dimension (height or
        width)

        Parameters
        ----------
        video_file: VideoFile
            the video file to watermark
        watermark_file: ImageFile
            the image file to watermark the video with
        watermarked_video_file_path: str
            absolute path to store the watermarked video
        watermark_to_video_ratio_along_smaller_video_dimension: float
            the ratio of the watermark size relative to the video size along the
            smaller of video's two size dimensions (width or height)
        corner: str
            the corner you want the watermark to be in. One of: "bottom_left",
            "bottom_right", "top_left", "top_right"
        opacity: float
            opacity of the src_img_file_path watermark on the video; must be between
            zero and one
        overwrite: bool
            Overwrites 'watermarked_video_file_path' if True; does not overwrite if
            False
        start_sec: float
            the time in seconds the trimmed media file begins
        end_sec: float
            the time in seconds the trimmed media file ends
        video_codec: str
            compression and decompression software for the video (libx264)
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use
        crop_x: int, optional
            x-coordinate of the top left corner of the crop area,
            none if no resizing
        crop_width: int, optional
            width of the crop area, none if no resizing
        crop_height: int, optional
            height of the crop area, none if no resizing

        Returns
        -------
        watermarked_video: Video
            Returns the watermarked and possibly cropped Video object
        """
        self.assert_valid_media_file(video_file, VideoFile)

        original_height = int(video_file.get_stream_info("v", "height"))
        crop_y = None

        if crop_height is not None:
            logging.debug("Watermark with resizing.")
            crop_y = max(original_height // 2 - crop_height // 2, 0)

        corner_commands = {
            "bottom_left": {
                "x": "0",
                "y": "H-overlay_h" if crop_height else "main_h-overlay_h",
            },
            "bottom_right": {
                "x": "W-overlay_w" if crop_width else "main_w-overlay_w",
                "y": "H-overlay_h" if crop_height else "main_h-overlay_h",
            },
            "top_left": {
                "x": "0",
                "y": "0",
            },
            "top_right": {
                "x": "W-overlay_w" if crop_width else "main_w-overlay_w",
                "y": "0",
            },
        }

        # video height > video width
        if original_height > int(video_file.get_stream_info("v", "width")):
            size_dim = "w"
        # video height <= video width
        else:
            size_dim = "h"
        logging.debug("entering watermarking and cropping")
        return self.watermark_and_crop_video(
            video_file=video_file,
            watermark_file=watermark_file,
            watermarked_video_file_path=watermarked_video_file_path,
            size_dim=size_dim,
            watermark_to_video_ratio_size_dim=(
                watermark_to_video_ratio_along_smaller_video_dimension
            ),
            x=corner_commands[corner]["x"],
            y=corner_commands[corner]["y"],
            opacity=opacity,
            overwrite=overwrite,
            start_sec=start_sec,
            end_sec=end_sec,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            preset=preset,
            num_threads=num_threads,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
        )

    def merge_audio_and_video(
        self,
        video_file: VideoFile,
        audio_file: AudioFile,
        merged_video_file_path: str,
        overwrite: bool = True,
        video_codec: str = "copy",
        audio_codec: str = "copy",
    ) -> VideoFile or None:
        """
        Merges an audio-only file and video-only file into a single video file

        - 'dest_merged_video_file_path' is overwritten if already exists
        - MoviePy reference:
        https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_tools.py

        Parameters
        ----------
        video_file: VideoFile
            the video file to merge
        audio_file: AudioFile
            the audio file to merge
        merged_video_file_path: str
            absolute path to store the merged video file
        overwrite: bool
            Overwrites 'audio_file_path' if True; does not overwrite if False
        audio_codec: str
            compression and decompression software for the audio (aac)
        video_codec: str
            compression and decompression software for the video (libx264)

        Returns
        -------
        VideoFile or None
            the merged video as a VideoFile object if successful; None if unsuccessful
        """
        self.assert_valid_media_file(audio_file, AudioFile)
        self.assert_valid_media_file(video_file, VideoFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                MediaFile(merged_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                merged_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            video_file.path,
            merged_video_file_path,
            "video_file path",
            "merged_video_file_path",
        )
        self._file_system_manager.assert_paths_not_equal(
            audio_file.path,
            merged_video_file_path,
            "audio_file path",
            "merged_video_file_path",
        )

        max_duration_diff = 3
        duration_diff = abs(video_file.get_duration() - audio_file.get_duration())
        if duration_diff > max_duration_diff:
            msg = (
                "Audio and video files cannot be merge. Audio file '{}' and video file "
                "'{}' have a duration difference of more than {} seconds."
                "".format(audio_file.path, video_file.path, max_duration_diff)
            )
            logging.error(msg)
            raise MediaEditorError(msg)

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                video_file.path,
                "-i",
                audio_file.path,
                "-c:v",
                video_codec,
                "-c:a",
                audio_codec,
                merged_video_file_path,
            ],
            capture_output=True,
            text=True,
        )

        msg = (
            "\n{'-' * 40}\n"
            + "video_file path: '{}'\n".format(video_file.path)
            + "audio_file path: '{}'\n".format(video_file.path)
            + "merged_video_file_path: '{}'\n".format(merged_video_file_path)
            + "audio_codec: '{}'\n".format(audio_codec)
            + "video_codec: '{}'\n".format(video_codec)
            + "Terminal return code: '{}'\n".format(result.returncode)
            + "Output: '{}'\n".format(result.stdout)
            + "Err Output: '{}'\n".format(result.stderr)
            + "\n{'-' * 40}\n"
        )
        # failure
        if result.returncode != SUCCESS:
            err_msg = (
                "Merging video file '{}' and audio file '{}' was unsuccessful. Here is "
                "some helpful troubleshooting information:\n"
                "".format(video_file.path, audio_file.path)
            ) + msg
            logging.error(err_msg)
            return None
        # success
        else:
            logging.debug(msg)
            merged_video_file = VideoFile(merged_video_file_path)
            return merged_video_file

    # def concatenate(
    #     self,
    #     media_files: list[TemporalMediaFile],
    #     concatenated_media_file_path: str,
    #     overwrite: bool = True,
    # ) -> MediaFile or None:
    #     """
    #     Concatenate media_files into a single media file.

    #     Parameters
    #     ----------
    #     media_files: list[TemporalMediaFile]
    #         list of media files to concatenate
    #     concatenated_media_file_path: str
    #         absolute path to store the concatenated media file
    #     overwrite: bool
    #         Overwrites'concatenated_media_file_path if True; does not overwrite if False

    #     Returns
    #     -------
    #     MediaFile or None
    #         the concatenated media file if successful; None if unsuccessful
    #     """
    #     if overwrite is True:
    #         self._file_system_manager.assert_parent_dir_exists(
    #             TemporalMediaFile(concatenated_media_file_path)
    #         )
    #     else:
    #         self._file_system_manager.assert_valid_path_for_new_fs_object(
    #             concatenated_media_file_path
    #         )
    #     # assert media_files exist
    #     for i, media in enumerate(media_files):
    #         media.assert_exists()
    #         self._file_system_manager.assert_paths_not_equal(
    #             media.path,
    #             concatenated_media_file_path,
    #             "temporal_media{} path".format(i),
    #             "concatenated_media_file_path",
    #         )

    #     # create a file containing the paths to each media file
    #     media_file_paths = ""
    #     for media_file in media_files:
    #         media_file_paths += "file '{}'\n".format(media_file.path)
    #     media_paths_file = File(
    #         os.path.join(
    #             K8S_PVC_DIR_PATH, "{}_media_file_paths.txt".format(uuid.uuid4().hex)
    #         )
    #     )
    #     # log contents of media_paths_file
    #     logging.debug("media_paths_file contents: %s", media_file_paths)
    #     media_paths_file.create(media_file_paths)
    #     logging.debug("media_paths_file path: %s", media_paths_file.path)

    #     # concatenate media_files
    #     logging.debug("Concatenating media files in editor")
    #     result = subprocess.run(
    #         [
    #             "ffmpeg",
    #             "-y",
    #             "-f",
    #             "concat",
    #             "-safe",
    #             "0",
    #             "-i",
    #             media_paths_file.path,
    #             # add to remove blank screen at beginning of output
    #             "-vf",
    #             "setpts=PTS-STARTPTS",
    #             concatenated_media_file_path,
    #         ]
    #     )
    #     logging.debug("Concatenation complete")
    #     media_paths_file.delete()

    #     msg = (
    #         "Terminal return code: '{}'\n"
    #         "Output: '{}'\n"
    #         "Err Output: '{}'\n"
    #         "".format(result.returncode, result.stdout, result.stderr)
    #     )
    #     # failure
    #     if result.returncode != SUCCESS:
    #         err_msg = (
    #             "Error in FFmpeg command for concatenating segments. Here is some "
    #             "helpful troubleshooting information:\n {}".format(msg)
    #         )
    #         logging.error(err_msg)
    #         return None

    #     # success
    #     else:
    #         media_file = self._create_media_file_of_same_type(
    #             concatenated_media_file_path, media_files[0]
    #         )
    #         media_file.assert_exists()
    #         return media_file

    def crop_video(
        self,
        original_video_file: VideoFile,
        cropped_video_file_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        start_sec: float = None,
        end_sec: float = None,
        audio_codec: str = "aac",
        video_codec: str = "libx264",
        crf: str = "18",
        preset: str = "veryfast",
        num_threads: str = "0",
        overwrite: bool = True,
    ) -> VideoFile or None:
        """
        Crop a video.

        Parameters
        ----------
        original_video_file: VideoFile
            the video file to crop
        cropped_video_file_path: str
            absolute path to store the cropped video file
        x: int
            x-coordinate of the top left corner of the cropped video
        y: int
            y-coordinate of the top left corner of the cropped video
        width: int
            width of the cropped video
        height: int
            height of the cropped video
        start_sec: float
            the time in seconds to begin the cropped video
        end_sec: float
            the time in seconds to end the cropped video
        audio_codec: str
            compression and decompression sfotware for the audio (aac)
        video_codec: str
            compression and decompression software for the video (libx264)
        crf: str
            constant rate factor - an encoding mode that adjusts the file data rate up
            or down to achieve a selected quality level rather than a specific data
            rate. CRF values range from 0 to 51, with lower numbers delivering higher
            quality scores
        preset: str
            the encoding speed to compression ratio. A slower preset will provide
            better compression (compression is quality per filesize)
        num_threads: str
            the number of threads to use for encoding
        overwrite: bool
            Overwrites 'cropped_video_file_path' if True; does not overwrite if False

        Returns
        -------
        VideoFile or None
            the cropped video if successful; None if unsuccessful
        """
        # check file inputs are valid
        self.assert_valid_media_file(original_video_file, VideoFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                VideoFile(cropped_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                cropped_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            original_video_file.path,
            cropped_video_file_path,
            "original_video_file path",
            "cropped_video_file_path",
        )

        # set valid start and end times
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = original_video_file.get_duration()
        self._assert_valid_trim_times(original_video_file, start_sec, end_sec)

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                original_video_file.path,
                "-ss",
                str(start_sec),
                "-to",
                str(end_sec),
                "-vf",
                "crop={}:{}:{}:{}".format(width, height, x, y),
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-c:a",
                audio_codec,
                "-map",
                "0",  # include all streams from input file to output file
                "-crf",
                crf,
                "-threads",
                num_threads,
                cropped_video_file_path,
            ],
            capture_output=True,
            text=True,
        )

        msg = (
            "Terminal return code: '{}'\n".format(result.returncode)
            + "Output: '{}'\n".format(result.stdout)
            + "Err Output: '{}'\n".format(result.stderr)
        )
        # failure
        if result.returncode != SUCCESS:
            err = (
                "Cropping video file '{}' to '{}' was unsuccessful. Here is some "
                "helpful troubleshooting information: {}"
                "".format(original_video_file.path, cropped_video_file_path, msg)
            )
            logging.error(err)
            return None
        # success
        else:
            cropped_video_file = self._create_media_file_of_same_type(
                cropped_video_file_path, original_video_file
            )
            # cropped_video_file.assert_exists()
            return cropped_video_file

    # def resize_video(
    #     self,
    #     original_video_file: VideoFile,
    #     resized_video_file_path: str,
    #     width: int,
    #     height: int,
    #     segments: list[dict],
    #     audio_codec: str = "aac",
    #     video_codec: str = "libx264",
    #     crf: str = "18",
    #     preset: str = "veryfast",
    #     num_threads: str = "0",
    #     overwrite: bool = True,
    # ) -> VideoFile or None:
    #     """
    #     Crop a series of videos from a video file to resize the video file

    #     Parameters
    #     ----------
    #     original_video_file: VideoFile
    #         the video file to crop
    #     resized_video_file_path: str
    #         absolute path to store the resized video file
    #     segments: list[dict]
    #         list of dictionaries where each dictionary is a distinct segment to crop
    #         the video. Each dictionary has the following keys:
    #         x: int
    #             x-coordinate of the top left corner of the cropped video segment
    #         y: int
    #             y-coordinate of the top left corner of the cropped video segment
    #         startTime: float
    #             the time in seconds to begin the cropped segment
    #         endTime: float
    #             the time in seconds to end the cropped segment
    #     audio_codec: str
    #         compression and decompression sfotware for the audio (aac)
    #     video_codec: str
    #         compression and decompression software for the video (libx264)
    #     crf: str
    #         constant rate factor - an encoding mode that adjusts the file data rate up
    #         or down to achieve a selected quality level rather than a specific data
    #         rate. CRF values range from 0 to 51, with lower numbers delivering higher
    #         quality scores
    #     preset: str
    #         the encoding speed to compression ratio. A slower preset will provide
    #         better compression (compression is quality per filesize)
    #     num_threads: str
    #         the number of threads to use for encoding
    #     overwrite: bool
    #         Overwrites 'resized_video_file_path' if True; does not overwrite if False

    #     Returns
    #     -------
    #     VideoFile or None
    #         the cropped video if successful; None if unsuccessful
    #     """
    #     self.assert_valid_media_file(original_video_file, VideoFile)
    #     if overwrite is True:
    #         self._file_system_manager.assert_parent_dir_exists(
    #             VideoFile(resized_video_file_path)
    #         )
    #     else:
    #         self._file_system_manager.assert_valid_path_for_new_fs_object(
    #             resized_video_file_path
    #         )
    #     self._file_system_manager.assert_paths_not_equal(
    #         original_video_file.path,
    #         resized_video_file_path,
    #         "original_video_file path",
    #         "resized_video_file_path",
    #     )

    #     # crop each segment
    #     cropped_video_files: list[VideoFile] = []
    #     for i, segment in enumerate(segments):
    #         cropped_video_file_path = os.path.join(
    #             K8S_PVC_DIR_PATH, "{}_segment_{}.mp4".format(uuid.uuid4().hex, i)
    #         )
    #         cropped_video_file = self.crop_video(
    #             original_video_file=original_video_file,
    #             cropped_video_file_path=cropped_video_file_path,
    #             x=segment["x"],
    #             y=segment["y"],
    #             width=width,
    #             height=height,
    #             start_sec=segment["startTime"],
    #             end_sec=segment["endTime"],
    #             audio_codec=audio_codec,
    #             video_codec=video_codec,
    #             crf=crf,
    #             preset=preset,
    #             num_threads=num_threads,
    #             overwrite=overwrite,
    #         )
    #         # failure
    #         if cropped_video_file is None:
    #             err = (
    #                 "Error in cropping video segment {} with segment information '{}'."
    #                 "".format(i, segment)
    #             )
    #             logging.error(err)
    #             return None
    #         # success
    #         else:
    #             cropped_video_files.append(cropped_video_file)

    #     # concatenate cropped segments
    #     resized_video_file = self.concatenate(
    #         media_files=cropped_video_files,
    #         concatenated_media_file_path=resized_video_file_path,
    #         overwrite=overwrite,
    #     )
    #     # delete cropped segments
    #     for cropped_video_file in cropped_video_files:
    #         cropped_video_file.delete()

    #     # failure
    #     if resized_video_file is None:
    #         return None
    #     # success
    #     else:
    #         resized_video_file.assert_exists()
    #         return resized_video_file

    def instantiate_as_temporal_media_file(
        self, media_file_path: str
    ) -> TemporalMediaFile:
        """
        Returns the media file as the correct type (e.g. VideoFile, AudioFile, etc.)

        Parameters
        ----------
        media_file_path: str
            Absolute path to the media file to instantiate

        Returns
        -------
        MediaFile
            the media file as the correct type (e.g. VideoFile, AudioFile, etc.)
        """
        media_file = TemporalMediaFile(media_file_path)
        media_file.assert_exists()

        if media_file.has_audio_stream() and media_file.has_video_stream():
            media_file = AudioVideoFile(media_file.path)
        elif media_file.has_audio_stream():
            media_file = AudioFile(media_file.path)
        elif media_file.has_video_stream():
            media_file = VideoFile(media_file.path)
        else:
            msg = (
                "File '{}' must be a VideoFile, AudioFile, or AudioVideoFile not {}."
                "".format(media_file.path, type(media_file))
            )
            logging.error(msg)
            raise MediaEditorError(msg)

        media_file.assert_exists()
        return media_file

    def check_valid_media_file(
        self, media_file: MediaFile, media_file_type
    ) -> str or None:
        """
        Checks if media_file is of the proper type and exists in the file system.
        Returns None if so, a descriptive error message if not.

        Parameters
        ----------
        media_file: MediaFile
            the media file to check
        media_file_type
            the type of media file to check for (e.g. VideoFile, AudioFile, ImageFile)

        Returns
        -------
        str or None
            None if media_file is of the proper type and exists in the file system, a
            descriptive error message if not
        """
        msg = self._type_checker.check_type(media_file, "media_file", media_file_type)
        if msg is not None:
            return msg

        msg = media_file.check_exists()
        if msg is not None:
            return msg

        return None

    def is_valid_media_file(self, media_file: MediaFile, media_file_type) -> bool:
        """
        Returns True if media_file is of the proper type and exists in the file system.
        Returns False if not.

        Parameters
        ----------
        media_file: MediaFile
            the media file to check
        media_file_type
            the type of media file to check for (e.g. VideoFile, AudioFile, ImageFile)

        Returns
        -------
        bool
            True if media_file is of the proper type and exists in the file system,
            False if not
        """
        return self.check_valid_media_file(media_file, media_file_type) is None

    def assert_valid_media_file(self, media_file: MediaFile, media_file_type) -> None:
        """
        Raises an error media_file is of the proper type and exists in the file system.
        Raises an error if not.

        Parameters
        ----------
        media_file: MediaFile
            the media file to check
        media_file_type
            the type of media file to check for (e.g. VideoFile, AudioFile, ImageFile)

        Raises
        ------
        MediaEditorError: media_file is not of the proper type or does not exist in the
            file system
        """
        msg = self.check_valid_media_file(media_file, media_file_type)
        if msg is not None:
            raise MediaEditorError(msg)

    def _check_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_sec: float,
        end_sec: float,
    ) -> str or None:
        """
        Checks if start_sec and end_sec are valid times to trim the media file. Returns
        None if so, a descriptive error message if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_sec: float
            the time in seconds the trimmed media file begins
        end_sec: float
            the time in seconds the trimmed media file ends

        Returns
        -------
        str or None
            None if start_sec and end_sec are valid for the media file, a descriptive
            error message if not
        """
        # check proper inputs
        if start_sec < 0:
            return "Start second ({} seconds) cannot be negative.".format(start_sec)
        if end_sec < 0:
            return "End second ({} seconds) cannot be negative.".format
        if start_sec > end_sec:
            return (
                "Start second ({} seconds) cannot exceed end second ({} seconds)."
                "".format(start_sec, end_sec)
            )

        duration = media_file.get_duration()
        if duration == -1:
            return (
                "Can't retrieve video duration from media file '{}'. Attempting to "
                "trim with given start_sec ({}) and end_sec ({}) regardless."
                "".format(duration, start_sec, end_sec)
            )
        elif start_sec > duration:
            return (
                "Start second ({} seconds) cannot exceed video duration ({} seconds)."
                "".format(start_sec, duration)
            )
        elif end_sec > duration + 1:
            return (
                "End second ({} seconds) cannot exceed video duration ({} seconds)."
                "".format(end_sec, duration)
            )

        return None

    def _is_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_sec: float,
        end_sec: float,
    ) -> bool:
        """
        Returns True if start_sec and end_sec are valid times to trim the media file.
        Returns False if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_sec: float
            the time in seconds the trimmed media file begins
        end_sec: float
            the time in seconds the trimmed media file ends

        Returns
        -------
        bool
            True if start_sec and end_sec are valid for the media file, False if not
        """
        return self._check_valid_trim_times(media_file, start_sec, end_sec) is None

    def _assert_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_sec: float,
        end_sec: float,
    ) -> None:
        """
        Raises an error if start_sec and end_sec are not valid times to trim the media
        file. Raises an error if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_sec: float
            the time in seconds the trimmed media file begins
        end_sec: float
            the time in seconds the trimmed media file ends

        Raises
        ------
        MediaEditorError: start_sec and end_sec are not valid for the media file
        """
        msg = self._check_valid_trim_times(media_file, start_sec, end_sec)
        if msg is not None:
            raise MediaEditorError(msg)

    def _create_media_file_of_same_type(
        self,
        file_path_to_create_media_file_from: str,
        media_file_to_copy_type_of: MediaFile,
    ) -> MediaFile:
        """
        Creates a MediaFile object with the same type as 'media_file_to_copy_type_of'
        from the file at 'file_path_to_create_media_file_from'

        Parameters
        ----------
        file_path_to_create_media_file_from: str
            absolute path to the file to create a MediaFile object from
        media_file_to_copy_type_of: MediaFile
            the media file to copy the type of

        Returns
        -------
        MediaFile
            the media file at 'file_path_to_create_media_file_from' as an MediaFile
            object of the same type as 'media_file_to_copy_type_of''
        """
        self._type_checker.assert_type(
            media_file_to_copy_type_of, "media_file_to_copy_type_of", MediaFile
        )

        if type(media_file_to_copy_type_of) is ImageFile:
            created_file = ImageFile(file_path_to_create_media_file_from)
        elif type(media_file_to_copy_type_of) is AudioFile:
            created_file = AudioFile(file_path_to_create_media_file_from)
        elif type(media_file_to_copy_type_of) is VideoFile:
            created_file = VideoFile(file_path_to_create_media_file_from)
        elif type(media_file_to_copy_type_of) is AudioVideoFile:
            created_file = AudioVideoFile(file_path_to_create_media_file_from)

        else:
            msg = (
                "media_file_to_copy_type_of '{}' must be a VideoFile, AudioFile, or "
                "ImageFile, not {}.".format(
                    media_file_to_copy_type_of.path, type(media_file_to_copy_type_of)
                )
            )
            logging.error(msg)
            raise MediaEditorError(msg)

        return created_file
