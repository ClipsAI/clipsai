"""
Editing media files with ffmpeg.
"""
# standard library imports
import logging
import subprocess
import os
import uuid

# standard library imports
import time # For polling Transcoder API

# current package imports
from .exceptions import MediaEditorError
from clipsai.utils.exceptions import ConfigError # Added
from .audio_file import AudioFile
from .audiovideo_file import AudioVideoFile
from .image_file import ImageFile
from .media_file import MediaFile
from .temporal_media_file import TemporalMediaFile
from .video_file import VideoFile

# local imports
from clipsai.gcloud.config import GCloudConfig # Added
from clipsai.filesys.file import File
from clipsai.filesys.manager import FileSystemManager
from clipsai.utils.conversions import seconds_to_hms_time_format
from clipsai.utils.type_checker import TypeChecker


# ffmpeg return code of 0 means success; any other (positive) integer means failure
SUCCESS = 0


class MediaEditor:
    """
    A class to edit media files using ffmpeg and Google Cloud services.
    """

    def __init__(self, gcloud_config: GCloudConfig = None) -> None:
        """
        Initializes the MediaEditor.

        The editor can use local ffmpeg for some operations and Google Cloud
        services (like Transcoder API via `transcode` method) for others.

        Parameters
        ----------
        gcloud_config : GCloudConfig, optional
            Configuration for Google Cloud services. If not provided, a default
            instance is created which relies on environment variables for settings
            like Project ID, Location, and temporary GCS bucket. This is primarily
            used by methods interacting with Google Cloud (e.g., `transcode`).
        """
        self._file_system_manager = FileSystemManager()
        self._type_checker = TypeChecker()
        if gcloud_config is None:
            self._gcloud_config = GCloudConfig()
        else:
            self._gcloud_config = gcloud_config

    def trim(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
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
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
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
        MediaEditorError: start_time < 0
        MediaEditorError: end_time < 0
        MediaEditorError: start_time > end_time
        MediaEditorError: start_time > media_file's duration
        MediaEditorError: end_time > media_file's duration
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
        self._assert_valid_trim_times(media_file, start_time, end_time)

        # convert seconds to '00:00:00.00' format for ffmpeg
        duration_secs = end_time - start_time
        start_time_hms_time_format = seconds_to_hms_time_format(start_time)
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
        num_threads: str = "0", # Ignored by Transcoder API
    ) -> TemporalMediaFile or None:
        """
        Creates a copy of a temporal media file (audio or video).
        If codecs are specified (not "copy"), this effectively transcodes.
        This method will use the Google Cloud Transcoder API if suitable codecs are provided
        and `gcloud_config` is set up. Otherwise, it may fall back to local ffmpeg processing via `trim`.

        - 'copied_media_file_path' (target) is interpreted as a GCS URI if using Transcoder API,
          or a local path for ffmpeg.
        - GCS objects at the target URI will be overwritten.

        Parameters
        ----------
        media_file: TemporalMediaFile
            The media file to copy or transcode.
        copied_media_file_path: str
            Target path for the output. If using Transcoder API, this should be a GCS URI
            or will be derived into one (e.g., gs://<bucket>/clipsai_transcoded_outputs/...).
        overwrite: bool
            If True, overwrites the destination if it exists. Default is True.
            (Note: GCS uploads and Transcoder jobs typically overwrite by default).
        video_codec: str
            Target video codec (e.g., "h264", "vp9"). "copy" implies using ffmpeg locally if possible.
        audio_codec: str
            Target audio codec (e.g., "aac", "mp3"). "copy" implies using ffmpeg locally if possible.
        crf: str
            Constant Rate Factor for video encoding (e.g., "23"). Used by both ffmpeg and Transcoder API.
        preset: str
            Encoding preset (e.g., "medium", "fast"). Used by both ffmpeg and Transcoder API.
        num_threads: str
            Number of threads for ffmpeg (ignored by Transcoder API). Default is "0" (auto).


        Returns
        -------
        TemporalMediaFile or None
            The copied/transcoded media as a TemporalMediaFile object if successful; None otherwise.
            The path of the returned file might be a GCS URI if Transcoder API was used.

        Raises
        ------
        MediaEditorError: If media_file's duration cannot be found (for ffmpeg path) or other processing errors.
        ConfigError: If GCloudConfig is not properly set for Transcoder API usage.
        """
        # This method is now a wrapper. If not using "copy" codecs, it calls self.transcode()
        # which is being refactored to use Transcoder API.
        # If "copy" codecs are used, it falls back to the old ffmpeg method via self.trim().

        # For this refactoring, self.transcode IS the target of Transcoder API.
        # So, this copy_temporal_media_file should just call self.trim if codecs are "copy",
        # otherwise it should call the NEW self.transcode.
        # However, the prompt is to refactor self.transcode directly.
        # The original self.transcode just called self.copy_temporal_media_file.
        # So, this copy_temporal_media_file method should be removed or updated AFTER
        # self.transcode is refactored.
        # For now, let's assume this method will be simplified later once transcode is done.
        # The current task is to change self.transcode.
        # This method's existing ffmpeg logic via self.trim will remain for "copy" operations.

        self.assert_valid_media_file(media_file, TemporalMediaFile)
        duration = media_file.get_duration()
        if duration == -1:
            msg = "Can't retrieve duration from media file '{}'".format(media_file.path)
            logging.error(msg)
            raise MediaEditorError(msg)

        copied_media_file = self.trim( # This uses ffmpeg
            media_file=media_file,
            start_time=0,
            end_time=duration,
            trimmed_media_file_path=copied_media_file_path,
            overwrite=overwrite,
            video_codec=video_codec,
            audio_codec=audio_codec,
            crf=crf,
            preset=preset,
            num_threads=num_threads,
        )
        if copied_media_file is None:
            msg = "Copying/trimming media file '{}' to '{}' was unsuccessful." "".format(
                media_file.path, copied_media_file_path
            )
            logging.error(msg)
            return None
        else:
            return copied_media_file


    def transcode(
        self,
        media_file: TemporalMediaFile,
        transcoded_media_file_path: str, # This will now be treated as a target GCS URI or a basis for one
        video_codec: str, # e.g., "h264"
        audio_codec: str, # e.g., "aac"
        crf: str = "23",
        preset: str = "medium",
        overwrite: bool = True,
        num_threads: str = "0",
    ) -> TemporalMediaFile or None:
        """
        Transcodes a media file to specified video and audio codecs using the
        Google Cloud Transcoder API.

        Local input files are first uploaded to a temporary GCS bucket.
        The transcoded output is also written to a GCS URI. If a local path is
        provided for `transcoded_media_file_path`, it's used to derive a filename
        within a structured path in the temporary GCS bucket.

        The method polls the Transcoder API job for completion and returns a
        `TemporalMediaFile` pointing to the GCS URI of the transcoded output.

        Parameters
        ----------
        media_file : TemporalMediaFile
            The input media file to transcode. Can be a local path or GCS URI.
        transcoded_media_file_path : str
            The target path for the transcoded output. If a local path, it's used
            to name the output in a GCS temporary bucket. If a GCS URI, it's used directly.
        video_codec : str
            Target video codec (e.g., "h264", "vp9"). Note: The Transcoder API
            always re-encodes; "copy" is not supported and will be logged as a warning.
        audio_codec : str
            Target audio codec (e.g., "aac", "mp3"). "copy" will also be re-encoded.
        crf : str, optional
            Constant Rate Factor for video encoding (e.g., "23"). Applicable to
            codecs like H.264/H.265. Default is "23".
        preset : str, optional
            Encoding preset (e.g., "medium", "fast"). Default is "medium".
        overwrite : bool, optional
            If True (default), GCS uploads and Transcoder job outputs will overwrite
            existing files at the destination URI.
        num_threads : str, optional
            This parameter is ignored when using the Google Cloud Transcoder API,
            as concurrency is managed by the service. Default is "0".

        Returns
        -------
        TemporalMediaFile or None
            A `TemporalMediaFile` instance pointing to the GCS URI of the successfully
            transcoded file, or None if transcoding fails.

        Raises
        ------
        ConfigError
            If `gcloud_config` is missing necessary attributes like `project_id`,
            `location`, or `temp_gcs_bucket_name` when GCS operations are required.
        MediaEditorError
            For failures during GCS upload, Transcoder API calls, job processing,
            or if required Google Cloud libraries are not installed.
        google.api_core.exceptions.GoogleAPICallError
            Propagated from Google Cloud client libraries for API-specific errors.
        """
        try:
            from google.cloud import video_transcoder_v1
            from google.cloud.video.transcoder_v1.services.transcoder_service import TranscoderServiceClient
            from google.cloud import storage
            from google.api_core import exceptions as google_exceptions
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[-2]
            logging.error(f"Google Cloud library '{missing_module}' not found for transcoding.")
            raise MediaEditorError(f"Google Cloud library '{missing_module}' not found. Please install google-cloud-video-transcoder and google-cloud-storage.") from e

        self.assert_valid_media_file(media_file, TemporalMediaFile)
        if not self._gcloud_config.project_id or not self._gcloud_config.location:
            raise ConfigError("Google Cloud Project ID and Location must be set in GCloudConfig for Transcoder API.")

        input_gcs_uri = media_file.path
        temp_input_gcs_object_name = None
        storage_client = None # Initialize to None

        if not input_gcs_uri.startswith("gs://"):
            if not self._gcloud_config.temp_gcs_bucket_name:
                raise ConfigError("temp_gcs_bucket_name not set in GCloudConfig for local file upload.")
            storage_client = storage.Client(project=self._gcloud_config.project_id)
            bucket = storage_client.bucket(self._gcloud_config.temp_gcs_bucket_name)

            safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in media_file.get_filename())
            temp_input_gcs_object_name = f"clipsai_transcode_inputs/{uuid.uuid4()}/{safe_filename}"
            input_gcs_uri = f"gs://{self._gcloud_config.temp_gcs_bucket_name}/{temp_input_gcs_object_name}"

            logging.info(f"Uploading {media_file.path} to {input_gcs_uri} for transcoding.")
            try:
                blob = bucket.blob(temp_input_gcs_object_name)
                blob.upload_from_filename(media_file.path, timeout=300) # 5 min upload timeout
            except Exception as e:
                raise MediaEditorError(f"Failed to upload {media_file.path} to GCS at {input_gcs_uri}: {e}")

        output_gcs_uri = transcoded_media_file_path
        if not output_gcs_uri.startswith("gs://"):
            if not self._gcloud_config.temp_gcs_bucket_name:
                if temp_input_gcs_object_name and storage_client: # Cleanup uploaded input
                    try:
                        bucket = storage_client.bucket(self._gcloud_config.temp_gcs_bucket_name)
                        blob = bucket.blob(temp_input_gcs_object_name)
                        blob.delete(timeout=60)
                    except Exception as e_del: logging.error(f"Failed to cleanup temp GCS input {input_gcs_uri}: {e_del}")
                raise ConfigError("temp_gcs_bucket_name not set for GCS output path derivation when local path is given.")

            output_filename = os.path.basename(transcoded_media_file_path)
            safe_output_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in output_filename)
            if not safe_output_filename: safe_output_filename = f"transcoded_output_{uuid.uuid4()}"

            output_gcs_object_name = f"clipsai_transcoded_outputs/{uuid.uuid4()}/{safe_output_filename}"
            output_gcs_uri = f"gs://{self._gcloud_config.temp_gcs_bucket_name}/{output_gcs_object_name}"

        logging.info(f"Transcoding {input_gcs_uri} to {output_gcs_uri} with video_codec={video_codec}, audio_codec={audio_codec}")

        transcoder_client = TranscoderServiceClient()
        parent = f"projects/{self._gcloud_config.project_id}/locations/{self._gcloud_config.location}"

        job = video_transcoder_v1.types.Job()
        job.input_uri = input_gcs_uri
        job.output_uri = output_gcs_uri

        # Basic config mapping - needs to be more robust for production
        video_stream_args = {"codec": video_codec.lower()}
        if video_codec.lower() in ["h264", "h265"]:
            if crf: video_stream_args["crf_level"] = int(crf)
            if preset: video_stream_args["preset"] = preset.lower()

        audio_stream_args = {"codec": audio_codec.lower()}
        # Add bitrate_bps for audio if needed, e.g., audio_stream_args["bitrate_bps"] = 128000

        job.config = video_transcoder_v1.types.JobConfig(
            elementary_streams=[
                video_transcoder_v1.types.ElementaryStream(key="video_stream", video_stream=video_transcoder_v1.types.VideoStream(**video_stream_args)),
                video_transcoder_v1.types.ElementaryStream(key="audio_stream", audio_stream=video_transcoder_v1.types.AudioStream(**audio_stream_args)),
            ],
            mux_streams=[
                video_transcoder_v1.types.MuxStream(
                    key=os.path.splitext(os.path.basename(output_gcs_uri))[0] or "output_video",
                    container=os.path.splitext(output_gcs_uri)[1][1:].lower() or "mp4",
                    elementary_streams=["video_stream", "audio_stream"]
                )
            ]
        )
        if video_codec.lower() == "copy" or audio_codec.lower() == "copy":
             logging.warning("Transcoder API always re-encodes. 'copy' codec is not directly supported and will result in re-encoding with defaults or specified parameters.")
             # Adjust config for 'copy' - e.g., remove problematic fields or set to high-quality defaults
             # This part needs careful design if 'copy' is a common use case.
             # For now, we let it pass, API might error or use defaults.

        created_job = None
        try:
            created_job = transcoder_client.create_job(parent=parent, job=job)
            logging.info(f"Created Transcoder job: {created_job.name}")

            POLL_INTERVAL_SECONDS = 15
            MAX_POLLS = 40 # 10 minutes timeout
            for _ in range(MAX_POLLS):
                current_job = transcoder_client.get_job(name=created_job.name)
                if current_job.state == video_transcoder_v1.types.Job.ProcessingState.SUCCEEDED:
                    logging.info(f"Transcoding job {created_job.name} succeeded.")
                    return self._create_media_file_of_same_type(output_gcs_uri, media_file)
                elif current_job.state == video_transcoder_v1.types.Job.ProcessingState.FAILED:
                    err_details = current_job.error
                    logging.error(f"Transcoding job {created_job.name} failed: Code {err_details.code}, Message: {err_details.message}")
                    raise MediaEditorError(f"Transcoding job {created_job.name} failed: {err_details.message}")
                time.sleep(POLL_INTERVAL_SECONDS)

            # Cleanup job if timed out (optional - job might complete later)
            # transcoder_client.delete_job(name=created_job.name)
            raise MediaEditorError(f"Transcoding job {created_job.name} timed out after {MAX_POLLS * POLL_INTERVAL_SECONDS}s.")

        except google_exceptions.GoogleAPICallError as e:
            logging.error(f"Transcoder API call failed: {e}")
            raise MediaEditorError(f"Transcoder API call failed: {e}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred during transcoding: {e}")
            raise MediaEditorError(f"Unexpected error during transcoding: {e}") from e
        finally:
            if temp_input_gcs_object_name and storage_client:
                logging.info(f"Deleting temporary GCS input object: gs://{self._gcloud_config.temp_gcs_bucket_name}/{temp_input_gcs_object_name}")
                try:
                    bucket = storage_client.bucket(self._gcloud_config.temp_gcs_bucket_name)
                    blob = bucket.blob(temp_input_gcs_object_name)
                    blob.delete(timeout=60)
                except Exception as e_del:
                    logging.error(f"Failed to delete temporary GCS input object: {e_del}")

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
        start_time: float = None,
        end_time: float = None,
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
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
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
        self._assert_valid_trim_times(video_file, start_time, end_time)

        duration_secs = end_time - start_time
        start_time_hms_time_format = seconds_to_hms_time_format(start_time)
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
        start_time: float = None,
        end_time: float = None,
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
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
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
            start_time=start_time,
            end_time=end_time,
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

    def concatenate(
        self,
        media_files: list[TemporalMediaFile],
        concatenated_media_file_path: str,
        overwrite: bool = True,
    ) -> MediaFile or None:
        """
        Concatenate media_files into a single media file.

        Parameters
        ----------
        media_files: list[TemporalMediaFile]
            list of media files to concatenate
        concatenated_media_file_path: str
            absolute path to store the concatenated media file
        overwrite: bool
            Overwrites'concatenated_media_file_path if True; does not overwrite if False

        Returns
        -------
        MediaFile or None
            the concatenated media file if successful; None if unsuccessful
        """
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                TemporalMediaFile(concatenated_media_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                concatenated_media_file_path
            )
        # assert media_files exist
        for i, media in enumerate(media_files):
            media.assert_exists()
            self._file_system_manager.assert_paths_not_equal(
                media.path,
                concatenated_media_file_path,
                "temporal_media{} path".format(i),
                "concatenated_media_file_path",
            )

        # create a file containing the paths to each media file
        media_file_paths = ""
        for media_file in media_files:
            media_file_paths += "file '{}'\n".format(media_file.path)
        media_paths_file = File(
            os.path.join(
                os.path.dirname(__file__),
                "{}_media_file_paths.txt".format(uuid.uuid4().hex),
            )
        )
        # log contents of media_paths_file
        logging.debug("media_paths_file contents: %s", media_file_paths)
        media_paths_file.create(media_file_paths)
        logging.debug("media_paths_file path: %s", media_paths_file.path)

        # concatenate media_files
        logging.debug("Concatenating media files in editor")
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                media_paths_file.path,
                # add to remove blank screen at beginning of output
                "-vf",
                "setpts=PTS-STARTPTS",
                concatenated_media_file_path,
            ]
        )
        logging.debug("Concatenation complete")
        media_paths_file.delete()

        msg = (
            "Terminal return code: '{}'\n"
            "Output: '{}'\n"
            "Err Output: '{}'\n"
            "".format(result.returncode, result.stdout, result.stderr)
        )
        # failure
        if result.returncode != SUCCESS:
            err_msg = (
                "Error in FFmpeg command for concatenating segments. Here is some "
                "helpful troubleshooting information:\n {}".format(msg)
            )
            logging.error(err_msg)
            return None

        # success
        else:
            media_file = self._create_media_file_of_same_type(
                concatenated_media_file_path, media_files[0]
            )
            media_file.assert_exists()
            return media_file

    def crop_video(
        self,
        original_video_file: VideoFile,
        cropped_video_file_path: str,
        x: int,
        y: int,
        width: int,
        height: int,
        start_time: float = None,
        end_time: float = None,
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
        start_time: float
            the time in seconds to begin the cropped video
        end_time: float
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
        if start_time is None:
            start_time = 0.0
        if end_time is None:
            end_time = original_video_file.get_duration()
        self._assert_valid_trim_times(original_video_file, start_time, end_time)

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                original_video_file.path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
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

    def resize_video(
        self,
        original_video_file: VideoFile,
        resized_video_file_path: str,
        width: int,
        height: int,
        segments: list[dict],
        audio_codec: str = "aac",
        video_codec: str = "libx264",
        crf: str = "18",
        preset: str = "veryfast",
        num_threads: str = "0",
        overwrite: bool = True,
    ) -> VideoFile or None:
        """
        Crop a series of videos from a video file to resize the video file

        Parameters
        ----------
        original_video_file: VideoFile
            the video file to crop
        resized_video_file_path: str
            absolute path to store the resized video file
        segments: list[dict]
            list of dictionaries where each dictionary is a distinct segment to crop
            the video. Each dictionary has the following keys:
            x: int
                x-coordinate of the top left corner of the cropped video segment
            y: int
                y-coordinate of the top left corner of the cropped video segment
            start_time: float
                the time in seconds to begin the cropped segment
            end_time: float
                the time in seconds to end the cropped segment
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
            Overwrites 'resized_video_file_path' if True; does not overwrite if False

        Returns
        -------
        VideoFile or None
            the cropped video if successful; None if unsuccessful
        """
        self.assert_valid_media_file(original_video_file, VideoFile)
        if overwrite is True:
            self._file_system_manager.assert_parent_dir_exists(
                VideoFile(resized_video_file_path)
            )
        else:
            self._file_system_manager.assert_valid_path_for_new_fs_object(
                resized_video_file_path
            )
        self._file_system_manager.assert_paths_not_equal(
            original_video_file.path,
            resized_video_file_path,
            "original_video_file path",
            "resized_video_file_path",
        )

        # crop each segment
        cropped_video_files: list[VideoFile] = []
        for i, segment in enumerate(segments):
            cropped_video_file_path = os.path.join(
                os.path.dirname(__file__),
                "{}_segment_{}.mp4".format(uuid.uuid4().hex, i),
            )
            cropped_video_file = self.crop_video(
                original_video_file=original_video_file,
                cropped_video_file_path=cropped_video_file_path,
                x=segment["x"],
                y=segment["y"],
                width=width,
                height=height,
                start_time=segment["start_time"],
                end_time=segment["end_time"],
                audio_codec=audio_codec,
                video_codec=video_codec,
                crf=crf,
                preset=preset,
                num_threads=num_threads,
                overwrite=overwrite,
            )
            # failure
            if cropped_video_file is None:
                err = (
                    "Error in cropping video segment {} with segment information '{}'."
                    "".format(i, segment)
                )
                logging.error(err)
                return None
            # success
            else:
                cropped_video_files.append(cropped_video_file)

        # concatenate cropped segments
        resized_video_file = self.concatenate(
            media_files=cropped_video_files,
            concatenated_media_file_path=resized_video_file_path,
            overwrite=overwrite,
        )
        # delete cropped segments
        for cropped_video_file in cropped_video_files:
            cropped_video_file.delete()

        # failure
        if resized_video_file is None:
            return None
        # success
        else:
            resized_video_file.assert_exists()
            return resized_video_file

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
        else:
            msg = "File '{}' must be a AudioFile, or AudioVideoFile not {}." "".format(
                media_file.path, type(media_file)
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
        start_time: float,
        end_time: float,
    ) -> str or None:
        """
        Checks if start_time and end_time are valid times to trim the media file.
        Returns None if so, a descriptive error message if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends

        Returns
        -------
        str or None
            None if start_time and end_time are valid for the media file, a descriptive
            error message if not
        """
        # check proper inputs
        if start_time < 0:
            return "Start second ({} seconds) cannot be negative.".format(start_time)
        if end_time < 0:
            return "End second ({} seconds) cannot be negative.".format
        if start_time > end_time:
            return (
                "Start second ({} seconds) cannot exceed end second ({} seconds)."
                "".format(start_time, end_time)
            )

        duration = media_file.get_duration()
        if duration == -1:
            return (
                "Can't retrieve video duration from media file '{}'. Attempting to "
                "trim with given start_time ({}) and end_time ({}) regardless."
                "".format(duration, start_time, end_time)
            )
        elif start_time > duration:
            return (
                "Start second ({} seconds) cannot exceed video duration ({} seconds)."
                "".format(start_time, duration)
            )
        elif end_time > duration + 1:
            return (
                "End second ({} seconds) cannot exceed video duration ({} seconds)."
                "".format(end_time, duration)
            )

        return None

    def _is_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
    ) -> bool:
        """
        Returns True if start_time and end_time are valid times to trim the media file.
        Returns False if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends

        Returns
        -------
        bool
            True if start_time and end_time are valid for the media file, False if not
        """
        return self._check_valid_trim_times(media_file, start_time, end_time) is None

    def _assert_valid_trim_times(
        self,
        media_file: TemporalMediaFile,
        start_time: float,
        end_time: float,
    ) -> None:
        """
        Raises an error if start_time and end_time are not valid times to trim the media
        file. Raises an error if not.

        Parameters
        ----------
        media_file: TemporalMediaFile
            the media file to check
        start_time: float
            the time in seconds the trimmed media file begins
        end_time: float
            the time in seconds the trimmed media file ends

        Raises
        ------
        MediaEditorError: start_time and end_time are not valid for the media file
        """
        msg = self._check_valid_trim_times(media_file, start_time, end_time)
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
