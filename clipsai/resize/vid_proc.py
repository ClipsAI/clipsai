"""Utilities for video processing."""

from __future__ import annotations
# standard library imports
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid # Add this

# current package imports
from .exceptions import VideoProcessingError
from clipsai.gcloud.config import GCloudConfig # Add this
from clipsai.utils.exceptions import ConfigError # Add this
from .img_proc import rgb_to_gray

# local imports
from clipsai.media.video_file import VideoFile

# third party imports
try:  # pragma: no cover - optional dependencies
    import av
    import cv2
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback stubs
    av = None  # type: ignore
    cv2 = None  # type: ignore
    np = None  # type: ignore


def extract_frames(
    video_file: VideoFile,
    extract_secs: list[int],
    grayscale: bool = False,
    downsample_factor: float = 1,
) -> list[np.ndarray]:
    """
    Extract frames from a video as a numpy array.

    Parameters
    ----------
    video_file: VideoFile
        The video file to extract frames from.
    extract_secs: list[int]
        The seconds to extract frames from.
    grayscale: bool
        Whether to convert the frames to grayscale.

    Returns
    -------
    list[np.array]
        The extracted frames as numpy arrays
    """
    # check valid extract seconds
    duration = video_file.get_duration()
    for extract_sec in extract_secs:
        if extract_sec > duration:
            err = "Extract second ({}) exceeds video duration ({})".format(
                extract_sec, duration
            )
            logging.error(err)
            raise VideoProcessingError(err)

    # find all the frames to process
    container = av.open(video_file.path)
    stream = container.streams.video[0]

    extract_times_pts = [
        int(extract_sec / stream.time_base) for extract_sec in extract_secs
    ]
    frames_to_process = []
    for extract_pts in extract_times_pts:
        # Seek to the nearest keyframe to our desired timestamp
        container.seek(extract_pts, stream=stream)
        prev_frame = None
        for frame in container.decode(stream):
            if frame.pts > extract_pts:
                frames_to_process.append(prev_frame or frame)
                break
            prev_frame = frame
    assert len(frames_to_process) == len(extract_secs)

    # define function for parallel processing
    def process_frame(frame):
        # read frame
        img = np.array(frame.to_image())

        # downsample frame
        if downsample_factor != 1:
            height_pixels = int(img.shape[0] / downsample_factor)
            width_pixels = int(img.shape[1] / downsample_factor)
            img = cv2.resize(img, (width_pixels, height_pixels))

        # color conversion
        if grayscale:
            img = rgb_to_gray(img).reshape(img.shape[0], img.shape[1])

        return img

    # process frames in parallel
    with ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(process_frame, frames_to_process))

    return processed_frames


def detect_scenes(
    video_file: VideoFile,
    gcloud_config: GCloudConfig,
    min_scene_duration: float = 0.25, # Parameter retained for signature compatibility, not directly used by Video AI in the same way
) -> list[float]:
    """
    Detect scene changes in a video using Google Cloud Video Intelligence API.
    Uploads local files to a temporary GCS location for processing.

    Parameters
    ----------
    video_file: VideoFile
        The video file to detect scene changes in.
    gcloud_config: GCloudConfig
        Google Cloud configuration object. Must have 'project_id' and
        'temp_gcs_bucket_name' set if uploading local files.
    min_scene_duration: float
        This parameter is noted for compatibility but the API's shot detection
        will primarily govern the output.

    Returns
    -------
    scene_changes: list[float]
        The seconds where scene changes occur (end time of each shot, except the last shot's end).

    Raises
    ------
    VideoProcessingError
        If the API call fails, video processing encounters an error, or GCS operations fail.
    ConfigError
        If GCS upload is needed but 'temp_gcs_bucket_name' is not set in gcloud_config.
    google.api_core.exceptions.GoogleAPICallError
        Propagated from the Google Cloud client library for API specific errors.
    ModuleNotFoundError
        If required Google Cloud libraries are not installed.
    """
    try:
        from google.cloud import videointelligence_v1 as videointelligence
        from google.cloud import storage
        from google.api_core import exceptions as google_exceptions
    except ModuleNotFoundError as e:
        missing_module = str(e).split("'")[-2] # basic way to get module name
        raise ModuleNotFoundError(
            f"Google Cloud library '{missing_module}' is not installed. "
            "Please install it (e.g., google-cloud-videointelligence, google-cloud-storage) "
            "to use scene detection with Google Cloud."
        )

    video_file.assert_exists()
    logging.info(f"Starting scene detection for {video_file.path} using Google Video Intelligence API.")

    client_options = None
    if gcloud_config.project_id:
        # Not typically needed if GOOGLE_CLOUD_PROJECT is set, but can be explicit
        # client_options = {"api_endpoint": f"{gcloud_config.location}-videointelligence.googleapis.com"} # if location specific endpoint desired
        pass # client usually infers project from credentials or environment

    # Using client_options if you need to specify regions, otherwise it's often not needed.
    # For Video Intelligence, the client generally works globally or picks up regionality from ADC/project if applicable.
    client = videointelligence.VideoIntelligenceServiceClient(client_options=client_options)
    storage_client = None

    input_uri = None
    gcs_object_name = None
    perform_gcs_upload = not video_file.path.startswith("gs://")

    if perform_gcs_upload:
        if not gcloud_config.temp_gcs_bucket_name:
            raise ConfigError(
                "Temporary GCS bucket ('temp_gcs_bucket_name') not configured in "
                "GCloudConfig, but it's required for uploading local video files."
            )
        if not gcloud_config.project_id:
             # Storage client might need project ID if not inferable from ADC for bucket access/creation context
            logging.warning("GCS Project ID not set in GCloudConfig. Storage operations might rely on ADC.")

        storage_client = storage.Client(project=gcloud_config.project_id)
        bucket = storage_client.bucket(gcloud_config.temp_gcs_bucket_name)

        # Sanitize filename for GCS and create a unique name
        safe_filename = "".join(c if c.isalnum() or c in ['.', '_', '-'] else '_' for c in video_file.get_filename())
        gcs_object_name = f"clipsai_temp/{uuid.uuid4()}/{safe_filename}"
        input_uri = f"gs://{gcloud_config.temp_gcs_bucket_name}/{gcs_object_name}"

        logging.info(f"Uploading {video_file.path} to {input_uri}")
        try:
            blob = bucket.blob(gcs_object_name)
            blob.upload_from_filename(video_file.path, timeout=300) # 5 min upload timeout
            logging.info(f"Successfully uploaded {video_file.path} to {input_uri}")
        except google_exceptions.GoogleAPICallError as e:
            logging.error(f"GCS upload failed for {video_file.path} to {input_uri}: {e}")
            raise # Re-raise GCS API error
        except Exception as e:
            logging.error(f"GCS upload failed for {video_file.path} to {input_uri}: {e}")
            raise VideoProcessingError(f"GCS upload failed for {video_file.path} to {input_uri}: {e}")
    else:
        input_uri = video_file.path
        logging.info(f"Using existing GCS URI: {input_uri}")


    features = [videointelligence.Feature.SHOT_CHANGE_DETECTION]
    request = videointelligence.AnnotateVideoRequest(
        features=features,
        input_uri=input_uri,
        # location_id=gcloud_config.location # Optional: if processing should be regional
    )

    operation = None
    try:
        logging.debug("Sending request to Video Intelligence API.")
        # For regionalized processing, specify location_id in the request or use regional client endpoint
        operation = client.annotate_video(request=request) # Pass project_id if needed by client config
        logging.info(f"Waiting for Video Intelligence API operation for {input_uri} to complete...")
        result = operation.result(timeout=600)  # 10 minutes timeout
        logging.info(f"Video Intelligence API operation for {input_uri} completed.")

    except google_exceptions.GoogleAPICallError as e:
        logging.error(f"Video Intelligence API call failed for {input_uri}: {e}")
        raise
    except Exception as e:
        logging.error(f"Video processing failed for {input_uri} (Operation: {operation.operation.name if operation else 'N/A'}): {e}")
        raise VideoProcessingError(f"Video processing failed for {input_uri}: {e}")
    finally:
        if perform_gcs_upload and gcs_object_name and storage_client:
            logging.info(f"Deleting temporary GCS object {input_uri}")
            try:
                bucket = storage_client.bucket(gcloud_config.temp_gcs_bucket_name)
                blob = bucket.blob(gcs_object_name)
                blob.delete(timeout=60) # 1 min delete timeout
                logging.info(f"Successfully deleted temporary GCS object {input_uri}")
            except google_exceptions.GoogleAPICallError as e:
                logging.error(f"Failed to delete temporary GCS object {input_uri}: {e}")
            except Exception as e:
                logging.error(f"Error during GCS cleanup for {input_uri}: {e}")


    scene_changes = []
    if result.annotation_results and result.annotation_results[0].shot_annotations:
        shot_annotations = result.annotation_results[0].shot_annotations
        for i, shot in enumerate(shot_annotations):
            if shot.end_time_offset:
                end_time_seconds = shot.end_time_offset.total_seconds()
                # Original PySceneDetect logic did not include the end time of the very last scene.
                # We replicate this by not adding the end_time_offset of the last shot.
                if i < len(shot_annotations) - 1:
                     scene_changes.append(round(end_time_seconds, 6))
            else:
                logging.warning(f"Shot annotation for {input_uri} missing end_time_offset: {shot}")

        # Video Intelligence API returns them in order, but sorting is harmless.
        scene_changes.sort()

    logging.info(f"Detected {len(scene_changes)} scene changes in {video_file.path} (processed as {input_uri}).")
    return scene_changes
