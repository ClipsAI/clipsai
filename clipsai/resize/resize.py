"""
Resize an asset's media to a 9:16 aspect ratio.
"""
# standard library imports
import logging

# current package imports
from .crops import Crops
from .resizer import Resizer
from .vid_proc import detect_scenes

# local package imports
# from clipsai.diarize.pyannote import PyannoteDiarizer # Removed import
from clipsai.media.audiovideo_file import AudioVideoFile
from clipsai.gcloud.config import GCloudConfig
from clipsai.transcribe.transcriber import Transcriber # New import
from clipsai.transcribe.transcription import Transcription # New import

def resize(
    video_file_path: str,
    # pyannote_auth_token: str, # REMOVED
    gcloud_config: GCloudConfig = None,
    transcriber_default_language_code: str = "en-US", # ADDED
    aspect_ratio: tuple[int, int] = (9, 16),
    min_segment_duration: float = 1.5,
    samples_per_segment: int = 13,
    face_detect_width: int = 960,
    # face_detect_margin: int = 20, # REMOVED from Resizer.__init__
    # face_detect_post_process: bool = False, # REMOVED from Resizer.__init__
    n_face_detect_batches: int = 8,
    min_scene_duration: float = 0.25,
    scene_merge_threshold: float = 0.25,
    time_precision: int = 6,
    device: str = None, # Still used by Resizer for face detection models
) -> Crops:
    """
    Resizes a video to a specified aspect ratio (default 9:16).
    This process involves:
    1. Transcription and Speaker Diarization using Google Cloud Speech-to-Text.
    2. Scene Detection using Google Cloud Video Intelligence API.
    3. Face Detection using Google Cloud Vision API (via Resizer).
    The results are combined to generate optimal crop coordinates.

    Parameters
    ----------
    video_file_path: str
        Absolute path to the video file.
    gcloud_config: GCloudConfig, optional
        Configuration for Google Cloud services. If None, a default GCloudConfig
        instance will be created (relying on environment variables for settings like
        project ID, GCS bucket for temporary files, etc.).
    transcriber_default_language_code: str, optional
        Default language code (e.g., "en-US") for transcription if not otherwise specified.
        Used by the Google Cloud Speech-to-Text transcriber.
    aspect_ratio: tuple[int, int] (width, height), default (9, 16)
        The target aspect ratio for resizing the video.
    min_segment_duration: float, optional
        Minimum duration in seconds for a speaker segment. Segments shorter than this,
        obtained from transcription-based diarization, might be filtered or ignored.
        Default is 1.5. Note: Its direct application might change based on how
        `get_speaker_segments` evolves.
    samples_per_segment: int, optional
        Number of frames to sample per speaker segment for face detection. Default is 13.
    face_detect_width: int, optional
        Width in pixels to which video frames are downscaled for face detection before
        being sent to the Google Cloud Vision API. Default is 960.
    # face_detect_margin: (Removed, was MTCNN specific)
    # face_detect_post_process: (Removed, was MTCNN specific)
    n_face_detect_batches: int, optional
        Number of batches for processing face detection. This may influence how frames
        are grouped if processed in batches, but the actual batching for API calls
        is handled within the Resizer's _detect_faces method. Default is 8.
    min_scene_duration: float, optional
        Minimum duration in seconds for a scene detected by Google Video Intelligence.
        Default is 0.25.
    scene_merge_threshold: float, optional
        Threshold in seconds for merging scene changes with speaker segments. Default is 0.25.
    time_precision: int, optional
        Precision (number of decimal places) for start and end times in speaker segments.
        Default is 6.
    device: str, optional
        PyTorch device ('cpu', 'cuda', etc.) for face detection models in Resizer.
        Default is None (auto-detection).

    Returns
    -------
    Crops
        An object containing information about the resized video
    """
    if gcloud_config is None:
        gcloud_config = GCloudConfig()

    media = AudioVideoFile(video_file_path)
    media.assert_has_audio_stream()
    media.assert_has_video_stream()

    # Transcription (includes diarization info)
    logging.info(f"TRANSCRIBING AND DIARIZING VIDEO ({media.get_filename()}) using Google Cloud Speech-to-Text")
    transcriber = Transcriber(
        gcloud_config=gcloud_config,
        default_language_code=transcriber_default_language_code
    )
    # The `transcribe` method in `Transcriber` takes `iso6391_lang_code` as an optional argument.
    # If `resize` needs to allow specifying this, it should be added as a parameter to `resize`
    # and passed here. For now, it will use the default_language_code or auto-detection logic
    # within the Transcriber based on its default `iso6391_lang_code=None`.
    transcription_obj = transcriber.transcribe(audio_file_path=video_file_path)

    logging.info(f"EXTRACTING SPEAKER SEGMENTS FROM TRANSCRIPTION for ({media.get_filename()})")
    diarized_segments = transcription_obj.get_speaker_segments(time_precision=time_precision)
    # Optional: Filter diarized_segments by min_segment_duration if still desired.
    # diarized_segments = [
    #     seg for seg in diarized_segments
    #     if (seg['end_time'] - seg['start_time']) >= min_segment_duration
    # ]

    # Scene Detection using Google Cloud Video Intelligence API
    logging.debug(f"DETECTING SCENES IN VIDEO ({media.get_filename()}) using Google Cloud API")
    scene_changes = detect_scenes(
        video_file=media,
        gcloud_config=gcloud_config,
        min_scene_duration=min_scene_duration
    )

    # Resizing
    logging.debug(f"RESIZING VIDEO ({media.get_filename()})")
    # device param is still relevant for potential local PyTorch ops in Resizer
    resizer = Resizer(
        gcloud_config=gcloud_config, # Pass GCloudConfig to Resizer
        device=device,
    )
    crops = resizer.resize(
        video_file=media,
        speaker_segments=diarized_segments,
        scene_changes=scene_changes,
        aspect_ratio=aspect_ratio,
        samples_per_segment=samples_per_segment,
        face_detect_width=face_detect_width,
        n_face_detect_batches=n_face_detect_batches,
        scene_merge_threshold=scene_merge_threshold,
    )
    resizer.cleanup()

    return crops
