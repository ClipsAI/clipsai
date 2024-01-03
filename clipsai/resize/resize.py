"""
Resize an asset's media to a 9:16 aspect ratio.
"""
# standard library imports
import logging

# current package imports
from .crops import Crops
from .facenet_mp import FaceNetMediaPipeResizer
from .detect_scenes import detect_scenes

# local package imports
from clipsai.diarize.pyannote import PyannoteDiarizer
from clipsai.media.audiovideo_file import AudioVideoFile


def resize(
    video_file_path: str,
    pyannote_auth_token: str,
    aspect_ratio: tuple[int, int] = (9, 16),
    min_segment_duration: float = 1.5,
    samples_per_segment: int = 13,
    face_detect_width: int = 960,
    face_detect_margin: int = 20,
    face_detect_post_process: bool = False,
    n_face_detect_batches: int = 8,
    min_scene_duration: float = 0.25,
    scene_merge_threshold: float = 0.25,
    time_precision: int = 6,
    device: str = "auto",
) -> Crops:
    """
    Resizes a video to a specified aspect ratio, with default being 9:16. It involves 
    speaker diarization, scene detection, and face detection for resizing.

    Parameters
    ----------
    video_file_path: str
        Absolute path to the video file.
    pyannote_auth_token: str
        Authentication token for Pyannote, obtained from HuggingFace.
    aspect_ratio: tuple[int, int] (width, height), default (9, 16)
        The target aspect ratio for resizing the video.
    min_segment_duration: float, default 1.5
        The minimum duration in seconds for a diarized speaker segment to be considered.
    samples_per_segment: int, default 13
        The number of samples to take per speaker segment for face detection.
    face_detect_width: int, default 960
        The width in pixels to which the video will be downscaled for face detection.
    face_detect_margin: int, default 20
        Margin around detected faces, used in the MTCNN face detector.
    face_detect_post_process: bool, default False
        If set to True, post-processing is applied to the face detection output to make 
        it appear more natural.
    n_face_detect_batches: int, default 8
        Number of batches for processing face detection when using GPUs.
    min_scene_duration: float, default 0.25
        Minimum duration in seconds for a scene to be considered during scene detection.
    scene_merge_threshold: float, default 0.25
        Threshold in seconds for merging scene changes with speaker segments.
    time_precision: int, default 6
        Precision (number of decimal places) for start and end times in diarization.
    device: str, default 'auto'
        The compute device ('auto', 'cpu', or 'cuda') for processing.

    Returns
    -------
    Crops
        An object containing information about the resized video
    """
    media = AudioVideoFile(video_file_path)
    media.assert_has_audio_stream()
    media.assert_has_video_stream()

    logging.debug("DIARIZING VIDEO ({})".format(media.get_filename()))
    diarizer = PyannoteDiarizer(auth_token=pyannote_auth_token, device=device)
    diarized_segments = diarizer.diarize(media, min_segment_duration, time_precision)

    logging.debug("DETECTING SCENES IN VIDEO ({})".format(media.get_filename()))
    scene_changes = detect_scenes(media, min_scene_duration)

    logging.debug("RESIZING VIDEO) ({})".format(media.get_filename()))
    resizer = FaceNetMediaPipeResizer(
        face_detect_margin=face_detect_margin,
        face_detect_post_process=face_detect_post_process,
        device=device
    )
    crops = resizer.resize(
        video_file=media,
        speaker_segments=diarized_segments,
        scene_changes=scene_changes,
        aspect_ratio=aspect_ratio,
        samples_per_segment=samples_per_segment,
        face_detect_width=face_detect_width,
        n_face_detect_batches=n_face_detect_batches,
        scene_merge_threshold=scene_merge_threshold
    )
    resizer.cleanup()

    return crops
    