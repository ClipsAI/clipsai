"""
Resize an asset's media to a 9:16 aspect ratio.
"""
# standard library imports
import logging

# current package imports
from .facenet_mp import FaceNetMediaPipeResizer
from .detect_scenes import detect_scenes

# local package imports
from diarize.pyannote import PyannoteDiarizer
from media.audiovideo_file import AudioVideoFile


def resize(
    video_file_path: str,
    pyannote_auth_token: str,
    aspect_ratio: tuple[int, int] = (9, 16),
    device: str = None,
) -> list[dict]:
    """
    Resizes a video to a desired aspect ratio, default aspect ration: 9:16.
    The video must have both an audio and video stream.

    Parameters
    ----------
    video_file_path: str
        absolute path to a video file
    pyannote_auth_token: str
        pyannote auth token created on HuggingFace
    aspect_ratio: tuple[int, int]
        desired aspect ratio, default: (9, 16)
    device: str
        device to use for diarization

    Returns
    -------
    list[dict]
        the resized speaker segments
    """
    media = AudioVideoFile(video_file_path)
    media.assert_has_audio_stream()
    media.assert_has_video_stream()

    logging.debug("DIARIZING VIDEO ({})".format(media.get_filename()))
    diarizer = PyannoteDiarizer(auth_token=pyannote_auth_token, device=device)
    diarized_segments = diarizer.diarize(media)

    logging.debug("DETECTING SCENES IN VIDEO ({})".format(media.get_filename()))
    scene_changes = detect_scenes(media)

    logging.debug("RESIZING VIDEO) ({})".format(media.get_filename()))
    resizer = FaceNetMediaPipeResizer(device)
    crops = resizer.resize(
        video_file=media,
        speaker_segments=diarized_segments,
        scene_changes=scene_changes,
        aspect_ratio=aspect_ratio,
    )
    resizer.cleanup()

    return crops
    
    