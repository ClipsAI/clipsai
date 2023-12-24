"""
Utilities for detecting scene changes in a video.
"""
# current package imports
from media.video_file import VideoFile

# third party imports
from scenedetect import detect, AdaptiveDetector


def detect_scenes(
    video_file: VideoFile,
    min_scene_secs: float = 0.25,
) -> list[float]:
    """
    Detect scene changes in a video.

    Parameters
    ----------
    video_file: VideoFile
        The video file to detect scene changes in.
    min_scene_secs: float
        The minimum length of a scene in seconds.

    Returns
    -------
    list[float]
        The seconds where scene changes occur.
    """
    detector = AdaptiveDetector(
        min_scene_len=min_scene_secs * video_file.get_frame_rate()
    )
    scene_list = detect(video_file.path, detector)

    scene_change_secs = []
    # don't include end time of last scene -> it's the end of the video
    for i in range(len(scene_list) - 1):
        scene = scene_list[i]
        scene_change_secs.append(round(scene[1].get_seconds(), 6))

    return scene_change_secs