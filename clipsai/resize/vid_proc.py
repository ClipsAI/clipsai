"""
Utilities for video processing.
"""
# standard library imports
from concurrent.futures import ThreadPoolExecutor
import logging

# current package imports
from .exceptions import VideoProcessingError
from .img_proc import rgb_to_gray

# local imports
from clipsai.media.video_file import VideoFile

# third party imports
import av
import cv2
import numpy as np
from scenedetect import detect, AdaptiveDetector


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
    min_scene_duration: float = 0.25,
) -> list[float]:
    """
    Detect scene changes in a video.

    Parameters
    ----------
    video_file: VideoFile
        The video file to detect scene changes in.
    min_scene_duration: float
        The minimum length of a scene in seconds.

    Returns
    -------
    scene_changes: list[float]
        The seconds where scene changes occur.
    """
    detector = AdaptiveDetector(
        min_scene_len=min_scene_duration * video_file.get_frame_rate()
    )
    scene_list = detect(video_file.path, detector)

    scene_changes = []
    # don't include end time of last scene -> it's the end of the video
    for i in range(len(scene_list) - 1):
        scene = scene_list[i]
        scene_changes.append(round(scene[1].get_seconds(), 6))

    return scene_changes
