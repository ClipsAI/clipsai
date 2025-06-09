"""Resizing/cropping a media file to a different aspect ratio."""

from __future__ import annotations

# Notes
# -----
# ROI is "region of interest"
# standard library imports
import logging

# current package imports
from .crops import Crops
from .exceptions import ResizerError
from .img_proc import calc_img_bytes
from .rect import Rect
from .segment import Segment
from .vid_proc import extract_frames

# local package imports
from clipsai.gcloud.config import GCloudConfig # Added import
from clipsai.media.editor import MediaEditor
from clipsai.media.video_file import VideoFile
from clipsai.utils import pytorch
from clipsai.utils.conversions import bytes_to_gibibytes

# 3rd party imports
import cv2
# MTCNN and mediapipe imports removed
from sklearn.cluster import KMeans
import torch
import numpy as np


class Resizer:
    """
    Calculates the optimal crop coordinates for resizing video segments.
    This class uses speaker segments (from transcription diarization) and scene changes
    (from Google Cloud Video Intelligence API) to define segments. For each segment,
    it samples frames, detects faces using the Google Cloud Vision API, and then
    determines a region of interest (ROI). This ROI is used to calculate the
    crop coordinates for the target aspect ratio.
    """

    def __init__(
        self,
        gcloud_config: GCloudConfig,
        device: str = None,
    ) -> None:
        """
        Initializes the Resizer.

        Face detection is performed using the Google Cloud Vision API.
        The `device` parameter is retained for potential future local PyTorch operations
        within the Resizer, but is not used for Google Cloud API calls.

        Parameters
        ----------
        gcloud_config : GCloudConfig
            Configuration object for Google Cloud services, used for initializing
            API clients (e.g., Vision API).
        device : str, optional
            PyTorch device ('cpu', 'cuda', etc.) for any local PyTorch-based
            operations that might be added or retained in the Resizer.
            Currently, primary computations like face detection are via Google Cloud.
            Default is None (auto-detects via `clipsai.utils.pytorch.get_compute_device`).
        """
        self._gcloud_config = gcloud_config
        if device is None:
            device = pytorch.get_compute_device()
        pytorch.assert_compute_device_available(device)
        logging.debug(f"Resizer initialized. Device for potential PyTorch ops: {device}")

        # self._face_detector = None # REMOVED MTCNN init
        # self._face_mesher = None # REMOVED MediaPipe init
        self._media_editor = MediaEditor()

    def resize(
        self,
        video_file: VideoFile,
        speaker_segments: list[dict],
        scene_changes: list[float],
        aspect_ratio: tuple = (9, 16),
        samples_per_segment: int = 13,
        face_detect_width: int = 960,
        n_face_detect_batches: int = 8,
        scene_merge_threshold: float = 0.25,
    ) -> Crops:
        """
        Calculates the coordinates to resize the video to for different
        segments given the diarized speaker segments and the desired aspect
        ratio.

        Parameters
        ----------
        video_file: VideoFile
            The video file to resize
        speaker_segments: list[dict]
            speakers: list[int]
                list of speakers (represented by int) talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        scene_changes: list[float]
            List of scene change times in seconds
        aspect_ratio: tuple[int, int]
            The (width,height) aspect ratio to resize the video to
        samples_per_segment: int
            Number of frames to sample per segment for face detection.
        face_detect_width: int
            The width to use for face detection
        n_face_detect_batches: int
            Number of batches for GPU face detection in a video file
        scene_merge_threshold: float
            The threshold in seconds for merging scene changes with speaker segments.
            Scene changes within this threshold of a segment's start or end time will
            cause the segment to be adjusted.

        Returns
        -------
        Crops
            the resized speaker segments
        """
        logging.debug(
            "Video Resolution: {}x{}".format(
                video_file.get_width_pixels(), video_file.get_height_pixels()
            )
        )
        # calculate resize dimensions
        resize_width, resize_height = self._calc_resize_width_and_height_pixels(
            original_width_pixels=video_file.get_width_pixels(),
            original_height_pixels=video_file.get_height_pixels(),
            resize_aspect_ratio=aspect_ratio,
        )

        logging.debug(
            "Merging {} speaker segments with {} scene changes.".format(
                len(speaker_segments), len(scene_changes)
            )
        )
        segments = self._merge_scene_change_and_speaker_segments(
            speaker_segments, scene_changes, scene_merge_threshold
        )
        logging.debug("Video has {} distinct segments.".format(len(segments)))

        logging.debug("Determining the first second with a face for each segment.")
        segments = self._find_first_sec_with_face_for_each_segment(
            segments, video_file, face_detect_width, n_face_detect_batches
        )

        logging.debug(
            "Determining the region of interest for {} segments.".format(len(segments))
        )
        segments = self._add_x_y_coords_to_each_segment(
            segments,
            video_file,
            resize_width,
            resize_height,
            samples_per_segment,
            face_detect_width,
            n_face_detect_batches,
        )

        logging.debug("Merging identical segments together.")
        unmerge_segments_length = len(segments)
        segments = self._merge_identical_segments(segments, video_file)
        logging.debug(
            "Merged {} identical segments.".format(
                unmerge_segments_length - len(segments)
            )
        )

        crop_segments = []
        for segment in segments:
            crop_segments.append(
                Segment(
                    speakers=segment["speakers"],
                    start_time=segment["start_time"],
                    end_time=segment["end_time"],
                    x=segment["x"],
                    y=segment["y"],
                )
            )

        crops = Crops(
            original_width=video_file.get_width_pixels(),
            original_height=video_file.get_height_pixels(),
            crop_width=resize_width,
            crop_height=resize_height,
            segments=crop_segments,
        )

        return crops

    def _calc_resize_width_and_height_pixels(
        self,
        original_width_pixels: int,
        original_height_pixels: int,
        resize_aspect_ratio: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Calculate the number of pixels along the width and height to resize the video
        to based on the desired aspect ratio.

        Parameters
        ----------
        original_pixels_width: int
            Number of pixels along the width of the original video.
        original_pixels_height: int
            Number of pixels along the height of the original video
        resize_aspect_ratio: tuple[int, int]
            The width:height aspect ratio to resize the video to

        Returns
        -------
        tuple[int, int]
            The number of pixels along the width and height to resize the video to
        """
        resize_ar_width, resize_ar_height = resize_aspect_ratio
        desired_aspect_ratio = resize_ar_width / resize_ar_height
        original_aspect_ratio = original_width_pixels / original_height_pixels

        # original aspect ratio is wider than desired aspect ratio
        if original_aspect_ratio > desired_aspect_ratio:
            resize_height_pixels = original_height_pixels
            resize_width_pixels = int(
                resize_height_pixels * resize_ar_width / resize_ar_height
            )
        # original aspect ratio is taller than desired aspect ratio
        else:
            resize_width_pixels = original_width_pixels
            resize_height_pixels = int(
                resize_width_pixels * resize_ar_height / resize_ar_width
            )

        return resize_width_pixels, resize_height_pixels

    def _merge_scene_change_and_speaker_segments(
        self,
        speaker_segments: list[dict],
        scene_changes: list[float],
        scene_merge_threshold: float,
    ) -> list[dict]:
        """
        Merge scene change segments with speaker segments based on a specified
        threshold.

        Parameters
        ----------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        scene_changes: list[float]
            List of scene change times in seconds.
        scene_merge_threshold: float
            The threshold in seconds for merging scene changes with speaker segments.
            Scene changes within this threshold of a segment's start or end time will
            cause the segment to be adjusted.

        Returns
        -------
        updated_speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
        """
        segments_idx = 0
        for scene_change_sec in scene_changes:
            segment = speaker_segments[segments_idx]
            while scene_change_sec > (segment["end_time"]):
                segments_idx += 1
                segment = speaker_segments[segments_idx]
            # scene change is close to speaker segment end -> merge the two
            if 0 < (segment["end_time"] - scene_change_sec) < scene_merge_threshold:
                segment["end_time"] = scene_change_sec
                if segments_idx == len(speaker_segments) - 1:
                    continue
                next_segment = speaker_segments[segments_idx + 1]
                next_segment["start_time"] = scene_change_sec
                continue
            # scene change is close to speaker segment start -> merge the two
            if 0 < (scene_change_sec - segment["start_time"]) < scene_merge_threshold:
                segment["start_time"] = scene_change_sec
                if segments_idx == 0:
                    continue
                prev_segment = speaker_segments[segments_idx - 1]
                prev_segment["end_time"] = scene_change_sec
                continue
            # scene change already exists
            if scene_change_sec == segment["end_time"]:
                continue
            # add scene change to segments
            new_segment = {
                "start_time": scene_change_sec,
                "speakers": segment["speakers"],
                "end_time": segment["end_time"],
            }
            segment["end_time"] = scene_change_sec
            speaker_segments = (
                speaker_segments[: segments_idx + 1]
                + [new_segment]
                + speaker_segments[segments_idx + 1 :]
            )

        return speaker_segments

    def _find_first_sec_with_face_for_each_segment(
        self,
        segments: list[dict],
        video_file: VideoFile,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> list[dict]:
        """
        Find the first frame in a segment with a face.

        Parameters
        ----------
        segments: list[dict]
            List of speaker segments (dictionaries), each with the following keys
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
        video_file: VideoFile
            The video file to analyze.
        n_face_detect_batches: int
            The number of batches to use for identifyinng faces from a video file

        Returns
        -------
        list[dict]
            List of speaker segments (dictionaries), each with the following keys
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                start_time: float
                    start time of the segment in seconds
                end_time: float
                    end time of the segment in seconds
                first_face_sec: float
                    the first second in the segment with a face
                found_face: bool
                    whether or not a face was found in the segment
        """
        for segment in segments:
            start_time = segment["start_time"]
            end_time = segment["end_time"]
            # start looking for faces an eighth of the way through the segment
            segment["first_face_sec"] = start_time + (end_time - start_time) / 8
            segment["found_face"] = False
            segment["is_analyzed"] = False

        batch_period = 1  # interval length to sample each segment at each iteration
        sample_period = 1  # interval between consecutive samples
        analyzed_segments = 0
        while analyzed_segments < len(segments):
            # select times to detect faces from
            detect_secs = []
            for segment in segments:
                if segment["is_analyzed"] is True:
                    continue
                segment_secs_left = segment["end_time"] - segment["first_face_sec"]
                num_samples = min(batch_period, segment_secs_left) // sample_period
                num_samples = max(1, int(num_samples))
                segment["num_samples"] = num_samples
                for i in range(num_samples):
                    detect_secs.append(segment["first_face_sec"] + i * sample_period)

            # detect faces
            n_batches = self._calc_n_batches(
                video_file=video_file,
                num_frames=len(detect_secs),
                face_detect_width=face_detect_width,
                n_face_detect_batches=n_face_detect_batches,
            )
            frames_per_batch = int(len(detect_secs) // n_batches + 1)
            face_detections = []
            for i in range(n_batches):
                frames = extract_frames(
                    video_file,
                    detect_secs[
                        i
                        * frames_per_batch : min(
                            (i + 1) * frames_per_batch, len(detect_secs)
                        )
                    ],
                )
                face_detections += self._detect_faces(frames, face_detect_width)

            # check if any faces were found for each segment
            idx = 0
            for segment in segments:
                # segment already analyzed
                if segment["is_analyzed"] is True:
                    continue
                segment_idx = idx
                # check if any faces were found
                for _ in range(segment["num_samples"]):
                    faces = face_detections[idx]
                    if faces is not None:
                        segment["found_face"] = True
                        break
                    segment["first_face_sec"] += sample_period
                    idx += 1
                # update segment analyzation status
                is_analyzed = (
                    segment["found_face"] is True
                    or segment["first_face_sec"] >= segment["end_time"] - 0.25
                )
                if is_analyzed:
                    segment["is_analyzed"] = True
                    analyzed_segments += 1
                idx = segment_idx + segment["num_samples"]

            # increase period for next iteration
            batch_period = (batch_period + 3) * 2

        for segment in segments:
            del segment["num_samples"]
            del segment["is_analyzed"]

        return segments

    def _calc_n_batches(
        self,
        video_file: VideoFile,
        num_frames: int,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> int:
        """
        Calculate the number of batches to use for extracting frames from a video file
        and detecting the face in each frame.

        Parameters
        ----------
        video_file: VideoFile
            The video file to analyze.
        num_frames: int
            The number of frames to analyze.
        face_detect_width: int
            The width to use for face detection.
        n_face_detect_batches: int
            Number of batches for GPU face detection in a video file.

        Returns
        -------
        int
            The number of batches to use.
        """
        # calculate memory needed to extract frames to CPU
        vid_height = video_file.get_height_pixels()
        vid_width = video_file.get_width_pixels()
        num_color_channels = 3
        bytes_per_frame = calc_img_bytes(vid_height, vid_width, num_color_channels)
        total_extract_bytes = num_frames * bytes_per_frame
        logging.debug(
            "Need {:.3f} GiB to extract (at most) {} frames".format(
                bytes_to_gibibytes(total_extract_bytes), num_frames
            )
        )

        # calculate memory needed to detect faces -> could be CPU or GPU
        downsample_factor = max(vid_width / face_detect_width, 1)
        face_detect_height = int(vid_height // downsample_factor)
        logging.debug(
            "Face detection dimensions: {}x{}".format(
                face_detect_height, face_detect_width
            )
        )
        bytes_per_frame = calc_img_bytes(
            face_detect_height, face_detect_width, num_color_channels
        )
        total_face_detect_bytes = num_frames * bytes_per_frame
        logging.debug(
            "Need {:.3f} GiB to detect faces from (at most) {} frames".format(
                bytes_to_gibibytes(total_face_detect_bytes), num_frames
            )
        )

        # calculate number of batches to use
        free_cpu_memory = pytorch.get_free_cpu_memory()
        if torch.cuda.is_available():
            n_extract_batches = int((total_extract_bytes // free_cpu_memory) + 1)
        else:
            total_extract_bytes += total_face_detect_bytes
            n_extract_batches = int((total_extract_bytes // free_cpu_memory) + 1)
            n_face_detect_batches = 0

        n_batches = int(max(n_extract_batches, n_face_detect_batches))
        cpu_mem_per_batch = bytes_to_gibibytes(total_extract_bytes // n_batches)
        if n_face_detect_batches == 0:
            gpu_mem_per_batch = 0
        else:
            gpu_mem_per_batch = bytes_to_gibibytes(total_face_detect_bytes // n_batches)
        logging.debug(
            "Using {} batches to extract and detect frames. Need {:.3f} GiB of CPU "
            "memory per batch and {:.3f} GiB of GPU memory per batch".format(
                n_batches,
                cpu_mem_per_batch,
                gpu_mem_per_batch,
            )
        )
        return n_batches

    def _detect_faces(
        self,
        frames: list[np.ndarray],
        face_detect_width: int,
    ) -> list[np.ndarray | None]:
        """
        Detects faces in a list of frames using the Google Cloud Vision API.

        Frames can be optionally downscaled to `face_detect_width` before being
        sent to the API to reduce processing time and cost. Detected face bounding
        box coordinates are scaled back to the original frame dimensions.

        Parameters
        ----------
        frames : list[np.ndarray]
            A list of video frames (as NumPy arrays) in which to detect faces.
        face_detect_width : int
            The target width to which frames are resized before being sent to the
            Vision API. If a frame's original width is smaller than this value,
            it is not upscaled. This helps standardize input for the API and can
            reduce payload size.

        Returns
        -------
        list[np.ndarray | None]
            A list corresponding to the input frames. Each element is either:
            - A NumPy array of shape (num_faces, 4), where each row represents a
              detected face's bounding box as [x1, y1, x2, y2] in original frame
              coordinates.
            - None, if no faces were detected in the corresponding frame or if an
              error occurred during its processing by the API.

        Raises
        ------
        ResizerError
            If the Google Cloud Vision library is not installed, if frame encoding
            fails, or if there's an API call failure or other processing error.
        """
        try:
            from google.cloud import vision
            from google.api_core import exceptions as google_exceptions
        except ModuleNotFoundError as e:
            logging.error(f"Google Cloud Vision library not found: {e}")
            raise ResizerError(f"Google Cloud Vision library not found. Please install google-cloud-vision.") from e

        if not frames:
            return []

        vision_client = vision.ImageAnnotatorClient(
            client_options={"project_id": self._gcloud_config.project_id} if self._gcloud_config.project_id else None
        )

        requests = []
        downsample_factors = []

        for frame_np in frames:
            original_height, original_width = frame_np.shape[:2]
            current_frame_for_api = frame_np
            downsample_factor = 1.0

            if face_detect_width < original_width:
                downsample_factor = original_width / face_detect_width
                detect_height = int(original_height / downsample_factor)
                current_frame_for_api = cv2.resize(frame_np, (face_detect_width, detect_height))

            downsample_factors.append(downsample_factor)

            success, encoded_image = cv2.imencode(".jpg", current_frame_for_api)
            if not success:
                logging.error("Failed to encode frame to JPEG for Vision API.")
                # Add a placeholder for this frame's result and continue
                # Or, raise an error immediately if one failure should stop all.
                # For batch, it's often better to collect individual errors.
                # Here, we'll effectively skip adding a request for this frame.
                # This needs to be reconciled with the length of `downsample_factors`
                # and the final `all_frame_detections` list.
                # A simpler approach for now: raise error or return Nones for all if one fails.
                raise ResizerError("Failed to encode a frame to JPEG for Vision API.")

            image = vision.Image(content=encoded_image.tobytes())
            feature = vision.Feature(type_=vision.Feature.Type.FACE_DETECTION, max_results=10)
            requests.append(vision.AnnotateImageRequest(image=image, features=[feature]))

        all_frame_detections = []
        if not requests: # If all frames failed encoding for some reason
            return [None] * len(frames)

        try:
            response_batch = vision_client.batch_annotate_images(requests=requests)

            for i, response_single_image in enumerate(response_batch.responses):
                if response_single_image.error.message:
                    logging.error(f"Vision API error for frame index {i}: {response_single_image.error.message}")
                    all_frame_detections.append(None)
                    continue

                frame_face_boxes = []
                for face_annotation in response_single_image.face_annotations:
                    vertices = face_annotation.bounding_poly.vertices
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]

                    # Bounding_poly can be from a rotated face, so min/max creates an axis-aligned box
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)

                    current_downsample_factor = downsample_factors[i]
                    x1 = int(x1 * current_downsample_factor)
                    y1 = int(y1 * current_downsample_factor)
                    x2 = int(x2 * current_downsample_factor)
                    y2 = int(y2 * current_downsample_factor)

                    # Clamp negative values and ensure box validity (x1<x2, y1<y2)
                    # Original frame dimensions are needed for more robust clamping if boxes can exceed them.
                    # For now, simple max(0, val) and ensuring x1<x2, y1<y2.
                    final_x1, final_y1 = max(0, x1), max(0, y1)
                    final_x2, final_y2 = max(final_x1, x2), max(final_y1, y2) # Ensure x2 > x1, y2 > y1

                    frame_face_boxes.append([final_x1, final_y1, final_x2, final_y2])

                if frame_face_boxes:
                    all_frame_detections.append(np.array(frame_face_boxes, dtype=np.int16))
                else:
                    all_frame_detections.append(None)

        except google_exceptions.GoogleAPICallError as e:
            logging.error(f"Google Cloud Vision API call failed: {e}")
            raise ResizerError(f"Vision API call failed: {e}") from e
        except Exception as e: # Catch other unexpected errors
            logging.error(f"Error processing face detection with Vision API: {e}")
            raise ResizerError(f"Error in Vision API face detection processing: {e}") from e

        # Ensure the output list matches the number of input frames if some requests failed early
        # This part is tricky if requests list itself became shorter than frames list.
        # The current logic assumes requests list is built for all encodable frames.
        # If an encoding error for one frame means we don't call batch_annotate_images,
        # then this needs adjustment. The ResizerError on encoding failure simplifies this.
        return all_frame_detections

    def _add_x_y_coords_to_each_segment(
        self,
        segments: list[dict],
        video_file: VideoFile,
        resize_width: int,
        resize_height: int,
        samples_per_segment: int,
        face_detect_width: int,
        n_face_detect_batches: int,
    ) -> list[dict]:
        """
        Add the x and y coordinates to resize each segment to.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            first_face_sec: float
                the first second in the segment with a face
            found_face: bool
                whether or not a face was found in the segment
        video_file: VideoFile
            The video file to analyze.
        resize_width: int
            The width to resize the video to.
        resize_height: int
            The height to resize the video to.
        samples_per_segment: int
            Number of samples to take per segment for face detection.
        face_detect_width: int
            Width to resize the frames to for face detection.
        n_face_detect_batches: int
            Number of batches to process for face detection.


        Returns
        -------
        updated_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        num_segments = len(segments)
        num_frames = num_segments * samples_per_segment
        n_batches = self._calc_n_batches(
            video_file, num_frames, face_detect_width, n_face_detect_batches
        )
        segments_per_batch = int(num_segments // n_batches + 1)
        segments_with_xy_coords = []
        for i in range(n_batches):
            logging.debug("Analyzing batch {} of {}.".format(i, n_batches))
            cur_segments = segments[
                i
                * segments_per_batch : min((i + 1) * segments_per_batch, len(segments))
            ]
            if len(cur_segments) == 0:
                logging.debug("No segments left to analyze. (Batch {})".format(i))
                break
            segments_with_xy_coords += self._add_x_y_coords_to_each_segment_batch(
                segments=cur_segments,
                video_file=video_file,
                resize_width=resize_width,
                resize_height=resize_height,
                samples_per_segment=samples_per_segment,
                face_detect_width=face_detect_width,
            )
        return segments_with_xy_coords

    def _add_x_y_coords_to_each_segment_batch(
        self,
        segments: list[dict],
        video_file: VideoFile,
        resize_width: int,
        resize_height: int,
        samples_per_segment: int,
        face_detect_width: int,
    ) -> list[dict]:
        """
        Add the x and y coordinates to resize each segment to for a given batch.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            first_face_sec: float
                the first second in the segment with a face
            found_face: bool
                whether or not a face was found in the segment
        video_file: VideoFile
            The video file to analyze.
        resize_width: int
            The width to resize the video to.
        resize_height: int
            The height to resize the video to.
        samples_per_segment: int
            Number of samples to take per segment for analyzing face locations.
        face_detect_width: int
            Width to which the video frames are resized for face detection.

        Returns
        -------
        updated_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            start_time: float
                start time of the segment in seconds
            end_time: float
                end time of the segment in seconds
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        fps = video_file.get_frame_rate()

        # define frames to analyze from each segment
        detect_secs = []
        for segment in segments:
            if segment["found_face"] is False:
                continue
            # define interval over which to analyze faces
            end_time = segment["end_time"]
            first_face_sec = segment["first_face_sec"]
            analyze_end_time = end_time - (end_time - first_face_sec) / 8
            # get sample locations
            frames_left = int((analyze_end_time - first_face_sec) * fps + 1)
            num_samples = min(frames_left, samples_per_segment)
            segment["num_samples"] = num_samples
            # add first face, sample the rest
            detect_secs.append(first_face_sec)
            sample_frames = np.sort(
                np.random.choice(range(1, frames_left), num_samples - 1, replace=False)
            )
            for sample_frame in sample_frames:
                detect_secs.append(first_face_sec + sample_frame / fps)

        # detect faces from each segment
        logging.debug("Extracting {} frames".format(len(detect_secs)))
        frames = extract_frames(video_file, detect_secs)
        logging.debug("Extracted {} frames".format(len(detect_secs)))
        face_detections = self._detect_faces(frames, face_detect_width)

        logging.debug("Calculating ROI for {} segments.".format(len(segments)))
        # find roi for each segment
        idx = 0
        for segment in segments:
            # find segment roi
            if segment["found_face"] is True:
                # Updated call to _calc_segment_roi (frames parameter removed)
                roi = self._calc_segment_roi(
                    face_detections=face_detections[idx : idx + segment["num_samples"]],
                )
                idx += segment["num_samples"]
                del segment["num_samples"]
            else:
                logging.debug("Using default ROI for segment {}".format(segment))
                roi = Rect(
                    x=(video_file.get_width_pixels()) // 4,
                    y=(video_file.get_height_pixels()) // 4,
                    width=(video_file.get_width_pixels()) // 2,
                    height=(video_file.get_height_pixels()) // 2,
                )
            del segment["found_face"]
            del segment["first_face_sec"]

            # add crop coordinates to segment
            crop = self._calc_crop(roi, resize_width, resize_height)
            segment["x"] = int(crop.x)
            segment["y"] = int(crop.y)
        logging.debug("Calculated ROI for {} segments.".format(len(segments)))

        return segments

    def _calc_segment_roi(
        self,
        face_detections: list[np.ndarray | None],
    ) -> Rect:
        """
        Calculates the primary Region of Interest (ROI) for a segment based on
        face detections from multiple sampled frames within that segment.

        The method uses K-Means clustering to group all detected bounding boxes
        from the sampled frames.
        - If only one cluster (k=1) is found, its average bounding box is the ROI.
        - If multiple clusters (k>1) are found, a heuristic is applied:
            1. The cluster with the most persistent face (most detections) is chosen.
            2. If there's a tie in persistence, the cluster whose average bounding
               box has the largest area is chosen.
        The ROI is the average bounding box of the selected cluster.

        Parameters
        ----------
        face_detections : list[np.ndarray | None]
            A list of face detection results for frames sampled from the segment.
            Each element corresponds to a frame:
            - np.ndarray of shape (num_faces, 4): bounding boxes [x1, y1, x2, y2]
            - None: if no faces were detected in that frame.

        Returns
        -------
        Rect
            The calculated Region of Interest (ROI) for the segment.

        Raises
        ------
        ResizerError
            If no faces are detected across all sampled frames (k=0), or if an
            internal error occurs in selecting a primary face cluster when k > 1.
        """
        segment_roi = None

        # preprocessing for kmeans
        bounding_boxes: list[np.ndarray] = []
        k = 0
        for face_detection in face_detections:
            if face_detection is None:
                continue
            k = max(k, len(face_detection))
            for bounding_box in face_detection:
                bounding_boxes.append(bounding_box)

        # no faces detected
        if k == 0:
            raise ResizerError("No faces detected in segment.")
        bounding_boxes = np.stack(bounding_boxes)

        # single face detected
        if k == 1:
            box = np.mean(bounding_boxes, axis=0).astype(np.int16)
            x1, y1, x2, y2 = box
            segment_roi = Rect(x1, y1, x2 - x1, y2 - y1)
            return segment_roi

        # use kmeans to group the same bounding boxes together
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=2, random_state=0).fit(
            bounding_boxes
        )
        bounding_box_labels = kmeans.labels_
        bounding_box_groups: list[list[dict]] = [[] for _ in range(k)]
        kmeans_idx = 0
        for i, face_detection in enumerate(face_detections):
            if face_detection is None:
                continue
            for bounding_box in face_detection:
                assert np.sum(bounding_box < 0) == 0
                bounding_box_label = bounding_box_labels[kmeans_idx]
                bounding_box_groups[bounding_box_label].append(
                    {"bounding_box": bounding_box, "frame": i}
                )
                kmeans_idx += 1

        # NEW LOGIC for k > 1 (replaces mouth_movement and its fallback):
        if k > 1:
            best_group_index = -1
            max_detections_in_group = -1
            max_area_for_tied_group = -1.0

            for i, group_data_list in enumerate(bounding_box_groups):
                num_detections_in_group = len(group_data_list)

                if num_detections_in_group == 0:
                    continue

                current_group_avg_box_sum = np.array([0.0, 0.0, 0.0, 0.0])
                for item_dict in group_data_list:
                    current_group_avg_box_sum += item_dict["bounding_box"]

                avg_box_coords = (current_group_avg_box_sum / num_detections_in_group).astype(np.int16)
                avg_box_width = avg_box_coords[2] - avg_box_coords[0]
                avg_box_height = avg_box_coords[3] - avg_box_coords[1]
                avg_box_width = max(0, avg_box_width)
                avg_box_height = max(0, avg_box_height)
                current_group_area = float(avg_box_width * avg_box_height)

                if num_detections_in_group > max_detections_in_group:
                    max_detections_in_group = num_detections_in_group
                    max_area_for_tied_group = current_group_area
                    best_group_index = i
                elif num_detections_in_group == max_detections_in_group:
                    if current_group_area > max_area_for_tied_group:
                        max_area_for_tied_group = current_group_area
                        best_group_index = i

            if best_group_index != -1:
                chosen_group_data_list = bounding_box_groups[best_group_index]
                avg_box_sum = np.array([0.0, 0.0, 0.0, 0.0])
                for item_dict in chosen_group_data_list:
                    avg_box_sum += item_dict["bounding_box"]
                avg_box_coords = (avg_box_sum / len(chosen_group_data_list)).astype(np.int16)
                segment_roi = Rect(
                    x=avg_box_coords[0],
                    y=avg_box_coords[1],
                    width=max(0, avg_box_coords[2] - avg_box_coords[0]),
                    height=max(0, avg_box_coords[3] - avg_box_coords[1])
                )
            else:
                logging.error("Internal error in _calc_segment_roi: No best face group found with k > 1. Defaulting to a full frame ROI or raising error.")
                raise ResizerError("Internal error: Could not determine primary face cluster in _calc_segment_roi with k > 1.")
        # End of new logic for k > 1. Note: segment_roi would already be set if k==1 from before.

        return segment_roi

    # _calc_mouth_movement and _calc_mouth_aspect_ratio methods are removed.

    def _calc_crop(
        self,
        roi: Rect,
        resize_width: int,
        resize_height: int,
    ) -> Rect:
        """
        Calculate the crop given the ROI location.

        Parameters
        ----------
        roi: Rect
            The rectangle containing the region of interest (ROI).

        Returns
        -------
        Rect
            The crop rectangle.
        """
        roi_x_center = roi.x + roi.width // 2
        roi_y_center = roi.y + roi.height // 2
        crop = Rect(
            x=max(roi_x_center - (resize_width // 2), 0),
            y=max(roi_y_center - (resize_height // 2), 0),
            width=resize_width,
            height=resize_height,
        )
        return crop

    def _merge_identical_segments(
        self,
        segments: list[dict],
        video_file: VideoFile,
    ) -> list[dict]:
        """
        Merge identical segments that are next to each other.

        Parameters
        ----------
        segments: list[dict]
            speakers: list[int]
                the speaker labels of the speakers talking in the segment
            start_time: float
                the start time of the segment
            end_time: float
                the end time of the segment
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        video_file: VideoFile
            The video file that the segments are from

        Returns
        -------
        list[dict]
            The merged segments.
        """
        idx = 0
        max_position_difference_ratio = 0.04
        video_width = video_file.get_width_pixels()
        video_height = video_file.get_height_pixels()

        for _ in range(len(segments) - 1):
            cur_x = segments[idx]["x"]
            next_x = segments[idx + 1]["x"]
            x_diff = abs(cur_x - next_x)
            if (x_diff / video_width) < max_position_difference_ratio:
                same_x = True
                segments[idx]["x"] = int((cur_x + next_x) // 2)
            else:
                same_x = False

            curr_y = segments[idx]["y"]
            next_y = segments[idx + 1]["y"]
            y_diff = abs(curr_y - next_y)
            if (y_diff / video_height) < max_position_difference_ratio:
                same_y = True
                segments[idx]["y"] = int((curr_y + next_y) // 2)
            else:
                same_y = False

            if same_x and same_y:
                segments[idx]["end_time"] = segments[idx + 1]["end_time"]
                segments = segments[: idx + 1] + segments[idx + 2 :]
            else:
                idx += 1
        return segments

    def cleanup(self) -> None:
        """
        Perform any necessary cleanup.
        (Face detector/mesher cleanup removed as they are no longer initialized here)
        """
        # del self._face_detector # REMOVED
        # self._face_detector = None # REMOVED
        # del self._face_mesher # REMOVED (if it was ever here)
        # self._face_mesher = None # REMOVED (if it was ever here)
        if torch.cuda.is_available():
            # This is a general PyTorch utility, can remain if other torch ops exist
            # or might be used by Resizer in other contexts.
            torch.cuda.empty_cache()
