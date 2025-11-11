"""
Face Tracking for Auto-Reframing
Option A: Horizontal canvas with movement tracking
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Tracks faces in horizontal video for dynamic reframing.

    Uses OpenCV Haar Cascades for face detection.
    Tracks movement to create smooth pan/zoom effects.
    """

    def __init__(self):
        """Initialize face tracker with Haar Cascade."""
        # Load pre-trained face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Tracking parameters
        self.smoothing_window = 10  # Frames to smooth tracking
        self.min_face_size = (80, 80)  # Minimum face size to detect

    def track_faces_in_video(
        self,
        video_path: str
    ) -> List[Dict[str, Any]]:
        """
        Track all faces across video frames.

        Args:
            video_path: Path to input video

        Returns:
            List of face tracking data per frame
        """
        try:
            cap = cv2.VideoCapture(video_path)

            frame_data = []
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in frame
                faces = self._detect_faces(frame)

                # Get primary face (largest or most centered)
                primary_face = self._get_primary_face(faces, frame.shape)

                frame_data.append({
                    "frame": frame_idx,
                    "faces": faces,
                    "primary_face": primary_face,
                    "timestamp": frame_idx / cap.get(cv2.CAP_PROP_FPS)
                })

                frame_idx += 1

            cap.release()

            logger.info(f"Tracked faces across {frame_idx} frames")

            # Apply smoothing to tracking data
            smoothed_data = self._smooth_tracking(frame_data)

            return smoothed_data

        except Exception as e:
            logger.error(f"Face tracking failed: {e}")
            raise

    def _detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a single frame.

        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size
        )

        return [tuple(face) for face in faces]

    def _get_primary_face(
        self,
        faces: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Select primary face from multiple detections.

        Strategy:
        1. If only one face, use it
        2. If multiple, prefer largest face
        3. If similar sizes, prefer most centered

        Returns:
            (x, y, w, h) of primary face, or None if no faces
        """
        if not faces:
            return None

        if len(faces) == 1:
            return faces[0]

        # Calculate scores for each face
        frame_h, frame_w = frame_shape[:2]
        frame_center = (frame_w // 2, frame_h // 2)

        scored_faces = []
        for (x, y, w, h) in faces:
            # Size score (area)
            size_score = w * h

            # Centrality score (distance from center)
            face_center = (x + w // 2, y + h // 2)
            distance = np.sqrt(
                (face_center[0] - frame_center[0]) ** 2 +
                (face_center[1] - frame_center[1]) ** 2
            )
            # Invert so closer = higher score
            centrality_score = 1.0 / (1.0 + distance / 100)

            # Combined score (70% size, 30% centrality)
            total_score = 0.7 * size_score + 0.3 * centrality_score * 10000

            scored_faces.append(((x, y, w, h), total_score))

        # Return highest scoring face
        best_face = max(scored_faces, key=lambda x: x[1])
        return best_face[0]

    def _smooth_tracking(
        self,
        frame_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply smoothing to face tracking data.

        Uses moving average over window to reduce jitter.
        """
        if len(frame_data) < self.smoothing_window:
            return frame_data

        smoothed = []

        for i, frame in enumerate(frame_data):
            # Get window of frames
            window_start = max(0, i - self.smoothing_window // 2)
            window_end = min(len(frame_data), i + self.smoothing_window // 2)

            window_faces = []
            for j in range(window_start, window_end):
                if frame_data[j]["primary_face"] is not None:
                    window_faces.append(frame_data[j]["primary_face"])

            if window_faces:
                # Average position
                avg_x = int(np.mean([f[0] for f in window_faces]))
                avg_y = int(np.mean([f[1] for f in window_faces]))
                avg_w = int(np.mean([f[2] for f in window_faces]))
                avg_h = int(np.mean([f[3] for f in window_faces]))

                smoothed_face = (avg_x, avg_y, avg_w, avg_h)
            else:
                smoothed_face = frame["primary_face"]

            smoothed.append({
                **frame,
                "primary_face_smoothed": smoothed_face
            })

        return smoothed

    def apply_tracking_to_video(
        self,
        input_path: str,
        output_path: str,
        tracking_data: List[Dict[str, Any]],
        target_size: Tuple[int, int] = (1080, 1920)
    ) -> str:
        """
        Apply face tracking to create reframed video.

        Args:
            input_path: Input video
            output_path: Output video
            tracking_data: Face tracking data from track_faces_in_video
            target_size: Output resolution (width, height)

        Returns:
            Path to output video
        """
        try:
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                target_size
            )

            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Get tracking data for this frame
                if frame_idx < len(tracking_data):
                    face_data = tracking_data[frame_idx]
                    primary_face = face_data.get("primary_face_smoothed")

                    if primary_face:
                        # Reframe around face
                        reframed = self._reframe_to_face(
                            frame,
                            primary_face,
                            target_size
                        )
                    else:
                        # No face, use center crop
                        reframed = self._center_crop(frame, target_size)
                else:
                    reframed = self._center_crop(frame, target_size)

                writer.write(reframed)
                frame_idx += 1

            cap.release()
            writer.release()

            logger.info(f"Applied face tracking: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Tracking application failed: {e}")
            raise

    def _reframe_to_face(
        self,
        frame: np.ndarray,
        face: Tuple[int, int, int, int],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Crop and resize frame to keep face centered."""
        x, y, w, h = face
        frame_h, frame_w = frame.shape[:2]
        target_w, target_h = target_size

        # Calculate crop region (with padding around face)
        padding = 1.5  # 50% padding around face

        crop_w = int(w * padding)
        crop_h = int(h * padding)

        # Maintain target aspect ratio
        target_aspect = target_w / target_h
        crop_aspect = crop_w / crop_h

        if crop_aspect > target_aspect:
            # Too wide, adjust height
            crop_h = int(crop_w / target_aspect)
        else:
            # Too tall, adjust width
            crop_w = int(crop_h * target_aspect)

        # Center on face
        center_x = x + w // 2
        center_y = y + h // 2

        crop_x1 = max(0, center_x - crop_w // 2)
        crop_y1 = max(0, center_y - crop_h // 2)
        crop_x2 = min(frame_w, crop_x1 + crop_w)
        crop_y2 = min(frame_h, crop_y1 + crop_h)

        # Crop
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize to target
        resized = cv2.resize(cropped, target_size)

        return resized

    def _center_crop(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Fallback: center crop when no face detected."""
        frame_h, frame_w = frame.shape[:2]
        target_w, target_h = target_size

        # Calculate crop region
        crop_h = frame_h
        crop_w = int(crop_h * (target_w / target_h))

        if crop_w > frame_w:
            crop_w = frame_w
            crop_h = int(crop_w * (target_h / target_w))

        x1 = (frame_w - crop_w) // 2
        y1 = (frame_h - crop_h) // 2

        cropped = frame[y1:y1+crop_h, x1:x1+crop_w]
        resized = cv2.resize(cropped, target_size)

        return resized
