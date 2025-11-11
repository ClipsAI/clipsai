"""
Canvas Rendering for Matrix Processing
Handles 3 reframe styles: original, flipped, blurry_bg
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CanvasRenderer:
    """
    Renders clips with different canvas styles.

    Styles:
    1. Original: Horizontal as-is
    2. Flipped: Horizontal mirror
    3. Blurry BG: Background blurred, foreground centered at 40%
    """

    def __init__(self):
        self.target_width = 1080  # 9:16 width
        self.target_height = 1920  # 9:16 height
        self.blur_amount = 50  # 50% blur

    def render_original(
        self,
        input_path: str,
        output_path: str
    ) -> str:
        """
        Render original style (horizontal as-is).
        Just converts to 9:16 with letterboxing.
        """
        try:
            cap = cv2.VideoCapture(input_path)

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (self.target_width, self.target_height)
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Letterbox to 9:16
                rendered = self._letterbox(frame)
                writer.write(rendered)

            cap.release()
            writer.release()

            logger.info(f"Rendered original: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Original render failed: {e}")
            raise

    def render_flipped(
        self,
        input_path: str,
        output_path: str
    ) -> str:
        """
        Render flipped style (horizontal mirror).
        """
        try:
            cap = cv2.VideoCapture(input_path)

            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (self.target_width, self.target_height)
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip horizontally
                flipped = cv2.flip(frame, 1)

                # Letterbox to 9:16
                rendered = self._letterbox(flipped)
                writer.write(rendered)

            cap.release()
            writer.release()

            logger.info(f"Rendered flipped: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Flipped render failed: {e}")
            raise

    def render_blurry_bg(
        self,
        input_path: str,
        output_path: str
    ) -> str:
        """
        Render blurry background style.

        - Background: Original at 100%, 50% blur
        - Foreground: Original centered at 40% scale
        - Final: 9:16 vertical
        """
        try:
            cap = cv2.VideoCapture(input_path)

            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (self.target_width, self.target_height)
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Create canvas
                canvas = np.zeros(
                    (self.target_height, self.target_width, 3),
                    dtype=np.uint8
                )

                # 1. Background: Blurred, full coverage
                bg = self._create_blurred_background(frame)
                canvas = bg

                # 2. Foreground: Original at 40% scale, centered
                fg = self._create_foreground(frame, scale=0.4)
                canvas = self._overlay_foreground(canvas, fg)

                writer.write(canvas)

            cap.release()
            writer.release()

            logger.info(f"Rendered blurry_bg: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Blurry BG render failed: {e}")
            raise

    def _letterbox(self, frame: np.ndarray) -> np.ndarray:
        """Add letterboxing to fit 9:16."""
        h, w = frame.shape[:2]

        # Calculate scaling
        scale = min(self.target_width / w, self.target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h))

        # Create canvas
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        # Center
        y_offset = (self.target_height - new_h) // 2
        x_offset = (self.target_width - new_w) // 2

        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

        return canvas

    def _create_blurred_background(self, frame: np.ndarray) -> np.ndarray:
        """Create blurred background that fills 9:16 canvas."""
        # Resize to fill canvas
        resized = cv2.resize(frame, (self.target_width, self.target_height))

        # Apply Gaussian blur (kernel size must be odd)
        kernel_size = 51
        blurred = cv2.GaussianBlur(resized, (kernel_size, kernel_size), 0)

        return blurred

    def _create_foreground(self, frame: np.ndarray, scale: float) -> np.ndarray:
        """Create foreground at specified scale."""
        h, w = frame.shape[:2]

        new_w = int(w * scale)
        new_h = int(h * scale)

        # Ensure dimensions fit in canvas
        if new_w > self.target_width:
            scale_down = self.target_width / new_w
            new_w = int(new_w * scale_down)
            new_h = int(new_h * scale_down)

        if new_h > self.target_height:
            scale_down = self.target_height / new_h
            new_w = int(new_w * scale_down)
            new_h = int(new_h * scale_down)

        foreground = cv2.resize(frame, (new_w, new_h))

        return foreground

    def _overlay_foreground(
        self,
        canvas: np.ndarray,
        foreground: np.ndarray
    ) -> np.ndarray:
        """Overlay foreground centered on canvas."""
        fg_h, fg_w = foreground.shape[:2]

        # Center position
        y_offset = (self.target_height - fg_h) // 2
        x_offset = (self.target_width - fg_w) // 2

        # Overlay
        canvas[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w] = foreground

        return canvas


def render_all_styles(
    input_path: str,
    output_dir: str,
    clip_id: str
) -> dict:
    """
    Render all 3 canvas styles for a clip.

    Args:
        input_path: Input video path
        output_dir: Output directory
        clip_id: Clip identifier

    Returns:
        Dict with paths to all 3 rendered versions
    """
    renderer = CanvasRenderer()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "original": renderer.render_original(
            input_path,
            str(output_dir / f"{clip_id}_original.mp4")
        ),
        "flipped": renderer.render_flipped(
            input_path,
            str(output_dir / f"{clip_id}_flipped.mp4")
        ),
        "blurry_bg": renderer.render_blurry_bg(
            input_path,
            str(output_dir / f"{clip_id}_blurry_bg.mp4")
        )
    }

    return results
