"""
Clip export and rendering using ffmpeg.
Handles video cutting, cropping, reframing, and thumbnail generation.
"""
import logging
import os
import subprocess
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ClipExporter:
    """
    Exports video clips using ffmpeg.
    Handles cutting, cropping, aspect ratio conversion, and thumbnails.
    """

    def __init__(self,
                 video_codec: str = "libx264",
                 audio_codec: str = "aac",
                 preset: str = "medium",
                 crf: int = 23):
        """
        Initialize clip exporter.

        Parameters
        ----------
        video_codec : str
            Video codec (default: libx264)
        audio_codec : str
            Audio codec (default: aac)
        preset : str
            Encoding preset: ultrafast, fast, medium, slow, veryslow
        crf : int
            Constant rate factor (quality): 0-51, lower = better quality
        """
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.preset = preset
        self.crf = crf

    def export_clip(self,
                   source_path: str,
                   output_path: str,
                   start_time: float,
                   end_time: float,
                   crop_box: Optional[Dict[str, int]] = None,
                   target_width: Optional[int] = None,
                   target_height: Optional[int] = None,
                   subtitle_path: Optional[str] = None,
                   subtitle_filter: Optional[str] = None) -> str:
        """
        Export a clip from source video.

        Parameters
        ----------
        source_path : str
            Path to source video
        output_path : str
            Output path for clip
        start_time : float
            Start time in seconds
        end_time : float
            End time in seconds
        crop_box : Dict[str, int], optional
            Crop box with x, y, width, height
        target_width : int, optional
            Target width (for aspect ratio conversion)
        target_height : int, optional
            Target height (for aspect ratio conversion)
        subtitle_path : str, optional
            Path to subtitle file for burn-in
        subtitle_filter : str, optional
            Custom subtitle filter string

        Returns
        -------
        str
            Path to exported clip
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y"]  # -y to overwrite

        # Input file
        cmd.extend(["-i", source_path])

        # Time range (using -ss before input for faster seeking)
        duration = end_time - start_time
        cmd.extend(["-ss", str(start_time), "-t", str(duration)])

        # Build video filter chain
        filters = []

        # Crop filter
        if crop_box:
            crop_filter = (
                f"crop={crop_box['width']}:{crop_box['height']}:"
                f"{crop_box['x']}:{crop_box['y']}"
            )
            filters.append(crop_filter)

        # Scale filter
        if target_width and target_height:
            # Use scale with SAR correction for clean output
            scale_filter = f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
            filters.append(scale_filter)

        # Subtitle filter
        if subtitle_path and os.path.exists(subtitle_path):
            if subtitle_filter:
                filters.append(subtitle_filter)
            else:
                # Default subtitle filter
                escaped_path = subtitle_path.replace('\\', '\\\\').replace(':', '\\:')
                filters.append(f"subtitles={escaped_path}")

        # Apply filters if any
        if filters:
            filter_chain = ",".join(filters)
            cmd.extend(["-vf", filter_chain])

        # Video encoding settings
        cmd.extend([
            "-c:v", self.video_codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
        ])

        # Audio encoding
        cmd.extend(["-c:a", self.audio_codec, "-b:a", "128k"])

        # Output file
        cmd.append(output_path)

        # Execute ffmpeg
        try:
            logger.info(f"Exporting clip: {output_path}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Clip exported successfully: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise RuntimeError(f"Failed to export clip: {e.stderr}")

    def generate_thumbnail(self,
                          video_path: str,
                          output_path: str,
                          timestamp: float = 1.0,
                          width: int = 1280) -> str:
        """
        Generate thumbnail from video.

        Parameters
        ----------
        video_path : str
            Path to video file
        output_path : str
            Output path for thumbnail
        timestamp : float
            Time in seconds to capture frame
        width : int
            Thumbnail width (maintains aspect ratio)

        Returns
        -------
        str
            Path to generated thumbnail
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            output_path
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Thumbnail generated: {output_path}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Thumbnail generation failed: {e.stderr}")
            raise RuntimeError(f"Failed to generate thumbnail: {e.stderr}")

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video metadata using ffprobe.

        Parameters
        ----------
        video_path : str
            Path to video file

        Returns
        -------
        Dict[str, Any]
            Video metadata: width, height, duration, fps, etc.
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            import json
            data = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                raise ValueError("No video stream found")

            # Parse frame rate safely without eval()
            def parse_frame_rate(r_frame_rate: str) -> float:
                """Safely parse ffprobe r_frame_rate string (e.g., '30000/1001')"""
                try:
                    if '/' in r_frame_rate:
                        num, den = r_frame_rate.split('/')
                        return float(num) / float(den)
                    else:
                        return float(r_frame_rate)
                except (ValueError, ZeroDivisionError):
                    return 30.0  # Default fallback

            return {
                "width": int(video_stream.get("width", 0)),
                "height": int(video_stream.get("height", 0)),
                "duration": float(data.get("format", {}).get("duration", 0)),
                "fps": parse_frame_rate(video_stream.get("r_frame_rate", "30/1")),
                "codec": video_stream.get("codec_name", "unknown"),
            }

        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get video info: {e}")
            return {}

    def calculate_aspect_crop(self,
                             source_width: int,
                             source_height: int,
                             target_aspect: str) -> Dict[str, int]:
        """
        Calculate crop box for aspect ratio conversion.

        Parameters
        ----------
        source_width : int
            Source video width
        source_height : int
            Source video height
        target_aspect : str
            Target aspect ratio (e.g., "9:16")

        Returns
        -------
        Dict[str, int]
            Crop box with x, y, width, height
        """
        # Parse target aspect ratio
        parts = target_aspect.split(":")
        target_w = int(parts[0])
        target_h = int(parts[1])
        target_ratio = target_w / target_h

        source_ratio = source_width / source_height

        if source_ratio > target_ratio:
            # Source is wider, crop horizontally
            new_width = int(source_height * target_ratio)
            new_height = source_height
            x = (source_width - new_width) // 2
            y = 0
        else:
            # Source is taller, crop vertically
            new_width = source_width
            new_height = int(source_width / target_ratio)
            x = 0
            y = (source_height - new_height) // 2

        return {
            "x": max(0, x),
            "y": max(0, y),
            "width": new_width,
            "height": new_height,
        }

    def batch_export(self,
                    export_tasks: list,
                    parallel: int = 1) -> list:
        """
        Export multiple clips (optionally in parallel).

        Parameters
        ----------
        export_tasks : list
            List of export task dictionaries
        parallel : int
            Number of parallel exports (default: 1, sequential)

        Returns
        -------
        list
            List of exported file paths
        """
        results = []

        if parallel == 1:
            # Sequential export
            for task in export_tasks:
                try:
                    output = self.export_clip(**task)
                    results.append(output)
                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    results.append(None)
        else:
            # Parallel export (basic implementation)
            # For production, use multiprocessing or concurrent.futures
            logger.warning("Parallel export not fully implemented, using sequential")
            for task in export_tasks:
                try:
                    output = self.export_clip(**task)
                    results.append(output)
                except Exception as e:
                    logger.error(f"Export failed: {e}")
                    results.append(None)

        successful = sum(1 for r in results if r is not None)
        logger.info(f"Batch export complete: {successful}/{len(export_tasks)} successful")
        return results

    def create_export_task(self,
                          variation,
                          source_path: str,
                          output_dir: str,
                          source_info: Dict[str, Any],
                          subtitle_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an export task dict for a variation.

        Parameters
        ----------
        variation : ClipVariation
            Variation to export
        source_path : str
            Source video path
        output_dir : str
            Output directory
        source_info : Dict[str, Any]
            Source video metadata
        subtitle_path : str, optional
            Subtitle file path

        Returns
        -------
        Dict[str, Any]
            Export task parameters
        """
        # Calculate crop box
        crop_box = self.calculate_aspect_crop(
            source_info["width"],
            source_info["height"],
            variation.aspect_ratio
        )

        # Parse target resolution
        aspect_parts = variation.aspect_ratio.split(":")
        aspect_w = int(aspect_parts[0])
        aspect_h = int(aspect_parts[1])

        # Target resolution (720p height for 9:16, etc.)
        if aspect_h >= aspect_w:
            # Portrait or square
            target_height = 1280
            target_width = int(target_height * aspect_w / aspect_h)
        else:
            # Landscape
            target_width = 1280
            target_height = int(target_width * aspect_h / aspect_w)

        output_path = os.path.join(output_dir, f"{variation.variation_id}.mp4")

        return {
            "source_path": source_path,
            "output_path": output_path,
            "start_time": variation.adjusted_start,
            "end_time": variation.adjusted_end,
            "crop_box": crop_box,
            "target_width": target_width,
            "target_height": target_height,
            "subtitle_path": subtitle_path,
        }


def create_exporter(config) -> ClipExporter:
    """
    Factory function to create exporter from config.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    ClipExporter
        Configured exporter instance
    """
    return ClipExporter(
        video_codec=config.get("export.video_codec", "libx264"),
        audio_codec=config.get("export.audio_codec", "aac"),
        preset=config.get("export.preset", "medium"),
        crf=config.get("export.crf", 23)
    )
