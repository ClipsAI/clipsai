"""
Temporal Variation Generator
Creates base, +4s, +35s variations with random frame offsets
"""
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemporalVariation:
    """Container for temporal variation metadata."""
    variation_id: str
    variation_type: str  # 'base', '+4s', '+35s'
    start_time: float
    end_time: float
    duration: float
    frame_offset: float  # Random 5-10 frames for duplicate detection


class TemporalVariationGenerator:
    """
    Generates temporal variations from base clips.

    Creates 3 variations per clip:
    - Base: Original duration
    - +4s: Extend end by 4 seconds
    - +35s: Extend end by 35 seconds (or max boundary)

    All variations get random 5-10 frame offset at START
    to defeat platform duplicate detection.
    """

    def __init__(self, fps: float = 60.0):
        """
        Initialize generator.

        Args:
            fps: Video frames per second (for frame offset calculation)
        """
        self.fps = fps
        self.min_frame_offset = 5
        self.max_frame_offset = 10

    def generate_variations(
        self,
        clip_id: str,
        base_start: float,
        base_end: float,
        video_duration: float
    ) -> List[TemporalVariation]:
        """
        Generate all 3 temporal variations.

        Args:
            clip_id: Base clip identifier
            base_start: Original clip start time
            base_end: Original clip end time
            video_duration: Total video duration (for boundary checking)

        Returns:
            List of 3 TemporalVariation objects
        """
        variations = []

        # 1. BASE VARIATION
        base_offset = self._random_frame_offset()
        variations.append(TemporalVariation(
            variation_id=f"{clip_id}_base",
            variation_type="base",
            start_time=base_start + base_offset,
            end_time=base_end,
            duration=base_end - base_start,
            frame_offset=base_offset
        ))

        # 2. +4S VARIATION
        plus4_offset = self._random_frame_offset()
        plus4_end = min(base_end + 4.0, video_duration)
        variations.append(TemporalVariation(
            variation_id=f"{clip_id}_plus4s",
            variation_type="+4s",
            start_time=base_start + plus4_offset,
            end_time=plus4_end,
            duration=plus4_end - (base_start + plus4_offset),
            frame_offset=plus4_offset
        ))

        # 3. +35S VARIATION
        plus35_offset = self._random_frame_offset()
        plus35_end = min(base_end + 35.0, video_duration)
        variations.append(TemporalVariation(
            variation_id=f"{clip_id}_plus35s",
            variation_type="+35s",
            start_time=base_start + plus35_offset,
            end_time=plus35_end,
            duration=plus35_end - (base_start + plus35_offset),
            frame_offset=plus35_offset
        ))

        logger.info(f"Generated 3 temporal variations for {clip_id}")

        return variations

    def _random_frame_offset(self) -> float:
        """
        Generate random frame offset in seconds.

        Returns random offset between 5-10 frames.
        At 60fps: 0.083s to 0.167s
        """
        frames = random.randint(self.min_frame_offset, self.max_frame_offset)
        offset_seconds = frames / self.fps
        return offset_seconds

    def generate_matrix_variations(
        self,
        clip_id: str,
        base_start: float,
        base_end: float,
        video_duration: float,
        reframe_styles: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate complete 3x3 matrix (3 temporal × 3 reframe).

        Args:
            clip_id: Base clip ID
            base_start: Start time
            base_end: End time
            video_duration: Total video duration
            reframe_styles: List of reframe styles ['original', 'flipped', 'blurry_bg']

        Returns:
            List of 9 variation dictionaries
        """
        # Get temporal variations
        temporal_vars = self.generate_variations(
            clip_id,
            base_start,
            base_end,
            video_duration
        )

        # Create matrix (3 temporal × 3 reframe = 9 total)
        matrix = []

        for temp_var in temporal_vars:
            for reframe_style in reframe_styles:
                variation = {
                    "variation_id": f"{temp_var.variation_id}_{reframe_style}",
                    "clip_id": clip_id,
                    "temporal_type": temp_var.variation_type,
                    "reframe_style": reframe_style,
                    "start_time": temp_var.start_time,
                    "end_time": temp_var.end_time,
                    "duration": temp_var.duration,
                    "frame_offset": temp_var.frame_offset
                }
                matrix.append(variation)

        logger.info(f"Generated {len(matrix)} matrix variations for {clip_id}")

        return matrix


def create_variations_for_clip(
    clip_id: str,
    start: float,
    end: float,
    video_duration: float,
    reframe_styles: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Helper function to create all variations for a clip.

    Args:
        clip_id: Clip identifier
        start: Start time
        end: End time
        video_duration: Total video duration
        reframe_styles: Reframe styles (default: all 3)

    Returns:
        List of variation dictionaries
    """
    if reframe_styles is None:
        reframe_styles = ['original', 'flipped', 'blurry_bg']

    generator = TemporalVariationGenerator()

    return generator.generate_matrix_variations(
        clip_id,
        start,
        end,
        video_duration,
        reframe_styles
    )
