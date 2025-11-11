"""
Clip variation generator for creating multiple versions of each clip.

Generates variations using:
- Temporal shifts (adjusting start time)
- Duration changes (15s, 30s, 45s, 60s)
- Aspect ratios (9:16, 1:1, 4:5)
- Smart cropping strategies
"""
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from itertools import product

from clipsai import Clip

logger = logging.getLogger(__name__)


@dataclass
class ClipVariation:
    """Container for a clip variation."""
    clip_id: str
    variation_id: str
    original_start: float
    original_end: float
    adjusted_start: float
    adjusted_end: float
    duration: float
    aspect_ratio: str
    temporal_shift: float
    crop_strategy: str
    metadata: Dict[str, Any]


class VariationGenerator:
    """
    Generates multiple variations of each clip.
    """

    def __init__(self,
                 temporal_shifts: List[float] = None,
                 durations: List[int] = None,
                 aspect_ratios: List[str] = None,
                 max_variations: int = 12):
        """
        Initialize variation generator.

        Parameters
        ----------
        temporal_shifts : List[float], optional
            Time shifts in seconds (e.g., [-1, -0.5, 0, 0.5, 1])
        durations : List[int], optional
            Target durations in seconds (e.g., [15, 30, 45, 60])
        aspect_ratios : List[str], optional
            Target aspect ratios (e.g., ["9:16", "1:1", "4:5"])
        max_variations : int
            Maximum variations per clip
        """
        self.temporal_shifts = temporal_shifts or [-1.0, -0.5, 0, 0.5, 1.0]
        self.durations = durations or [15, 30, 45, 60]
        self.aspect_ratios = aspect_ratios or ["9:16", "1:1", "4:5"]
        self.max_variations = max_variations

        # Crop strategies for different aspect ratios
        self.crop_strategies = {
            "9:16": "center_portrait",  # TikTok/Reels/Shorts
            "1:1": "center_square",     # Instagram feed
            "4:5": "center_portrait_wide",  # Instagram optimized
            "16:9": "letterbox",        # YouTube Shorts (landscape)
        }

    def generate_variations(self,
                           clip: Clip,
                           base_id: str,
                           source_duration: float,
                           source_aspect: str = "16:9") -> List[ClipVariation]:
        """
        Generate all variations for a clip.

        Parameters
        ----------
        clip : Clip
            Original clip from ClipFinder
        base_id : str
            Base identifier for this clip
        source_duration : float
            Total duration of source video
        source_aspect : str
            Source video aspect ratio

        Returns
        -------
        List[ClipVariation]
            All generated variations
        """
        variations = []
        variation_count = 0

        clip_duration = clip.end_time - clip.start_time

        # Generate combinations of temporal shifts and durations
        for temporal_shift, target_duration in product(self.temporal_shifts, self.durations):
            if variation_count >= self.max_variations:
                break

            # Skip if target duration is too different from clip length
            if abs(target_duration - clip_duration) > clip_duration * 0.8:
                continue

            # Calculate adjusted times
            adjusted_start = clip.start_time + temporal_shift
            adjusted_end = adjusted_start + target_duration

            # Validate bounds
            if adjusted_start < 0 or adjusted_end > source_duration:
                continue

            # Create variation for each aspect ratio
            for aspect_ratio in self.aspect_ratios:
                if variation_count >= self.max_variations:
                    break

                crop_strategy = self.crop_strategies.get(aspect_ratio, "center")

                variation = ClipVariation(
                    clip_id=base_id,
                    variation_id=f"{base_id}_t{temporal_shift:+.1f}_d{target_duration}_{aspect_ratio.replace(':', 'x')}",
                    original_start=clip.start_time,
                    original_end=clip.end_time,
                    adjusted_start=adjusted_start,
                    adjusted_end=adjusted_end,
                    duration=target_duration,
                    aspect_ratio=aspect_ratio,
                    temporal_shift=temporal_shift,
                    crop_strategy=crop_strategy,
                    metadata={
                        "original_duration": clip_duration,
                        "source_aspect": source_aspect,
                    }
                )

                variations.append(variation)
                variation_count += 1

        logger.info(f"Generated {len(variations)} variations for clip {base_id}")
        return variations

    def generate_smart_variations(self,
                                 clip: Clip,
                                 base_id: str,
                                 source_duration: float,
                                 hook_score: float = 5.0) -> List[ClipVariation]:
        """
        Generate variations using smart strategies based on hook score.

        Higher-scoring clips get more aggressive variations.
        Lower-scoring clips get more conservative variations (trying to find better hooks).

        Parameters
        ----------
        clip : Clip
            Original clip
        base_id : str
            Base identifier
        source_duration : float
            Source video duration
        hook_score : float
            VVSA hook score (0-10)

        Returns
        -------
        List[ClipVariation]
            Smart variations optimized for hook score
        """
        variations = []
        clip_duration = clip.end_time - clip.start_time

        # Strategy based on hook score
        if hook_score >= 7.0:
            # High score: keep hook, vary duration and aspect
            shifts = [0, -0.5, 0.5]  # Small adjustments
            durations = self.durations
        elif hook_score >= 5.0:
            # Medium score: try more temporal shifts
            shifts = [-1.0, -0.5, 0, 0.5, 1.0]
            durations = [d for d in self.durations if abs(d - clip_duration) < 30]
        else:
            # Low score: aggressive temporal shifts to find better hook
            shifts = [-2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
            durations = [d for d in self.durations if d <= clip_duration]

        variation_count = 0

        for temporal_shift in shifts:
            for target_duration in durations:
                if variation_count >= self.max_variations:
                    break

                adjusted_start = clip.start_time + temporal_shift
                adjusted_end = adjusted_start + target_duration

                # Validate bounds
                if adjusted_start < 0 or adjusted_end > source_duration:
                    continue

                # Skip if duration mismatch is too large
                if abs(target_duration - clip_duration) > clip_duration:
                    continue

                # For each valid time window, create aspect ratio variations
                for aspect_ratio in self.aspect_ratios:
                    if variation_count >= self.max_variations:
                        break

                    crop_strategy = self.crop_strategies.get(aspect_ratio, "center")

                    variation = ClipVariation(
                        clip_id=base_id,
                        variation_id=f"{base_id}_smart_t{temporal_shift:+.1f}_d{target_duration}_{aspect_ratio.replace(':', 'x')}",
                        original_start=clip.start_time,
                        original_end=clip.end_time,
                        adjusted_start=adjusted_start,
                        adjusted_end=adjusted_end,
                        duration=target_duration,
                        aspect_ratio=aspect_ratio,
                        temporal_shift=temporal_shift,
                        crop_strategy=crop_strategy,
                        metadata={
                            "original_duration": clip_duration,
                            "hook_score": hook_score,
                            "strategy": "smart_variation",
                        }
                    )

                    variations.append(variation)
                    variation_count += 1

        logger.info(
            f"Generated {len(variations)} smart variations for clip {base_id} "
            f"(hook_score={hook_score:.1f})"
        )
        return variations

    def parse_aspect_ratio(self, aspect_str: str) -> Tuple[int, int]:
        """
        Parse aspect ratio string to width:height tuple.

        Parameters
        ----------
        aspect_str : str
            Aspect ratio like "9:16" or "16:9"

        Returns
        -------
        Tuple[int, int]
            (width_ratio, height_ratio)
        """
        try:
            parts = aspect_str.split(":")
            return (int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            logger.warning(f"Invalid aspect ratio: {aspect_str}, using 9:16")
            return (9, 16)

    def calculate_crop_box(self,
                          source_width: int,
                          source_height: int,
                          target_aspect: str,
                          strategy: str = "center") -> Dict[str, int]:
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
        strategy : str
            Cropping strategy: "center", "top", "bottom", "left", "right"

        Returns
        -------
        Dict[str, int]
            Crop box with x, y, width, height
        """
        target_w, target_h = self.parse_aspect_ratio(target_aspect)
        target_ratio = target_w / target_h
        source_ratio = source_width / source_height

        if source_ratio > target_ratio:
            # Source is wider, crop horizontally
            new_width = int(source_height * target_ratio)
            new_height = source_height

            if strategy == "left":
                x = 0
            elif strategy == "right":
                x = source_width - new_width
            else:  # center
                x = (source_width - new_width) // 2

            y = 0

        else:
            # Source is taller, crop vertically
            new_width = source_width
            new_height = int(source_width / target_ratio)

            x = 0

            if strategy == "top":
                y = 0
            elif strategy == "bottom":
                y = source_height - new_height
            else:  # center
                y = (source_height - new_height) // 2

        return {
            "x": max(0, x),
            "y": max(0, y),
            "width": new_width,
            "height": new_height,
        }

    def optimize_variations(self,
                           variations: List[ClipVariation],
                           target_count: int = 300) -> List[ClipVariation]:
        """
        Optimize variation list to reach target count.

        Balances across:
        - Aspect ratios (platform diversity)
        - Durations (content diversity)
        - Temporal shifts (hook optimization)

        Parameters
        ----------
        variations : List[ClipVariation]
            All generated variations
        target_count : int
            Target number of variations

        Returns
        -------
        List[ClipVariation]
            Optimized subset
        """
        if len(variations) <= target_count:
            return variations

        # Strategy: Ensure balanced distribution
        # Priority: aspect_ratio > duration > temporal_shift

        optimized = []

        # Group by aspect ratio
        by_aspect = {}
        for var in variations:
            aspect = var.aspect_ratio
            if aspect not in by_aspect:
                by_aspect[aspect] = []
            by_aspect[aspect].append(var)

        # Distribute evenly across aspect ratios
        per_aspect = target_count // len(by_aspect)

        for aspect, aspect_vars in by_aspect.items():
            # Further distribute by duration
            by_duration = {}
            for var in aspect_vars:
                dur = var.duration
                if dur not in by_duration:
                    by_duration[dur] = []
                by_duration[dur].append(var)

            per_duration = per_aspect // len(by_duration)

            for dur, dur_vars in by_duration.items():
                # Take up to per_duration variations
                optimized.extend(dur_vars[:per_duration])

        # Fill remaining slots with best temporal shifts (centered)
        if len(optimized) < target_count:
            remaining = [v for v in variations if v not in optimized]
            # Sort by temporal shift proximity to 0
            remaining.sort(key=lambda v: abs(v.temporal_shift))
            optimized.extend(remaining[:target_count - len(optimized)])

        logger.info(f"Optimized {len(variations)} variations to {len(optimized)}")
        return optimized[:target_count]


def create_variation_generator(config) -> VariationGenerator:
    """
    Factory function to create variation generator from config.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    VariationGenerator
        Configured generator instance
    """
    return VariationGenerator(
        temporal_shifts=config.get("variations.temporal_shifts"),
        durations=config.get("variations.durations"),
        aspect_ratios=config.get("variations.aspect_ratios"),
        max_variations=config.get("variations.max_variations_per_clip", 12)
    )
