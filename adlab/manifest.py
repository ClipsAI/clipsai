"""
Manifest generation and JSONL export for clip metadata.
Provides deduplication and batch writing utilities.
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ManifestWriter:
    """
    Writes clip metadata to JSONL manifest files.
    Supports deduplication and incremental updates.
    """

    def __init__(self, output_dir: str):
        """
        Initialize manifest writer.

        Parameters
        ----------
        output_dir : str
            Output directory for manifest files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_manifest(self,
                      clips_data: List[Dict[str, Any]],
                      manifest_path: str,
                      deduplicate: bool = True) -> str:
        """
        Write clips metadata to JSONL manifest.

        Parameters
        ----------
        clips_data : List[Dict[str, Any]]
            List of clip metadata dictionaries
        manifest_path : str
            Output path for manifest file
        deduplicate : bool
            Whether to deduplicate entries

        Returns
        -------
        str
            Path to written manifest
        """
        if deduplicate:
            clips_data = self.deduplicate_clips(clips_data)

        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

        with open(manifest_path, 'w', encoding='utf-8') as f:
            for clip in clips_data:
                # Add metadata
                clip["_manifest_version"] = "1.0"
                clip["_created_at"] = datetime.now().isoformat()

                json_line = json.dumps(clip, ensure_ascii=False)
                f.write(json_line + '\n')

        logger.info(f"Wrote {len(clips_data)} clips to manifest: {manifest_path}")
        return manifest_path

    def read_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        """
        Read clips metadata from JSONL manifest.

        Parameters
        ----------
        manifest_path : str
            Path to manifest file

        Returns
        -------
        List[Dict[str, Any]]
            List of clip metadata
        """
        clips = []

        if not os.path.exists(manifest_path):
            logger.warning(f"Manifest not found: {manifest_path}")
            return clips

        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    clip_data = json.loads(line)
                    clips.append(clip_data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")

        logger.info(f"Read {len(clips)} clips from manifest: {manifest_path}")
        return clips

    def append_to_manifest(self,
                          clips_data: List[Dict[str, Any]],
                          manifest_path: str,
                          deduplicate: bool = True) -> str:
        """
        Append clips to existing manifest.

        Parameters
        ----------
        clips_data : List[Dict[str, Any]]
            New clips to append
        manifest_path : str
            Path to manifest file
        deduplicate : bool
            Whether to deduplicate against existing entries

        Returns
        -------
        str
            Path to updated manifest
        """
        # Read existing clips
        existing_clips = []
        if os.path.exists(manifest_path):
            existing_clips = self.read_manifest(manifest_path)

        # Combine
        all_clips = existing_clips + clips_data

        # Deduplicate if requested
        if deduplicate:
            all_clips = self.deduplicate_clips(all_clips)

        # Write updated manifest
        return self.write_manifest(all_clips, manifest_path, deduplicate=False)

    def deduplicate_clips(self, clips_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate clips based on video path and timing.

        Parameters
        ----------
        clips_data : List[Dict[str, Any]]
            Clips to deduplicate

        Returns
        -------
        List[Dict[str, Any]]
            Deduplicated clips
        """
        seen: Set[str] = set()
        unique_clips = []

        for clip in clips_data:
            # Create hash from key properties
            clip_hash = self._compute_clip_hash(clip)

            if clip_hash not in seen:
                seen.add(clip_hash)
                unique_clips.append(clip)
            else:
                logger.debug(f"Skipping duplicate clip: {clip.get('variation_id', 'unknown')}")

        removed = len(clips_data) - len(unique_clips)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate clips")

        return unique_clips

    def _compute_clip_hash(self, clip: Dict[str, Any]) -> str:
        """
        Compute hash for clip deduplication.

        Uses: video_path, start_time, end_time, aspect_ratio

        Parameters
        ----------
        clip : Dict[str, Any]
            Clip metadata

        Returns
        -------
        str
            Hash string
        """
        # Extract key fields
        video_path = clip.get("video_path", "")
        start_time = clip.get("start_time", 0)
        end_time = clip.get("end_time", 0)
        aspect_ratio = clip.get("aspect_ratio", "")

        # Create hash input
        hash_input = f"{video_path}|{start_time:.2f}|{end_time:.2f}|{aspect_ratio}"

        return hashlib.md5(hash_input.encode()).hexdigest()

    def create_clip_entry(self,
                         variation,
                         video_path: str,
                         source_video: str,
                         hook_score,
                         title_variants: list,
                         captions: Dict[str, str],
                         thumbnail_path: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized clip manifest entry.

        Parameters
        ----------
        variation : ClipVariation
            Clip variation object
        video_path : str
            Path to exported video file
        source_video : str
            Path to source video
        hook_score : HookScore
            VVSA hook score
        title_variants : list
            List of TitleVariant objects
        captions : Dict[str, str]
            Caption file paths (srt, vtt)
        thumbnail_path : str, optional
            Thumbnail image path
        metadata : Dict[str, Any], optional
            Additional metadata

        Returns
        -------
        Dict[str, Any]
            Clip manifest entry
        """
        entry = {
            # Identity
            "clip_id": variation.clip_id,
            "variation_id": variation.variation_id,

            # Video files
            "video_path": video_path,
            "source_video": source_video,
            "thumbnail_path": thumbnail_path,

            # Timing
            "start_time": variation.adjusted_start,
            "end_time": variation.adjusted_end,
            "duration": variation.duration,
            "temporal_shift": variation.temporal_shift,

            # Format
            "aspect_ratio": variation.aspect_ratio,
            "crop_strategy": variation.crop_strategy,

            # Hook scoring
            "hook_score": {
                "overall": hook_score.overall_score,
                "visual": hook_score.visual_score,
                "audio": hook_score.audio_score,
                "text": hook_score.text_score,
                "llm": hook_score.llm_score,
                "reasoning": hook_score.reasoning,
                "strengths": hook_score.strengths,
                "improvements": hook_score.improvements,
            },

            # Titles (A/B variants)
            "titles": [
                {
                    "variant": tv.variant_id,
                    "text": tv.text,
                    "hook_style": tv.hook_style,
                    "target_audience": tv.target_audience,
                    "predicted_ctr": tv.predicted_ctr,
                    "tags": tv.tags,
                }
                for tv in title_variants
            ],

            # Captions
            "captions": captions,

            # Metadata
            "metadata": metadata or {},
        }

        return entry

    def generate_summary(self, manifest_path: str) -> Dict[str, Any]:
        """
        Generate summary statistics from manifest.

        Parameters
        ----------
        manifest_path : str
            Path to manifest file

        Returns
        -------
        Dict[str, Any]
            Summary statistics
        """
        clips = self.read_manifest(manifest_path)

        if not clips:
            return {"total_clips": 0}

        # Calculate statistics
        hook_scores = [c.get("hook_score", {}).get("overall", 0) for c in clips]
        durations = [c.get("duration", 0) for c in clips]

        aspect_ratios = {}
        for clip in clips:
            ar = clip.get("aspect_ratio", "unknown")
            aspect_ratios[ar] = aspect_ratios.get(ar, 0) + 1

        summary = {
            "total_clips": len(clips),
            "unique_base_clips": len(set(c.get("clip_id") for c in clips)),
            "hook_score_avg": sum(hook_scores) / len(hook_scores) if hook_scores else 0,
            "hook_score_min": min(hook_scores) if hook_scores else 0,
            "hook_score_max": max(hook_scores) if hook_scores else 0,
            "duration_avg": sum(durations) / len(durations) if durations else 0,
            "duration_min": min(durations) if durations else 0,
            "duration_max": max(durations) if durations else 0,
            "aspect_ratios": aspect_ratios,
            "created_at": datetime.now().isoformat(),
        }

        return summary

    def write_summary(self, manifest_path: str, summary_path: Optional[str] = None) -> str:
        """
        Write summary statistics to JSON file.

        Parameters
        ----------
        manifest_path : str
            Path to manifest file
        summary_path : str, optional
            Output path for summary (default: manifest_path with .summary.json)

        Returns
        -------
        str
            Path to summary file
        """
        if summary_path is None:
            summary_path = manifest_path.replace('.jsonl', '.summary.json')

        summary = self.generate_summary(manifest_path)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary written to: {summary_path}")
        return summary_path

    def filter_by_score(self,
                       clips_data: List[Dict[str, Any]],
                       min_score: float = 6.0) -> List[Dict[str, Any]]:
        """
        Filter clips by minimum hook score.

        Parameters
        ----------
        clips_data : List[Dict[str, Any]]
            Clips to filter
        min_score : float
            Minimum hook score threshold

        Returns
        -------
        List[Dict[str, Any]]
            Filtered clips
        """
        filtered = [
            clip for clip in clips_data
            if clip.get("hook_score", {}).get("overall", 0) >= min_score
        ]

        logger.info(f"Filtered to {len(filtered)}/{len(clips_data)} clips with score >= {min_score}")
        return filtered

    def sort_by_score(self, clips_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sort clips by hook score (descending).

        Parameters
        ----------
        clips_data : List[Dict[str, Any]]
            Clips to sort

        Returns
        -------
        List[Dict[str, Any]]
            Sorted clips
        """
        return sorted(
            clips_data,
            key=lambda c: c.get("hook_score", {}).get("overall", 0),
            reverse=True
        )

    def export_csv(self, manifest_path: str, csv_path: Optional[str] = None) -> str:
        """
        Export manifest to CSV format.

        Parameters
        ----------
        manifest_path : str
            Path to JSONL manifest
        csv_path : str, optional
            Output CSV path

        Returns
        -------
        str
            Path to CSV file
        """
        if csv_path is None:
            csv_path = manifest_path.replace('.jsonl', '.csv')

        clips = self.read_manifest(manifest_path)

        if not clips:
            logger.warning("No clips to export to CSV")
            return csv_path

        import csv

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Extract flattened fields
            fieldnames = [
                "variation_id", "video_path", "start_time", "end_time", "duration",
                "aspect_ratio", "hook_score", "title_A", "title_B"
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for clip in clips:
                row = {
                    "variation_id": clip.get("variation_id", ""),
                    "video_path": clip.get("video_path", ""),
                    "start_time": clip.get("start_time", 0),
                    "end_time": clip.get("end_time", 0),
                    "duration": clip.get("duration", 0),
                    "aspect_ratio": clip.get("aspect_ratio", ""),
                    "hook_score": clip.get("hook_score", {}).get("overall", 0),
                }

                # Add title variants
                titles = clip.get("titles", [])
                if len(titles) > 0:
                    row["title_A"] = titles[0].get("text", "")
                if len(titles) > 1:
                    row["title_B"] = titles[1].get("text", "")

                writer.writerow(row)

        logger.info(f"Exported CSV: {csv_path}")
        return csv_path


def create_manifest_writer(config) -> ManifestWriter:
    """
    Factory function to create manifest writer from config.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    ManifestWriter
        Configured writer instance
    """
    output_dir = config.get("export.output_dir", "./output")
    return ManifestWriter(output_dir=output_dir)
