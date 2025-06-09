"""Google-based clip finder stub.

This module provides ``GoogleClipFinder`` which uses Google's APIs to
suggest clips of fixed length. The clip length can be 30, 60, or 90
seconds. When ``clip_length='auto'`` an appropriate length is chosen
based on the video duration or API returned scores.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

from .clip import Clip


ClipLength = Literal[30, 60, 90, "auto"]


class GoogleClipFinder:
    """Find clips within a video using Google APIs."""

    def __init__(self, clip_length: ClipLength = "auto") -> None:
        if clip_length not in (30, 60, 90, "auto"):
            raise ValueError("clip_length must be 30, 60, 90, or 'auto'")
        self._clip_length = clip_length

    # -- Internal helpers -------------------------------------------------

    def _fetch_length_scores(self, video_file_path: str) -> Optional[dict[int, float]]:
        """Fetch score per clip length from Google API.

        In production this would call Google's API. During testing it is
        patched to return deterministic results.
        """
        raise NotImplementedError

    def _fetch_clip_starts(
        self, video_file_path: str, clip_length: int
    ) -> Iterable[float]:
        """Fetch clip start times for a fixed length from Google API."""
        raise NotImplementedError

    def _choose_auto_length(
        self, video_length: float, scores: Optional[dict[int, float]]
    ) -> int:
        if scores:
            best = max(scores, key=scores.get)
            if best in (30, 60, 90):
                return best
        if video_length <= 45:
            return 30
        if video_length <= 75:
            return 60
        return 90

    # -- Public API -------------------------------------------------------

    def find_clips(self, video_file_path: str, video_length: float) -> list[Clip]:
        """Return a list of :class:`Clip` objects for the video."""
        clip_length: int
        if self._clip_length == "auto":
            scores = self._fetch_length_scores(video_file_path)
            clip_length = self._choose_auto_length(video_length, scores)
        else:
            clip_length = int(self._clip_length)

        start_times = self._fetch_clip_starts(video_file_path, clip_length)
        clips = [
            Clip(start, min(start + clip_length, video_length), -1, -1)
            for start in start_times
        ]
        return clips
