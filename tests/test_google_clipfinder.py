import pytest
from unittest.mock import patch, MagicMock

from clipsai.clip.google_clipfinder import GoogleClipFinder


def test_invalid_clip_length():
    with pytest.raises(ValueError):
        GoogleClipFinder(clip_length=15)


def test_auto_uses_duration_when_no_scores():
    finder = GoogleClipFinder(clip_length="auto")
    with patch.object(finder, "_fetch_length_scores", return_value=None), patch.object(
        finder, "_fetch_clip_starts", return_value=[0]
    ) as mock_fetch:
        finder.find_clips("path.mp4", video_length=40)
        mock_fetch.assert_called_with("path.mp4", 30)


def test_auto_uses_scores_when_available():
    finder = GoogleClipFinder(clip_length="auto")
    scores = {30: 0.1, 60: 0.8, 90: 0.3}
    with patch.object(
        finder, "_fetch_length_scores", return_value=scores
    ), patch.object(finder, "_fetch_clip_starts", return_value=[0]) as mock_fetch:
        finder.find_clips("path.mp4", video_length=120)
        mock_fetch.assert_called_with("path.mp4", 60)


def test_explicit_clip_length_passed_through():
    finder = GoogleClipFinder(clip_length=90)
    with patch.object(finder, "_fetch_clip_starts", return_value=[0]) as mock_fetch:
        finder.find_clips("path.mp4", video_length=200)
        mock_fetch.assert_called_with("path.mp4", 90)
