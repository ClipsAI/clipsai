from unittest.mock import MagicMock, patch

import pytest

from clipsai.clip.clip import Clip
from clipsai.clip.exceptions import ClipFinderError
from clipsai.clip.pegasus_clipfinder import (
    PegasusClipFinder,
    PegasusClipFinderConfigManager,
)
from clipsai.transcribe.transcription import Transcription


@pytest.fixture
def config_manager():
    return PegasusClipFinderConfigManager()


@pytest.fixture
def transcription():
    transcription = MagicMock(spec=Transcription)
    transcription.end_time = 600.0
    # map a start time to its char index and an end time to its char index
    transcription.find_char_index.side_effect = lambda t, kind: int(t)
    return transcription


@pytest.fixture
def finder():
    return PegasusClipFinder(api_key="test-key", min_clip_duration=15, max_clips=5)


# --- config validation ---
def test_config_manager_valid_config(config_manager):
    config = {
        "model_name": "pegasus1.5",
        "min_clip_duration": 15,
        "max_clip_duration": 900,
        "max_clips": 10,
    }
    assert config_manager.check_valid_config(config) is None


def test_config_manager_invalid_model(config_manager):
    config = {
        "model_name": "not-a-model",
        "min_clip_duration": 15,
        "max_clip_duration": 900,
        "max_clips": 10,
    }
    assert isinstance(config_manager.check_valid_config(config), str)


def test_config_manager_invalid_durations(config_manager):
    config = {
        "model_name": "pegasus1.5",
        "min_clip_duration": 100,
        "max_clip_duration": 50,
        "max_clips": 10,
    }
    assert isinstance(config_manager.check_valid_config(config), str)


def test_config_manager_missing_keys(config_manager):
    assert isinstance(config_manager.check_valid_config({}), str)


# --- constructor ---
def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("TWELVELABS_API_KEY", raising=False)
    with pytest.raises(ClipFinderError):
        PegasusClipFinder()


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TWELVELABS_API_KEY", "env-key")
    assert PegasusClipFinder()._api_key == "env-key"


# --- response parsing ---
def test_parse_plain_json(finder):
    assert finder._parse_response('[{"start_time": 1, "end_time": 2}]') == [
        {"start_time": 1, "end_time": 2}
    ]


def test_parse_json_wrapped_in_prose(finder):
    text = 'Here are the clips:\n```json\n[{"start_time": 1, "end_time": 2}]\n```'
    assert finder._parse_response(text) == [{"start_time": 1, "end_time": 2}]


def test_parse_empty_raises(finder):
    with pytest.raises(ClipFinderError):
        finder._parse_response("")


def test_parse_no_array_raises(finder):
    with pytest.raises(ClipFinderError):
        finder._parse_response("no json here")


# --- ranges to clips ---
def test_ranges_to_clips_maps_times_to_chars(finder, transcription):
    ranges = [{"start_time": 10.0, "end_time": 40.0}]
    clips = finder._ranges_to_clips(ranges, transcription)
    assert clips == [Clip(10.0, 40.0, 10, 40)]


def test_ranges_to_clips_filters_too_short(finder, transcription):
    # 5s < min_clip_duration of 15s
    ranges = [{"start_time": 10.0, "end_time": 15.0}]
    assert finder._ranges_to_clips(ranges, transcription) == []


def test_ranges_to_clips_clamps_to_media_end(finder, transcription):
    ranges = [{"start_time": 500.0, "end_time": 9999.0}]
    clips = finder._ranges_to_clips(ranges, transcription)
    assert clips[0].end_time == transcription.end_time


def test_ranges_to_clips_skips_malformed(finder, transcription):
    ranges = [{"start_time": 10.0}, {"start_time": 20.0, "end_time": 60.0}]
    clips = finder._ranges_to_clips(ranges, transcription)
    assert clips == [Clip(20.0, 60.0, 20, 60)]


def test_ranges_to_clips_sorted(finder, transcription):
    ranges = [
        {"start_time": 100.0, "end_time": 140.0},
        {"start_time": 10.0, "end_time": 50.0},
    ]
    clips = finder._ranges_to_clips(ranges, transcription)
    assert [c.start_time for c in clips] == [10.0, 100.0]


# --- find_clips wiring (no network) ---
def test_find_clips_requires_exactly_one_source(finder, transcription):
    with pytest.raises(ClipFinderError):
        finder.find_clips(transcription)
    with pytest.raises(ClipFinderError):
        finder.find_clips(transcription, video_url="u", video_id="i")


def test_find_clips_calls_pegasus_and_maps(finder, transcription):
    fake_response = MagicMock()
    fake_response.data = '[{"start_time": 10.0, "end_time": 40.0}]'
    fake_client = MagicMock()
    fake_client.analyze.return_value = fake_response

    with patch("twelvelabs.TwelveLabs", return_value=fake_client):
        clips = finder.find_clips(transcription, video_id="vid-123")

    fake_client.analyze.assert_called_once()
    _, kwargs = fake_client.analyze.call_args
    assert kwargs["video_id"] == "vid-123"
    assert kwargs["model_name"] == "pegasus1.5"
    assert clips == [Clip(10.0, 40.0, 10, 40)]
