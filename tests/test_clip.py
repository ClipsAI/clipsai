import pytest
from unittest.mock import patch, MagicMock
from clipsai.clip.clip import clip
from clipsai.clip.clip_input_validator import ClipInputValidator
from clipsai.clip.texttile_config_manager import TextTileClipFinderConfigManager
from clipsai.transcribe.transcription import Transcription
from clipsai.clip.clip_model import Clip


@pytest.fixture
def clip_input_validator():
    return ClipInputValidator()


@pytest.fixture
def texttile_config_manager():
    return TextTileClipFinderConfigManager()


@pytest.fixture
def mock_texttile_clip_finder():
    with patch('clipsai.clip.texttile.TextTileClipFinder') as mock:
        yield mock


@pytest.fixture
def valid_transcription():
    transcription = MagicMock(spec=Transcription)
    transcription.end_time = 800.0
    transcription.get_sentence_info.return_value = [{"sentence": "Example sentence"}]
    return transcription


# Testing ClipInputValidator
def test_clip_input_validator_valid_data(clip_input_validator):
    valid_input_data = {
        "computeDevice": "cpu",
        "cutoffPolicy": "high",
        "embeddingAggregationPoolMethod": "max",
        "minClipTime": 15,
        "maxClipTime": 900,
        "smoothingWidth": 3,
        "windowComparePoolMethod": "mean",
    }
    assert clip_input_validator.check_valid_input_data(valid_input_data) is None


def test_clip_input_validator_invalid_data(clip_input_validator):
    invalid_input_data = {
        "computeDevice": "unknown_device",
        "cutoffPolicy": "invalid_policy",
        "embeddingAggregationPoolMethod": "invalid_method",
        "minClipTime": -5,
        "maxClipTime": 5,
        "smoothingWidth": 1,
        "windowComparePoolMethod": "invalid_method",
    }
    assert isinstance(clip_input_validator.check_valid_input_data(invalid_input_data), str)


# Testing TextTileClipFinderConfigManager
def test_texttile_config_manager_valid_config(texttile_config_manager):
    valid_config = {
        "cutoff_policy": "high",
        "embedding_aggregation_pool_method": "max",
        "max_clip_duration_secs": 900,
        "min_clip_duration_secs": 15,
        "smoothing_width": 3,
        "window_compare_pool_method": "mean",
    }
    assert texttile_config_manager.check_valid_config(valid_config) is None


def test_texttile_config_manager_invalid_config(texttile_config_manager):
    invalid_config = {
        "cutoff_policy": "invalid_policy",
        "embedding_aggregation_pool_method": "invalid_method",
        "max_clip_duration_secs": 5,
        "min_clip_duration_secs": 10,
        "smoothing_width": 1,
        "window_compare_pool_method": "invalid_method",
    }
    assert isinstance(texttile_config_manager.check_valid_config(invalid_config), str)


# Testing clip function
def test_clip_with_valid_input(valid_transcription, mock_texttile_clip_finder):
    mock_clip_finder = mock_texttile_clip_finder.return_value
    mock_clip = {"startTime": 100, "endTime": 200, "startChar": 0, "endChar": 10}
    mock_clip_finder.find_clips.return_value = [mock_clip]
    result = clip(
        transcription=valid_transcription,
        min_clip_duration=15,
        max_clip_duration=900,
        device="cpu"
    )

    assert isinstance(result, list)
