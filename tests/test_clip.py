import pytest
from unittest.mock import MagicMock
from clipsai.clip.clipfinder import ClipFinderConfigManager
from clipsai.clip.texttiler import TextTilerConfigManager
from clipsai.transcribe.transcription import Transcription


@pytest.fixture
def clip_finder_config_manager():
    return ClipFinderConfigManager()


@pytest.fixture
def texttiler_config_manager():
    return TextTilerConfigManager()


@pytest.fixture
def valid_transcription():
    transcription = MagicMock(spec=Transcription)
    transcription.end_time = 800.0
    transcription.get_sentence_info.return_value = [{"sentence": "Example sentence"}]
    return transcription


# Testing ClipConfigManager
def test_clip_finder_config_manager_valid_config(
    clip_finder_config_manager: ClipFinderConfigManager,
):
    config = {
        "cutoff_policy": "high",
        "embedding_aggregation_pool_method": "max",
        "min_clip_duration": 15,
        "max_clip_duration": 900,
        "smoothing_width": 3,
        "window_compare_pool_method": "mean",
    }
    assert clip_finder_config_manager.check_valid_config(config) is None


def test_clip_finder_config_manager_invalid_config(
    clip_finder_config_manager: ClipFinderConfigManager,
):
    config = {
        "cutoff_policy": "invalid_policy",
        "embedding_aggregation_pool_method": "invalid_method",
        "min_clip_duration": -5,
        "max_clip_duration": 5,
        "smoothing_width": 1,
        "window_compare_pool_method": "invalid_method",
    }
    assert isinstance(clip_finder_config_manager.check_valid_config(config), str)


# Testing TextTileClipFinderConfigManager
def test_texttiler_config_manager_valid_config(
    texttiler_config_manager: TextTilerConfigManager,
):
    config = {
        "k": 5,
        "cutoff_policy": "high",
        "embedding_aggregation_pool_method": "max",
        "smoothing_width": 3,
        "window_compare_pool_method": "mean",
    }
    assert texttiler_config_manager.check_valid_config(config) is None


def test_texttiler_config_manager_invalid_config(
    texttiler_config_manager: TextTilerConfigManager,
):
    config = {
        "k": 1,
        "cutoff_policy": "invalid_policy",
        "embedding_aggregation_pool_method": "invalid_method",
        "smoothing_width": 1,
        "window_compare_pool_method": "invalid_method",
    }
    assert isinstance(texttiler_config_manager.check_valid_config(config), str)
