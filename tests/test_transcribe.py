import pytest
from unittest.mock import patch, Mock
from datetime import datetime

from clipsai.filesys.json_file import JSONFile
from clipsai.media.audio_file import AudioFile
from clipsai.media.audiovideo_file import AudioVideoFile
from clipsai.media.editor import MediaEditor
from clipsai.media.exceptions import MediaEditorError
from clipsai.transcribe.exceptions import TranscriptionError
from clipsai.transcribe.transcriber import TranscriberConfigManager, Transcriber
from clipsai.transcribe.transcription import Transcription


@pytest.fixture
def transcriber_config_manager():
    return TranscriberConfigManager()


@pytest.fixture
def media_editor():
    return MediaEditor()


@pytest.fixture
def mock_whisperx_transcriber():
    with patch("transcribe.transcribe.WhisperXTranscriber") as mock:
        yield mock


@pytest.fixture
def mock_media_editor():
    with patch("transcribe.transcribe.MediaEditor") as mock:
        yield mock


# Testing TranscriberConfigManager
def test_assert_valid_config(transcriber_config_manager: TranscriberConfigManager):
    config = {
        "language": "en",
        "precision": "float16",
        "model_size": "medium",
    }
    transcriber_config_manager.assert_valid_config(config)


# Testing MediaEditor
@patch("media.temporal_media_file.TemporalMediaFile.assert_exists")
def test_instantiate_as_audio_file(mock_assert_exists, media_editor: MediaEditor):
    mock_assert_exists.return_value = None
    with patch(
        "media.temporal_media_file.TemporalMediaFile.has_audio_stream",
        return_value=True,
    ), patch(
        "media.temporal_media_file.TemporalMediaFile.has_video_stream",
        return_value=False,
    ):
        result = media_editor.instantiate_as_temporal_media_file("path/to/audio.mp3")
    assert isinstance(result, AudioFile)


@patch("media.temporal_media_file.TemporalMediaFile.assert_exists")
def test_instantiate_as_audio_video_file(mock_assert_exists, media_editor: MediaEditor):
    mock_assert_exists.return_value = None
    with patch(
        "media.temporal_media_file.TemporalMediaFile.has_audio_stream",
        return_value=True,
    ), patch(
        "media.temporal_media_file.TemporalMediaFile.has_video_stream",
        return_value=True,
    ):
        result = media_editor.instantiate_as_temporal_media_file("path/to/video.mp4")
    assert isinstance(result, AudioVideoFile)


@patch("media.temporal_media_file.TemporalMediaFile.assert_exists")
def test_instantiate_invalid_file(mock_assert_exists, media_editor: MediaEditor):
    mock_assert_exists.return_value = None
    with patch(
        "media.temporal_media_file.TemporalMediaFile.has_audio_stream",
        return_value=False,
    ), patch(
        "media.temporal_media_file.TemporalMediaFile.has_video_stream",
        return_value=False,
    ):
        with pytest.raises(MediaEditorError):
            media_editor.instantiate_as_temporal_media_file("path/to/invalid.file")


# Testing Transcription
valid_transcription_data = {
    "source_software": "TestSoftware",
    "time_created": datetime.now(),
    "language": "en",
    "num_speakers": 2,
    "char_info": [
        {"char": "H", "start_time": 0.0, "end_time": 0.2, "speaker": 1},
    ],
}


def test_init_with_valid_dict():
    transcription = Transcription(valid_transcription_data)
    assert transcription.language == "en"


def test_init_with_valid_json_file():
    transcription = Transcription(valid_transcription_data)
    assert isinstance(transcription, Transcription)


def test_init_with_invalid_data():
    with pytest.raises(TypeError):
        Transcription("invalid_data")


def test_get_source_software():
    transcription = Transcription(valid_transcription_data)
    assert transcription.source_software == "TestSoftware"


def test_get_time_spawned():
    transcription = Transcription(valid_transcription_data)
    assert isinstance(transcription.created_time, datetime)


def test_get_char_info_with_time_filter():
    transcription = Transcription(valid_transcription_data)
    char_info = transcription.get_char_info(start_time=0.0, end_time=0.2)
    assert len(char_info) > 0


def test_find_char_index():
    transcription = Transcription(valid_transcription_data)
    index = transcription.find_char_index(0.1, "start")
    assert index >= 0


def test_store_as_json_file():
    mock_json_file = JSONFile("path/to/output.json")
    transcription = Transcription(valid_transcription_data)

    with patch("filesys.json_file.JSONFile.assert_has_file_extension"), patch(
        "filesys.manager.FileSystemManager.assert_parent_dir_exists"
    ), patch("filesys.json_file.JSONFile.delete"), patch(
        "filesys.json_file.JSONFile.create", return_value=mock_json_file
    ), patch(
        "filesys.json_file.JSONFile.assert_exists"
    ):
        json_file = transcription.store_as_json_file("path/to/output.json")

        json_file_class_name = json_file.__class__.__name__
        mock_json_file_class_name = mock_json_file.__class__.__name__

        assert json_file_class_name == mock_json_file_class_name


def test_invalid_times_exception():
    transcription = Transcription(valid_transcription_data)
    with pytest.raises(TranscriptionError):
        transcription.get_char_info(start_time=-1, end_time=5)


@patch("clipsai.transcribe.transcriber.whisperx", new_callable=Mock)
def test_transcribe_short_duration_raises(mock_whisperx):
    mock_whisperx.load_model.return_value = Mock()
    transcriber = Transcriber()
    fake_media = Mock()
    fake_media.path = "fake.mp3"
    fake_media.get_duration.return_value = 100.0
    fake_media.assert_exists = Mock(return_value=None)
    fake_media.assert_has_audio_stream = Mock(return_value=None)
    with patch("clipsai.transcribe.transcriber.MediaEditor") as mock_editor:
        mock_editor.return_value.instantiate_as_temporal_media_file.return_value = fake_media
        with pytest.raises(ValueError):
            transcriber.transcribe("fake.mp3")


@patch("clipsai.transcribe.transcriber.whisperx", new_callable=Mock)
def test_transcribe_long_duration_raises(mock_whisperx):
    mock_whisperx.load_model.return_value = Mock()
    transcriber = Transcriber()
    fake_media = Mock()
    fake_media.path = "fake_long.mp3"
    fake_media.get_duration.return_value = 8000.0
    fake_media.assert_exists = Mock(return_value=None)
    fake_media.assert_has_audio_stream = Mock(return_value=None)
    with patch("clipsai.transcribe.transcriber.MediaEditor") as mock_editor:
        mock_editor.return_value.instantiate_as_temporal_media_file.return_value = fake_media
        with pytest.raises(ValueError):
            transcriber.transcribe("fake_long.mp3")
