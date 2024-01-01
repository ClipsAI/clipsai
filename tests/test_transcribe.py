import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from clipsai.media.editor import MediaEditor
from clipsai.media.audio_file import AudioFile
from clipsai.media.audiovideo_file import AudioVideoFile
from clipsai.filesys.json_file import JSONFile
from clipsai.exceptions import InvalidInputDataError
from clipsai.transcribe.exceptions import TranscriptionError
from clipsai.media.exceptions import MediaEditorError
from clipsai.transcribe.transcribe_input_validator import TranscribeInputValidator
from clipsai.transcribe.transcribe import transcribe
from clipsai.transcribe.transcription import Transcription


@pytest.fixture
def transcribe_input_validator():
    return TranscribeInputValidator()


@pytest.fixture
def media_editor():
    return MediaEditor()


@pytest.fixture
def mock_whisperx_transcriber():
    with patch('clipsai.transcribe.transcribe.WhisperXTranscriber') as mock:
        yield mock


@pytest.fixture
def mock_media_editor():
    with patch('clipsai.transcribe.transcribe.MediaEditor') as mock:
        yield mock


# Testing TranscribeInputValidator
def test_impute_input_data_defaults(transcribe_input_validator):
    input_data = {
        "mediaFilePath": "path/to/media.mp3",
        "computeDevice": "auto",
        "languageCode": "auto"
    }
    result = transcribe_input_validator.impute_input_data_defaults(input_data)
    assert result["computeDevice"] is None
    assert result["languageCode"] is None
    assert "precision" in result
    assert "whisperModelSize" in result


def test_assert_valid_input_data(transcribe_input_validator):
    valid_input_data = {
        "mediaFilePath": "path/to/media.mp4",
        "computeDevice": "cpu",
        "languageCode": "en",
        "precision": "float16",
        "whisperModelSize": "medium"
    }
    transcribe_input_validator.assert_valid_input_data(valid_input_data)


def test_invalid_media_file_path(transcribe_input_validator):
    invalid_input_data = {
        "mediaFilePath": "path/to/media.txt"
    }
    with pytest.raises(InvalidInputDataError):
        transcribe_input_validator.assert_valid_input_data(invalid_input_data)


def test_invalid_compute_device(transcribe_input_validator):
    invalid_input_data = {
        "mediaFilePath": "path/to/media.mp3",
        "computeDevice": "invalid_device"
    }
    with pytest.raises(InvalidInputDataError):
        transcribe_input_validator.assert_valid_input_data(invalid_input_data)


# Testing MediaEditor
@patch('clipsai.media.temporal_media_file.TemporalMediaFile.assert_exists')
def test_instantiate_as_audio_file(mock_assert_exists, media_editor):
    mock_assert_exists.return_value = None
    with patch('clipsai.media.temporal_media_file.TemporalMediaFile.has_audio_stream', return_value=True), \
         patch('clipsai.media.temporal_media_file.TemporalMediaFile.has_video_stream', return_value=False):
        result = media_editor.instantiate_as_temporal_media_file('path/to/audio.mp3')
    assert isinstance(result, AudioFile)


@patch('clipsai.media.temporal_media_file.TemporalMediaFile.assert_exists')
def test_instantiate_as_audio_video_file(mock_assert_exists, media_editor):
    mock_assert_exists.return_value = None
    with patch('clipsai.media.temporal_media_file.TemporalMediaFile.has_audio_stream', return_value=True), \
         patch('clipsai.media.temporal_media_file.TemporalMediaFile.has_video_stream', return_value=True):
        result = media_editor.instantiate_as_temporal_media_file('path/to/video.mp4')
    assert isinstance(result, AudioVideoFile)


@patch('clipsai.media.temporal_media_file.TemporalMediaFile.assert_exists')
def test_instantiate_invalid_file(mock_assert_exists, media_editor):
    mock_assert_exists.return_value = None
    with patch('clipsai.media.temporal_media_file.TemporalMediaFile.has_audio_stream', return_value=False), \
         patch('clipsai.media.temporal_media_file.TemporalMediaFile.has_video_stream', return_value=False):
        with pytest.raises(MediaEditorError):
            media_editor.instantiate_as_temporal_media_file('path/to/invalid.file')


# Testing transcribe
def test_transcribe_successful(mock_media_editor, mock_whisperx_transcriber):
    mock_transcription = {
        'charInfo': [],
        'language': 'en',
        'numSpeakers': 1,
        'sourceSoftware': 'whisperx',
        'timeCreated': '2023-01-01T00:00:00'
    }
    mock_transcriber_instance = mock_whisperx_transcriber.return_value
    mock_transcriber_instance.transcribe.return_value = MagicMock(**mock_transcription)

    result = transcribe(
        media_file_path="path/to/media.mp3",
        language_code="en",
        device="cpu"
    )

    assert isinstance(result, MagicMock)


def test_transcribe_invalid_input():
    result = transcribe("path/to/invalid.txt", "en", "cpu")
    assert isinstance(result, dict)
    assert result.get("state") == "failed"


def test_transcribe_exception_handling(mock_media_editor, mock_whisperx_transcriber):
    mock_transcriber_instance = mock_whisperx_transcriber.return_value
    mock_transcriber_instance.transcribe.side_effect = Exception("Transcription failed")

    result = transcribe("path/to/media.mp3", "en", "cpu")
    assert result["state"] == "failed"


# Testing Transcription
valid_transcription_data = {
    "sourceSoftware": "TestSoftware",
    "timeCreated": datetime.now(),
    "language": "en",
    "numSpeakers": 2,
    "charInfo": [
        {"char": "H", "startTime": 0.0, "endTime": 0.2, "speaker": 1},
    ]
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

    with patch('filesys.json_file.JSONFile.assert_has_file_extension'), \
         patch('filesys.manager.FileSystemManager.assert_parent_dir_exists'), \
         patch('filesys.json_file.JSONFile.delete'), \
         patch('filesys.json_file.JSONFile.create', return_value=mock_json_file), \
         patch('filesys.json_file.JSONFile.assert_exists'):

        json_file = transcription.store_as_json_file("path/to/output.json")

        json_file_class_name = json_file.__class__.__name__
        mock_json_file_class_name = mock_json_file.__class__.__name__

        assert json_file_class_name == mock_json_file_class_name

def test_invalid_times_exception():
    transcription = Transcription(valid_transcription_data)
    with pytest.raises(TranscriptionError):
        transcription.get_char_info(start_time=-1, end_time=5)
