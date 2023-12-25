# standard library imports
from unittest.mock import patch, Mock

# local package imports
from clipsai.diarize.pyannote import PyannoteDiarizer

# third party imports
from pyannote.core import Segment, Annotation
import pytest


@pytest.fixture
def mock_pipeline():
    with patch('pyannote.audio.Pipeline.from_pretrained') as mock:
        mock.return_value = Mock()
        yield mock

@pytest.fixture
def mock_diarizer(mock_pipeline):
    diarizer = PyannoteDiarizer(auth_token='mock_token')
    diarizer.pipeline = mock_pipeline
    return diarizer

@pytest.fixture
def mock_annotation():
    annotation = Annotation()
    annotation[Segment(0, 10)] = 'speaker_0'
    annotation[Segment(11, 20)] = 'speaker_1'
    annotation[Segment(21, 30)] = 'speaker_0'
    return annotation

@pytest.fixture
def mock_audio_file():
    mock_audio_file = Mock()
    mock_audio_file.get_duration.return_value = 30.0
    return mock_audio_file

def test_relabel_speakers_on_discontiguous_speaker_labels(mock_diarizer):
    initial_segments = [
        {"speakers": [2], "startTime": 0.0, "endTime": 10.0},
        {"speakers": [5], "startTime": 10.0, "endTime": 20.0},
        {"speakers": [2], "startTime": 20.0, "endTime": 30.0}
    ]
    
    relabeled_segments = mock_diarizer._relabel_speakers(
        speaker_segments=initial_segments, 
        unique_speakers={2, 5},
    )

    assert len(relabeled_segments) == len(initial_segments)
    
    result = []
    correct_speaker_labels = {0, 1}
    for segment in relabeled_segments:
        for speaker in segment['speakers']:
            result.append(speaker in correct_speaker_labels)

    # all speakers were relabeled correctly        
    assert all(result) 

def test_relabel_speakers_on_contiguous_speaker_labels(mock_diarizer):
    initial_segments = [
        {"speakers": [0], "startTime": 0.0, "endTime": 10.0},
        {"speakers": [1], "startTime": 10.0, "endTime": 20.0},
        {"speakers": [0], "startTime": 20.0, "endTime": 30.0},
    ]

    relabeled_segments = mock_diarizer._relabel_speakers(
        speaker_segments=initial_segments,
        unique_speakers={0, 1},
    )

    assert len(relabeled_segments) == len(initial_segments)
    assert relabeled_segments == initial_segments

def test_relabel_speakers_handles_empty_speaker_lists(mock_diarizer):
    initial_segments = []

    relabeled_segments = mock_diarizer._relabel_speakers(
        speaker_segments=initial_segments,
        unique_speakers=set(),
    )

    assert len(relabeled_segments) == len(initial_segments)
    assert relabeled_segments == []

def test_relabel_speakers_handles_unlabeled_speaker(mock_diarizer):
    initial_segments = [
        {"speakers": [], "startTime": 0.0, "endTime": 10.0},
        {"speakers": [1], "startTime": 10.0, "endTime": 20.0},
    ]

    relabeled_segments = mock_diarizer._relabel_speakers(
        speaker_segments=initial_segments,
        unique_speakers={1},
    )

    assert len(relabeled_segments) == len(initial_segments)
    assert relabeled_segments[0]["speakers"] == []
    assert relabeled_segments[1]["speakers"] == [0]

def test_adjust_segments(mock_diarizer, mock_annotation, mock_audio_file):
    segments = mock_diarizer._adjust_segments(
        pyannote_segments=mock_annotation,
        duration=mock_audio_file.get_duration()
    )
    
    assert len(segments) == 3 

    assert segments[0]['speakers'] == [0]
    assert segments[0]['startTime'] == 0
    assert segments[0]['endTime'] == 11

    assert segments[1]['speakers'] == [1]
    assert segments[1]['startTime'] == 11
    assert segments[1]['endTime'] == 21

    assert segments[2]['speakers'] == [0]
    assert segments[2]['startTime'] == 21
    assert segments[2]['endTime'] == mock_audio_file.get_duration()

def test_adjust_segments_handles_overlapping_segments(mock_diarizer, mock_annotation, mock_audio_file):
    mock_annotation[Segment(8, 15)] = 'speaker_2'  
    
    segments = mock_diarizer._adjust_segments(
        pyannote_segments=mock_annotation,
        duration=mock_audio_file.get_duration()
    )

    assert len(segments) == 4
    
    assert segments[0]['speakers'] == [0]
    assert segments[0]['startTime'] == 0
    assert segments[0]['endTime'] == 8

    assert segments[1]['speakers'] == [2]
    assert segments[1]['startTime'] == 8
    assert segments[1]['endTime'] == 11

    assert segments[2]['speakers'] == [1]
    assert segments[2]['startTime'] == 11
    assert segments[2]['endTime'] == 21

    assert segments[3]['speakers'] == [0]
    assert segments[3]['startTime'] == 21
    assert segments[3]['endTime'] == mock_audio_file.get_duration()

def test_adjust_segments_discards_short_segments(mock_diarizer, mock_annotation, mock_audio_file):
    mock_annotation[Segment(15, 16)] = 'speaker_1'

    segments = mock_diarizer._adjust_segments(
        pyannote_segments=mock_annotation,
        duration=mock_audio_file.get_duration()
    )

    assert len(segments) == 3
    
    assert segments[0]['speakers'] == [0]
    assert segments[0]['startTime'] == 0
    assert segments[0]['endTime'] == 11

    assert segments[1]['speakers'] == [1]
    assert segments[1]['startTime'] == 11
    assert segments[1]['endTime'] == 21

    assert segments[2]['speakers'] == [0]
    assert segments[2]['startTime'] == 21
    assert segments[2]['endTime'] == mock_audio_file.get_duration()


def test_adjust_segments_merges_contiguous_segments_with_same_speakers(mock_diarizer, mock_annotation, mock_audio_file):
    mock_annotation[Segment(10, 12)] = 'speaker_1'

    segments = mock_diarizer._adjust_segments(
        pyannote_segments=mock_annotation,
        duration=mock_audio_file.get_duration()
    )

    assert len(segments) == 3

    assert segments[0]['speakers'] == [0]
    assert segments[0]['startTime'] == 0
    assert segments[0]['endTime'] == 10

    assert segments[1]['speakers'] == [1]
    assert segments[1]['startTime'] == 10
    assert segments[1]['endTime'] == 21

    assert segments[2]['speakers'] == [0]
    assert segments[2]['startTime'] == 21
    assert segments[2]['endTime'] == mock_audio_file.get_duration()

def test_adjust_segments_on_empty_annotation(mock_diarizer, mock_audio_file):
    mock_annotation = Annotation()

    segments = mock_diarizer._adjust_segments(
        pyannote_segments=mock_annotation,
        duration=mock_audio_file.get_duration()
    )

    assert len(segments) == 1

    assert segments[0]['speakers'] == []
    assert segments[0]['startTime'] == 0
    assert segments[0]['endTime'] == mock_audio_file.get_duration()