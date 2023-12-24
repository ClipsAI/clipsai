import pytest
from unittest.mock import patch, Mock
from clipsai.diarize.pyannote import PyannoteDiarizer
from pyannote.core import Segment, Annotation


@pytest.fixture
def mock_annotation():
    annotation = Annotation()
    annotation[Segment(0, 10)] = 'speaker_0'
    annotation[Segment(11, 20)] = 'speaker_1'
    annotation[Segment(21, 30)] = 'speaker_0'
    return annotation

@pytest.fixture
def mock_annotation_short_segment():
    annotation = Annotation()
    annotation[Segment(0, 5)] = 'speaker_0'
    annotation[Segment(5, 6)] = 'speaker_0'
    annotation[Segment(11, 20)] = 'speaker_1'
    annotation[Segment(21, 30)] = 'speaker_0'
    return annotation

@pytest.fixture
def mock_annotation_short_segment():
    annotation = Annotation()
    annotation[Segment(0, 5)] = 'speaker_0'
    annotation[Segment(5, 12)] = 'speaker_0'
    annotation[Segment(11, 20)] = 'speaker_1'
    annotation[Segment(21, 30)] = 'speaker_0'
    return annotation

@pytest.fixture
def mock_audio_file():
    mock_audio_file = Mock()
    mock_audio_file.get_duration.return_value = 30.0
    return mock_audio_file

def test_adjust_segments(mock_annotation, mock_audio_file):
    diarizer = PyannoteDiarizer(auth_token="mock_token")
    segments = diarizer._adjust_segments(
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
    assert segments[2]['endTime'] == 30

    assert segments[-1]['endTime'] == mock_audio_file.get_duration()

def test_relabel_speakers_discontiguous_speaker_labels():
    initial_segments = [
        {"speakers": [2], "startTime": 0.0, "endTime": 10.0},
        {"speakers": [5], "startTime": 10.0, "endTime": 20.0},
        {"speakers": [2], "startTime": 20.0, "endTime": 30.0}
    ]
    
    diarizer = PyannoteDiarizer(auth_token="mock_token")
    relabeled_segments = diarizer._relabel_speakers(
        speaker_segments=initial_segments, 
        unique_speakers={2, 5},
    )

    assert len(relabeled_segments) == len(initial_segments)
    
    result = []
    correct_speaker_labels = {0, 1}
    for segment in relabeled_segments:
        for speaker in segment['speakers']:
            result.append(speaker in correct_speaker_labels)

    # asset all speakers were relabeled correctly        
    assert all(result) 

def test_relabel_speakers_on_contiguous_speaker_labels():
    initial_segments = [
        {"speakers": [0], "startTime": 0.0, "endTime": 10.0},
        {"speakers": [1], "startTime": 10.0, "endTime": 20.0},
        {"speakers": [0], "startTime": 20.0, "endTime": 30.0},
    ]

    diarizer = PyannoteDiarizer(auth_token="mock_token")
    relabeled_segments = diarizer._relabel_speakers(
        speaker_segments=initial_segments,
        unique_speakers={0, 1},
    )

    assert len(relabeled_segments) == len(initial_segments)
    assert relabeled_segments == initial_segments

def test_relabel_speakers_handles_empty_speaker_lists():
    initial_segments = []

    diarizer = PyannoteDiarizer(auth_token="mock_token")
    relabeled_segments = diarizer._relabel_speakers(
        speaker_segments=initial_segments,
        unique_speakers=set(),
    )

    assert len(relabeled_segments) == len(initial_segments)
    assert relabeled_segments == []
