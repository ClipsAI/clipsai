# standard library imports
from unittest.mock import patch, Mock

# local package imports
from clipsai.diarize.pyannote import PyannoteDiarizer

# third party imports
import pandas as pd
from pyannote.core import Segment, Annotation
import pytest


@pytest.fixture
def mock_diarizer():
    with patch('pyannote.audio.Pipeline.from_pretrained', return_value=Mock()):
        diarizer = PyannoteDiarizer(auth_token='mock_token')
        diarizer.pipeline = Mock()
        return diarizer

@pytest.fixture
def mock_audio_file():
    mock_audio_file = Mock()
    mock_audio_file.path.return_value = 'mock_audio.mp3'
    mock_audio_file.get_duration.return_value = 30.0
    return mock_audio_file


@pytest.mark.parametrize("annotation_data, expected_output", [
    # Test 1: Segments with gaps between them
    (
        [
            {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
            {"segment": Segment(12, 20), "label": "speaker_1", "track": "_"},
            {"segment": Segment(21, 30), "label": "speaker_0", "track": "_"}
        ],
        [
            {"speakers": [0], "startTime": 0, "endTime": 12},
            {"speakers": [1], "startTime": 12, "endTime": 21},
            {"speakers": [0], "startTime": 21, "endTime": 30}
        ]
    ),

    # Test 2: overlapping segments
    (
        [
            {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
            {"segment": Segment(8, 12), "label": "speaker_2", "track": "_"},
            {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
            {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"}
        ],
        [
            {"speakers": [0], "startTime": 0, "endTime": 8},
            {"speakers": [2], "startTime": 8, "endTime": 10},
            {"speakers": [1], "startTime": 10, "endTime": 20},
            {"speakers": [0], "startTime": 20, "endTime": 30}
        ]
    ),

    # Test 3: discarding short segments
    (
        [
            {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
            {"segment": Segment(11, 20), "label": "speaker_1", "track": "_"},
            {"segment": Segment(15, 16), "label": "speaker_1", "track": "_"},
            {"segment": Segment(21, 30), "label": "speaker_0", "track": "_"}
        ],
        [
            {"speakers": [0], "startTime": 0, "endTime": 11},
            {"speakers": [1], "startTime": 11, "endTime": 21},
            {"speakers": [0], "startTime": 21, "endTime": 30}
        ]
    ),

    # Test 4: merge contiguous segments with same speakers
    (
        [
            {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
            {"segment": Segment(10, 12), "label": "speaker_1", "track": "_"},
            {"segment": Segment(12, 15), "label": "speaker_1", "track": "_"},
            {"segment": Segment(15, 20), "label": "speaker_1", "track": "_"},
            {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"}
        ],
        [
            {"speakers": [0], "startTime": 0, "endTime": 10},
            {"speakers": [1], "startTime": 10, "endTime": 20},
            {"speakers": [0], "startTime": 20, "endTime": 30}
        ]
    ),

    # Test 5: handles empty annotation
    (
        [],
        [{"speakers": [], "startTime": 0, "endTime": 30}]
    ),

    # Test 6: relabel speakers with discontiguous speaker labels
    (
        [
            {"segment": Segment(0, 10), "label": "speaker_2", "track": "_"},
            {"segment": Segment(10, 20), "label": "speaker_5", "track": "_"},
            {"segment": Segment(20, 30), "label": "speaker_2", "track": "_"}
        ],
        [
            {"speakers": [0], "startTime": 0, "endTime": 10},
            {"speakers": [1], "startTime": 10, "endTime": 20},
            {"speakers": [0], "startTime": 20, "endTime": 30}
        ]
    ),

    # Test 7: relabeling speakers not required with contiguous speaker labels
    (
        [
            {"segment": Segment(0, 10), "label": "speaker_0", "track": "_"},
            {"segment": Segment(10, 20), "label": "speaker_1", "track": "_"},
            {"segment": Segment(20, 30), "label": "speaker_0", "track": "_"}
        ],
        [
            {"speakers": [0], "startTime": 0, "endTime": 10},
            {"speakers": [1], "startTime": 10, "endTime": 20},
            {"speakers": [0], "startTime": 20, "endTime": 30}
        ]
    ),

    # Test 8: handles unlabeled speaker
    (
        [{"segment": Segment(0, 30), "label": "_", "track": "_"}],
        [{"speakers": [], "startTime": 0, "endTime": 30}]
    ),    
])

def test_diarize(mock_diarizer, mock_audio_file, annotation_data, expected_output):
    # handle empty annotation to prevent KeyError
    if not annotation_data:
        annotation = Annotation()
    else:
        df = pd.DataFrame(annotation_data)
        annotation = Annotation().from_df(df)

    mock_diarizer.pipeline.return_value = annotation
    output_segments = mock_diarizer.diarize(mock_audio_file)

    assert output_segments == expected_output