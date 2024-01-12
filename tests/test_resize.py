# standard library imports
from unittest.mock import patch, MagicMock

# local package imports
from clipsai.media.video_file import VideoFile
from clipsai.resize.resizer import Resizer
from clipsai.resize.rect import Rect


# third party imports
import pytest


@pytest.mark.parametrize(
    "original_width, original_height, aspect_ratio, expected",
    [
        # Wider aspect ratio
        (1920, 1080, (9, 16), (607, 1080)),
        (1280, 720, (9, 16), (405, 720)),
        # Taller aspect ratio
        (1080, 1920, (16, 9), (1080, 607)),
        (720, 1280, (16, 9), (720, 405)),
        # Extreme aspect ratios
        (1920, 1080, (1, 100), (10, 1080)),
        (1920, 1080, (100, 1), (1920, 19)),
        # Equal aspect ratio
        (1920, 1080, (16, 9), (1920, 1080)),
        (1280, 720, (16, 9), (1280, 720)),
        # Equal aspect ratio = Small dimensions
        (320, 240, (4, 3), (320, 240)),
        (10, 10, (1, 1), (10, 10)),
        # Equal aspect ratio = Large dimensions
        (8000, 4500, (16, 9), (8000, 4500)),
        (4500, 8000, (9, 16), (4500, 8000)),
    ],
)
def test_calc_resize_width_and_height_pixels(
    original_width: int,
    original_height: int,
    aspect_ratio: tuple[int, int],
    expected: tuple[int, int],
):
    resizer = Resizer()
    result = resizer._calc_resize_width_and_height_pixels(
        original_width_pixels=original_width,
        original_height_pixels=original_height,
        resize_aspect_ratio=aspect_ratio,
    )
    assert result == expected


# Test cases
@pytest.mark.parametrize(
    "speaker_segments, scene_changes, expected",
    [
        # Test with no scene changes
        (
            [{"speakers": [0], "start_time": 0, "end_time": 10}],
            [],
            [{"speakers": [0], "start_time": 0, "end_time": 10}],
        ),
        # Test with scene change matching the end of a segment
        (
            [{"speakers": [0], "start_time": 0, "end_time": 5}],
            [5],
            [{"speakers": [0], "start_time": 0, "end_time": 5}],
        ),
        # Test with scene change within a segment
        (
            [{"speakers": [0], "start_time": 0, "end_time": 10}],
            [5],
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [0], "start_time": 5, "end_time": 10},
            ],
        ),
        # Test with multiple segments and scene changes
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [3, 8],
            [
                {"speakers": [0], "start_time": 0, "end_time": 3},
                {"speakers": [0], "start_time": 3, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 8},
                {"speakers": [1], "start_time": 8, "end_time": 10},
            ],
        ),
        # Test with scene changes at segment boundaries
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [5],
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
        ),
        # Test with scene change very close to segment start
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [4.8],
            [
                {"speakers": [0], "start_time": 0, "end_time": 4.8},
                {"speakers": [1], "start_time": 4.8, "end_time": 10},
            ],
        ),
        # Test with scene change very close to segment end
        (
            [
                {"speakers": [0], "start_time": 0, "end_time": 5},
                {"speakers": [1], "start_time": 5, "end_time": 10},
            ],
            [5.1],
            [
                {"speakers": [0], "start_time": 0, "end_time": 5.1},
                {"speakers": [1], "start_time": 5.1, "end_time": 10},
            ],
        ),
    ],
)
def test_merge_scene_change_and_speaker_segments(
    speaker_segments: list[dict], scene_changes: list[float], expected: list[dict]
):
    resizer = Resizer()
    result = resizer._merge_scene_change_and_speaker_segments(
        speaker_segments=speaker_segments,
        scene_changes=scene_changes,
        scene_merge_threshold=0.25,
    )
    assert result == expected


@pytest.mark.parametrize(
    (
        "width, height, num_frames, gpu_available, face_detect_width,"
        "n_face_detect_batches, expected_batches"
    ),
    [
        # Scenario 1: CPU only, small video
        (640, 480, 100, False, 960, 8, 1),
        # Scenario 2: CPU only, large video
        (1920, 1080, 100, False, 960, 8, 1),
        # Scenario 3: GPU available, small video
        (640, 480, 100, True, 960, 8, 8),
        # Scenario 4: GPU available, large video
        (1920, 1080, 100, True, 960, 8, 8),
    ],
)
def test_calc_n_batches(
    width: int,
    height: int,
    num_frames: int,
    gpu_available: bool,
    face_detect_width: int,
    n_face_detect_batches: int,
    expected_batches: int,
):
    # Setup the mock video file object
    mock_video_file = MagicMock(spec=VideoFile)
    mock_video_file.get_width_pixels.return_value = width
    mock_video_file.get_height_pixels.return_value = height

    resizer = Resizer()

    # Mock pytorch.get_free_cpu_memory ~7.5 GiB
    with patch("torch.cuda.is_available", return_value=gpu_available), patch(
        "utils.pytorch.get_free_cpu_memory", return_value=8000000000
    ):
        n_batches = resizer._calc_n_batches(
            video_file=mock_video_file,
            num_frames=num_frames,
            face_detect_width=face_detect_width,
            n_face_detect_batches=n_face_detect_batches,
        )

        assert n_batches == expected_batches


@pytest.mark.parametrize(
    "roi, resize_width, resize_height, expected_crop",
    [
        # Test case 1
        (Rect(400, 300, 200, 200), 200, 200, Rect(400, 300, 200, 200)),
        # Test case 2
        (Rect(0, 0, 100, 100), 200, 200, Rect(0, 0, 200, 200)),
        # Test case 3
        (Rect(800, 600, 100, 100), 200, 200, Rect(750, 550, 200, 200)),
        # Test case 4
        (Rect(800, 600, 100, 100), 200, 400, Rect(750, 450, 200, 400)),
    ],
)
def test_calc_crop(roi, resize_width, resize_height, expected_crop):
    resizer = Resizer()
    actual_crop = resizer._calc_crop(roi, resize_width, resize_height)
    assert actual_crop == expected_crop


@pytest.mark.parametrize(
    "segments, expected",
    [
        # Test case 1: No identical segments
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 200, "y": 0, "start_time": 10, "end_time": 20},
            ],
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 200, "y": 0, "start_time": 10, "end_time": 20},
            ],
        ),
        # Test case 2: Two identical segments
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 0, "start_time": 10, "end_time": 20},
            ],
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 20}],
        ),
        # Test case 3: Multiple identical segments
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 0, "start_time": 10, "end_time": 20},
                {"x": 100, "y": 0, "start_time": 20, "end_time": 30},
            ],
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 30}],
        ),
        # Test case 4: Identical X but different Y
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 50, "start_time": 10, "end_time": 20},
            ],
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 100, "y": 50, "start_time": 10, "end_time": 20},
            ],
        ),
        # Test case 5: Single segment
        (
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 10}],
            [{"x": 100, "y": 0, "start_time": 0, "end_time": 10}],
        ),
        # Test case 6: Empty list
        ([], []),
        # Test case 7: Segments with very slight differences in X
        (
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 10},
                {"x": 101, "y": 0, "start_time": 10, "end_time": 20},
            ],
            [
                {"x": 100, "y": 0, "start_time": 0, "end_time": 20},
            ],
        ),
    ],
)
def test_merge_identical_segments(segments, expected):
    mock_video_file = MagicMock(spec=VideoFile)
    mock_video_file.get_width_pixels.return_value = 1000
    mock_video_file.get_height_pixels.return_value = 1000

    resizer = Resizer()
    merged_segments = resizer._merge_identical_segments(segments, mock_video_file)
    assert merged_segments == expected
