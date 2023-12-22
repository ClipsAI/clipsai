"""
Defines an abstract class for finding clips within some media.
"""
# local package imports
from ..utils.pytorch import get_compute_device, assert_compute_device_available


class ClipFinder:
    """
    An abstract class defining classes that finding clips within some media.
    """

    def __init__(
        self,
        min_clip_duration_secs: int,
        max_clip_duration_secs: int,
        device: str = None,
    ) -> None:
        """
        Parameters
        ----------
        min_clip_duration_secs: int
            minimum clip length for a clip to be created
        max_clip_duration_secs: int
            max clip length for a clip to be created
        """
        self._min_clip_duration_secs = min_clip_duration_secs
        self._max_clip_duration_secs = max_clip_duration_secs

        if device is None:
            device = get_compute_device()
        assert_compute_device_available(device)
        self._device = device
