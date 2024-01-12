"""
A class to represent the Crops object.
"""
# current package imports
from .segment import Segment


class Crops:
    """
    Represents the crop and resize information of a video, including original
    dimensions, crop dimensions, and segments of the video that were cropped.

    Segments are represented by Segment instances. They provide the x and y position
    of the video to start and crop from, as well as the speakers present in the segment,
    and the start and end time of the segment.

    Attributes
    ----------
    original_width (int): Original width of the video.
    original_height (int): Original height of the video.
    crop_width (int): Width of the cropped video.
    crop_height (int): Height of the cropped video.
    segments (list[CropSegment]): List of cropped segments.
    """

    def __init__(
        self,
        original_width: int,
        original_height: int,
        crop_width: int,
        crop_height: int,
        segments: list["Segment"],
    ) -> None:
        """
        Initializes a Crops instance.

        Parameters
        ----------
        original_width: int
            Original width of the video.
        original_height: int
            Original height of the video.
        crop_width: int
            Width of the cropped video.
        crop_height: int
            Height of the cropped video.
        segments: list[Segment]
            List of cropped segments.
        """
        self._original_width = original_width
        self._original_height = original_height
        self._crop_width = crop_width
        self._crop_height = crop_height
        self._segments = segments

    @property
    def original_width(self) -> int:
        """
        The width of the original video.
        """
        return self._original_width

    @property
    def original_height(self) -> int:
        """
        The height of the original video.
        """
        return self._original_height

    @property
    def crop_width(self) -> int:
        """
        The width of the cropped video.
        """
        return self._crop_width

    @property
    def crop_height(self) -> int:
        """
        The height of the cropped video.
        """
        return self._crop_height

    @property
    def segments(self) -> list["Segment"]:
        """
        The list of Segments.
        """
        return self._segments

    def copy(self) -> "Crops":
        """
        Returns a copy of the Crops instance.
        """
        return Crops(
            self._original_width,
            self._original_height,
            self._crop_width,
            self._crop_height,
            [segment.copy() for segment in self._segments],
        )

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the Crops instance.
        """
        return {
            "original_width": self._original_width,
            "original_height": self._original_height,
            "crop_width": self._crop_width,
            "crop_height": self._crop_height,
            "segments": [segment.to_dict() for segment in self._segments],
        }

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the Crops instance,
        detailing original and resized dimensions, and segment information.
        """
        segments_str = ", ".join([str(segment) for segment in self.segments])
        return (
            f"Crops(Original: ({self.original_width}x{self.original_height}), "
            f"Resized: ({self.crop_width}x{self.crop_height}), Segments: "
            f"[{segments_str}])"
        )

    def __eq__(self, __value: object) -> bool:
        """
        Returns True if the Crops instance is equal to the given value, False otherwise.
        """
        if not isinstance(__value, Crops):
            return False
        return (
            self.original_width == __value.original_width
            and self.original_height == __value.original_height
            and self.crop_width == __value.crop_width
            and self.crop_height == __value.crop_height
            and self.segments == __value.segments
        )

    def __ne__(self, __value: object) -> bool:
        """
        Returns True if the Crops instance is not equal to the given value, False
        otherwise.
        """
        return not self.__eq__(__value)

    def __bool__(self) -> bool:
        """
        Returns True if the Crops instance is not empty, False otherwise.
        """
        return bool(self.segments)
