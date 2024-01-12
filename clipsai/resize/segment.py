"""
A class to represent the Segment object.
"""


class Segment:
    """
    Represents a segment of a video that was cropped, including the speakers present,
    timing of the segment, and the crop coordinates.

    Attributes
    ----------
        speakers (list[int]): List of speaker IDs present in the segment.
        start_time (float): Start time of the segment in seconds.
        end_time (float): End time of the segment in seconds.
        x (int): The x coordinate of the top left corner of the segment.
        y (int): The y coordinate of the top left corner of the segment.
    """

    def __init__(
        self,
        speakers: list[int],
        start_time: float,
        end_time: float,
        x: int,
        y: int,
    ) -> None:
        """
        Initializes a CropSegment instance.

        Parameters
        ----------
        speakers: list[int]
            List of speaker IDs present in the segment.
        start_time: float
            Start time of the segment in seconds.
        end_time: float
            End time of the segment in seconds.
        x: int
            The x coordinate of the top left corner of the segment.
        y: int
            The y coordinate of the top left corner of the segment.
        """
        self._speakers = speakers
        self._start_time = start_time
        self._end_time = end_time
        self._x = x
        self._y = y

    @property
    def speakers(self) -> list[int]:
        """
        Returns a list of speaker identifiers in this segment. Each identifier
        uniquely represents a speaker in the video.
        """
        return self._speakers

    @property
    def start_time(self) -> float:
        """
        The start time of the segment.
        """
        return self._start_time

    @property
    def end_time(self) -> float:
        """
        The end time of the segment.
        """
        return self._end_time

    @property
    def x(self) -> int:
        """
        The x coordinate of the top left corner of the resized segment frame.
        """
        return self._x

    @property
    def y(self) -> int:
        """
        The y coordinate of the top left corner of the segment.
        """
        return self._y

    def copy(self) -> "Segment":
        """
        Returns a copy of the Segment instance.
        """
        return Segment(
            speakers=self._speakers.copy(),
            start_time=self._start_time,
            end_time=self._end_time,
            x=self._x,
            y=self._y,
        )

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the Segment instance.
        """
        return {
            "speakers": self._speakers,
            "start_time": self._start_time,
            "end_time": self._end_time,
            "x": self._x,
            "y": self._y,
        }

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the CropSegment instance,
        detailing speakers, time stamps, and crop coordinates.
        """
        return (
            f"Segment(speakers: {self._speakers}, start: {self._start_time}, "
            f"end: {self._end_time}, coordinates: ({self._x}, {self._y}))"
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the CropSegment instance.
        """
        return (
            f"Segment(speakers: {self._speakers}, start: {self._start_time}, "
            f"end: {self._end_time}, coordinates: ({self._x}, {self._y}))"
        )

    def __eq__(self, __value: object) -> bool:
        """
        Returns True if the CropSegment instance is equal to the given value, False
        otherwise.
        """
        if not isinstance(__value, Segment):
            return False
        return (
            self._speakers == __value.speakers
            and self._start_time == __value.start_time
            and self._end_time == __value.end_time
            and self._x == __value.x
            and self._y == __value.y
        )

    def __ne__(self, __value: object) -> bool:
        """
        Returns True if the CropSegment instance is not equal to the given value, False
        otherwise.
        """
        return not self.__eq__(__value)

    def __bool__(self) -> bool:
        """
        Returns True if the CropSegment instance is not empty, False otherwise.
        """
        return (
            bool(self._speakers)
            and bool(self._start_time)
            and bool(self._end_time)
            and bool(self._x)
            and bool(self._y)
        )
