"""
A class to represent the clip object.
"""


class Clip:
    """
    Represents a clip of a video or audio file.

    Attributes
    ----------
    start_time (float): The start time of the clip in seconds.
    end_time (float): The end time of the clip in seconds.
    start_char (int): The start character in the transcription of the clip.
    end_char (int): The end character in the transcription of the clip.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_char: int,
        end_char: int,
    ):
        """
        Initializes the Clip class attributes with the given values.

        Parameters
        ----------
        start_time: float
            The start time of the clip in seconds.
        end_time: float
            The end time of the clip in seconds.
        start_char: int
            The start character in the transcription of the clip.
        end_char: int
            The end character in the transcription of the clip.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._start_char = start_char
        self._end_char = end_char

    @property
    def start_time(self) -> float:
        """
        Returns the start time of the clip in seconds.
        """
        return self._start_time

    @property
    def end_time(self) -> float:
        """
        Returns the end time of the clip in seconds.
        """
        return self._end_time

    @property
    def start_char(self) -> int:
        """
        Returns the start character in the transcription of the clip.
        """
        return self._start_char

    @property
    def end_char(self) -> int:
        """
        Returns the end character in the transcription of the clip.
        """
        return self._end_char

    def copy(self) -> "Clip":
        """
        Returns a copy of the clip.
        """
        return Clip(self._start_time, self._end_time, self._start_char, self._end_char)

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the clip.
        """
        return {
            "start_time": self._start_time,
            "end_time": self._end_time,
            "start_char": self._start_char,
            "end_char": self._end_char,
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the clip.
        """
        return (
            f"Clip(start_time={self._start_time}, end_time={self._end_time}, "
            f"start_char={self._start_char}, end_char={self._end_char})"
        )

    def __eq__(self, __other: object) -> bool:
        """
        Returns True if the clip is equal to the given value, False otherwise.

        Parameters
        ----------
        other: object
            The value to compare the clip to.
        """
        if not isinstance(__other, Clip):
            return False
        return (
            self._start_time == __other.start_time
            and self._end_time == __other.end_time
            and self._start_char == __other.start_char
            and self._end_char == __other.end_char
        )

    def __ne__(self, __other: object) -> bool:
        """
        Returns True if the clip is not equal to the given value, False otherwise.

        Parameters
        ----------
        other: object
            The value to compare the clip to.
        """
        return not self.__eq__(__other)

    def __bool__(self) -> bool:
        """
        Returns True if the clip is not empty, False otherwise.
        """
        return (
            bool(self._start_time)
            and bool(self._end_time)
            and bool(self._start_char)
            and bool(self._end_char)
        )
