"""
Type definition for SpeakerSegment.
"""

class SpeakerSegment:
    """
    Represents a segment of audio with associated speaker information.
    """

    def __init__(self, speakers: list[int], startTime: float, endTime: float) -> None:
        """Initializes a SpeakerSegment instance."""
        self._speakers = speakers
        self._startTime = startTime
        self._endTime = endTime

    @property
    def speakers(self) -> list[int]:
        """
        The list of speakers in the segment.
        """
        return self._speakers

    @property
    def startTime(self) -> float:
        """
        The start time of the segment.
        """
        return self._startTime

    @property
    def endTime(self) -> float:
        """
        The end time of the segment.
        """
        return self._endTime
