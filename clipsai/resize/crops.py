"""
Type definition for Crops and CropSegment.
"""


class Crops:
    """
    Represents the crop and resize information of a video, including original
    dimensions, crop dimensions, and segments of the video that were cropped.

    Segments are represented by CropSegment instances. They provide the x and y position
    of the video to start and crop from, as well as the speakers present in the segment,
    and the start and end time of the segment.

    Attributes:
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
        segments: list["CropSegment"]
        
    ) -> None:
        """
        Initializes a Crops instance.
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
    def segments(self) -> list["CropSegment"]:
        """
        The list of CropSegments.
        """
        return self._segments
    
    def add_segment(self, segment: "CropSegment") -> None:
        """
        Add a new segment to the crops.

        Args:
            segment (CropSegment): The segment to add.
        """
        self._segments.append(segment)

    def remove_segment(self, index: int) -> None:
        """
        Remove a segment from the crops by index.

        Args:
            index (int): Index of the segment to remove.
        """
        if 0 <= index < len(self._segments):
            self._segments.pop(index)
    
    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the Crops instance.

        Returns
        -------
        dict[int, int, int, int, list[dict]]
            originalWidth: int
                original width of the video
            originalHeight: int
                original height of the video
            resizeWidth: int
                resized width of the video
            resizeHeight: int
                resized height of the video
            segments: list
                list of speaker segments (dictionaries) with the following keys
                    speakers: list[int]
                        the speaker labels of the speakers talking in the segment
                    startTime: float
                        the start time of the segment
                    endTime: float
                        the end time of the segment
                    x: int
                        x-coordinate of the top left corner of the resized segment
                    y: int
                        y-coordinate of the top left corner of the resized segment
        """
        return {
            "original_width": self._original_width,
            "original_height": self._original_height,
            "crop_width": self._crop_width,
            "crop_height": self._crop_height,
            "segments": [segment.to_dict() for segment in self._segments]
        }
    
    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the Crops instance,
        detailing original and resized dimensions, and segment information.
        """
        segments_str = ", ".join([str(segment) for segment in self.segments])
        return f"Crops(Original: ({self.original_width}x{self.original_height}), " \
               f"Resized: ({self.crop_width}x{self.crop_height}), Segments: " \
                f"[{segments_str}])"
    
    def __eq__(self, __value: object) -> bool:
        """
        Returns True if the Crops instance is equal to the given value, False otherwise.
        """
        if not isinstance(__value, Crops):
            return False
        return self.original_width == __value.original_width and \
               self.original_height == __value.original_height and \
               self.crop_width == __value.crop_width and \
               self.crop_height == __value.crop_height and \
               self.segments == __value.segments
    
    def __ne__(self, __value: object) -> bool:
        """
        Returns True if the Crops instance is not equal to the given value, False 
        otherwise.
        """
        return not self.__eq__(__value)


class CropSegment:
    """
    Represents a segment of a video that was cropped, including the speakers present,
    timing of the segment, and the crop coordinates.

    Attributes:
        speakers (list[int]): List of speaker IDs present in the segment.
        startTime (float): Start time of the segment in seconds.
        endTime (float): End time of the segment in seconds.
        x (int): The x coordinate of the top left corner of the segment.
        y (int): The y coordinate of the top left corner of the segment.
    """

    def __init__(
        self,
        speakers: list[int],
        startTime: float,
        endTime: float,
        x: int,
        y: int,
    ) -> None:
        """
        Initializes a CropSegment instance.
        """
        self._speakers = speakers
        self._startTime = startTime
        self._endTime = endTime
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
    
    @property
    def x(self) -> int:
        """
        The x coordinate of the top left corner of the segment.
        """
        return self._x
    
    @property
    def y(self) -> int:
        """
        The y coordinate of the top left corner of the segment.
        """
        return self._y
    
    def update_coordinates(self, x: int, y: int) -> None:
        """
        Update the crop coordinates of the segment.

        Args:
            x (int): The new x coordinate.
            y (int): The new y coordinate.
        """
        self._x = x
        self._y = y
    
    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the CropSegment instance.

        Returns
        -------
        segments: list[dict]
            speakers: list[int]
                the speaker labels of the speakers talking in the segment
            startTime: float
                the start time of the segment
            endTime: float
                the end time of the segment
            x: int
                x-coordinate of the top left corner of the resized segment
            y: int
                y-coordinate of the top left corner of the resized segment
        """
        return {
            "speakers": self._speakers,
            "startTime": self._startTime,
            "endTime": self._endTime,
            "x": self._x,
            "y": self._y
        }
    
    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the CropSegment instance,
        detailing speakers, time stamps, and crop coordinates.
        """
        return f"CropSegment(Speakers: {self.speakers}, Start: {self.startTime}, " \
               f"End: {self.endTime}, Coordinates: ({self.x}, {self.y}))"
    
    def __eq__(self, __value: object) -> bool:
        """
        Returns True if the CropSegment instance is equal to the given value, False 
        otherwise.
        """
        if not isinstance(__value, CropSegment):
            return False
        return self.speakers == __value.speakers and \
               self.startTime == __value.startTime and \
               self.endTime == __value.endTime and \
               self.x == __value.x and \
               self.y == __value.y
    
    def __ne__(self, __value: object) -> bool:
        """
        Returns True if the CropSegment instance is not equal to the given value, False 
        otherwise.
        """
        return not self.__eq__(__value)
