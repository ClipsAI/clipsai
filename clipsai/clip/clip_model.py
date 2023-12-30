class Clip:
    """
    A class to represent the clip object.
    """

    def __init__(self, start_time, end_time, start_char, end_char):
        """
        Constructs all the necessary attributes for the clip object.

        Parameters
        ----------
        start_time: float
            The start time of the clip in seconds.
        end_time: float
            The end time of the clip in seconds.
        start_char: int
            The start character of the clip.
        end_char: int
            The end character of the clip.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._start_char = start_char
        self._end_char = end_char

    @property
    def start_time(self):
        """
        Returns the start time of the clip in seconds.

        Returns
        -------
        float
            The start time of the clip in seconds.
        """
        return self._start_time

    @property
    def end_time(self):
        """
        Returns the end time of the clip in seconds.

        Returns
        -------
        float
            The end time of the clip in seconds.
        """
        return self._end_time

    @property
    def start_char(self):
        """
        Returns the start character of the clip.

        Returns
        -------
        int
            The start character of the clip.
        """
        return self._start_char

    @property
    def end_char(self):
        """
        Returns the end character of the clip.

        Returns
        -------
        int
            The end character of the clip.
        """
        return self._end_char
