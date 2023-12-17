"""
Working with srt files in the local file system.
"""
# current package imports
from .file import File


class SrtFile(File):
    """
    A class for working with srt files in the local file system.
    """

    def __init__(self, srt_file_path: str) -> None:
        """
        Initialize Srt File

        Parameters
        ----------
        srt_file_path: str
            absolute path of a srt file to set SrtFile's path to

        Returns
        -------
        None
        """
        super().__init__(srt_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'SrtFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'SrtFile' as a string.
        """
        return "SrtFile"

    def check_exists(self) -> str or None:
        """
        Checks that SrtFile exists in the file system. Returns None if so, a
        descriptive error message if not.

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if SrtFile exists in the file system, a descriptive error
            message if not.
        """
        # check if the path is a valid File
        error = super().check_exists()
        if error is not None:
            return error

        file_extension = self.get_file_extension()
        if file_extension != "srt":
            return (
                "'{}' is a valid {} but is not a valid {} because it has file "
                "extension '{}' instead of 'srt'.""".format(
                    self._path,
                    super().get_type(),
                    self.get_type(),
                    file_extension
                )
            )

    def create(self, subtitles: list[dict]) -> None:
        """
        Creates a new srt file with the given srt data.

        Parameters
        ----------
        subtitles: list[dict]
            The subtitles to create the srt file from.
        Each entry in the subtitles list should be a dictionary with the following keys:
            - 'text': The text of the subtitle.
            - 'startTime': The start time of the subtitle, in seconds.
            - 'endTime': The end time of the subtitle, in seconds.

        Returns
        -------
        None
        """
        srt_data = ""
        for i, subtitle in enumerate(subtitles):
            text = subtitle["text"]
            start_time = self._convert_time_to_srt_format(subtitle["startTime"])
            end_time = self._convert_time_to_srt_format(subtitle["endTime"])
            srt_data += "{}\n{} --> {}\n{}\n\n".format(
                i + 1, start_time, end_time, text
            )
        self.assert_does_not_exist()
        super().create(srt_data)

    def _convert_time_to_srt_format(self, time_in_seconds: float) -> str:
        """
        Converts the given time in seconds to the SRT time format "hh:mm:ss,ms"

        Parameters
        ----------
        time_in_seconds: float
            The time value to convert, in seconds

        Returns
        -------
        str
            The time value formatted as a string in SRT time format
        """
        hours = int(time_in_seconds // 3600)
        minutes = int((time_in_seconds // 60) % 60)
        seconds = int(time_in_seconds % 60)
        milliseconds = int((time_in_seconds * 1000) % 1000)
        return "{:02}:{:02}:{:02},{:03}".format(hours, minutes, seconds, milliseconds)
