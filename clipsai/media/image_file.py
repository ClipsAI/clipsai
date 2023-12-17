"""
Working with image files.
"""
# current package imports
from .media_file import MediaFile


class ImageFile(MediaFile):
    """
    A class for working with image files.
    """

    def __init__(self, image_file_path: str) -> None:
        """
        Initialize ImageFile

        Parameters
        ----------
        image_file_path: str
            absolute path to an image file

        Returns
        -------
        None
        """
        super().__init__(image_file_path)

    def get_type(self) -> str:
        """
        Returns the object type 'ImageFile' as a string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            Object type 'ImageFile' as a string.
        """
        return "ImageFile"

    def check_exists(self) -> None:
        """
        Checks that the ImageFile exists in the file system. Returns None if so, a
        descriptive error message if not

        Parameters
        ----------
        None

        Returns
        -------
        str or None
            None if the ImageFile still exists in the file system, a descriptive error
            message if not
        """
        # check if it's a media file
        msg = super().check_exists()
        if msg is not None:
            return msg

        # check if it's an image file
        media_file = MediaFile(self._path)
        if media_file.has_audio_stream():
            return (
                "'{}' is a valid {} but is not a valid {} since the file contains an "
                "audio stream.".format(self._path, super().get_type(), self.get_type())
            )

        return None

    def get_stream_info(self, stream_field: str) -> str or None:
        """
        Gets stream information

        Parameters
        ----------
        stream_field: str
            the information about the stream you want to know ('duration' for duration
            in seconds, 'r_frame_rate' for frame rate as a precise fraction, 'width' for
            number of horizontal pixels, 'height' for number of vertical pixels,
            'pix_fmt' for pixel format, 'bit_rate' for bit rate)

        Returns
        -------
        str
            stream information
        """
        self.assert_exists()
        return super().get_stream_info("v:0", stream_field)
