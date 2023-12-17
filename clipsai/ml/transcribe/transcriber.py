"""
Defines an abstract class for transcribing audio files.
"""
# standard library imports
import abc

# local imports
from media.audio_file import AudioFile
from transcription.transcription import Transcription


class Transcriber(abc.ABC):
    """
    Abstract base class defining classes that transcribe audio files.
    """

    @abc.abstractmethod
    def transcribe(self, media_file: AudioFile) -> Transcription:
        """
        Transcribes of the media file

        Parameters
        ----------
        media_file: AudioFile
            the media file to transcribe

        Returns
        -------
        Transcription
            the transcription
        """
        pass
