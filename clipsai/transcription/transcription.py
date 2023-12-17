"""
Defines an abstract class for transcription classes.

Notes
-----
- Transcription 'language' should be specified using the ISO 639-1 codes:
    https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
"""
# standard library imports
import abc
from datetime import datetime

# local imports
from filesys.json_file import JsonFile


class Transcription(abc.ABC):
    """
    Abstract class defining transcription classes.
    """

    @abc.abstractmethod
    def get_source_software(self) -> str:
        """
        Returns the name of the software used to transcribe the audio

        Parameters
        ----------
        None

        Returns
        -------
        str
            Name of the software used to transcribe the audio
        """
        pass

    @abc.abstractmethod
    def get_time_spawned(self) -> datetime:
        """
        Returns the time the transcription was created

        Parameters
        ----------
        None

        Returns
        -------
        time_spawned: datetime
            the time created as a datetime object
        """
        pass

    @abc.abstractmethod
    def get_language(self) -> str:
        """
        Returns the transcription language

        Parameters
        ----------
        None

        Returns
        -------
        str
            ISO 639-1 code of the transcription language
        """
        pass

    @abc.abstractmethod
    def get_text(self, predicted: bool) -> str:
        """
        Returns the full text of the transcription

        Parameters
        ----------
        predicted: bool
            predicted text if True, edited text if False

        Returns
        -------
        text: str
            the full text of the predicted or edited transcription
        """
        pass

    @abc.abstractmethod
    def store_as_json_file(self, file_path: str) -> JsonFile:
        """
        Stores the transcription as a json file

        - If 'file_path' already exists, it is overwritten only if the transcription
        metadata is compatible with the existing file

        Parameters
        ----------
        file_path: str
            absolute file path to store the transcription as a json file

        Returns
        -------
        None
        """
        pass
