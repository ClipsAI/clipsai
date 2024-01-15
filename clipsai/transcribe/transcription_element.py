"""
A base class to represent an element (sentence, word, or character) in a transcription.
"""


class TranscriptionElement:
    """
    Represents an element (sentence, word, or character) in a transcription.

    Attributes
    ----------
    start_time (float): The start time of the element in seconds.
    end_time (float): The end time of the element in seconds.
    start_char (int): The start character in the transcription of the element.
    end_char (int): The end character in the transcription of the element.
    text (str): The text of the element.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_char: int,
        end_char: int,
        text: str,
    ):
        """
        Constructs all the necessary attributes for the element object.

        Parameters
        ----------
        start_time: float
            The start time of the element in seconds.
        end_time: float
            The end time of the element in seconds.
        start_char: int
            The start character in the transcription of the element.
        end_char: int
            The end character in the transcription of the element.
        text: str
            The text of the element.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._start_char = start_char
        self._end_char = end_char
        self._text = text

    @property
    def start_time(self) -> float:
        """
        Returns the start time of the element in seconds.
        """
        return self._start_time

    @property
    def end_time(self) -> float:
        """
        Returns the end time of the element in seconds.
        """
        return self._end_time

    @property
    def start_char(self) -> int:
        """
        Returns the start character in the transcription of the element.
        """
        return self._start_char

    @property
    def end_char(self) -> int:
        """
        Returns the end character in the transcription of the element.
        """
        return self._end_char

    @property
    def text(self) -> str:
        """
        Returns the text of the element.
        """
        return self._text

    def to_dict(self) -> dict:
        """
        Returns the attributes of the element as a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            The attributes of the element in a dictionary with the following keys:
                start_time: float
                    start time of the element in seconds
                end_time: float
                    end time of the element in seconds
                start_char: int
                    start character of the transcription of the element in the full text
                end_char: int
                    end character of the transcription of the element in the full text
                text: str
                    text of the element
        """
        return {
            "start_time": self._start_time,
            "end_time": self._end_time,
            "start_char": self._start_char,
            "end_char": self._end_char,
            "text": self._text,
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the element.
        """
        return self._text

    def __eq__(self, other: "TranscriptionElement") -> bool:
        """
        Returns True if the element is equal to the other element, False otherwise.
        """
        return (
            self._start_time == other.start_time
            and self._end_time == other.end_time
            and self._start_char == other.start_char
            and self._end_char == other.end_char
            and self._text == other.text
        )

    def __ne__(self, __value: object) -> bool:
        """
        Returns True if the element is not equal to the other element, False otherwise.
        """
        return not self.__eq__(__value)

    def __bool__(self) -> bool:
        """
        Returns True if the element is not empty, False otherwise.
        """
        return bool(self._text)


class Sentence(TranscriptionElement):
    """
    Represents a sentence in a transcription.

    Attributes
    ----------
    start_time (float): The start time of the sentence in seconds.
    end_time (float): The end time of the sentence in seconds.
    start_char (int): The start character in the transcription of the sentence.
    end_char (int): The end character in the transcription of the sentence.
    text (str): The text of the sentence.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_char: int,
        end_char: int,
        text: str,
    ):
        """
        Constructs all the necessary attributes for the sentence object.

        Parameters
        ----------
        start_time: float
            The start time of the sentence in seconds.
        end_time: float
            The end time of the sentence in seconds.
        start_char: int
            The index of the sentence's start character in the full text
        end_char: int
            The index of the sentence's end character in the full text
        text: str
            The text of the sentence.
        """
        super().__init__(start_time, end_time, start_char, end_char, text)


class Word(TranscriptionElement):
    """
    Represents a word in a transcription.

    Attributes
    ----------
    start_time (float): The start time of the word in seconds.
    end_time (float): The end time of the word in seconds.
    start_char (int): The start character in the transcription of the word.
    end_char (int): The end character in the transcription of the word.
    text (str): The text of the word.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_char: int,
        end_char: int,
        text: str,
    ):
        """
        Constructs all the necessary attributes for the word object.

        Parameters
        ----------
        start_time: float
            The start time of the word in seconds.
        end_time: float
            The end time of the word in seconds.
        start_char: int
            The index of the word's start character in the full text
        end_char: int
            The index of the word's end character in the full text
        text: str
            The text of the word.
        """
        super().__init__(start_time, end_time, start_char, end_char, text)


class Character:
    """
    Represents a character in a transcription.

    Attributes
    ----------
    start_time (float): The start time of the character in seconds.
    end_time (float): The end time of the character in seconds.
    word_index (int): The index of the word in the transcription of the character.
    sentence_index (int): The index of the sentence in the transcription of the
        character.
    text (str): The text of the character.
    """

    def __init__(
        self,
        start_time: float,
        end_time: float,
        word_index: int,
        sentence_index: int,
        text: str,
    ):
        """
        Constructs all the necessary attributes for the character object.

        Parameters
        ----------
        start_time: float
            The start time of the character in seconds.
        end_time: float
            The end time of the character in seconds.
        word_index: int
            The index of the word in the transcription of the character.
        sentence_index: int
            The index of the sentence in the transcription of the character.
        text: str
            The text of the character.
        """
        self._start_time = start_time
        self._end_time = end_time
        self._word_index = word_index
        self._sentence_index = sentence_index
        self._text = text

    @property
    def start_time(self) -> float:
        """
        Returns the start time of the character in seconds.
        """
        return self._start_time

    @property
    def end_time(self) -> float:
        """
        Returns the end time of the character in seconds.
        """
        return self._end_time

    @property
    def word_index(self) -> int:
        """
        Returns the index of the word in the transcription of the character.
        """
        return self._word_index

    @property
    def sentence_index(self) -> int:
        """
        Returns the index of the sentence in the transcription of the character.
        """
        return self._sentence_index

    @property
    def text(self) -> str:
        """
        Returns the text of the character.
        """
        return self._text

    def to_dict(self) -> dict:
        """
        Returns the attributes of the element as a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            The attributes of the element in a dictionary with the following keys:
                start_time: float
                    start time of the element in seconds
                end_time: float
                    end time of the element in seconds
                word_index: int
                    index of the word in the transcription that contains this character
                sentence_index: int
                    index of the sentence in the transcription that contains this
                    character
                text: str
                    text of the element
        """
        return {
            "start_time": self._start_time,
            "end_time": self._end_time,
            "word_index": self._word_index,
            "sentence_index": self._sentence_index,
            "text": self._text,
        }

    def __str__(self) -> str:
        """
        Returns a string representation of the character.
        """
        return self._text

    def __eq__(self, other: "Character") -> bool:
        """
        Returns True if the element is equal to the other character, False otherwise.
        """
        return (
            self._start_time == other.start_time
            and self._end_time == other.end_time
            and self._word_index == other.word_index
            and self._sentence_index == other.sentence_index
            and self._text == other.text
        )

    def __ne__(self, __value: object) -> bool:
        """
        Returns True if the element is not equal to the other element, False otherwise.
        """
        return not self.__eq__(__value)

    def __bool__(self) -> bool:
        """
        Returns True if the element is not empty, False otherwise.
        """
        return bool(self._text)
