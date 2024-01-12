"""
Transcriptions generated using WhisperX.

Notes
-----
- Character, word, and sentence level time stamps are available
- NLTK used for tokenizing sentences
- WhisperX GitHub: https://github.com/m-bain/whisperX
"""
# standard library imports
from __future__ import annotations
from datetime import datetime
import logging

# current package imports
from .exceptions import TranscriptionError
from .transcription_element import Sentence, Word, Character

# local imports
from clipsai.filesys.json_file import JSONFile
from clipsai.filesys.manager import FileSystemManager
from clipsai.utils.type_checker import TypeChecker

# 3rd party imports
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


class Transcription:
    """
    A class for whisperx transcription data viewing, storage, and manipulation.
    """

    def __init__(
        self,
        transcription: dict or JSONFile,
    ) -> None:
        """
        Initialize Transcription Class.

        Parameters
        ----------
        transcription: dict or JSONFile
            - a dictionary object containing whisperx transcription
            - a JSONFile containing a whisperx transcription

        Returns
        -------
        None
        """
        self._fs_manager = FileSystemManager()

        # the below are set in __init_from_json_file() or __init_from_dict()
        self._source_software = None
        self._created_time = None
        self._language = None
        self._num_speakers = None
        self._char_info = None
        # derived from char_info data
        self._text = None
        self._word_info = None
        self._sentence_info = None

        self._type_checker = TypeChecker()
        self._type_checker.assert_type(transcription, "transcription", (dict, JSONFile))

        if isinstance(transcription, JSONFile):
            self._init_from_json_file(transcription)
        else:
            self._init_from_dict(transcription)

    @property
    def source_software(self) -> str:
        """
        Returns the name of the software used to transcribe the audio
        """
        return self._source_software

    @property
    def created_time(self) -> datetime:
        """
        Returns the time the transcription was created
        """
        return self._created_time

    @property
    def language(self) -> str:
        """
        Returns the language of the transcription.
        """
        return self._language

    @property
    def start_time(self) -> float:
        """
        Returns the start time of the transcript in seconds.
        """
        return 0.0

    @property
    def end_time(self) -> float:
        """
        Returns the end time of the transcript in seconds.
        """
        char_info = self.get_char_info()
        for i in range(len(char_info) - 1, -1, -1):
            if char_info[i]["end_time"] is not None:
                return char_info[i]["end_time"]
            if char_info[i]["start_time"] is not None:
                return char_info[i]["start_time"]

    @property
    def text(self) -> str:
        """
        Returns the full text of the transcription
        """
        return self._text

    @property
    def characters(self) -> list[Character]:
        """
        Returns a list of characters from the text. The characters are represented as
        Character objects. The list is ordered by the start time of the characters.
        """
        chars = []
        for char_info in self.get_char_info():
            chars.append(
                Character(
                    start_time=char_info["start_time"],
                    end_time=char_info["end_time"],
                    word_index=char_info["work_index"],
                    sentence_index=char_info["sentence_index"],
                    text=char_info["char"],
                )
            )
        return chars

    @property
    def words(self) -> list[Word]:
        """
        Returns a list words from the text. The words are represented as Word objects.
        The lis is ordered by the start time of the words.
        """
        words = []
        for word_info in self.get_word_info():
            words.append(
                Word(
                    start_time=word_info["start_time"],
                    end_time=word_info["end_time"],
                    start_char=word_info["start_char"],
                    end_char=word_info["end_char"],
                    text=word_info["word"],
                )
            )
        return words

    @property
    def sentences(self) -> list[Sentence]:
        """
        Returns a list of sentences from the text. The sentences are represented as
        Sentence objects. The list is ordered by the start time of the sentences.
        """
        sentences = []
        for sentence_info in self.get_sentence_info():
            sentences.append(
                Sentence(
                    start_time=sentence_info["start_time"],
                    end_time=sentence_info["end_time"],
                    start_char=sentence_info["start_char"],
                    end_char=sentence_info["end_char"],
                )
            )
        return sentences

    def get_char_info(
        self,
        start_time: float = None,
        end_time: float = None,
    ) -> list:
        """
        Returns the character info of the transcription

        Parameters
        ----------
        start_time: float
            start time of the character info in seconds.
            If None, returns all character info
        end_time: float
            end time of the character info in seconds.
            If None, returns all character info

        Returns
        -------
        list[dict]
            list of dictionaries where each dictionary contains
            info about a single character in the text
        """
        self._assert_valid_times(start_time, end_time)
        char_info = self._char_info

        # return all char info
        if start_time is None and end_time is None:
            return char_info
        # return subset of char info
        else:
            start_index = self.find_char_index(start_time, type_of_time="start")
            end_index = self.find_char_index(end_time, type_of_time="end")
            return char_info[start_index : end_index + 1]

    def get_word_info(
        self,
        start_time: float = None,
        end_time: float = None,
    ) -> list:
        """
        Returns the word info of the text

        Parameters
        ----------
        start_time: float
            start time of the word info in seconds.
            If None, returns all word info
        end_time: float
            end time of the word info in seconds.
            If None, returns all word info

        Returns
        -------
        list[dict]
            list of dictionaries where each dictionary contains
            info about a single word in the text
        """
        self._assert_valid_times(start_time, end_time)

        # get all word info
        word_info = self._word_info

        # return all word info
        if start_time is None and end_time is None:
            return word_info
        # return subset of word info
        else:
            start_index = self.find_word_index(start_time, type_of_time="start")
            end_index = self.find_word_index(end_time, type_of_time="end")
            return word_info[start_index : end_index + 1]

    def get_sentence_info(
        self,
        start_time: float = None,
        end_time: float = None,
    ) -> list:
        """
        Returns the sentence information of the text.

        Parameters
        ----------
        start_time: float
            start time of the sentence info in seconds. If None, returns all word info
        end_time: float
            end time of the sentence info in seconds. If None, returns all word info

        Returns
        -------
        list[dict]
            list of dictionaries where each dictionary contains info about a single
            sentence in the text
        """
        self._assert_valid_times(start_time, end_time)
        sentence_info = self._sentence_info

        # return all word info
        if start_time is None and end_time is None:
            return sentence_info
        # return subset of word info
        else:
            start_index = self.find_sentence_index(start_time, type_of_time="start")
            end_index = self.find_sentence_index(end_time, type_of_time="end")
            return sentence_info[start_index : end_index + 1]

    def find_char_index(self, target_time: float, type_of_time: str) -> int:
        """
        Finds the index in the transcript's character info who's start or end time is
        closest to 'target_time' (seconds).

        Parameters
        ----------
        target_time: float
            The time in seconds to search for.
        type_of_time: str
            A string that specifies the type of time we're searching for.
            If 'start', the function returns the index of character with the closest
            start time before 'target_time'.
            If 'end', the function returns the index of the character with the closest
            end time after target time.

        Returns
        -------
        int
            The index of char_info that is closest to 'target_time'
        """
        return self._find_index(self.get_char_info(), target_time, type_of_time)

    def find_word_index(self, target_time: float, type_of_time: str) -> int:
        """
        Finds the index in the transcript's word info who's start or end time is closest
        to 'target_time' (seconds)

        Parameters
        ----------
        target_time: float
            The time in seconds to search for.
        type_of_time: str
            A string that specifies the type of time we're searching for.
            If 'start', the function returns the index of word with the closest start
            time before 'target_time'.
            If 'end', the function returns the index of the word with the closest end
            time after target time.

        Returns
        -------
        int
            The index of word_info that is closest to 'target_time'.
        """
        return self._find_index(self.get_word_info(), target_time, type_of_time)

    def find_sentence_index(self, target_time: float, type_of_time: str) -> int:
        """
        Finds the index in the transcript's sentence info who's start or end time is
        closest to 'target_time' (seconds).

        Parameters
        ----------
        target_time: float
            The time in seconds to search for.
        type_of_time: str
            A string that specifies the type of time we're searching for.
            If 'start', the function returns the index of sentence with the closest
            start time before 'target_time'.
            If 'end', the function returns the index of the sentence with the closest
            end time after target time.

        Returns
        -------
        int
            The index of word_info that is closest to 'target_time'
        """
        return self._find_index(self.get_sentence_info(), target_time, type_of_time)

    def store_as_json_file(self, file_path: str) -> JSONFile:
        """
        Stores the transcription as a json file. 'file_path' is overwritten if already
        exists.

        Parameters
        ----------
        file_path: str
            absolute file path to store the transcription as a json file

        Returns
        -------
        JSONFile
        """
        json_file = JSONFile(file_path)
        json_file.assert_has_file_extension("json")
        self._fs_manager.assert_parent_dir_exists(json_file)

        # delete file if it exists
        json_file.delete()

        # only store necessary data
        char_info_needed_for_storage = []
        for char_info in self._char_info:
            char_info_needed_for_storage.append(
                {
                    "char": char_info["char"],
                    "start_time": char_info["start_time"],
                    "end_time": char_info["end_time"],
                    "speaker": char_info["speaker"],
                }
            )

        transcription_dict = {
            "source_software": self._source_software,
            "time_created": str(self._created_time),
            "language": self._language,
            "num_speakers": self._num_speakers,
            "char_info": char_info_needed_for_storage,
        }

        json_file.create(transcription_dict)
        return json_file

    def print_char_info(self) -> None:
        """
        Pretty prints the character info for easy viewing

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        title = "Character Info"
        print(title)
        print("-" * len(title))
        for i, char_info in enumerate(self.get_char_info()):
            print("char: {}".format(char_info["char"]))
            print("start_time: {}".format(char_info["start_time"]), end=" | ")
            print("end_time: {}".format(char_info["end_time"]))
            print("index: {}".format(i), end=" | ")
            print("word_index: {}".format(char_info["work_index"]), end=" | ")
            print("sentence_index: {}\n".format(char_info["sentence_index"]))

    def print_word_info(self) -> None:
        """
        Pretty prints the word info for easy viewing

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        title = "Word Info"
        print(title)
        print("-" * len(title))
        for i, word_info in enumerate(self.get_word_info()):
            print("word: '{}'".format(word_info["word"]), end=" | ")
            print("word_index: {}".format(i))
            print("speaker: {}".format(word_info["speaker"]))
            print("start_time: {}".format(word_info["start_time"]), end=" | ")
            print("end_time: {}".format(word_info["end_time"]))
            print("start_char: {}".format(word_info["start_char"]), end=" | ")
            print("end_char: {}\n".format(word_info["end_char"]))

    def print_sentence_info(self) -> None:
        """
        Pretty prints the sentence info for easy viewing

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        title = "Sentence Info"
        print(title)
        print("-" * len(title))
        for i, sentence_info in enumerate(self.get_sentence_info()):
            print("sentence: '{}'".format(sentence_info["sentence"]))
            print("sentence_index: {}".format(i))
            print("start_char: {}".format(sentence_info["start_char"]), end=" | ")
            print("end_char: {}".format(sentence_info["end_char"]))
            print("start_time: {}".format(sentence_info["start_time"]), end=" | ")
            print("end_time: {}\n".format(sentence_info["end_time"]))

    def _find_index(
        self, transcript_info: list[dict], target_time: float, type_of_time: str
    ) -> int:
        """
        Finds the index in some transcript info who's start or end time is closest to
        'target_time' (seconds).

        Parameters
        ----------
        transcript_info: list[dict]
            list of dictionaries where each dictionary contains info about a single
            character, word, or sentence in the text
        target_time: float
            The time in seconds to search for.
        type_of_time: str
            A string that specifies the type of time we're searching for.
            If 'start', the function returns the index with the closest start time
            before 'target_time'.
            If 'end', the function returns the index with the closest end time after
            target time.

        Returns
        -------
        int
            The index that is closest to 'target_time'
        """
        transcript_start = self.start_time
        transcript_end = self.end_time
        if (transcript_start <= target_time <= transcript_end) is False:
            err = (
                "target_time '{}' seconds is not within the range of the transcript "
                "times: {} - {}".format(target_time, self.start_time, self.end_time)
            )
            logging.error(err)
            raise TranscriptionError(err)

        left, right = 0, len(transcript_info) - 1
        while left <= right:
            mid = left + (right - left) // 2
            start_time = transcript_info[mid]["start_time"]
            end_time = transcript_info[mid]["end_time"]

            if start_time <= target_time <= end_time:
                return mid
            elif target_time > end_time:
                left = mid + 1
            elif target_time < start_time:
                right = mid - 1

        if type_of_time == "start":
            return left - 1 if left == len(transcript_info) else left
        else:
            return right + 1 if right == -1 else right

    def _init_from_json_file(self, json_file: JSONFile) -> None:
        """
        Initializes the transcription object from an existing json file

        Parameters
        ----------
        json_file: JSONFile
            a json file with whisperx transcription data

        Returns
        -------
        None
        """
        self._type_checker.assert_type(json_file, "json_file", JSONFile)
        json_file.assert_exists()
        transcription_data = json_file.read()
        self._init_from_dict(transcription_data)

    def _init_from_dict(self, transcription: dict) -> None:
        """
        Initializes the transcription object from a dictionary

        Parameters
        ----------
        transcription: dict
            a dictionary containing all the fields needed to initialize
            WhisperXTranscription

        Returns
        -------
        None

        Raises
        ------
        ValueError: transcript_dict doesn't contain proper fields for initialization
        TypeError: transcript_dict contains fields of the wrong type
        """
        self._assert_valid_transcription_data(transcription)

        if isinstance(transcription["time_created"], str):
            transcription["time_created"] = datetime.strptime(
                transcription["time_created"], "%Y-%m-%d %H:%M:%S.%f"
            )

        self._created_time = transcription["time_created"]
        self._source_software = transcription["source_software"]
        self._language = transcription["language"]
        self._num_speakers = transcription["num_speakers"]
        self._char_info = transcription["char_info"]
        # derived data
        self._build_text()
        self._build_word_info()
        self._build_sentence_info()

    def _assert_valid_transcription_data(self, transcription: dict) -> None:
        """
        Raises exceptions if the json file contains incompatible data

        Parameters
        ----------
        transcription: dict
            transcription data to be checked

        Returns
        -------
        None
        """

        # ensure transcription has valid keys and datatypes
        transcription_keys_correct_data_types = {
            "source_software": (str),
            "time_created": (datetime, str),
            "language": (str),
            "num_speakers": (int, type(None)),
            "char_info": (list),
        }
        self._type_checker.assert_dict_elems_type(
            transcription, transcription_keys_correct_data_types
        )

        # ensure char_info contains dictionaries
        for char_info in transcription["char_info"]:
            self._type_checker.assert_type(char_info, "char_info", dict)

        # ensure char_info has valid keys and datatypes
        char_dict_keys_correct_data_types = {
            "char": (str),
            "start_time": (float, type(None)),
            "end_time": (float, type(None)),
            "speaker": (int, type(None)),
        }
        for char_dict in transcription["char_info"]:
            self._type_checker.are_dict_elems_of_type(
                char_dict,
                char_dict_keys_correct_data_types,
            )

    def _build_text(self) -> str:
        """
        Builds the text from the char_info

        Parameters
        ----------
        None

        Returns
        -------
        str:
            the full text built from the char_info
        """
        text = ""
        for char_info in self.get_char_info():
            text += char_info["char"]

        self._text = text

    def _build_word_info(self) -> list[dict]:
        """
        Builds the word_info from the char_info

        Parameters
        ----------
        None

        Returns
        -------
        list[dict]:
            the word_info built from the char_info
        """
        char_info = self.get_char_info()

        # final destination for word_info
        word_info = []

        # current word
        cur_word = ""
        cur_word_start_char_idx = None
        cur_word_start_time = None
        cur_word_end_time = None

        # helper variables
        cur_word_idx = 0
        prev_char_info = {
            "char": " ",  # set to space so first char is always a word start
            "start_time": None,
            "end_time": None,
            "speaker": None,
        }
        last_recorded_time = 0

        for i, cur_char_info in enumerate(char_info):
            cur_char = cur_char_info["char"]
            prev_char = prev_char_info["char"]

            if self._is_word_start(prev_char, cur_char):
                cur_word = ""
                cur_word_start_char_idx = i
                if cur_char_info["start_time"] is not None:
                    cur_word_start_time = cur_char_info["start_time"]
                else:
                    cur_word_start_time = last_recorded_time

            if self._is_word_end(prev_char, cur_char):
                new_word_info = {
                    "word": cur_word,
                    "start_char": cur_word_start_char_idx,
                    # prev_char is the actual last char of this word but python
                    # slicing is non-inclusive so we use the index of cur_char (+1)
                    "end_char": i,
                    "start_time": cur_word_start_time,
                    "end_time": cur_word_end_time,
                    "speaker": None,
                }
                word_info.append(new_word_info)

                cur_word_idx += 1
                # reset word info
                cur_word_start_char_idx = None
                cur_word = ""

            # update char info
            cur_char_info["work_index"] = cur_word_idx

            # update word info
            if cur_char_info["end_time"] is not None:
                last_recorded_time = cur_char_info["end_time"]
            elif cur_char_info["start_time"] is not None:
                last_recorded_time = cur_char_info["start_time"]

            cur_word_end_time = last_recorded_time
            cur_word += cur_char
            prev_char_info = cur_char_info

        # last word
        new_word_info = {
            "word": cur_word,
            "start_char": cur_word_start_char_idx,
            # i is the actual last char index of this word but python
            # slicing is non-inclusive so we increment by 1
            "end_char": i + 1,
            "start_time": cur_word_start_time,
            "end_time": cur_word_end_time,
            "speaker": None,
        }
        word_info.append(new_word_info)
        self._char_info = char_info
        self._word_info = word_info

    def _is_space(self, char: str) -> bool:
        """
        Returns whether the character is a space

        Parameters
        ----------
        char: str
            the character to check

        Returns
        -------
        bool:
            whether the character is a space
        """
        return char == " "

    def _is_word_start(self, prev_char: str, char: str) -> bool:
        """
        Returns whether the character is the start of a word

        Parameters
        ----------
        char: str
            the character to check
        prev_char: str
            the previous character

        Returns
        -------
        bool:
            whether the character is the start of a word
        """
        is_word_start = self._is_space(prev_char) is True
        is_word_start = is_word_start and (self._is_space(char) is False)
        return is_word_start

    def _is_word_end(self, char: str, next_char: str) -> bool:
        """
        Returns whether the character is the end of a word

        Parameters
        ----------
        char: str
            the character to check
        next_char: str
            the prev character

        Returns
        -------
        bool:
            whether the character is the end of a word
        """
        is_word_end = self._is_space(char) is False
        is_word_end = is_word_end and (self._is_space(next_char) is True)
        return is_word_end

    def _build_sentence_info(self) -> None:
        """
        Builds the sentence_info from the char_info

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        char_info = self.get_char_info()
        sentences = sent_tokenize(self.text)

        # final destination for sentence_info
        sentence_info = []

        # current sentence
        cur_sentence_start_char_idx = None
        cur_sentence_start_time = None

        # helper variables
        cur_char_idx = 0
        last_recorded_time = 0.0

        for i, cur_sentence in enumerate(sentences):
            # nltk tokenizer doesn't include spaces in between sentences
            # need increment the char_idx by 1 for each sentence to account for this
            if char_info[cur_char_idx]["char"] == " ":
                char_info[cur_char_idx]["sentence_index"] = i
                cur_char_idx += 1

            for j, sentence_char in enumerate(cur_sentence):
                cur_char_info = char_info[cur_char_idx]
                # realign cur_char_idx with sentence if needed
                if cur_sentence[j] != cur_char_info["char"]:
                    cur_char_idx = self._realign_char_idx_with_sentence(
                        char_info, cur_char_idx, cur_sentence[j], 3
                    )

                # sentence start time and start index
                if j == 0:
                    cur_sentence_start_char_idx = cur_char_idx
                    if cur_char_info["start_time"] is not None:
                        cur_sentence_start_time = cur_char_info["start_time"]
                    else:
                        cur_sentence_start_time = last_recorded_time

                if cur_char_info["end_time"] is not None:
                    last_recorded_time = cur_char_info["end_time"]
                elif cur_char_info["start_time"] is not None:
                    last_recorded_time = cur_char_info["start_time"]

                # update char_info
                cur_char_info["sentence_index"] = i

                cur_char_idx += 1

            new_sentence_info = {
                "sentence": cur_sentence,
                "start_char": cur_sentence_start_char_idx,
                "start_time": cur_sentence_start_time,
                "end_char": cur_char_idx,
                "end_time": last_recorded_time,
            }
            sentence_info.append(new_sentence_info)

        self._char_info = char_info
        self._sentence_info = sentence_info

        return sentence_info

    def _realign_char_idx_with_sentence(
        self,
        char_info: list[dict],
        char_idx: int,
        correct_char: str,
        search_window_size: int,
    ) -> int:
        """
        Realigns the char_idx so that char_info[char_idx] == correct_char

        Parameters
        ----------
        char_info: list[dict]
            char_info list
        char_idx: int
            index of character to start searching from
        correct_char: str
            the character that should be at char_info[char_idx]
        search_window_size: int
            the number of characters to search in each direction

        Returns
        -------
        correct_char_idx: int or None
            the char_idx scuh that char_info[char_idx] == correct_char
        """
        logging.debug(
            "Realigning char_idx '{}' with the correct starting character "
            "'{}' for the sentence.".format(char_idx, correct_char)
        )

        if char_idx < 0 or char_idx >= len(char_info):
            err_msg = (
                "char_idx must be between 0 and {} (length of char_info), not '{}'"
                "".format(len(char_info), char_idx)
            )
            logging.error(err_msg)
            raise ValueError(err_msg)
        if search_window_size <= 1:
            err_msg = "search_window_size must be greater than 0, not '{}'" "".format(
                search_window_size
            )
            logging.error(err_msg)
            raise ValueError(err_msg)

        for offset in range(1, search_window_size * 2):
            offset *= -1
            if char_info[char_idx + offset]["char"] == correct_char:
                return char_idx + offset

        # realignment failed
        err_msg = (
            "Realigning char_idx '{}' with the correct starting character '{}' for the "
            "sentence failed.".format(char_idx, correct_char)
        )
        raise TranscriptionError(err_msg)

    def _assert_valid_times(self, start_time: float, end_time: float) -> None:
        """
        Raises an error if the start_time and end_time are invalid for the transcript.

        Parameters
        ----------
        start_time: float
            start time of the transcript in seconds
        end_time: float
            end time of the transcript in seconds

        Returns
        -------
        None
        """
        # start time and end time must both be None or both be floats
        if type(start_time) is not type(end_time):
            err = (
                "start_time and end_time must both be None or both be floats, not "
                "'{}' (start_time) and '{}' (end_time)".format(start_time, end_time)
            )
            logging.error(err)
            raise TranscriptionError(err)

        if start_time is None and end_time is None:
            return

        # start time must be positive
        if start_time < 0:
            err = "start_time must be greater than or equal to 0."
            logging.error(err)
            raise TranscriptionError(err)

        # end time can't exceed transcription end time
        if end_time > self.end_time:
            err = (
                "end_time ({} seconds) must be less than or equal to the transcript's "
                "end time ({} seconds)".format(end_time, self.end_time)
            )
            logging.error(err)
            raise TranscriptionError(err)

        # start time must be less than end time
        if start_time >= end_time:
            err = (
                "start_time ({} seconds) must be less than end_time ({} seconds)."
                "".format(start_time, end_time)
            )
            logging.error(err)
            raise TranscriptionError(err)

    def __str__(self) -> str:
        """
        Tells Python interpreter how to print the object

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return self.text
