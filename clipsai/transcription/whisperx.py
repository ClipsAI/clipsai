"""
Transcriptions from WhisperX.

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
from .exceptions import WhisperXTranscriptionError
from .transcription import Transcription

# local imports
from filesys.json_file import JsonFile
from filesys.manager import FileSystemManager
from filesys.pdf_file import PdfFile
from filesys.srt_file import SrtFile
from utils.type_checker import TypeChecker

# 3rd party imports
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt")


class WhisperXTranscription(Transcription):
    """
    A class for whisperx transcription data viewing, storage, and manipulation.
    """

    def __init__(
        self,
        transcription: dict or JsonFile,
    ) -> None:
        """
        Initialize WhisperXTranscription

        Parameters
        ----------
        transcription: dict or JsonFile
            - a dictionary object containing whisperx transcription
            - a JsonFile containing a whisperx transcription

        Returns
        -------
        None
        """
        self._fs_manager = FileSystemManager()

        # set in __init_from_json_file() or __init_from_dict()
        self._source_software = None
        self._time_spawned = None
        self._language = None
        self._num_speakers = None
        self._char_info_predicted = None
        self._char_info_edited = None
        # derived from char_info data
        self._text_predicted = None
        self._text_edited = None
        self._word_info_predicted = None
        self._word_info_edited = None
        self._sentence_info_predicted = None
        self._sentence_info_edited = None

        self._type_checker = TypeChecker()
        self._type_checker.assert_type(transcription, "transcription", (dict, JsonFile))

        if isinstance(transcription, JsonFile):
            self._init_from_json_file(transcription)
        else:
            self._init_from_dict(transcription)

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
        return self._source_software

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
        return self._time_spawned

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
        return self._language

    def get_num_speakers(self) -> int:
        """
        Returns the number of speakers in the transcription

        Parameters
        ----------
        None

        Returns
        -------
        int
            the number of speakers in the transcription
        """
        return self._num_speakers

    def get_text(self, predicted: bool) -> str:
        """
        Returns the full text of either the predicted or edited transcription

        Parameters
        ----------
        predicted: bool
            predicted text if True, edited text if False

        Returns
        -------
        text: str
            the full text of the predicted or edited transcription
        """
        if predicted is True:
            return self._text_predicted
        else:
            return self._text_edited

    def get_char_info(
        self,
        predicted: bool,
        start_time: float = None,
        end_time: float = None,
    ) -> list:
        """
        Returns the character info of either the predicted or edited transcription

        Parameters
        ----------
        predicted: bool
            predicted character info if True, edited character info if False
        start_time: float
            start time of the character info in seconds. If None, returns all character
            info
        end_time: float
            end time of the character info in seconds. If None, returns all character
            info

        Returns
        -------
        list[dict]
            list of dictionaries where each dictionary contains info about a single
            character in the text
        """
        self._assert_valid_times(start_time, end_time)

        if predicted is True:
            char_info = self._char_info_predicted
        else:
            char_info = self._char_info_edited

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
        predicted: bool,
        start_time: float = None,
        end_time: float = None,
    ) -> list:
        """
        Returns the word info of either the predicted or edited text

        Parameters
        ----------
        predicted: bool
            predicted word info if True, edited word info if False
        start_time: float
            start time of the word info in seconds. If None, returns all word info
        end_time: float
            end time of the word info in seconds. If None, returns all word info

        Returns
        -------
        list[dict]
            list of dictionaries where each dictionary contains info about a single
            word in the text
        """
        self._assert_valid_times(start_time, end_time)

        # get all word info
        if predicted is True:
            word_info = self._word_info_predicted
        else:
            word_info = self._word_info_edited

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
        predicted: bool,
        start_time: float = None,
        end_time: float = None,
    ) -> list:
        """
        Returns the sentence information of either the predicted or edited text

        Parameters
        ----------
        predicted: bool
            predicted sentence info if True, edited sentence info if False
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

        if predicted is True:
            sentence_info = self._sentence_info_predicted
        else:
            sentence_info = self._sentence_info_edited

        # return all word info
        if start_time is None and end_time is None:
            return sentence_info
        # return subset of word info
        else:
            start_index = self.find_sentence_index(start_time, type_of_time="start")
            end_index = self.find_sentence_index(end_time, type_of_time="end")
            return sentence_info[start_index : end_index + 1]

    def get_sentences(self, predicted: bool) -> list:
        """
        Returns the sentences of either the predicted or edited text.

        Parameters
        ----------
        predicted: bool
            predicted sentences if True, edited sentences if False

        Returns
        -------
        list[str]
            list of sentences
        """
        sentences = []
        for sentence_info in self.get_sentence_info(predicted):
            sentences.append(sentence_info["sentence"])
        return sentences

    def get_start_time(self, predicted: bool) -> float:
        """
        Returns the start time of the transcript in seconds.

        Parameters
        ----------
        predicted: bool
            predicted start time if True, edited start time if False

        Returns
        -------
        float
            the start time of the transcript in seconds
        """
        return 0.0

    def get_end_time(self, predicted: bool) -> float:
        """
        Returns the end time of the transcript in seconds.

        Parameters
        ----------
        predicted: bool
            predicted end time if True, edited end time if False

        Returns
        -------
        float
            the end time of the transcript in seconds
        """
        char_info = self.get_char_info(predicted)
        for i in range(len(char_info) - 1, -1, -1):
            if char_info[i]["endTime"] is not None:
                return char_info[i]["endTime"]
            if char_info[i]["startTime"] is not None:
                return char_info[i]["startTime"]

    def store_as_json_file(self, file_path: str) -> JsonFile:
        """
        Stores the transcription as a json file. 'file_path' is overwritten if already
        exists.

        Parameters
        ----------
        file_path: str
            absolute file path to store the transcription as a json file

        Returns
        -------
        None
        """
        json_file = JsonFile(file_path)
        json_file.assert_has_file_extension("json")
        self._fs_manager.assert_parent_dir_exists(json_file)

        # delete file if it exists
        json_file.delete()

        # only store necessary data
        char_info_predicted_needed_for_storage = []
        for char_info in self._char_info_predicted:
            char_info_predicted_needed_for_storage.append(
                {
                    "char": char_info["char"],
                    "startTime": char_info["startTime"],
                    "endTime": char_info["endTime"],
                    "speaker": char_info["speaker"],
                }
            )

        char_info_edited_needed_for_storage = []
        for char_info in self._char_info_edited:
            char_info_edited_needed_for_storage.append(
                {
                    "char": char_info["char"],
                    "startTime": char_info["startTime"],
                    "endTime": char_info["endTime"],
                    "speaker": char_info["speaker"],
                }
            )

        transcription_dict = {
            "sourceSoftware": self._source_software,
            "timeSpawned": str(self._time_spawned),
            "language": self._language,
            "numSpeakers": self._num_speakers,
            "charInfoPredicted": char_info_predicted_needed_for_storage,
            "charInfoEdited": char_info_edited_needed_for_storage,
        }

        json_file.create(transcription_dict)
        return json_file

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
        return self._find_index(
            self.get_char_info(predicted=False), target_time, type_of_time
        )

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
        return self._find_index(
            self.get_word_info(predicted=False), target_time, type_of_time
        )

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
        return self._find_index(
            self.get_sentence_info(predicted=False), target_time, type_of_time
        )

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
        transcript_start = self.get_start_time(False)
        transcript_end = self.get_end_time(False)
        if (transcript_start <= target_time <= transcript_end) is False:
            err = (
                "target_time '{}' seconds is not within the range of the transcript "
                "times: {} - {}".format(
                    target_time, self.get_start_time(), self.get_end_time()
                )
            )
            logging.error(err)
            raise WhisperXTranscriptionError(err)

        left, right = 0, len(transcript_info) - 1
        while left <= right:
            mid = left + (right - left) // 2
            start_time = transcript_info[mid]["startTime"]
            end_time = transcript_info[mid]["endTime"]

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

    def store_as_srt_file(
        self,
        file_path: str,
        start_time: float = None,
        end_time: float = None,
    ) -> SrtFile:
        """
        Stores the transcription as a srt file. Overwrites 'dest_file_path' if already
        exists.

        Parameters
        ----------
        file_path: str
            absolute file path to store the transcription as a srt file
        start_time: float
            start time of the transcript in seconds
        end_time: float
            end time of the transcript in seconds

        Returns
        -------
        File
            the srt file
        """
        # check valid file path
        srt_file = SrtFile(file_path)
        srt_file.assert_has_file_extension("srt")
        self._fs_manager.assert_parent_dir_exists(srt_file)

        # delete old file to write new one
        srt_file.delete()

        # build subtitles
        word_info = self.get_word_info(False, start_time, end_time)
        subtitles = self._build_subtitles(word_info, max_subtitle_length=32)

        # create srt file
        srt_file.create(subtitles)
        srt_file.assert_exists()
        logging.info("srt file created at {}".format(srt_file.get_path()))
        return srt_file

    def store_as_pdf_file(
        self,
        file_path: str,
        start_time: float = None,
        end_time: float = None,
    ) -> PdfFile:
        """
        Stores the transcription as a pdf file. Overwrites 'dest_file_path' if already
        exists.

        Parameters
        ----------
        file_path: str
            absolute file path to store the transcription as a PDF file.
        start_time: float
            start time of the transcript to store as a pdf file in seconds
        end_time: float
            start time of the transcript to store as a pdf file in seconds

        Returns
        -------
        None
        """
        pdf_file = PdfFile(file_path)
        pdf_file.assert_has_file_extension("pdf")
        self._fs_manager.assert_parent_dir_exists(pdf_file)

        # delete old file to write new one
        pdf_file.delete()

        # build paragraphs
        sentence_info = self.get_sentence_info(False, start_time, end_time)
        sentences = [sent["sentence"] for sent in sentence_info]
        num_sentences_per_paragraph = 5
        paragraphs = []
        for i in range(0, len(sentences), num_sentences_per_paragraph):
            paragraphs.append(" ".join(sentences[i : i + num_sentences_per_paragraph]))

        pdf_file.create(paragraphs)
        pdf_file.assert_exists()
        return pdf_file

    def print_char_info(self, predicted: bool) -> None:
        """
        Pretty prints the character info for easy viewing

        Parameters
        ----------
        predicted: bool
            prints predicted character info if True, edited character info if False

        Returns
        -------
        None
        """
        if predicted is True:
            title = "charInfoPredicted"
        else:
            title = "charInfoEdited"
        print(title)
        print("-" * len(title))
        for i, char_info in enumerate(self.get_char_info(predicted)):
            print("char: {}".format(char_info['char']))
            print("startTime: {}".format(char_info['startTime']), end=" | ")
            print("endTime: {}".format(char_info['endTime']))
            print("index: {}".format(i), end=" | ")
            print("wordIndex: {}".format(char_info['wordIdx']), end=" | ")
            print("sentenceIndex: {}\n".format(char_info['sentenceIdx']))

    def print_word_info(self, predicted: bool) -> None:
        """
        Pretty prints the word info for easy viewing

        Parameters
        ----------
        predicted: bool
            prints predicted word info if True, edited word info if False

        Returns
        -------
        None
        """
        title = "wordInfoPredicted" if predicted is True else "wordInfoEdited"
        print(title)
        print("-" * len(title))
        for i, word_info in enumerate(self.get_word_info(predicted)):
            print("word: '{}'".format(word_info['word']), end=" | ")
            print("wordIndex: {}".format(i))
            print("speaker: {}".format(word_info['speaker']))
            print("startTime: {}".format(word_info['startTime']), end=" | ")
            print("endTime: {}".format(word_info['endTime']))
            print("startChar: {}".format(word_info['startChar']), end=" | ")
            print("endChar: {}\n".format(word_info['endChar']))

    def print_sentence_info(self, predicted: bool) -> None:
        """
        Pretty prints the sentence info for easy viewing

        Parameters
        ----------
        predicted: bool
            prints predicted sentence info if True, edited sentence info if False

        Returns
        -------
        None
        """
        title = "sentenceInfoPredicted" if predicted is True else "sentenceInfoEdited"
        print(title)
        print("-" * len(title))
        for i, sentence_info in enumerate(self.get_sentence_info(predicted)):
            print("sentence: '{}'".format(sentence_info['sentence']))
            print("sentenceIndex: {}".format(i))
            print("startChar: {}".format(sentence_info['startChar']), end=" | ")
            print("endChar: {}".format(sentence_info['endChar']))
            print("startTime: {}".format(sentence_info['startTime']), end=" | ")
            print("endTime: {}\n".format(sentence_info['endTime']))

    def _init_from_json_file(self, json_file: JsonFile) -> None:
        """
        Initializes the transcription object from an existing json file

        Parameters
        ----------
        json_file: JsonFile
            a json file with whisperx transcription data

        Returns
        -------
        None
        """
        self._type_checker.assert_type(json_file, "json_file", JsonFile)
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

        if isinstance(transcription["timeSpawned"], str):
            transcription["timeSpawned"] = datetime.strptime(
                transcription["timeSpawned"], "%Y-%m-%d %H:%M:%S.%f"
            )

        self._time_spawned = transcription["timeSpawned"]
        self._source_software = transcription["sourceSoftware"]
        self._language = transcription["language"]
        self._num_speakers = transcription["numSpeakers"]
        self._char_info_predicted = transcription["charInfoPredicted"]
        self._char_info_edited = transcription["charInfoEdited"]
        # derived data
        self._build_text(predicted=True)
        self._build_text(predicted=False)
        self._build_word_info(predicted=True)
        self._build_word_info(predicted=False)
        self._build_sentence_info(predicted=True)
        self._build_sentence_info(predicted=False)

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
            "sourceSoftware": (str),
            "timeSpawned": (datetime, str),
            "language": (str),
            "numSpeakers": (int, type(None)),
            "charInfoPredicted": (list),
            "charInfoEdited": (list),
        }
        self._type_checker.assert_dict_elems_type(
            transcription, transcription_keys_correct_data_types
        )

        # ensure charInfoPredicted and charInfoEdited contain dictionaries
        for char_info in transcription["charInfoPredicted"]:
            self._type_checker.assert_type(char_info, "char_info", dict)
        for char_info in transcription["charInfoEdited"]:
            self._type_checker.assert_type(char_info, "char_info", dict)

        # ensure charInfoPredicted and charInfoEdited have valid keys and datatypes
        char_dict_keys_correct_data_types = {
            "char": (str),
            "startTime": (float, type(None)),
            "endTime": (float, type(None)),
            "speaker": (int, type(None)),
        }
        for char_dict in transcription["charInfoPredicted"]:
            self._type_checker.are_dict_elems_of_type(
                char_dict,
                char_dict_keys_correct_data_types,
            )
        for char_dict in transcription["charInfoEdited"]:
            self._type_checker.are_dict_elems_of_type(
                char_dict,
                char_dict_keys_correct_data_types,
            )

    def _build_text(self, predicted: bool) -> str:
        """
        Builds the text from the char_info

        Parameters
        ----------
        predicted: bool
            whether to build the predicted text or edited text

        Returns
        -------
        str:
            the full text built from the char_info
        """
        text = ""
        for char_info in self.get_char_info(predicted):
            text += char_info["char"]

        if predicted is True:
            self._text_predicted = text
        else:  # predicted == False:
            self._text_edited = text

    def _build_word_info(self, predicted: bool) -> list[dict]:
        """
        Builds the word_info from the char_info

        Parameters
        ----------
        predicted: bool
            whether to build the predicted text or edited text

        Returns
        -------
        list[dict]:
            the word_info built from the char_info
        """
        char_info = self.get_char_info(predicted)

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
            "startTime": None,
            "endTime": None,
            "speaker": None,
        }
        last_recorded_time = 0

        for i, cur_char_info in enumerate(char_info):
            cur_char = cur_char_info["char"]
            prev_char = prev_char_info["char"]

            if self._is_word_start(prev_char, cur_char):
                cur_word = ""
                cur_word_start_char_idx = i
                if cur_char_info["startTime"] is not None:
                    cur_word_start_time = cur_char_info["startTime"]
                else:
                    cur_word_start_time = last_recorded_time

            if self._is_word_end(prev_char, cur_char):
                new_word_info = {
                    "word": cur_word,
                    "startChar": cur_word_start_char_idx,
                    # prev_char is the actual last char of this word but python
                    # slicing is non-inclusive so we use the index of cur_char (+1)
                    "endChar": i,
                    "startTime": cur_word_start_time,
                    "endTime": cur_word_end_time,
                    "speaker": None,
                }
                word_info.append(new_word_info)

                cur_word_idx += 1
                # reset word info
                cur_word_start_char_idx = None
                cur_word = ""

            # update char info
            cur_char_info["wordIdx"] = cur_word_idx

            # update word info
            if cur_char_info["endTime"] is not None:
                last_recorded_time = cur_char_info["endTime"]
            elif cur_char_info["startTime"] is not None:
                last_recorded_time = cur_char_info["startTime"]

            cur_word_end_time = last_recorded_time
            cur_word += cur_char
            prev_char_info = cur_char_info

        # last word
        new_word_info = {
            "word": cur_word,
            "startChar": cur_word_start_char_idx,
            # i is the actual last char index of this word but python
            # slicing is non-inclusive so we increment by 1
            "endChar": i + 1,
            "startTime": cur_word_start_time,
            "endTime": cur_word_end_time,
            "speaker": None,
        }
        word_info.append(new_word_info)

        if predicted is True:
            self._char_info_predicted = char_info
            self._word_info_predicted = word_info
        else:  # predicted == False:
            self._char_info_edited = char_info
            self._word_info_edited = word_info

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

    def _build_sentence_info(self, predicted: bool) -> None:
        """
        Builds the sentence_info from the char_info

        Parameters
        ----------
        predicted: bool
            whether to build the predicted or edited sentence_info

        Returns
        -------
        None
        """
        char_info = self.get_char_info(predicted)
        sentences = sent_tokenize(self.get_text(predicted))

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
                char_info[cur_char_idx]["sentenceIdx"] = i
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
                    if cur_char_info["startTime"] is not None:
                        cur_sentence_start_time = cur_char_info["startTime"]
                    else:
                        cur_sentence_start_time = last_recorded_time

                if cur_char_info["endTime"] is not None:
                    last_recorded_time = cur_char_info["endTime"]
                elif cur_char_info["startTime"] is not None:
                    last_recorded_time = cur_char_info["startTime"]

                # update char_info
                cur_char_info["sentenceIdx"] = i

                cur_char_idx += 1

            new_sentence_info = {
                "sentence": cur_sentence,
                "startChar": cur_sentence_start_char_idx,
                "startTime": cur_sentence_start_time,
                "endChar": cur_char_idx,
                "endTime": last_recorded_time,
            }
            sentence_info.append(new_sentence_info)

        if predicted is True:
            self._char_info_predicted = char_info
            self._sentence_info_predicted = sentence_info
        else:  # predicted == False:
            self._char_info_edited = char_info
            self._sentence_info_edited = sentence_info

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
        logging.info(
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
            err_msg = (
                "search_window_size must be greater than 0, not '{}'"
                "".format(search_window_size)
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
        raise WhisperXTranscriptionError(err_msg)

    def _build_subtitles(self, word_info: list, max_subtitle_length: int) -> list:
        """
        Constructs subtitles from word-level information considering a maximum line
        length.

        Groups words together to form subtitle lines, each with start and end times,
        without exceeding the specified maximum line length. The resulting structure is:
            [
                {
                    'text': str,         # Subtitle text
                    'startTime': float,  # Start time of the subtitle
                    'endTime': float,    # End time of the subtitle
                },
            ]

        Parameters
        ----------
        word_info: list[dict]
            A list of dictionaries where each dictionary contains info about a single
            word in the transcript. Data structure is as follows:
                [
                    {
                        'word': str,
                        'startTime': float,
                        'endTime': float,
                        'startChar': int,
                        'endChar': int,
                        'speaker': str,
                    },
                ]

        max_subtitle_length: int
            The maximum allowed characters in a single subtitle line.

        Returns
        -------
        list
            A list of dictionaries containing the subtitles.
        """
        subtitle = ""
        subtitles = []
        start_time = word_info[0]["startTime"]

        for i, word_info_data in enumerate(word_info):
            word = word_info_data["word"]
            subtitle_length = len(subtitle) + len(word)

            # start new line
            if subtitle_length > max_subtitle_length:
                end_time = word_info[i - 1]["endTime"]
                subtitles.append(
                    {
                        "text": subtitle.strip(),
                        "startTime": start_time,
                        "endTime": end_time,
                    }
                )

                subtitle = word + " "
                start_time = word_info_data["startTime"]
            # add word to current line
            else:
                subtitle += word + " "

        # final word
        subtitle += word
        end_time = word_info_data["endTime"]
        subtitles.append(
            {
                "text": subtitle.strip(),
                "startTime": start_time,
                "endTime": end_time,
            }
        )

        return subtitles

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
            raise WhisperXTranscriptionError(err)

        if start_time is None and end_time is None:
            return

        # start time must be positive
        if start_time < 0:
            err = "start_time must be greater than or equal to 0."
            logging.error(err)
            raise WhisperXTranscriptionError(err)

        # end time can't exceed transcription end time
        if end_time > self.get_end_time(predicted=False):
            err = (
                "end_time ({} seconds) must be less than or equal to the transcript's "
                "end time ({} seconds)".format(
                    end_time, self.get_end_sec(predicted=False)
                )
            )
            logging.error(err)
            raise WhisperXTranscriptionError(err)

        # start time must be less than end time
        if start_time >= end_time:
            err = (
                "start_time ({} seconds) must be less than end_time ({} seconds)."
                "".format(start_time, end_time)
            )
            logging.error(err)
            raise WhisperXTranscriptionError(err)

    def __str__(self) -> None:
        """
        Tells Python interpreter how to print the object

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        transcription = (
            "sourceSoftware: {}\ntimeSpawned: {}\nlanguage: {}\nnumSpeakers: {}\n"
            "".format(
                self.get_source_software(),
                self.get_time_spawned(),
                self.get_language(),
                self.get_num_speakers()
            )
        )
        return transcription
