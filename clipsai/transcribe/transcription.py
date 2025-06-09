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
try:  # pragma: no cover - optional dependency
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download("punkt")
except ModuleNotFoundError:  # pragma: no cover - fallback stub
    nltk = None  # type: ignore
    def sent_tokenize(text: str):  # type: ignore
        return [text]


class Transcription:
    """
    A class for transcription data viewing, storage, and manipulation.
    It is designed to handle data primarily from Google Cloud Speech-to-Text,
    including character-level information with speaker tags and interpolated timings.
    Character timings are interpolated from word timings provided by the STT API.
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
            - A dictionary object containing transcription data, typically from
              Google Cloud Speech-to-Text (see `Transcriber.transcribe` for format).
            - A JSONFile containing such transcription data.

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
        The software used for transcribing.
        """
        return self._source_software

    @property
    def created_time(self) -> datetime:
        """
        The time when the transcription was created.
        """
        return self._created_time

    @property
    def language(self) -> str:
        """
        The ISO 639-1 language code of the transcription's language.
        """
        return self._language

    @property
    def start_time(self) -> float:
        """
        The start time of the transcript in seconds.
        """
        return 0.0

    @property
    def end_time(self) -> float:
        """
        The end time of the transcript in seconds.
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
        The full textual content of the transcription.
        """
        return self._text

    @property
    def characters(self) -> list[Character]:
        """
        A list of characters from the text as Character objects and ordered by start
        time.
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
        A list of words from the text as Word objects and ordered by start time.
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
        A list of sentences from the text as Sentence objects and ordered by start time.
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

    def get_speaker_segments(self, time_precision: int = 6) -> list[dict]:
        """
        Generates speaker segments from the character information, which includes speaker tags.

        This method processes the character-level speaker tags to produce a list of
        segments, each representing a contiguous block of speech by a particular speaker
        or an unlabeled segment. Speaker tags are re-labeled to be zero-indexed and
        contiguous.

        Parameters
        ----------
        time_precision : int, optional
            The number of decimal places to round start and end times to, by default 6.

        Returns
        -------
        list[dict]
            A list of speaker segments. Each segment is a dictionary with keys:
            'speakers': list[int]
                A list containing the single integer label for the speaker of this segment.
                Empty if the segment has no assigned speaker.
            'start_time': float
                The start time of the segment in seconds.
            'end_time': float
                The end time of the segment in seconds.
        """
        if not hasattr(self, '_char_info') or not self._char_info:
            logging.warning("Transcription: _char_info is not populated or is empty. Cannot get speaker segments.")
            return []

        # Ensure char_info has been built (it should be by __init__)
        # If _char_info could be None or empty, handle appropriately.
        # Assuming _char_info is a list of dicts, each with 'start_time', 'end_time', 'speaker'

        # First pass: Group characters by speaker tag and time contiguity
        # This pass aims to create initial segments based on changes in speaker or significant time gaps.
        initial_segments_by_chars = []
        current_segment_chars = []
        for char_data in self._char_info:
            if char_data.get("start_time") is None or char_data.get("end_time") is None:
                if current_segment_chars:
                    initial_segments_by_chars.append(list(current_segment_chars)) # Store a copy
                    current_segment_chars.clear()
                logging.debug(f"Transcription: Skipping char_data due to missing time: {char_data.get('char')}")
                continue

            if not current_segment_chars:
                current_segment_chars.append(char_data)
            else:
                prev_char_data = current_segment_chars[-1]
                if prev_char_data.get("end_time") is None: # Should not happen if list is clean
                    initial_segments_by_chars.append(list(current_segment_chars))
                    current_segment_chars.clear()
                    current_segment_chars.append(char_data) # Start new with current valid char
                    continue

                # Define a threshold for what constitutes a significant time gap to break a segment
                time_gap_threshold = 0.5  # seconds; consider if this should be configurable
                is_speaker_change = char_data.get("speaker") != prev_char_data.get("speaker")
                is_time_gap = (char_data["start_time"] - prev_char_data["end_time"]) > time_gap_threshold

                if is_speaker_change or is_time_gap:
                    initial_segments_by_chars.append(list(current_segment_chars))
                    current_segment_chars.clear()
                current_segment_chars.append(char_data)

        if current_segment_chars:
            initial_segments_by_chars.append(list(current_segment_chars))

        if not initial_segments_by_chars:
            return []

        # Second pass: Convert groups of characters into segment dictionaries and collect original speaker tags
        processed_segments_info = []
        original_speaker_tags = set()
        for char_group in initial_segments_by_chars:
            if not char_group:
                continue

            first_char = char_group[0]
            last_char = char_group[-1]
            speaker_tag = first_char.get("speaker") # Assumed consistent within the group from pass 1

            # This check might be redundant if pass 1 filters Nones, but good for safety
            if first_char.get("start_time") is None or last_char.get("end_time") is None:
                logging.warning(f"Transcription: Segment group missing critical timing, skipping.")
                continue

            # Ensure end_time is not before start_time for the segment
            segment_start_time = first_char["start_time"]
            segment_end_time = last_char["end_time"]
            if segment_end_time < segment_start_time:
                logging.warning(f"Transcription: Segment group has end_time ({segment_end_time}) before start_time ({segment_start_time}), adjusting or skipping.")
                # Option: skip, or set end_time = start_time, or use last valid end_time from group
                segment_end_time = segment_start_time # Simplest fix: zero-duration segment

            processed_segments_info.append({
                "original_speaker_tag": speaker_tag,
                "start_time": segment_start_time,
                "end_time": segment_end_time,
            })
            if speaker_tag is not None:
                original_speaker_tags.add(speaker_tag)

        # Relabel speaker tags to be zero-indexed and contiguous
        sorted_unique_tags = sorted(list(original_speaker_tags))
        speaker_relabel_map = {tag: i for i, tag in enumerate(sorted_unique_tags)}

        # Apply relabeling
        relabeled_segments = []
        for seg_info in processed_segments_info:
            original_tag = seg_info["original_speaker_tag"]
            speaker_list = []
            if original_tag is not None and original_tag in speaker_relabel_map:
                speaker_list.append(speaker_relabel_map[original_tag])

            relabeled_segments.append({
                "speakers": speaker_list,
                "start_time": round(seg_info["start_time"], time_precision),
                "end_time": round(seg_info["end_time"], time_precision),
            })

        if not relabeled_segments:
            return []

        # Third pass: Merge consecutive segments if the *newly labeled* speaker is the same
        # and they are temporally close (approximating some of PyannoteDiarizer._adjust_segments)
        final_merged_segments = []
        current_merged_segment = relabeled_segments[0]

        for i in range(1, len(relabeled_segments)):
            next_segment = relabeled_segments[i]

            current_speakers = current_merged_segment["speakers"]
            next_speakers = next_segment["speakers"]

            is_same_speaker_type = (not current_speakers and not next_speakers) or \
                                   (current_speakers and next_speakers and \
                                    current_speakers[0] == next_speakers[0])

            merge_time_gap_threshold = 0.1  # seconds; smaller threshold for merging
            # Ensure end_time of current is not after start_time of next before subtraction
            time_diff = next_segment["start_time"] - current_merged_segment["end_time"]
            is_contiguous_time = (time_diff >= 0 and time_diff < merge_time_gap_threshold) or \
                                 (current_merged_segment["end_time"] >= next_segment["start_time"]) # Overlap or tiny gap

            if is_same_speaker_type and is_contiguous_time:
                current_merged_segment["end_time"] = max(current_merged_segment["end_time"], next_segment["end_time"])
            else:
                final_merged_segments.append(current_merged_segment)
                current_merged_segment = next_segment

        final_merged_segments.append(current_merged_segment)

        return final_merged_segments

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
        type_of_time: start | end
            start: returns the index of the word with the closest start time before
            target_time.
            end: returns the index of the word with the closest end time after target
            time.

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
        type_of_time: start | end
            start: returns the index of the sentence with the closest start time before
            target_time.
            end: returns the index of the sentence with the closest end time after
            target time.

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
