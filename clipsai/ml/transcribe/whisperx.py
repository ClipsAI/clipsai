"""
Transcribe audio files using whisperx.

Notes
-----
- WhisperX GitHub: https://github.com/m-bain/whisperX
"""
# standard library imports
from datetime import datetime
import logging

# current package imports
from .exceptions import NoSpeechError
from .whisperx_config_manager import WhisperXTranscriberConfigManager
from .transcriber import Transcriber

# local imports
from ...media.audio_file import AudioFile
from ...transcription.whisperx import WhisperXTranscription
from ...utils.type_checker import TypeChecker
from ...utils.pytorch import assert_valid_torch_device

# third party imports
import torch
import whisperx


class WhisperXTranscriber(Transcriber):
    """
    A class to transcribe using whisperx.
    """

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        precision: str = None,
    ) -> None:
        """
        Parameters
        ----------
        model_size: str
            One of the model sizes implemented by whisper/whisperx
        device: 'cpu' | 'cuda'
            Hardware to run the machine learning model on
        precision: 'float16' | 'int8'
            Precision to perform prediction with
        """
        self._config_manager = WhisperXTranscriberConfigManager()
        self._type_checker = TypeChecker()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if precision is None:
            precision = "float16" if torch.cuda.is_available() else "int8"
        if model_size is None:
            model_size = "large-v2" if torch.cuda.is_available() else "tiny"

        # check valid inputs
        assert_valid_torch_device(device)
        self._config_manager.assert_valid_model_size(model_size)
        self._config_manager.assert_valid_precision(precision)

        self._precision = precision
        self._device = device
        self._model_size = model_size
        self._model = whisperx.load_model(
            whisper_arch=self._model_size,
            device=self._device,
            compute_type=self._precision,
        )

    def transcribe(
        self,
        media_file: AudioFile,
        iso6391_lang_code: str or None = None,
        batch_size: int = 16,
    ) -> WhisperXTranscription:
        """
        Transcribes the media file

        Parameters
        ----------
        media_file: AudioFile
            the media file to transcribe
        iso6391_lang_code: str or None
            ISO 639-1 language code to transcribe in
        batch_size: int = 16
            reduce if low in GPU memory (not actually sure what it does though -Ben)
        Returns
        -------
        WhisperXTranscription
            the media file transcription
        """
        self._type_checker.assert_type(media_file, "media_file", (AudioFile))
        media_file.assert_exists()
        media_file.assert_has_audio_stream()

        if iso6391_lang_code is not None:
            self._config_manager.assert_valid_language(iso6391_lang_code)

        # if iso6391_lang_code is None, whisperx will try to detect the language
        transcription = self._model.transcribe(
            media_file.path, language=iso6391_lang_code, batch_size=batch_size
        )

        # align whisper output to get word level times
        model_a, metadata = whisperx.load_align_model(
            language_code=transcription["language"],
            device=self._device,
        )
        aligned_transcription = whisperx.align(
            transcription["segments"],
            model_a,
            metadata,
            media_file.path,
            self._device,
            return_char_alignments=True,
        )

        """
        ALIGNED_TRANSCRIPTION DATA STRUCTURE
        ------------------------------------
        s = number of segments in the transcription
        w = number of words in the transcription
        n_s = number of chars in the transcription of segment s
        m_s = number of words in the transcription of segment s
        aligned_transcription = {
            "segments":
            [
                {<segment-0>},
                {
                    "start": float (start time in seconds)
                    "end": float (start time in seconds)
                    "text": str (text transcription for that segment)
                    "words":
                    [
                        {<word-0>},
                        {
                            "word": str (word transcription)
                            "start": float (start time in seconds)
                            "end": float (start time in seconds)
                            "score": float (score for that word)
                        },
                        {<word-m_s>},
                    ]
                    "chars":
                    [
                        {<char-0>},
                        {
                            "char": str (char transcription)
                            "start": float (start time in seconds)
                            "end": float (start time in seconds)
                            "score": float (score for that char)
                        }
                        {<char-n_s>},
                    ]
                },
                {segment_n},
            ]
            "word_segments":
            [
                {<word-segment-0>},
                {
                    "word":
                    "start":
                    "end":
                    "score":
                },
                {word-segment-w},
            ]
        }
        """
        if len(aligned_transcription["segments"]) == 0:
            err = "Media file '{}' contains no active speech.".format(media_file.path)
            logging.error(err)
            raise NoSpeechError(err)

        # final destination for transcript information
        predicted_char_info = []

        # remove global first character -> always a space
        try:
            del aligned_transcription["segments"][0]["chars"][0]
        except Exception as e:
            print("Error:", str(e))
            print("Aligned Transcription:", aligned_transcription)
            raise Exception(str(e))

        for i, segment in enumerate(aligned_transcription["segments"]):
            segment_chars = segment["chars"]

            # iterate through each char in the segment
            for j, char in enumerate(segment_chars):
                char_start_time = (
                    float(char["start"]) if "start" in char.keys() else None
                )
                char_end_time = float(char["end"]) if "end" in char.keys() else None

                # character information
                new_char_dic = {
                    "char": char["char"],
                    "startTime": char_start_time,
                    "endTime": char_end_time,
                    "speaker": None,
                }
                predicted_char_info.append(new_char_dic)

        transcription_dict = {
            "sourceSoftware": "whisperx-v3",
            "timeSpawned": datetime.now(),
            "language": transcription["language"],
            "numSpeakers": None,
            "charInfoPredicted": predicted_char_info,
            "charInfoEdited": predicted_char_info,
        }
        return WhisperXTranscription(transcription_dict)

    def detect_language(self, media_file: AudioFile) -> str:
        """
        Detects the language of the media file

        Parameters
        ----------
        media_file: AudioFile
            the media file to detect the language of

        Returns
        -------
        str
            the ISO 639-1 language code of the media file
        """
        self._type_checker.assert_type(media_file, "media_file", (AudioFile))
        media_file.assert_exists()
        media_file.assert_has_audio_stream()

        audio = whisperx.load_audio(media_file.path)
        language = self._model.detect_language(audio)
        return language
