"""
Diarize an audio file using pyannote/speaker-diarization-3.1

Notes
-----
- Real-time factor is around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for the neural
inference part) and one Intel Cascade Lake 6248 CPU (for the clustering part).
In other words, it takes approximately 1.5 minutes to process a one hour conversation.

- The technical details of the model are described in
 https://huggingface.co/pyannote/speaker-diarization-3.0

- pyannote/speaker-diarization allows setting a number of speakers to detect. Could be
viable to analyze different subsections of the video, detect the number of faces, and
use that as the number of speakers to detect.
"""
# standard library imports
import logging

# local package imports
from ..media.audio_file import AudioFile
from ..utils.pytorch import get_compute_device, assert_compute_device_available

# third party imports
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
import torch


class PyannoteDiarizer:
    """
    A class for diarizing audio files using pyannote.
    """

    def __init__(self, auth_token: str, device: str = "auto") -> None:
        """
        Initialize PyannoteDiarizer

        Parameters
        ----------
        device: str
            device to use for diarization

        Returns
        -------
        None
        """
        if device == "auto":
            device = get_compute_device()
        assert_compute_device_available(device)

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token,
        ).to(torch.device(device))
        logging.debug("Pyannote using device: {}".format(self.pipeline.device))

    def diarize(
        self,
        audio_file: AudioFile,
        min_segment_duration: float = 1.5,
        time_precision: int = 6
    ) -> list[dict]:
        """
        Diarizes the audio file.

        Parameters
        ----------
        audio_file: AudioFile
            the audio file to diarize
        time_precision: int
            The number of decimal places for rounding the start and end times of 
            segments.
        min_segment_duration: float
            The minimum duration (in seconds) for a segment to be considered valid.

        Returns
        -------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            startTime: float
                start time of the segment in seconds
            endTime: float
                end time of the segment in seconds
        """

        wav_file = audio_file.extract_audio(
            extracted_audio_file_path=audio_file.convert_to_wav_path(),
            audio_codec="pcm_s16le",
        )

        pyannote_segments: Annotation = self.pipeline({"audio": wav_file.path})

        adjusted_speaker_segments = self._adjust_segments(
            pyannote_segments=pyannote_segments,
            min_segment_duration=min_segment_duration,
            duration=audio_file.get_duration(),
            time_precision=time_precision,
        )

        wav_file.delete()

        return adjusted_speaker_segments

    def _adjust_segments(
        self,
        pyannote_segments: Annotation,
        min_segment_duration: float,
        duration: float,
        time_precision: int,
    ) -> list[dict]:
        """
        Adjusts and merges speaker segments to achieve an unbroken, non-overlapping
        sequence of speaker segments with at least one person speaking in each segment.

        Parameters
        ----------
        pyannote_segments: Annotation
            the pyannote speaker segments
        duration: float
            duration of the audio being diarized.
        time_precision: int
            The number of decimal places for rounding the start and end times of 
            segments.
        min_segment_duration: float
            The minimum duration (in seconds) for a segment to be considered valid.

        Returns
        -------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            startTime: float
                start time of the segment in seconds
            endTime: float
                end time of the segment in seconds
        """    
        cur_end_sec = None
        cur_speaker = None
        cur_start_sec = 0.000
        adjusted_speaker_segments = []
        unique_speakers: set[int] = set()

        for segment, _, speaker_label in pyannote_segments.itertracks(True):
            next_start_sec = segment.start
            next_end_sec = segment.end
            if speaker_label.split("_")[1] == "":
                next_speaker = None
            else:
                next_speaker = int(speaker_label.split("_")[1])

            # skip segments that are too short
            if next_end_sec - next_start_sec < min_segment_duration:
                continue

            # first identified speaker
            if cur_speaker is None:
                cur_speaker = next_speaker
                cur_end_sec = next_end_sec
                continue

            # same speaker as next segment -> merge segments and continue
            if cur_speaker == next_speaker:
                cur_end_sec = max(cur_end_sec, next_end_sec)
                continue

            # Different speaker than next segment
            # 1) The next speaker begins before the current speaker ends -> cut short
            # the end of the current speaker's segment be the start of the next
            # speaker segment.
            # 2) The next speaker begins after the current speaker ends -> extend the
            # current speaker's segment to end at the start of the next speaker segment.
            cur_end_sec = next_start_sec
            if cur_speaker is not None:
                speakers = [cur_speaker]
                unique_speakers.add(cur_speaker)
            else:
                speakers = []
            adjusted_speaker_segments.append({
                "speakers": speakers,
                "startTime": round(cur_start_sec, time_precision),
                "endTime": round(cur_end_sec, time_precision),
            })

            cur_speaker = next_speaker
            cur_start_sec = next_start_sec
            cur_end_sec = next_end_sec

        # explicitly add the last segment
        if cur_speaker is not None:
            speakers = [cur_speaker]
            unique_speakers.add(cur_speaker)
        else:
            speakers = []
        adjusted_speaker_segments.append({
            "speakers": speakers,
            "startTime": round(cur_start_sec, time_precision),
            "endTime": round(duration, time_precision),
        })

        adjusted_speaker_segments = self._relabel_speakers(
            adjusted_speaker_segments,
            unique_speakers
        )
        return adjusted_speaker_segments

    def _relabel_speakers(
        self,
        speaker_segments: list[dict],
        unique_speakers: set[int]
    ) -> dict[int, int]:
        """
        Relabels speaker segments so that the speaker labels are contiguous.

        Some speakers may have been skipped if their segments were too short. Thus,
        we could end up with a set of speaker labels like {0, 1, 3}. This function
        relabels the speakers to remove gaps so that our set of speaker labels would
        be contiguous, e.g. {0, 1, 2}.

        Parameters
        ----------
        speaker_segments: list[dict]
            speakers: list[int]
                list of speaker numbers for the speakers talking in the segment
            startTime: float
                start time of the segment in seconds
            endTime: float
                end time of the segment in seconds
        unique_speakers: set[int]
            set of unique speaker labels in the speaker segments

        Returns
        -------
        updated_speaker_segments: list[dict]
            list of speaker segments where the speakers are relabeled so that the
            speaker labels are contiguous. Each dictionary contains the following keys:
                speakers: list[int]
                    list of speaker numbers for the speakers talking in the segment
                startTime: float
                    start time of the segment in seconds
                endTime: float
                    end time of the segment in seconds
        """
        # no speakers
        if len(unique_speakers) == 0:
            return speaker_segments

        unique_speakers = sorted(list(unique_speakers))
        # speaker labels are already contiguous
        if len(unique_speakers) == unique_speakers[-1] + 1:
            return speaker_segments

        # create mapping from old speaker labels to new speaker labels
        relabel_speaker_map = {}
        for i in range(len(unique_speakers)):
            new_speaker_num = i
            old_speaker_num = unique_speakers[i]
            relabel_speaker_map[old_speaker_num] = new_speaker_num

        # relabel
        for segment in speaker_segments:
            relabeled_speakers = []
            for speaker in segment["speakers"]:
                relabeled_speakers.append(relabel_speaker_map[speaker])
            segment["speakers"] = relabeled_speakers

        return speaker_segments

    def cleanup(self) -> None:
        """
        Remove the diarization pipeline from memory and explicity free up GPU memory.
        """
        del self.pipeline
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
