"""
Transcribing a asset's media and storing the transcription in Cloud Storage.
"""
# standard library imports
import logging
import os

# local package imports
from ..utils.k8s import K8S_PVC_DIR_PATH
from ..media.audio_file import AudioFile
from ..transcription.whisperx import WhisperXTranscription

# machine learning imports
from ..ml.transcribe.whisperx import WhisperXTranscriber


class Transcriber():
    """
    A class for transcribing an asset's media
    """

    def build(
        self,
        temporal_media_file: AudioFile,
        transcribe_config: dict,
    ) -> WhisperXTranscription:
        """
        Transcribes the media file.

        Parameters
        ----------
        temporal_media_file: AudioFile
            the media file to transcribe
        transcribe_config: dict
            dictionary containing the configuration settings for the transcriber

        Returns
        -------
        WhisperXTranscription
            the transcription of the asset media
        """
        logging.info("USING WHISPER TO TRANSCRIBE MEDIA FILE")

        # transcribe
        transcriber = WhisperXTranscriber(
            transcribe_config["model_size"],
            transcribe_config["device"],
            transcribe_config["precision"],
        )
        transcription = transcriber.transcribe(
            temporal_media_file,
            transcribe_config["language"],
        )

        # create transcription and subtitle files
        transcription_file_path = os.path.join(K8S_PVC_DIR_PATH, "transcription.json")
        transcription_file = transcription.store_as_json_file(transcription_file_path)
        # logging.info("creating subtitles file")
        # subtitles_file_path = os.path.join(K8S_PVC_DIR_PATH, "subtitles.srt")
        # subtitles_file = transcription.store_as_srt_file(
        #     subtitles_file_path
        # )

        # # upload file to Cloud Storage
        # asset.store_subtitles(subtitles_file, overwrite=True)

        # TODO: send back to user instead of storing
        # asset.store_transcript(transcription_file, overwrite=True)

        transcription_file.delete()
        logging.info("TRANSCRIPTION STAGE COMPLETE")
        return transcription
