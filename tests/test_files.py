"""
TestFiles module.

A class for retrieving test files and their paths

Classes
-------
TestFiles: A class for retrieving test files and their paths
"""
# standard library imports
import os

# local imports
from clipsai.filesys.dir import Dir
from clipsai.filesys.json_file import JSONFile
from clipsai.media.audio_file import AudioFile
from clipsai.media.audiovideo_file import AudioVideoFile
from clipsai.media.image_file import ImageFile
from clipsai.media.media_file import MediaFile
from clipsai.media.temporal_media_file import TemporalMediaFile
from clipsai.media.video_file import VideoFile


class TestFiles:
    """
    A class for retrieving test files and their paths

    Attributes
    ----------
    None

    Methods
    -------
    get_paths(self) -> list[str]:
        Gets valid file system object paths

    get_invalid_paths(self) -> list[str]:
        Gets invalid file system object paths

    get_file_paths(self) -> list[str]:
        Gets valid file system file paths

    get_invalid_file_paths(self) -> list[str]:
        Gets invalid file paths

    get_json_file_paths(self) -> list[str]:
        Gets valid json file paths

    get_invalid_json_file_paths(self) -> list[str]:
        Gets invalid json file paths

    get_dir_paths(self) -> list[str]:
        Gets valid file system directory paths

    get_invalid_dir_paths(self) -> list[str]:
        Gets invalid directory paths

    get_test_files_dir(self) -> Dir:
        Gets the 'test_files' directory

    get_media_dir(self) -> Dir:
        Gets the 'media' directory housing media test files

    get_media_file_paths(self) -> list[str]:
        Gets all media file paths

    get_invalid_media_file_paths(self) -> list[str]:
        Gets invalid media file paths

    get_media_files(self) -> list[MediaFile]:
        Gets all media test files

    get_image_dir(self) -> Dir:
        Gets the 'image' directory housing image test files

    get_image_file_paths(self) -> list[str]:
        Gets all image file paths

    get_image_files(self) -> list[ImageFile]:
        Gets all image test files

    get_jpeg_image_dir(self) -> Dir:
        Gets the 'jpeg' directory housing jpeg image test files

    get_jpeg_image_file_paths(self) -> list[str]:
        Gets 'jpeg' test file paths

    get_jpeg_image_files(self) -> list[ImageFile]:
        Gets 'jpeg' test files

    get_png_image_dir(self) -> Dir:
        Gets the 'png' directory housing png image test files

    get_png_image_file_paths(self) -> list[str]:
        Gets 'png' test file paths

    get_png_image_files(self) -> list[ImageFile]:
        Gets 'png' test files

    get_temporal_media_file_paths(self) -> list[str]:
        Gets all temporal media (audio and video) file paths

    get_temporal_media_files(self) -> list[TemporalMediaFile]:
        Gets all temporal media (audio and video) test files

    get_audio_dir(self) -> Dir:
        Gets the 'audio' directory housing audio test files

    get_audio_file_paths(self) -> list[str]:
        Gets all audio file paths

    get_audio_files(self) -> list[AudioFile]:
        Gets all audio test files

    get_mp3_audio_dir(self) -> Dir:
        Gets the 'mp3' directory housing mp3 audio test files

    get_mp3_audio_file_paths(self) -> list[str]:
        Gets 'mp3' test file paths

    get_mp3_audio_files(self) -> list[AudioFile]:
        Gets 'mp3' test files

    get_mp4_audio_dir(self) -> Dir:
        Gets the 'mp4' directory housing mp4 audio test files

    get_mp4_audio_file_paths(self) -> list[str]:
        Gets 'mp4' test file paths

    get_mp4_audio_files(self) -> list[AudioFile]:
        Gets 'mp4' test files

    get_wav_audio_dir(self) -> Dir:
        Gets the 'wav' directory housing wav audio test files

    get_wav_audio_file_paths(self) -> list[str]:
        Gets 'wav' test file paths

    get_wav_audio_files(self) -> list[AudioFile]:
        Gets 'wav' test files

    get_video_dir(self) -> Dir:
        Gets the 'video' directory housing video test files

    get_video_file_paths(self) -> list[str]:
        Gets all video file paths

    get_video_files(self) -> list[VideoFile]:
        Gets all video test files

    get_mp4_video_dir(self) -> Dir:
        Gets the 'mp4' directory housing mp4 video test files

    get_mp4_video_file_paths(self) -> list[str]:
        Gets 'mp4' test file paths

    get_mp4_video_files(self) -> list[VideoFile]:
        Gets 'mp4' test files

    get_mov_video_dir(self) -> Dir:
        Gets the 'mov' directory housing mov video test files

    get_mov_video_file_paths(self) -> list[str]:
        Gets 'mov' test file paths

    get_mov_video_files(self) -> list[VideoFile]:
        Gets 'mov' test files

    get_transcription_dir(self) -> Dir:
        Gets the 'transcribe' directory housing transcription test files

    get_transcription_file_paths(self) -> list[str]:
        Gets all transcription file paths

    get_transcription_files(self) -> list[JSONFile]:
        Gets all transcription test files
    """

    def get_paths(self) -> list[str]:
        """
        Gets valid file system object paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            valid file system object paths
        """
        return self.get_file_paths() + self.get_dir_paths()

    def get_invalid_paths(self) -> list[str]:
        """
        Gets invalid file system object paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid file system object paths
        """
        return [
            "invalid/file/path",
            "ben/likes/writing/random/shit",
            "jajaj/jajaaj/arglebargle",
            "this/is/way/too/much/fun",
            "lebron/the/goat",
        ]

    def get_file_paths(self) -> list[str]:
        """
        Gets valid file system file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            valid file system file paths
        """
        return self.get_transcription_file_paths() + self.get_media_file_paths()

    def get_invalid_file_paths(self) -> list[str]:
        """
        Gets invalid file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid file paths
        """
        return self.get_invalid_paths() + self.get_dir_paths()

    def get_json_file_paths(self) -> list[str]:
        """
        Gets valid json file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            valid json file paths
        """
        return self.get_transcription_file_paths()

    def get_invalid_json_file_paths(self) -> list[str]:
        """
        Gets invalid json file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid json file paths
        """
        return self.get_invalid_file_paths() + self.get_media_file_paths()

    def get_dir_paths(self) -> list[str]:
        """
        Gets valid file system directory paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            valid file system directory paths
        """
        return [
            self.get_media_dir().get_path(),
            self.get_test_files_dir().get_path(),
            self.get_transcription_dir().get_path(),
        ]

    def get_invalid_dir_paths(self) -> list[str]:
        """
        Gets invalid directory paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid directory paths
        """
        return self.get_file_paths()

    def get_test_files_dir(self) -> Dir:
        """
        Gets the 'test_files' directory

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'test_files' directory
        """
        return Dir(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_files"))
        )

    def get_media_dir(self) -> Dir:
        """
        Gets the 'media' directory housing media test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'media' directory housing media test files
        """
        return Dir(os.path.join(self.get_test_files_dir().get_path(), "media"))

    def get_media_file_paths(self) -> list[str]:
        """
        Gets all media file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            all media file paths
        """
        return (
            self.get_image_file_paths()
            + self.get_audio_file_paths()
            + self.get_video_file_paths()
        )

    def get_invalid_media_file_paths(self) -> list[str]:
        """
        Gets invalid media file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid media file paths
        """
        return self.get_invalid_file_paths() + self.get_transcription_file_paths()

    def get_media_files(self) -> list[MediaFile]:
        """
        Gets all media test files

        Parameters
        ----------
        None

        Returns
        -------
        list[MediaFile]
            all media test files
        """
        media_file_paths = self.get_media_file_paths()
        media_files = []
        for file_path in media_file_paths:
            media_files.append(MediaFile(file_path))
        return media_files

    def get_image_dir(self) -> Dir:
        """
        Gets the 'image' directory housing image test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'image' directory housing image test files
        """
        return Dir(os.path.join(self.get_media_dir().get_path(), "image"))

    def get_image_file_paths(self) -> list[str]:
        """
        Gets all image file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            all image file paths
        """
        return self.get_jpeg_image_file_paths() + self.get_png_image_file_paths()

    def get_image_files(self) -> list[ImageFile]:
        """
        Gets all image test files

        Parameters
        ----------
        None

        Returns
        -------
        list[ImageFile]
            all image test files
        """
        return self.get_jpeg_image_files() + self.get_png_image_files()

    def get_invalid_image_file_paths(self) -> list[str]:
        """
        Gets invalid image file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid image file paths
        """
        return (
            self.get_invalid_media_file_paths()
            + self.get_audio_file_paths()
            + self.get_video_file_paths()
        )

    def get_jpeg_image_dir(self) -> Dir:
        """
        Gets the 'jpeg' directory housing jpeg image test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'jpeg' directory housing jpeg image test files
        """
        return Dir(os.path.join(self.get_image_dir().get_path(), "jpeg"))

    def get_jpeg_image_file_paths(self) -> list[str]:
        """
        Gets 'jpeg' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'jpeg' test file paths
        """
        jpeg_image_dir = self.get_jpeg_image_dir()
        return jpeg_image_dir.get_file_paths_with_extension("jpeg")

    def get_jpeg_image_files(self) -> list[ImageFile]:
        """
        Gets 'jpeg' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[ImageFile]
            'jpeg' test files
        """
        jpeg_image_file_paths = self.get_jpeg_image_file_paths()
        jpeg_image_files = []
        for file_path in jpeg_image_file_paths:
            jpeg_image_files.append(ImageFile(file_path))
        return jpeg_image_files

    def get_png_image_dir(self) -> Dir:
        """
        Gets the 'png' directory housing png image test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'png' directory housing png image test files
        """
        return Dir(os.path.join(self.get_image_dir().get_path(), "png"))

    def get_png_image_file_paths(self) -> list[str]:
        """
        Gets 'png' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'png' test file paths
        """
        png_image_dir = self.get_png_image_dir()
        return png_image_dir.get_file_paths_with_extension("png")

    def get_png_image_files(self) -> list[ImageFile]:
        """
        Gets 'png' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[ImageFile]
            'png' test files
        """
        png_image_file_paths = self.get_png_image_file_paths()
        png_image_files = []
        for file_path in png_image_file_paths:
            png_image_files.append(ImageFile(file_path))
        return png_image_files

    def get_temporal_media_file_paths(self) -> list[str]:
        """
        Gets all temporal media (audio and video) file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            all temporal media file paths
        """
        return self.get_audio_file_paths() + self.get_video_file_paths()

    def get_temporal_media_files(self) -> list[TemporalMediaFile]:
        """
        Gets all temporal media (audio and video) test files

        Parameters
        ----------
        None

        Returns
        -------
        list[TemporalMediaFile]
            all temporal media test files
        """
        temporal_media_file_paths = self.get_temporal_media_file_paths()
        temporal_media_files = []
        for file_path in temporal_media_file_paths:
            temporal_media_files.append(TemporalMediaFile(file_path))
        return temporal_media_files

    def get_invalid_temporal_media_file_paths(self) -> list[str]:
        """
        Gets invalid temporal media file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid temporal media file paths
        """
        return self.get_invalid_media_file_paths() + self.get_image_file_paths()

    def get_audio_dir(self) -> Dir:
        """
        Gets the 'audio' directory housing audio test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'audio' directory housing audio test files
        """
        return Dir(os.path.join(self.get_media_dir().get_path(), "audio"))

    def get_audio_file_paths(self) -> list[str]:
        """
        Gets all audio file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            all audio file paths
        """
        return (
            self.get_mp3_audio_file_paths()
            + self.get_mp4_audio_file_paths()
            + self.get_wav_audio_file_paths()
        )

    def get_audio_files(self) -> list[AudioFile]:
        """
        Gets all audio test files

        Parameters
        ----------
        None

        Returns
        -------
        list[AudioFile]
            all audio test files
        """
        return (
            self.get_mp3_audio_files()
            + self.get_mp4_audio_files()
            + self.get_wav_audio_files()
        )

    def get_invalid_audio_file_paths(self) -> list[str]:
        """
        Gets invalid audio file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid audio file paths
        """
        return (
            self.get_invalid_media_file_paths()
            + self.get_image_file_paths()
            + self.get_video_file_paths()
        )

    def get_mp3_audio_dir(self) -> Dir:
        """
        Gets the 'mp3' directory housing mp3 audio test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'mp3' directory housing mp3 audio test files
        """
        return Dir(os.path.join(self.get_audio_dir().get_path(), "mp3"))

    def get_mp3_audio_file_paths(self) -> list[str]:
        """
        Gets 'mp3' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'mp3' test file paths
        """
        mp3_audio_dir = self.get_mp3_audio_dir()
        return mp3_audio_dir.get_file_paths_with_extension("mp3")

    def get_mp3_audio_files(self) -> list[AudioFile]:
        """
        Gets 'mp3' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[AudioFile]
            'mp3' test files
        """
        mp3_audio_file_paths = self.get_mp3_audio_file_paths()
        mp3_audio_files = []
        for file_path in mp3_audio_file_paths:
            mp3_audio_files.append(AudioFile(file_path))
        return mp3_audio_files

    def get_mp4_audio_dir(self) -> Dir:
        """
        Gets the 'mp4' directory housing mp4 audio test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'mp4' directory housing mp4 audio test files
        """
        return Dir(os.path.join(self.get_audio_dir().get_path(), "mp4"))

    def get_mp4_audio_file_paths(self) -> list[str]:
        """
        Gets 'mp4' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'mp4' test file paths
        """
        mp4_audio_dir = self.get_mp4_audio_dir()
        return mp4_audio_dir.get_file_paths_with_extension("mp4")

    def get_mp4_audio_files(self) -> list[AudioFile]:
        """
        Gets 'mp4' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[AudioFile]
            'mp4' test files
        """
        mp4_audio_file_paths = self.get_mp4_audio_file_paths()
        mp4_audio_files = []
        for file_path in mp4_audio_file_paths:
            mp4_audio_files.append(AudioFile(file_path))
        return mp4_audio_files

    def get_wav_audio_dir(self) -> Dir:
        """
        Gets the 'wav' directory housing wav audio test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'wav' directory housing wav audio test files
        """
        return Dir(os.path.join(self.get_audio_dir().get_path(), "wav"))

    def get_wav_audio_file_paths(self) -> list[str]:
        """
        Gets 'wav' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'wav' test file paths
        """
        wav_audio_dir = self.get_wav_audio_dir()
        return wav_audio_dir.get_file_paths_with_extension("wav")

    def get_wav_audio_files(self) -> list[AudioFile]:
        """
        Gets 'wav' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[AudioFile]
            'wav' test files
        """
        wav_audio_file_paths = self.get_wav_audio_file_paths()
        wav_audio_files = []
        for file_path in wav_audio_file_paths:
            wav_audio_files.append(AudioFile(file_path))
        return wav_audio_files

    def get_video_dir(self) -> Dir:
        """
        Gets the 'video' directory housing video test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'video' directory housing video test files
        """
        return Dir(os.path.join(self.get_media_dir().get_path(), "video"))

    def get_video_file_paths(self) -> list[str]:
        """
        Gets all video file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            all video file paths
        """
        return self.get_mp4_video_file_paths() + self.get_mov_video_file_paths()

    def get_video_files(self) -> list[VideoFile]:
        """
        Gets all video test files

        Parameters
        ----------
        None

        Returns
        -------
        list[VideoFile]
            all video test files
        """
        return self.get_mp4_video_files() + self.get_mov_video_files()

    def get_invalid_video_file_paths(self) -> list[str]:
        """
        Gets invalid video file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            Invalid video file paths
        """
        return (
            self.get_invalid_media_file_paths()
            + self.get_image_file_paths()
            + self.get_audio_file_paths()
        )

    def get_mp4_video_dir(self) -> Dir:
        """
        Gets the 'mp4' directory housing mp4 video test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'mp4' directory housing mp4 video test files
        """
        return Dir(os.path.join(self.get_video_dir().get_path(), "mp4"))

    def get_mp4_video_file_paths(self) -> list[str]:
        """
        Gets 'mp4' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'mp4' test file paths
        """
        mp4_video_dir = self.get_mp4_video_dir()
        return mp4_video_dir.get_file_paths_with_extension("mp4")

    def get_mp4_video_files(self) -> list[VideoFile]:
        """
        Gets 'mp4' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[VideoFile]
            'mp4' test files
        """
        mp4_video_file_paths = self.get_mp4_video_file_paths()
        mp4_video_files = []
        for file_path in mp4_video_file_paths:
            mp4_video_files.append(VideoFile(file_path))
        return mp4_video_files

    def get_mov_video_dir(self) -> Dir:
        """
        Gets the 'mov' directory housing mov video test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'mov' directory housing mov video test files
        """
        return Dir(os.path.join(self.get_video_dir().get_path(), "mov"))

    def get_mov_video_file_paths(self) -> list[str]:
        """
        Gets 'mov' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'mov' test file paths
        """
        mov_video_dir = self.get_mov_video_dir()
        return mov_video_dir.get_file_paths_with_extension("mov")

    def get_mov_video_files(self) -> list[VideoFile]:
        """
        Gets 'mov' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[VideoFile]
            'mov' test files
        """
        mov_video_file_paths = self.get_mov_video_file_paths()
        mov_video_files = []
        for file_path in mov_video_file_paths:
            mov_video_files.append(VideoFile(file_path))
        return mov_video_files

    def get_audiovideo_dir(self) -> Dir:
        """
        Gets the 'audiovideo' directory housing audiovideo test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'audiovideo' directory housing audiovideo test files
        """
        return Dir(os.path.join(self.get_media_dir().get_path(), "audiovideo"))

    def get_mp4_audiovideo_dir(self) -> Dir:
        """
        Gets the 'mp4' directory housing mp4 audiovideo test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'mp4' directory housing mp4 audiovideo test files
        """
        return Dir(os.path.join(self.get_audiovideo_dir().get_path(), "mp4"))

    def get_mp4_audiovideo_file_paths(self) -> list[str]:
        """
        Gets 'mp4' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'mp4' test file paths
        """
        mp4_audiovideo_dir = self.get_mp4_audiovideo_dir()
        return mp4_audiovideo_dir.get_file_paths_with_extension("mp4")

    def get_mp4_audiovideo_files(self) -> list[AudioVideoFile]:
        """
        Gets 'mp4' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[AudioVideoFile]
            'mp4' test files
        """
        mp4_audiovideo_file_paths = self.get_mp4_audiovideo_file_paths()
        mp4_audiovideo_files = []
        for file_path in mp4_audiovideo_file_paths:
            mp4_audiovideo_files.append(AudioVideoFile(file_path))
        return mp4_audiovideo_files

    def get_mov_audiovideo_dir(self) -> Dir:
        """
        Gets the 'mov' directory housing mov audiovideo test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'mov' directory housing mov audiovideo test files
        """
        return Dir(os.path.join(self.get_audiovideo_dir().get_path(), "mov"))

    def get_mov_audiovideo_file_paths(self) -> list[str]:
        """
        Gets 'mov' test file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            'mov' test file paths
        """
        mov_audiovideo_dir = self.get_mov_audiovideo_dir()
        return mov_audiovideo_dir.get_file_paths_with_extension("mov")

    def get_mov_audiovideo_files(self) -> list[AudioVideoFile]:
        """
        Gets 'mov' test files

        Parameters
        ----------
        None

        Returns
        -------
        list[AudioVideoFile]
            'mov' test files
        """
        mov_audiovideo_file_paths = self.get_mov_audiovideo_file_paths()
        mov_audiovideo_files = []
        for file_path in mov_audiovideo_file_paths:
            mov_audiovideo_files.append(AudioVideoFile(file_path))
        return mov_audiovideo_files

    def get_transcription_dir(self) -> Dir:
        """
        Gets the 'transcribe' directory housing transcription test files

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            the 'transcribe' directory housing transcription test files
        """
        return Dir(os.path.join(self.get_test_files_dir().get_path(), "transcription"))

    def get_transcription_file_paths(self) -> list[str]:
        """
        Gets all transcription file paths

        Parameters
        ----------
        None

        Returns
        -------
        list[str]
            all transcription file paths
        """
        return self.get_transcription_dir().get_file_paths_with_extension("json")

    def get_transcription_files(self) -> list[JSONFile]:
        """
        Gets all transcription test files

        Parameters
        ----------
        None

        Returns
        -------
        list[JSONFile]
            all transcription test files
        """
        transcription_file_paths = self.get_transcription_file_paths()
        transcription_files = []
        for file_path in transcription_file_paths:
            transcription_files.append(JSONFile(file_path))
        return transcription_files

    def get_webscrape_dir(self) -> Dir:
        """
        Gets the 'webscrape' directory which test webscraping files should be
        downloaded to

        Parameters
        ----------
        None

        Returns
        -------
        Dir
            The 'webscrape' directory which test webscraping files should be
            downloaded to.
        """
        return Dir(os.path.join(self.get_test_files_dir().get_path(), "webscrape"))
