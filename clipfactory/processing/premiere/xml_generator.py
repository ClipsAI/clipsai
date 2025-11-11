"""
Premiere Pro XML Generator
Exports clips to Premiere-compatible FCP XML 7.0 format for modern Premiere Pro (2020+)
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from lxml import etree as ET
import logging
import platform
import os

logger = logging.getLogger(__name__)


class PremiereXMLGenerator:
    """Generate FCP XML 7.0 format for Premiere Pro 2020+ compatibility."""

    def __init__(
        self,
        fps: int = 60,
        audio_sample_rate: int = 48000,
        resolution: Tuple[int, int] = (1920, 1080),
        video_codec: str = "H.264",
        audio_codec: str = "AAC"
    ):
        """
        Initialize XML generator with configurable parameters.

        Args:
            fps: Frame rate (default: 60)
            audio_sample_rate: Audio sample rate in Hz (default: 48000)
            resolution: Video resolution as (width, height) tuple (default: 1920x1080)
            video_codec: Video codec name (default: H.264)
            audio_codec: Audio codec name (default: AAC)
        """
        self.timeline_fps = fps
        self.audio_sample_rate = audio_sample_rate
        self.video_width, self.video_height = resolution
        self.video_codec = video_codec
        self.audio_codec = audio_codec

    def generate_xml(
        self,
        clips: List[Dict[str, Any]],
        source_video_path: str,
        output_path: str,
        max_duration_seconds: Optional[float] = None
    ) -> str:
        """
        Generate FCP XML 7.0 file for Premiere Pro.

        Args:
            clips: List of clip dictionaries with start/end times
            source_video_path: Path to source video file
            output_path: Where to save XML file
            max_duration_seconds: Maximum timeline duration (default: auto-calculate from clips)

        Returns:
            Path to generated XML file

        Raises:
            ValueError: If clips validation fails
            FileNotFoundError: If source video doesn't exist
        """
        try:
            # Validate inputs
            self._validate_clips(clips)
            self._validate_source_path(source_video_path)

            # Calculate total duration if not specified
            if max_duration_seconds is None:
                max_duration_seconds = self._calculate_timeline_duration(clips)

            # Create FCP XML 7.0 structure
            root = self._create_root_element()

            # Add project wrapper
            project = self._create_project(root, max_duration_seconds)

            # Create sequence
            sequence = self._create_sequence(project, max_duration_seconds)

            # Add media tracks
            media = ET.SubElement(sequence, "media")
            video = ET.SubElement(media, "video")
            video_track = self._create_video_track(video)

            audio = ET.SubElement(media, "audio")
            audio_track = self._create_audio_track(audio)

            # Add clips to timeline
            for i, clip in enumerate(clips):
                self._add_clip_to_track(
                    video_track,
                    audio_track,
                    clip,
                    source_video_path,
                    i
                )

            # Write to file with proper DOCTYPE
            tree = ET.ElementTree(root)

            # Create custom DOCTYPE for FCP XML 7.0
            doctype = '<!DOCTYPE xmeml>'

            tree.write(
                output_path,
                pretty_print=True,
                xml_declaration=True,
                encoding='UTF-8',
                doctype=doctype
            )

            logger.info(f"Generated FCP XML 7.0: {output_path}")
            logger.info(f"  - {len(clips)} clips")
            logger.info(f"  - Resolution: {self.video_width}x{self.video_height}")
            logger.info(f"  - Frame rate: {self.timeline_fps} fps")
            logger.info(f"  - Audio: {self.audio_sample_rate} Hz")

            return output_path

        except Exception as e:
            logger.error(f"XML generation failed: {e}")
            raise

    def _create_root_element(self) -> ET.Element:
        """Create root xmeml element with FCP XML 7.0 attributes."""
        root = ET.Element(
            "xmeml",
            version="5"  # FCP XML 7.0 uses version 5
        )
        return root

    def _create_project(self, root: ET.Element, duration_seconds: float) -> ET.Element:
        """Create project wrapper with metadata."""
        project = ET.SubElement(root, "project")

        # Project name
        ET.SubElement(project, "name").text = "ClipFactory_Project"

        # Project children (will contain sequences)
        children = ET.SubElement(project, "children")

        return children

    def _create_sequence(self, parent: ET.Element, duration_seconds: float) -> ET.Element:
        """Create sequence with full metadata."""
        sequence = ET.SubElement(parent, "sequence")

        # Basic sequence info
        ET.SubElement(sequence, "uuid").text = "clipfactory-sequence-001"
        ET.SubElement(sequence, "name").text = "ClipFactory_Sequence"
        ET.SubElement(sequence, "duration").text = str(self._frames_from_seconds(duration_seconds))

        # Rate information
        rate = ET.SubElement(sequence, "rate")
        ET.SubElement(rate, "timebase").text = str(self.timeline_fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Timecode information
        timecode = ET.SubElement(sequence, "timecode")
        tc_rate = ET.SubElement(timecode, "rate")
        ET.SubElement(tc_rate, "timebase").text = str(self.timeline_fps)
        ET.SubElement(tc_rate, "ntsc").text = "FALSE"
        ET.SubElement(timecode, "string").text = "00:00:00:00"
        ET.SubElement(timecode, "frame").text = "0"
        ET.SubElement(timecode, "displayformat").text = "NDF"  # Non-Drop Frame

        return sequence

    def _create_video_track(self, video: ET.Element) -> ET.Element:
        """Create video track with format specifications."""
        # Format specs
        format_elem = ET.SubElement(video, "format")

        samplecharacteristics = ET.SubElement(format_elem, "samplecharacteristics")
        ET.SubElement(samplecharacteristics, "width").text = str(self.video_width)
        ET.SubElement(samplecharacteristics, "height").text = str(self.video_height)

        # Pixel aspect ratio
        ET.SubElement(samplecharacteristics, "pixelaspectratio").text = "square"

        # Frame rate
        rate = ET.SubElement(samplecharacteristics, "rate")
        ET.SubElement(rate, "timebase").text = str(self.timeline_fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Codec
        ET.SubElement(samplecharacteristics, "codec").text = self.video_codec

        # Color depth
        ET.SubElement(samplecharacteristics, "depth").text = "24"

        # Create track
        track = ET.SubElement(video, "track")

        return track

    def _create_audio_track(self, audio: ET.Element) -> ET.Element:
        """Create audio track with format specifications."""
        # Format specs
        format_elem = ET.SubElement(audio, "format")

        samplecharacteristics = ET.SubElement(format_elem, "samplecharacteristics")
        ET.SubElement(samplecharacteristics, "depth").text = "16"
        ET.SubElement(samplecharacteristics, "samplerate").text = str(self.audio_sample_rate)

        # Create track
        track = ET.SubElement(audio, "track")

        # Output configuration
        outputconfig = ET.SubElement(track, "outputconfig")
        layout = ET.SubElement(outputconfig, "layout")
        ET.SubElement(layout, "numchannels").text = "2"  # Stereo

        return track

    def _add_clip_to_track(
        self,
        video_track: ET.Element,
        audio_track: ET.Element,
        clip: Dict[str, Any],
        source_path: str,
        clip_index: int
    ):
        """Add a single clip to video and audio tracks with full metadata."""

        start_frame = self._frames_from_seconds(clip['start_time'])
        end_frame = self._frames_from_seconds(clip['end_time'])
        duration_frames = end_frame - start_frame

        # Calculate timeline position (stack clips sequentially with gap)
        gap_frames = self._frames_from_seconds(0.5)  # 0.5 second gap
        timeline_position = clip_index * (duration_frames + gap_frames)

        # Generate unique ID for this clip
        clip_id = f"clipfactory-clip-{clip_index:03d}"
        file_id = f"clipfactory-file-{clip_index:03d}"

        # Video clip item
        video_clip = self._create_video_clipitem(
            video_track,
            clip_id,
            clip_index,
            timeline_position,
            duration_frames,
            start_frame,
            end_frame,
            source_path,
            file_id
        )

        # Audio clip item
        audio_clip = self._create_audio_clipitem(
            audio_track,
            f"{clip_id}-audio",
            clip_index,
            timeline_position,
            duration_frames,
            start_frame,
            end_frame,
            source_path,
            file_id
        )

    def _create_video_clipitem(
        self,
        track: ET.Element,
        clip_id: str,
        clip_index: int,
        timeline_position: int,
        duration_frames: int,
        start_frame: int,
        end_frame: int,
        source_path: str,
        file_id: str
    ) -> ET.Element:
        """Create video clip item with full metadata."""

        clipitem = ET.SubElement(track, "clipitem", id=clip_id)

        # Basic info
        ET.SubElement(clipitem, "uuid").text = clip_id
        ET.SubElement(clipitem, "name").text = f"Clip_{clip_index:03d}"
        ET.SubElement(clipitem, "enabled").text = "TRUE"
        ET.SubElement(clipitem, "start").text = str(timeline_position)
        ET.SubElement(clipitem, "end").text = str(timeline_position + duration_frames)
        ET.SubElement(clipitem, "in").text = str(start_frame)
        ET.SubElement(clipitem, "out").text = str(end_frame)

        # Rate
        rate = ET.SubElement(clipitem, "rate")
        ET.SubElement(rate, "timebase").text = str(self.timeline_fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Media type
        ET.SubElement(clipitem, "mediatype").text = "video"

        # File reference with full metadata
        file_elem = self._create_file_reference(clipitem, source_path, file_id, "video")

        return clipitem

    def _create_audio_clipitem(
        self,
        track: ET.Element,
        clip_id: str,
        clip_index: int,
        timeline_position: int,
        duration_frames: int,
        start_frame: int,
        end_frame: int,
        source_path: str,
        file_id: str
    ) -> ET.Element:
        """Create audio clip item with full metadata."""

        clipitem = ET.SubElement(track, "clipitem", id=clip_id)

        # Basic info
        ET.SubElement(clipitem, "uuid").text = clip_id
        ET.SubElement(clipitem, "name").text = f"Clip_{clip_index:03d}_Audio"
        ET.SubElement(clipitem, "enabled").text = "TRUE"
        ET.SubElement(clipitem, "start").text = str(timeline_position)
        ET.SubElement(clipitem, "end").text = str(timeline_position + duration_frames)
        ET.SubElement(clipitem, "in").text = str(start_frame)
        ET.SubElement(clipitem, "out").text = str(end_frame)

        # Rate
        rate = ET.SubElement(clipitem, "rate")
        ET.SubElement(rate, "timebase").text = str(self.timeline_fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Media type
        ET.SubElement(clipitem, "mediatype").text = "audio"

        # File reference
        file_elem = self._create_file_reference(clipitem, source_path, file_id, "audio")

        # Source track (which audio channel)
        sourcetrack = ET.SubElement(clipitem, "sourcetrack")
        ET.SubElement(sourcetrack, "mediatype").text = "audio"
        ET.SubElement(sourcetrack, "trackindex").text = "1"

        return clipitem

    def _create_file_reference(
        self,
        parent: ET.Element,
        source_path: str,
        file_id: str,
        media_type: str
    ) -> ET.Element:
        """Create file reference with platform-specific path and full metadata."""

        file_elem = ET.SubElement(parent, "file", id=file_id)

        # File name
        ET.SubElement(file_elem, "name").text = Path(source_path).name

        # Platform-specific file path
        pathurl = self._get_platform_file_url(source_path)
        ET.SubElement(file_elem, "pathurl").text = pathurl

        # Rate
        rate = ET.SubElement(file_elem, "rate")
        ET.SubElement(rate, "timebase").text = str(self.timeline_fps)
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Duration (use a large value as source duration)
        # In real usage, this should be the actual file duration
        ET.SubElement(file_elem, "duration").text = str(self._frames_from_seconds(3600))  # 1 hour default

        # Media metadata
        media = ET.SubElement(file_elem, "media")

        if media_type == "video":
            # Video characteristics
            video = ET.SubElement(media, "video")
            samplecharacteristics = ET.SubElement(video, "samplecharacteristics")
            ET.SubElement(samplecharacteristics, "width").text = str(self.video_width)
            ET.SubElement(samplecharacteristics, "height").text = str(self.video_height)

            # Codec
            codec = ET.SubElement(samplecharacteristics, "codec")
            ET.SubElement(codec, "name").text = self.video_codec
            ET.SubElement(codec, "appspecificdata").text = ""

            # Rate
            vid_rate = ET.SubElement(samplecharacteristics, "rate")
            ET.SubElement(vid_rate, "timebase").text = str(self.timeline_fps)
            ET.SubElement(vid_rate, "ntsc").text = "FALSE"

        # Audio characteristics (always include for compatibility)
        audio = ET.SubElement(media, "audio")
        samplecharacteristics = ET.SubElement(audio, "samplecharacteristics")
        ET.SubElement(samplecharacteristics, "depth").text = "16"
        ET.SubElement(samplecharacteristics, "samplerate").text = str(self.audio_sample_rate)

        # Audio codec
        codec = ET.SubElement(samplecharacteristics, "codec")
        ET.SubElement(codec, "name").text = self.audio_codec

        return file_elem

    def _get_platform_file_url(self, file_path: str) -> str:
        """
        Convert file path to platform-specific file:// URL.

        Windows: file:///C:/path/to/file.mp4
        Mac/Linux: file:///absolute/path/to/file.mp4
        """
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)

        system = platform.system()

        if system == "Windows":
            # Windows: file:///C:/path/to/file.mp4
            # Replace backslashes with forward slashes
            abs_path = abs_path.replace("\\", "/")
            # Ensure it starts with file:///
            if abs_path[1] == ":":  # Has drive letter
                return f"file:///{abs_path}"
            else:
                return f"file://{abs_path}"
        else:
            # Mac/Linux: file:///absolute/path/to/file.mp4
            return f"file://{abs_path}"

    def _validate_clips(self, clips: List[Dict[str, Any]]) -> None:
        """
        Validate clip data for consistency.

        Raises:
            ValueError: If validation fails
        """
        if not clips:
            raise ValueError("No clips provided")

        for i, clip in enumerate(clips):
            # Check required fields
            if 'start_time' not in clip or 'end_time' not in clip:
                raise ValueError(f"Clip {i}: Missing start_time or end_time")

            start = clip['start_time']
            end = clip['end_time']

            # Validate times
            if start < 0:
                raise ValueError(f"Clip {i}: start_time cannot be negative ({start})")

            if end <= start:
                raise ValueError(f"Clip {i}: end_time ({end}) must be greater than start_time ({start})")

            if end - start < 0.01:  # Minimum 10ms duration
                raise ValueError(f"Clip {i}: Clip duration too short ({end - start}s)")

        # Check for overlapping clips in timeline (assuming sequential layout)
        logger.debug(f"Validated {len(clips)} clips successfully")

    def _validate_source_path(self, source_path: str) -> None:
        """
        Validate that source video file exists.

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source video not found: {source_path}")

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source path is not a file: {source_path}")

    def _calculate_timeline_duration(self, clips: List[Dict[str, Any]]) -> float:
        """Calculate total timeline duration based on clips."""
        total_duration = 0
        gap_seconds = 0.5

        for clip in clips:
            clip_duration = clip['end_time'] - clip['start_time']
            total_duration += clip_duration + gap_seconds

        # Add extra buffer at end
        total_duration += 5.0

        return total_duration

    def _frames_from_seconds(self, seconds: float) -> int:
        """Convert seconds to frame count."""
        return int(seconds * self.timeline_fps)


def export_clips_to_premiere(
    clips: List[Dict[str, Any]],
    source_video: str,
    output_xml: str,
    fps: int = 60,
    resolution: Tuple[int, int] = (1920, 1080),
    audio_sample_rate: int = 48000,
    video_codec: str = "H.264",
    audio_codec: str = "AAC"
) -> str:
    """
    Helper function to export clips to Premiere FCP XML 7.0 format.

    Args:
        clips: List of clips with start/end times
        source_video: Path to source video
        output_xml: Output XML path
        fps: Frame rate (default: 60)
        resolution: Video resolution as (width, height) tuple (default: 1920x1080)
        audio_sample_rate: Audio sample rate in Hz (default: 48000)
        video_codec: Video codec name (default: H.264)
        audio_codec: Audio codec name (default: AAC)

    Returns:
        Path to generated XML file
    """
    generator = PremiereXMLGenerator(
        fps=fps,
        audio_sample_rate=audio_sample_rate,
        resolution=resolution,
        video_codec=video_codec,
        audio_codec=audio_codec
    )
    return generator.generate_xml(clips, source_video, output_xml)
