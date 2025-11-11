"""
Caption handling for viral clips.
Supports SRT/VTT export and optional burn-in.
"""
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from clipsai import Transcription

logger = logging.getLogger(__name__)


class CaptionHandler:
    """
    Handles caption generation and export for clips.
    Supports SRT, VTT formats and optional burn-in.
    """

    def __init__(self, style: str = "minimal"):
        """
        Initialize caption handler.

        Parameters
        ----------
        style : str
            Caption style: "minimal", "highlighted", "animated"
        """
        self.style = style

    def generate_srt(self,
                    transcription: Transcription,
                    start_time: float,
                    end_time: float,
                    output_path: str) -> str:
        """
        Generate SRT caption file for a clip.

        Parameters
        ----------
        transcription : Transcription
            Full transcription
        start_time : float
            Clip start time (seconds)
        end_time : float
            Clip end time (seconds)
        output_path : str
            Output path for SRT file

        Returns
        -------
        str
            Path to generated SRT file
        """
        # Extract words in time range
        words = self._extract_words_in_range(transcription, start_time, end_time)

        # Generate SRT content
        srt_content = self._words_to_srt(words, start_time)

        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        logger.info(f"Generated SRT: {output_path}")
        return output_path

    def generate_vtt(self,
                    transcription: Transcription,
                    start_time: float,
                    end_time: float,
                    output_path: str) -> str:
        """
        Generate WebVTT caption file for a clip.

        Parameters
        ----------
        transcription : Transcription
            Full transcription
        start_time : float
            Clip start time (seconds)
        end_time : float
            Clip end time (seconds)
        output_path : str
            Output path for VTT file

        Returns
        -------
        str
            Path to generated VTT file
        """
        # Extract words in time range
        words = self._extract_words_in_range(transcription, start_time, end_time)

        # Generate VTT content
        vtt_content = self._words_to_vtt(words, start_time)

        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)

        logger.info(f"Generated VTT: {output_path}")
        return output_path

    def _extract_words_in_range(self,
                                transcription: Transcription,
                                start_time: float,
                                end_time: float) -> List[Dict[str, Any]]:
        """
        Extract words within time range from transcription.

        Parameters
        ----------
        transcription : Transcription
            Full transcription
        start_time : float
            Start time (seconds)
        end_time : float
            End time (seconds)

        Returns
        -------
        List[Dict[str, Any]]
            Words with timing information
        """
        words = []
        current_word = {"text": "", "start": None, "end": None}

        char_info = transcription.get_char_info()

        for char_data in char_info:
            char_start = char_data.get("start_time")
            char_end = char_data.get("end_time")
            char = char_data.get("char", "")

            # Skip if outside range
            if char_start is not None and char_start < start_time:
                continue
            if char_start is not None and char_start >= end_time:
                break

            # Build words
            if char == " ":
                if current_word["text"]:
                    words.append(current_word)
                    current_word = {"text": "", "start": None, "end": None}
            else:
                current_word["text"] += char
                if current_word["start"] is None:
                    current_word["start"] = char_start
                current_word["end"] = char_end

        # Add final word
        if current_word["text"]:
            words.append(current_word)

        # Adjust times relative to clip start
        for word in words:
            if word["start"] is not None:
                word["start"] -= start_time
            if word["end"] is not None:
                word["end"] -= start_time

        return words

    def _words_to_srt(self, words: List[Dict[str, Any]], offset: float = 0) -> str:
        """
        Convert words to SRT format.

        Groups words into 2-3 word phrases for better readability.

        Parameters
        ----------
        words : List[Dict[str, Any]]
            Words with timing
        offset : float
            Time offset (not used since we already adjusted)

        Returns
        -------
        str
            SRT formatted content
        """
        if not words:
            return ""

        srt_lines = []
        entry_num = 1

        # Group words into phrases (2-3 words each)
        words_per_caption = 3
        for i in range(0, len(words), words_per_caption):
            phrase_words = words[i:i + words_per_caption]
            if not phrase_words:
                continue

            # Build phrase
            phrase_text = " ".join([w["text"] for w in phrase_words])

            # Get timing
            start_time = phrase_words[0]["start"]
            end_time = phrase_words[-1]["end"]

            if start_time is None or end_time is None:
                continue

            # Format times
            start_str = self._format_srt_time(start_time)
            end_str = self._format_srt_time(end_time)

            # Build SRT entry
            srt_lines.append(f"{entry_num}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(phrase_text)
            srt_lines.append("")  # Blank line

            entry_num += 1

        return "\n".join(srt_lines)

    def _words_to_vtt(self, words: List[Dict[str, Any]], offset: float = 0) -> str:
        """
        Convert words to WebVTT format.

        Parameters
        ----------
        words : List[Dict[str, Any]]
            Words with timing
        offset : float
            Time offset

        Returns
        -------
        str
            VTT formatted content
        """
        vtt_lines = ["WEBVTT", ""]

        # Group words into phrases
        words_per_caption = 3
        for i in range(0, len(words), words_per_caption):
            phrase_words = words[i:i + words_per_caption]
            if not phrase_words:
                continue

            phrase_text = " ".join([w["text"] for w in phrase_words])

            start_time = phrase_words[0]["start"]
            end_time = phrase_words[-1]["end"]

            if start_time is None or end_time is None:
                continue

            start_str = self._format_vtt_time(start_time)
            end_str = self._format_vtt_time(end_time)

            vtt_lines.append(f"{start_str} --> {end_str}")
            vtt_lines.append(phrase_text)
            vtt_lines.append("")

        return "\n".join(vtt_lines)

    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def get_burn_in_filter(self,
                          srt_path: str,
                          style: Optional[str] = None) -> str:
        """
        Generate ffmpeg filter for burning in subtitles.

        Parameters
        ----------
        srt_path : str
            Path to SRT file
        style : str, optional
            Caption style override

        Returns
        -------
        str
            FFmpeg subtitles filter string
        """
        style = style or self.style

        # Escape path for ffmpeg
        escaped_path = srt_path.replace('\\', '\\\\').replace(':', '\\:')

        if style == "highlighted":
            # Bold, larger font with background
            return (
                f"subtitles={escaped_path}:"
                "force_style='FontName=Arial,FontSize=24,Bold=1,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BackColour=&H80000000,BorderStyle=4,Outline=2,"
                "Shadow=1,MarginV=50'"
            )
        elif style == "minimal":
            # Clean, simple style
            return (
                f"subtitles={escaped_path}:"
                "force_style='FontName=Arial,FontSize=20,Bold=0,"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                "BorderStyle=1,Outline=1,MarginV=40'"
            )
        else:
            # Default style
            return f"subtitles={escaped_path}"

    def create_caption_bundle(self,
                            transcription: Transcription,
                            start_time: float,
                            end_time: float,
                            output_dir: str,
                            base_name: str) -> Dict[str, str]:
        """
        Create both SRT and VTT files for a clip.

        Parameters
        ----------
        transcription : Transcription
            Full transcription
        start_time : float
            Clip start time
        end_time : float
            Clip end time
        output_dir : str
            Output directory
        base_name : str
            Base filename (without extension)

        Returns
        -------
        Dict[str, str]
            Paths to generated caption files
        """
        os.makedirs(output_dir, exist_ok=True)

        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        vtt_path = os.path.join(output_dir, f"{base_name}.vtt")

        self.generate_srt(transcription, start_time, end_time, srt_path)
        self.generate_vtt(transcription, start_time, end_time, vtt_path)

        return {
            "srt": srt_path,
            "vtt": vtt_path,
        }


def create_caption_handler(config) -> CaptionHandler:
    """
    Factory function to create caption handler from config.

    Parameters
    ----------
    config : Config
        AdLab configuration

    Returns
    -------
    CaptionHandler
        Configured handler instance
    """
    style = config.get("captions.style", "minimal")
    return CaptionHandler(style=style)
