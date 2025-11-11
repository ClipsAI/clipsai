#!/usr/bin/env python3
"""
Example usage of the FCP XML 7.0 Premiere Pro export functionality
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from clipfactory.processing.premiere.xml_generator import (
    PremiereXMLGenerator,
    export_clips_to_premiere
)


def example_basic_usage():
    """Example 1: Basic usage with defaults (backward compatible)."""
    print("=" * 70)
    print("Example 1: Basic Usage (Backward Compatible)")
    print("=" * 70)

    clips = [
        {'start_time': 0.0, 'end_time': 5.0},
        {'start_time': 10.0, 'end_time': 15.0},
        {'start_time': 20.0, 'end_time': 30.0},
    ]

    # Simple usage - uses defaults (60fps, 1920x1080, 48000Hz)
    export_clips_to_premiere(
        clips=clips,
        source_video="/path/to/your/video.mp4",
        output_xml="/path/to/output/clips_1080p60.xml"
    )

    print("✓ Generated 1080p 60fps XML with default settings")
    print()


def example_4k_project():
    """Example 2: 4K project at 30fps."""
    print("=" * 70)
    print("Example 2: 4K Project (3840x2160 @ 30fps)")
    print("=" * 70)

    clips = [
        {'start_time': 0.0, 'end_time': 10.0},
        {'start_time': 15.0, 'end_time': 25.0},
        {'start_time': 30.0, 'end_time': 45.0},
    ]

    export_clips_to_premiere(
        clips=clips,
        source_video="/path/to/your/4k_video.mp4",
        output_xml="/path/to/output/clips_4k30.xml",
        fps=30,
        resolution=(3840, 2160)
    )

    print("✓ Generated 4K 30fps XML")
    print()


def example_cinematic():
    """Example 3: Cinematic 24fps project."""
    print("=" * 70)
    print("Example 3: Cinematic (1920x1080 @ 24fps)")
    print("=" * 70)

    clips = [
        {'start_time': 0.0, 'end_time': 8.0},
        {'start_time': 12.0, 'end_time': 20.0},
    ]

    export_clips_to_premiere(
        clips=clips,
        source_video="/path/to/your/cinematic_video.mp4",
        output_xml="/path/to/output/clips_24fps.xml",
        fps=24
    )

    print("✓ Generated 24fps cinematic XML")
    print()


def example_advanced_configuration():
    """Example 4: Advanced configuration with custom generator."""
    print("=" * 70)
    print("Example 4: Advanced Configuration")
    print("=" * 70)

    # Create custom generator instance
    generator = PremiereXMLGenerator(
        fps=30,
        resolution=(2560, 1440),  # 1440p
        audio_sample_rate=48000,
        video_codec="H.264",
        audio_codec="AAC"
    )

    clips = [
        {'start_time': 0.0, 'end_time': 5.0},
        {'start_time': 7.0, 'end_time': 12.0},
        {'start_time': 15.0, 'end_time': 25.0},
        {'start_time': 30.0, 'end_time': 40.0},
    ]

    generator.generate_xml(
        clips=clips,
        source_video_path="/path/to/your/video.mp4",
        output_path="/path/to/output/clips_1440p30.xml",
        max_duration_seconds=None  # Auto-calculate from clips
    )

    print("✓ Generated custom 1440p 30fps XML")
    print()


def example_multiple_projects():
    """Example 5: Generate multiple project variants."""
    print("=" * 70)
    print("Example 5: Multiple Project Variants")
    print("=" * 70)

    # Same clips, different export formats
    clips = [
        {'start_time': 0.0, 'end_time': 10.0},
        {'start_time': 15.0, 'end_time': 25.0},
        {'start_time': 30.0, 'end_time': 45.0},
    ]

    source_video = "/path/to/your/video.mp4"

    # Web/YouTube version - 1080p 60fps
    export_clips_to_premiere(
        clips=clips,
        source_video=source_video,
        output_xml="/path/to/output/youtube_1080p60.xml",
        fps=60,
        resolution=(1920, 1080)
    )
    print("✓ Generated YouTube version (1080p60)")

    # Social media version - 720p 30fps
    export_clips_to_premiere(
        clips=clips,
        source_video=source_video,
        output_xml="/path/to/output/social_720p30.xml",
        fps=30,
        resolution=(1280, 720)
    )
    print("✓ Generated social media version (720p30)")

    # Premium version - 4K 60fps
    export_clips_to_premiere(
        clips=clips,
        source_video=source_video,
        output_xml="/path/to/output/premium_4k60.xml",
        fps=60,
        resolution=(3840, 2160)
    )
    print("✓ Generated premium version (4K60)")

    print()


def example_with_clip_metadata():
    """Example 6: Clips with additional metadata."""
    print("=" * 70)
    print("Example 6: Clips with Additional Metadata")
    print("=" * 70)

    # Clips can include additional metadata (will be preserved)
    clips = [
        {
            'start_time': 0.0,
            'end_time': 5.0,
            'score': 0.95,
            'reason': 'High engagement moment',
            'category': 'action'
        },
        {
            'start_time': 10.0,
            'end_time': 15.0,
            'score': 0.88,
            'reason': 'Emotional peak',
            'category': 'emotional'
        },
        {
            'start_time': 20.0,
            'end_time': 30.0,
            'score': 0.92,
            'reason': 'Key dialogue',
            'category': 'dialogue'
        },
    ]

    export_clips_to_premiere(
        clips=clips,
        source_video="/path/to/your/video.mp4",
        output_xml="/path/to/output/analyzed_clips.xml"
    )

    print("✓ Generated XML from analyzed clips")
    print("  (Additional metadata preserved in clip dictionary)")
    print()


def example_validation_handling():
    """Example 7: Handling validation errors."""
    print("=" * 70)
    print("Example 7: Validation Error Handling")
    print("=" * 70)

    generator = PremiereXMLGenerator()

    # Example 1: Invalid clip times
    try:
        invalid_clips = [
            {'start_time': 10.0, 'end_time': 5.0}  # End before start
        ]
        generator.generate_xml(
            clips=invalid_clips,
            source_video_path="/path/to/video.mp4",
            output_path="/path/to/output.xml"
        )
    except ValueError as e:
        print(f"✓ Validation caught invalid times: {e}")

    # Example 2: Missing file
    try:
        valid_clips = [
            {'start_time': 0.0, 'end_time': 5.0}
        ]
        generator.generate_xml(
            clips=valid_clips,
            source_video_path="/nonexistent/video.mp4",
            output_path="/path/to/output.xml"
        )
    except FileNotFoundError as e:
        print(f"✓ Validation caught missing file: {e}")

    # Example 3: Proper error handling in production
    clips = [
        {'start_time': 0.0, 'end_time': 5.0}
    ]
    source = "/path/to/video.mp4"
    output = "/path/to/output.xml"

    try:
        generator.generate_xml(
            clips=clips,
            source_video_path=source,
            output_path=output
        )
        print("✓ XML generated successfully")
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ Error: {e}")
        # Handle error appropriately in your application
        pass

    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "FCP XML 7.0 USAGE EXAMPLES" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    print("These examples demonstrate the upgraded Premiere Pro XML export")
    print("functionality with FCP XML 7.0 format compatibility.")
    print()
    print("Note: Update file paths before running these examples!")
    print()

    # Run examples (commented out to avoid file errors)
    # Uncomment and update paths to actually run

    # example_basic_usage()
    # example_4k_project()
    # example_cinematic()
    # example_advanced_configuration()
    # example_multiple_projects()
    # example_with_clip_metadata()
    # example_validation_handling()

    print("=" * 70)
    print("Configuration Options Summary")
    print("=" * 70)
    print("""
Available parameters:
- fps: Frame rate (default: 60)
  Common values: 24 (cinematic), 30 (standard), 60 (smooth)

- resolution: (width, height) tuple (default: (1920, 1080))
  Common values: (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)

- audio_sample_rate: Audio sample rate in Hz (default: 48000)
  Common values: 44100, 48000

- video_codec: Video codec name (default: "H.264")
  Common values: "H.264", "H.265", "ProRes"

- audio_codec: Audio codec name (default: "AAC")
  Common values: "AAC", "MP3", "PCM"

- max_duration_seconds: Timeline duration (default: auto-calculate)
  Set to None for automatic calculation based on clips
    """)

    print("=" * 70)
    print("For more details, see:")
    print("  - UPGRADE_NOTES.md - Complete upgrade documentation")
    print("  - test_xml_generator.py - Test suite and validation")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
