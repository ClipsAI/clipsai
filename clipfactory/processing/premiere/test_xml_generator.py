#!/usr/bin/env python3
"""
Test script for Premiere Pro FCP XML 7.0 generator
Generates sample XML and validates structure
"""

import sys
import os
from pathlib import Path

# Add clipfactory to path
sys.path.insert(0, str(Path(__file__).parent))

from clipfactory.processing.premiere.xml_generator import PremiereXMLGenerator
import tempfile


def generate_sample_clips(num_clips: int = 10):
    """Generate sample clip data."""
    clips = []

    for i in range(num_clips):
        # Each clip is 5 seconds long
        start_time = i * 5.5  # 5 seconds + 0.5 gap
        end_time = start_time + 5.0

        clips.append({
            'start_time': start_time,
            'end_time': end_time,
            'score': 0.9,  # Optional metadata
            'reason': f'Test clip {i+1}'
        })

    return clips


def test_xml_generation():
    """Test XML generation with sample data."""

    print("=" * 70)
    print("Premiere Pro FCP XML 7.0 Generator - Test Script")
    print("=" * 70)
    print()

    # Create sample clips
    print("1. Generating sample clips...")
    clips = generate_sample_clips(10)
    print(f"   Created {len(clips)} clips")
    print()

    # Create a temporary test video file
    print("2. Creating test video reference...")
    test_video_path = "/tmp/test_video.mp4"

    # Create a dummy file for testing
    Path(test_video_path).touch()
    print(f"   Test video: {test_video_path}")
    print()

    # Test with different configurations
    print("3. Testing XML generation with multiple configurations...")
    print()

    configs = [
        {
            'name': '1080p 60fps (Default)',
            'fps': 60,
            'resolution': (1920, 1080),
            'audio_sample_rate': 48000
        },
        {
            'name': '4K 30fps',
            'fps': 30,
            'resolution': (3840, 2160),
            'audio_sample_rate': 48000
        },
        {
            'name': '720p 24fps',
            'fps': 24,
            'resolution': (1280, 720),
            'audio_sample_rate': 44100
        }
    ]

    for i, config in enumerate(configs, 1):
        print(f"   Test {i}: {config['name']}")
        print(f"   - Resolution: {config['resolution'][0]}x{config['resolution'][1]}")
        print(f"   - FPS: {config['fps']}")
        print(f"   - Audio: {config['audio_sample_rate']} Hz")

        # Create generator
        generator = PremiereXMLGenerator(
            fps=config['fps'],
            resolution=config['resolution'],
            audio_sample_rate=config['audio_sample_rate']
        )

        # Generate XML
        output_file = f"/tmp/test_premiere_{config['fps']}fps.xml"

        try:
            result = generator.generate_xml(
                clips=clips,
                source_video_path=test_video_path,
                output_path=output_file
            )

            # Check file exists and has content
            if os.path.exists(result):
                file_size = os.path.getsize(result)
                print(f"   ✓ XML generated: {result} ({file_size} bytes)")
            else:
                print(f"   ✗ Failed to generate XML")

        except Exception as e:
            print(f"   ✗ Error: {e}")

        print()

    # Read and display sample output
    print("=" * 70)
    print("4. Sample XML Output (First 100 lines)")
    print("=" * 70)
    print()

    sample_file = "/tmp/test_premiere_60fps.xml"
    if os.path.exists(sample_file):
        with open(sample_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:100], 1):
                print(f"{i:3d}: {line}", end='')

    print()
    print("=" * 70)
    print()

    # Validate XML structure
    print("5. Validating XML Structure...")
    validate_xml_structure(sample_file)
    print()

    # Test validation features
    print("6. Testing Validation Features...")
    test_validation()
    print()

    print("=" * 70)
    print("Test Complete!")
    print("=" * 70)
    print()
    print(f"Generated XML files:")
    for config in configs:
        xml_file = f"/tmp/test_premiere_{config['fps']}fps.xml"
        if os.path.exists(xml_file):
            print(f"  - {xml_file}")
    print()

    # Cleanup
    os.remove(test_video_path)


def validate_xml_structure(xml_file: str):
    """Validate that XML contains required FCP XML 7.0 elements."""

    try:
        with open(xml_file, 'r') as f:
            content = f.read()

        # Check for required elements
        required_elements = [
            ('<?xml version', 'XML declaration'),
            ('<!DOCTYPE xmeml>', 'DOCTYPE declaration'),
            ('<xmeml version="5">', 'XMEML version 5 (FCP XML 7.0)'),
            ('<project>', 'Project wrapper'),
            ('<sequence>', 'Sequence element'),
            ('<rate>', 'Rate information'),
            ('<timecode>', 'Timecode information'),
            ('<displayformat>', 'Timecode display format'),
            ('<media>', 'Media container'),
            ('<video>', 'Video track'),
            ('<audio>', 'Audio track'),
            ('<samplecharacteristics>', 'Sample characteristics'),
            ('<width>', 'Video width'),
            ('<height>', 'Video height'),
            ('<samplerate>', 'Audio sample rate'),
            ('<clipitem', 'Clip items'),
            ('<pathurl>file://', 'Platform-specific file URLs'),
            ('<uuid>', 'UUID elements'),
            ('<codec>', 'Codec information'),
        ]

        print()
        for element, description in required_elements:
            if element in content:
                print(f"   ✓ {description}")
            else:
                print(f"   ✗ MISSING: {description}")

        # Count clips
        clip_count = content.count('<clipitem')
        print()
        print(f"   Total clip items: {clip_count}")
        print(f"   Expected: 20 (10 video + 10 audio)")

        if clip_count == 20:
            print(f"   ✓ Correct number of clips")
        else:
            print(f"   ✗ Incorrect clip count")

    except Exception as e:
        print(f"   ✗ Validation error: {e}")


def test_validation():
    """Test validation features."""

    test_video_path = "/tmp/test_video_validation.mp4"
    Path(test_video_path).touch()

    generator = PremiereXMLGenerator()

    # Test 1: Empty clips
    print("\n   Test 1: Empty clips list")
    try:
        generator._validate_clips([])
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")

    # Test 2: Missing start_time
    print("\n   Test 2: Missing start_time")
    try:
        generator._validate_clips([{'end_time': 10.0}])
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")

    # Test 3: Negative start_time
    print("\n   Test 3: Negative start_time")
    try:
        generator._validate_clips([{'start_time': -1.0, 'end_time': 10.0}])
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")

    # Test 4: End time before start time
    print("\n   Test 4: End time before start time")
    try:
        generator._validate_clips([{'start_time': 10.0, 'end_time': 5.0}])
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly rejected: {e}")

    # Test 5: Non-existent file
    print("\n   Test 5: Non-existent source file")
    try:
        generator._validate_source_path("/nonexistent/path/video.mp4")
        print("   ✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"   ✓ Correctly rejected: {e}")

    # Test 6: Valid clips
    print("\n   Test 6: Valid clips")
    try:
        generator._validate_clips([
            {'start_time': 0.0, 'end_time': 5.0},
            {'start_time': 5.0, 'end_time': 10.0},
        ])
        print("   ✓ Valid clips accepted")
    except Exception as e:
        print(f"   ✗ Should not have raised error: {e}")

    # Cleanup
    os.remove(test_video_path)


if __name__ == "__main__":
    test_xml_generation()
