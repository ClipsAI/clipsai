# Premiere Pro XML Export - FCP XML 7.0 Upgrade

## Overview

The Premiere Pro XML export has been upgraded from XMEML v4 (year 2000) to **FCP XML 7.0 format** (version 5), ensuring compatibility with **Premiere Pro 2020 and later versions**.

---

## What Changed

### 1. XML Format Upgraded to FCP XML 7.0

**Before:**
```xml
<xmeml version="4">
  <sequence>
    <!-- Basic structure -->
  </sequence>
</xmeml>
```

**After:**
```xml
<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE xmeml>
<xmeml version="5">
  <project>
    <name>ClipFactory_Project</name>
    <children>
      <sequence>
        <!-- Full metadata structure -->
      </sequence>
    </children>
  </project>
</xmeml>
```

**Key improvements:**
- Added DOCTYPE declaration
- Changed version from 4 to 5 (FCP XML 7.0 standard)
- Added proper `<project>` wrapper
- Added `<children>` container for sequences

---

### 2. Fixed File Paths (Platform-Specific)

**Before:**
```xml
<pathurl>file://localhost/path/to/video.mp4</pathurl>
```

**After:**
- **Windows:** `file:///C:/path/to/video.mp4`
- **Mac/Linux:** `file:///absolute/path/to/video.mp4`

The system now automatically detects the platform and formats URLs correctly.

---

### 3. Added Complete Metadata

#### Video Metadata
```xml
<video>
  <format>
    <samplecharacteristics>
      <width>1920</width>
      <height>1080</height>
      <pixelaspectratio>square</pixelaspectratio>
      <rate>
        <timebase>60</timebase>
        <ntsc>FALSE</ntsc>
      </rate>
      <codec>H.264</codec>
      <depth>24</depth>
    </samplecharacteristics>
  </format>
</video>
```

#### Audio Metadata
```xml
<audio>
  <format>
    <samplecharacteristics>
      <depth>16</depth>
      <samplerate>48000</samplerate>
    </samplecharacteristics>
  </format>
  <track>
    <outputconfig>
      <layout>
        <numchannels>2</numchannels>
      </layout>
    </outputconfig>
  </track>
</audio>
```

#### Timecode Information
```xml
<timecode>
  <rate>
    <timebase>60</timebase>
    <ntsc>FALSE</ntsc>
  </rate>
  <string>00:00:00:00</string>
  <frame>0</frame>
  <displayformat>NDF</displayformat>
</timecode>
```

#### Clip-Level Metadata
```xml
<clipitem id="clipfactory-clip-000">
  <uuid>clipfactory-clip-000</uuid>
  <name>Clip_000</name>
  <enabled>TRUE</enabled>
  <start>0</start>
  <end>300</end>
  <in>0</in>
  <out>300</out>
  <rate>
    <timebase>60</timebase>
    <ntsc>FALSE</ntsc>
  </rate>
  <mediatype>video</mediatype>
  <file id="clipfactory-file-000">
    <!-- File metadata with codec information -->
  </file>
</clipitem>
```

---

### 4. Configurable Parameters

**Before:** Hardcoded values
- FPS: 60 (hardcoded)
- Duration limit: 2 hours (hardcoded)
- Resolution: Not specified
- Audio rate: 48000 Hz (hardcoded)

**After:** Fully configurable

```python
from clipfactory.processing.premiere.xml_generator import PremiereXMLGenerator

# Create generator with custom settings
generator = PremiereXMLGenerator(
    fps=30,                          # Frame rate
    resolution=(3840, 2160),         # 4K resolution
    audio_sample_rate=48000,         # Audio sample rate
    video_codec="H.264",             # Video codec
    audio_codec="AAC"                # Audio codec
)

# Generate XML
generator.generate_xml(
    clips=clips,
    source_video_path="/path/to/video.mp4",
    output_path="/path/to/output.xml",
    max_duration_seconds=None  # Auto-calculate from clips
)
```

**Or use the helper function:**

```python
from clipfactory.processing.premiere.xml_generator import export_clips_to_premiere

export_clips_to_premiere(
    clips=clips,
    source_video="/path/to/video.mp4",
    output_xml="/path/to/output.xml",
    fps=60,
    resolution=(1920, 1080),
    audio_sample_rate=48000
)
```

---

### 5. Added Validation

The generator now validates:

#### Clip Validation
- ✓ Clips list is not empty
- ✓ All clips have `start_time` and `end_time`
- ✓ Start time is non-negative
- ✓ End time is greater than start time
- ✓ Clip duration is at least 10ms

#### File Validation
- ✓ Source video file exists
- ✓ Source path is a file (not directory)

**Example error messages:**
```python
# Empty clips
ValueError: No clips provided

# Missing start_time
ValueError: Clip 0: Missing start_time or end_time

# Negative start_time
ValueError: Clip 0: start_time cannot be negative (-1.0)

# Invalid time range
ValueError: Clip 0: end_time (5.0) must be greater than start_time (10.0)

# File not found
FileNotFoundError: Source video not found: /path/to/video.mp4
```

---

## Usage Examples

### Basic Usage (Backward Compatible)
```python
from clipfactory.processing.premiere.xml_generator import export_clips_to_premiere

clips = [
    {'start_time': 0.0, 'end_time': 5.0},
    {'start_time': 10.0, 'end_time': 15.0},
]

export_clips_to_premiere(
    clips=clips,
    source_video="/path/to/video.mp4",
    output_xml="/path/to/output.xml"
)
# Uses default: 60fps, 1920x1080, 48000Hz
```

### Advanced Usage (Custom Settings)
```python
from clipfactory.processing.premiere.xml_generator import PremiereXMLGenerator

# 4K 30fps project
generator = PremiereXMLGenerator(
    fps=30,
    resolution=(3840, 2160),
    audio_sample_rate=48000,
    video_codec="H.264",
    audio_codec="AAC"
)

clips = [
    {'start_time': 0.0, 'end_time': 10.0},
    {'start_time': 15.0, 'end_time': 25.0},
]

generator.generate_xml(
    clips=clips,
    source_video_path="/path/to/4k_video.mp4",
    output_path="/path/to/4k_project.xml"
)
```

### Different Frame Rates
```python
# 24fps cinematic
export_clips_to_premiere(
    clips=clips,
    source_video="/path/to/video.mp4",
    output_xml="/path/to/24fps.xml",
    fps=24
)

# 30fps standard
export_clips_to_premiere(
    clips=clips,
    source_video="/path/to/video.mp4",
    output_xml="/path/to/30fps.xml",
    fps=30
)

# 60fps smooth
export_clips_to_premiere(
    clips=clips,
    source_video="/path/to/video.mp4",
    output_xml="/path/to/60fps.xml",
    fps=60
)
```

---

## Testing

Run the test suite to validate XML generation:

```bash
cd /home/user/clipsai
python3 clipfactory/processing/premiere/test_xml_generator.py
```

**Test coverage:**
- ✓ XML generation with multiple configurations (1080p, 4K, 720p)
- ✓ Different frame rates (24fps, 30fps, 60fps)
- ✓ XML structure validation (all required elements present)
- ✓ Clip validation (empty clips, missing fields, negative times, etc.)
- ✓ File validation (non-existent files, invalid paths)
- ✓ Platform-specific file URLs

---

## Compatibility

### Premiere Pro Versions
- ✅ **Premiere Pro 2020 and later** - Full support
- ✅ **Premiere Pro 2019** - Should work
- ⚠️ **Premiere Pro 2018 and earlier** - May require FCP XML import plugin

### Operating Systems
- ✅ **Windows** - Uses `file:///C:/path` format
- ✅ **macOS** - Uses `file:///absolute/path` format
- ✅ **Linux** - Uses `file:///absolute/path` format

### Video Formats
Tested and working with:
- MP4 (H.264/AAC)
- MOV (H.264/AAC)
- Any format with standard codecs

---

## Migration Guide

### Existing Code (No Changes Needed)
Your existing code will continue to work without modifications:

```python
# This still works with default settings
export_clips_to_premiere(clips, source_video, output_xml)
```

### Optional Enhancements
Take advantage of new features:

```python
# Specify custom resolution for 4K projects
export_clips_to_premiere(
    clips, source_video, output_xml,
    resolution=(3840, 2160)
)

# Use different frame rate
export_clips_to_premiere(
    clips, source_video, output_xml,
    fps=30
)
```

---

## Technical Details

### XML Structure Hierarchy
```
xmeml (version="5")
└── project
    ├── name
    └── children
        └── sequence
            ├── uuid
            ├── name
            ├── duration
            ├── rate
            ├── timecode
            └── media
                ├── video
                │   ├── format
                │   │   └── samplecharacteristics
                │   └── track
                │       └── clipitem (multiple)
                └── audio
                    ├── format
                    │   └── samplecharacteristics
                    └── track
                        └── clipitem (multiple)
```

### File Reference Structure
```xml
<file id="clipfactory-file-000">
  <name>video.mp4</name>
  <pathurl>file:///path/to/video.mp4</pathurl>
  <rate>...</rate>
  <duration>216000</duration>
  <media>
    <video>
      <samplecharacteristics>
        <width>1920</width>
        <height>1080</height>
        <codec>
          <name>H.264</name>
        </codec>
        <rate>...</rate>
      </samplecharacteristics>
    </video>
    <audio>
      <samplecharacteristics>
        <depth>16</depth>
        <samplerate>48000</samplerate>
        <codec>
          <name>AAC</name>
        </codec>
      </samplecharacteristics>
    </audio>
  </media>
</file>
```

---

## Known Issues / Limitations

1. **File Duration**: Currently set to 1 hour default for source files. In production, this should be extracted from the actual video file metadata.

2. **Color Space**: Not specified in current implementation. May need to be added for HDR content.

3. **Pixel Aspect Ratio**: Currently hardcoded to "square". Anamorphic formats may need custom values.

---

## Future Enhancements

Potential improvements for future versions:

1. **Automatic Video Metadata Extraction**
   - Use ffprobe to extract actual duration, resolution, frame rate
   - Auto-detect codec information

2. **Color Space Support**
   - Add color space metadata for HDR/SDR
   - Support for Rec.709, Rec.2020, etc.

3. **Multiple Audio Tracks**
   - Support for multi-channel audio
   - Separate audio track management

4. **Transitions and Effects**
   - Add basic transitions between clips
   - Support for simple effects

5. **Markers and Comments**
   - Add clip markers
   - Include comments from clip analysis

---

## Support

For issues or questions:
1. Check test output: `python3 clipfactory/processing/premiere/test_xml_generator.py`
2. Verify XML structure in generated files
3. Test import in Premiere Pro

---

## Change Log

### Version 2.0 (Current)
- ✅ Upgraded to FCP XML 7.0 format (version 5)
- ✅ Fixed platform-specific file paths
- ✅ Added complete video/audio metadata
- ✅ Made all parameters configurable
- ✅ Added comprehensive validation
- ✅ Removed hardcoded 2-hour duration limit
- ✅ Added proper timecode information
- ✅ Included codec metadata

### Version 1.0 (Previous)
- Used XMEML v4 (year 2000)
- Basic structure only
- Hardcoded parameters
- Limited metadata
