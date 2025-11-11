# AdLab: Viral Clip Factory

A smart extension for ClipsAI that automatically generates 300-500+ optimized short-form clips from long videos using VVSA-style hook scoring.

## 🎯 What It Does

Given a long video (up to 2-3 hours), AdLab will:

1. **Find candidate moments** using ClipsAI's WhisperX + TextTiling
2. **Score each hook** (first 3 seconds) using VVSA methodology (0-10)
3. **Generate variations** with different:
   - Start times (temporal shifts)
   - Durations (15s, 30s, 45s, 60s)
   - Aspect ratios (9:16, 1:1, 4:5)
4. **Create titles** with A/B testing variants
5. **Export everything**:
   - 300-500 MP4 clips
   - Captions (SRT/VTT)
   - Thumbnails
   - JSONL manifest with all metadata

## 🚀 Quick Start

### Prerequisites

```bash
# 1. Install ffmpeg (required for video processing)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# 2. Get an Anthropic API key
# Visit: https://console.anthropic.com/
```

### Installation

```bash
# Install ClipsAI (if not already installed)
pip install -e .

# Install AdLab dependencies
pip install -r requirements.adlab.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Create Config

```bash
# Generate example config
python -m adlab.run init-config

# Edit config.yaml with your settings
nano config.yaml
```

### Process a Video

```bash
# Process a single video
python -m adlab.run process video.mp4

# With custom config
python -m adlab.run process video.mp4 --config config.yaml

# Specify output directory
python -m adlab.run process video.mp4 --output ./my_clips

# Target specific number of clips
python -m adlab.run process video.mp4 --target 500 --max 600

# Dry run (analyze only, don't export)
python -m adlab.run process video.mp4 --dry-run
```

### Batch Processing

```bash
# Process multiple videos
python -m adlab.run batch "videos/*.mp4"

# With custom output
python -m adlab.run batch "videos/*.mp4" --output ./all_clips
```

## 📊 Output Structure

```
output/
├── videos/
│   ├── clip_0001_t-1.0_d30_9x16.mp4
│   ├── clip_0001_t0.0_d30_9x16.mp4
│   ├── clip_0001_t0.5_d30_9x16.mp4
│   └── ...
├── thumbnails/
│   ├── clip_0001_t-1.0_d30_9x16.jpg
│   └── ...
├── captions/
│   ├── clip_0001_t-1.0_d30_9x16.srt
│   ├── clip_0001_t-1.0_d30_9x16.vtt
│   └── ...
├── manifest.jsonl         # Complete metadata
├── manifest.summary.json  # Statistics
└── manifest.csv           # Spreadsheet format
```

## 🎬 5-Minute Smoke Test

Test the system with a short video:

```bash
# 1. Download a test video (or use your own 2-5 minute clip)
wget https://example.com/test_video.mp4 -O test.mp4

# 2. Run AdLab
python -m adlab.run process test.mp4 --target 20 --max 30

# 3. Check output
ls -lh output/videos/
cat output/manifest.summary.json
```

Expected results:
- 20-30 video clips
- Processing time: 5-10 minutes
- Hook scores in manifest

## ⚙️ Configuration

### Key Settings

```yaml
# config.yaml

anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  model: claude-3-5-sonnet-20241022

vvsa:
  hook_duration: 3.0        # Analyze first 3 seconds
  min_score: 6.0            # Filter clips below this score
  weights:
    visual: 0.3             # Visual analysis weight
    audio: 0.2              # Audio analysis weight
    text: 0.3               # Text/transcript weight
    llm: 0.2                # LLM analysis weight

variations:
  temporal_shifts: [-1.0, -0.5, 0, 0.5, 1.0]  # Time adjustments (seconds)
  durations: [15, 30, 45, 60]                  # Clip lengths
  aspect_ratios: ["9:16", "1:1", "4:5"]        # TikTok, Instagram, etc.
  max_variations_per_clip: 12

processing:
  min_clip_duration: 10     # Minimum clip length
  max_clip_duration: 90     # Maximum clip length
  target_clips: 300         # Goal number of clips
  max_clips: 500            # Hard limit

export:
  output_dir: ./output
  video_codec: libx264      # H.264
  audio_codec: aac
  preset: medium            # Encoding speed: ultrafast, fast, medium, slow
  crf: 23                   # Quality: 0-51, lower = better
```

## 🧠 VVSA Hook Scoring

AdLab evaluates the first 3 seconds of each clip using **VVSA** (Viral Video Success Analysis):

### Scoring Components (0-10 each)

1. **Text Score** (heuristic)
   - Curiosity keywords ("secret", "hack", "revealed")
   - Questions
   - Strong action verbs
   - Emotional triggers
   - Optimal length (5-12 words)

2. **Audio Score** (heuristic)
   - Pacing/word density
   - Energy markers (exclamation points)
   - Engagement cues (questions)

3. **Visual Score** (placeholder)
   - Currently returns neutral 5.0
   - Future: scene detection, motion, faces

4. **LLM Score** (Claude)
   - Pattern interrupt
   - Curiosity gap
   - Emotional resonance
   - Clarity of promise

### Overall Score

Weighted average of all components:
```
overall = (0.3 × visual) + (0.2 × audio) + (0.3 × text) + (0.2 × llm)
```

Clips below `min_score` (default: 6.0) are filtered out.

## 📈 Manifest Format

Each clip entry in `manifest.jsonl`:

```json
{
  "clip_id": "clip_0001",
  "variation_id": "clip_0001_t-0.5_d30_9x16",
  "video_path": "output/videos/clip_0001_t-0.5_d30_9x16.mp4",
  "source_video": "input_video.mp4",
  "thumbnail_path": "output/thumbnails/clip_0001_t-0.5_d30_9x16.jpg",
  "start_time": 45.5,
  "end_time": 75.5,
  "duration": 30,
  "temporal_shift": -0.5,
  "aspect_ratio": "9:16",
  "crop_strategy": "center_portrait",
  "hook_score": {
    "overall": 7.8,
    "visual": 5.0,
    "audio": 6.5,
    "text": 8.2,
    "llm": 9.1,
    "reasoning": "Strong curiosity gap with clear value proposition",
    "strengths": ["Unexpected opening", "Clear promise"],
    "improvements": ["Could be more concise"]
  },
  "titles": [
    {
      "variant": "A",
      "text": "This Changed Everything...",
      "hook_style": "curiosity",
      "target_audience": "general",
      "predicted_ctr": 7.5,
      "tags": ["viral", "trending", "fyp", "mustwatch"]
    },
    {
      "variant": "B",
      "text": "The Secret Nobody Tells You About...",
      "hook_style": "revelation",
      "target_audience": "engaged",
      "predicted_ctr": 8.2,
      "tags": ["secret", "revealed", "truth", "fyp"]
    }
  ],
  "captions": {
    "srt": "output/captions/clip_0001_t-0.5_d30_9x16.srt",
    "vtt": "output/captions/clip_0001_t-0.5_d30_9x16.vtt"
  },
  "metadata": {
    "transcript": "Full text of what was said in this clip..."
  }
}
```

## 🔧 Architecture

### Design Principles

1. **Minimal Dependencies**: <20 new packages (actually ~5)
2. **No Docker Required**: Runs directly via Python
3. **GPU Optional**: CPU fallback for all operations
4. **Preserve ClipsAI**: Extension, not replacement
5. **Direct FFmpeg**: No heavy video wrappers

### Components

```
adlab/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── llm.py               # Anthropic Claude wrapper
├── vvsa.py              # Hook scoring system
├── variations.py        # Temporal/spatial/aspect strategies
├── titles.py            # Title + tag generation
├── captions.py          # SRT/VTT export
├── export.py            # FFmpeg rendering
├── manifest.py          # JSONL writer + deduplication
└── run.py               # Typer CLI + orchestration
```

### Integration with ClipsAI

```python
from clipsai import Transcriber, ClipFinder, Transcription, Clip

# AdLab uses these existing APIs:
transcriber = Transcriber()          # WhisperX transcription
transcription = transcriber.transcribe(video_path)

clip_finder = ClipFinder()           # TextTiling segmentation
clips = clip_finder.find_clips(transcription)

# Then extends with:
from adlab import VVSAScorer, VariationGenerator, ClipExporter

scorer = VVSAScorer()                # Hook scoring
variations = generator.generate()     # Create variations
exporter.export_clip()                # Render with ffmpeg
```

## 🎯 Use Cases

### Content Creator Workflow

1. **Record** a 2-hour livestream
2. **Process** with AdLab: `python -m adlab.run process stream.mp4`
3. **Review** manifest.csv in spreadsheet
4. **Select** top 50 clips by hook score
5. **Schedule** posts across TikTok, Reels, Shorts
6. **A/B test** title variants
7. **Track** which perform best

### Agency/Editor Workflow

1. **Batch process** all client videos
2. **Filter** by minimum score (7.0+)
3. **Export** to Premiere Pro (future feature)
4. **Add** brand watermarks
5. **Deliver** to client

### Researcher Workflow

1. **Analyze** what makes hooks effective
2. **Export** manifest data
3. **Correlate** scores with performance
4. **Refine** VVSA weights

## 🐛 Troubleshooting

### FFmpeg Not Found

```bash
# Install ffmpeg
brew install ffmpeg  # macOS
sudo apt-get install ffmpeg  # Ubuntu
```

### API Rate Limits

If you hit Anthropic rate limits, reduce batch size:
```yaml
processing:
  target_clips: 100  # Process fewer clips
```

Or disable LLM scoring:
```yaml
vvsa:
  weights:
    llm: 0  # Turn off LLM scoring
```

### Out of Memory

For very long videos (3+ hours), process in chunks or reduce batch size.

### Slow Processing

- Use GPU for WhisperX: `device: cuda`
- Reduce video quality: `crf: 28` (faster encoding)
- Use faster preset: `preset: fast`

## 📝 Development

### Running Tests

```bash
# Run smoke test
pytest tests/test_adlab_smoke.py -v

# Test specific component
pytest tests/test_adlab_smoke.py::test_vvsa_scoring -v
```

### Adding New Features

1. **New variation strategy**: Edit `adlab/variations.py`
2. **Custom hook scoring**: Edit `adlab/vvsa.py`
3. **Different LLM**: Edit `adlab/llm.py`
4. **New export format**: Edit `adlab/export.py`

## 🤝 Contributing

AdLab is built as an extension to ClipsAI. Keep these principles:

1. **Don't modify ClipsAI core**: Import, don't rewrite
2. **Keep dependencies minimal**: <20 total
3. **Direct FFmpeg preferred**: Avoid heavy wrappers
4. **Config over hardcoding**: Use config.yaml

## 📚 Resources

- **ClipsAI**: https://github.com/ClipsAI/clipsai
- **VVSA Methodology**: Hook scoring for viral videos
- **Anthropic Claude**: https://www.anthropic.com/
- **FFmpeg**: https://ffmpeg.org/

## 📄 License

MIT License - Same as ClipsAI

## 🙏 Credits

- Built on top of [ClipsAI](https://github.com/ClipsAI/clipsai)
- Uses [WhisperX](https://github.com/m-bain/whisperX) for transcription
- Powered by [Anthropic Claude](https://www.anthropic.com/)

---

**Happy clipping! 🎬**

For support, open an issue on GitHub.
