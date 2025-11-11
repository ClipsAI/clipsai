"""
Smoke tests for AdLab viral clip factory.
Tests basic functionality without requiring actual video files or API keys.
"""
import pytest
import os
import tempfile
from pathlib import Path

# AdLab imports
from adlab.config import Config, create_example_config
from adlab.vvsa import VVSAScorer, HookScore
from adlab.variations import VariationGenerator, ClipVariation
from adlab.titles import TitleGenerator
from adlab.captions import CaptionHandler
from adlab.export import ClipExporter
from adlab.manifest import ManifestWriter

# Mock ClipsAI objects for testing
class MockClip:
    """Mock ClipsAI Clip object."""
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time


class MockTranscription:
    """Mock ClipsAI Transcription object."""
    def __init__(self, text="Test transcript", duration=60.0):
        self.text = text
        self.end_time = duration
        self.language = "en"

        # Create mock char_info
        self.char_data = []
        words = text.split()
        time = 0.0

        for word in words:
            for char in word:
                self.char_data.append({
                    "char": char,
                    "start_time": time,
                    "end_time": time + 0.1,
                    "speaker": None,
                })
                time += 0.1

            # Add space
            self.char_data.append({
                "char": " ",
                "start_time": time,
                "end_time": time + 0.05,
                "speaker": None,
            })
            time += 0.05

    def get_char_info(self):
        return self.char_data


class TestConfig:
    """Test configuration management."""

    def test_config_creation(self, tmp_path):
        """Test creating a config object."""
        config = Config()
        assert config is not None

    def test_config_defaults(self):
        """Test default values are set."""
        config = Config()
        assert config.get("vvsa.hook_duration") == 3.0
        assert config.get("processing.target_clips") == 300
        assert config.get("export.video_codec") == "libx264"

    def test_config_get_set(self):
        """Test getting and setting config values."""
        config = Config()
        config.set("test.value", 123)
        assert config.get("test.value") == 123

    def test_example_config_creation(self, tmp_path):
        """Test creating example config file."""
        output_path = str(tmp_path / "test_config.yaml")
        create_example_config(output_path)
        assert os.path.exists(output_path)


class TestVVSAScorer:
    """Test VVSA hook scoring."""

    def test_scorer_creation(self):
        """Test creating a scorer."""
        scorer = VVSAScorer()
        assert scorer is not None
        assert scorer.hook_duration == 3.0

    def test_text_scoring(self):
        """Test text-based hook scoring."""
        scorer = VVSAScorer()

        # High score text (curiosity + question)
        high_score = scorer._score_text_hook(
            "What's the secret to viral videos?"
        )
        assert high_score > 6.0

        # Low score text (too short)
        low_score = scorer._score_text_hook("Hi")
        assert low_score < 5.0

        # Medium score text
        medium_score = scorer._score_text_hook(
            "This is a normal sentence without any hooks"
        )
        assert 4.0 <= medium_score <= 6.0

    def test_extract_hook_text(self):
        """Test extracting hook text from transcription."""
        transcription = MockTranscription("This is a test transcript for hooks")
        scorer = VVSAScorer(hook_duration=2.0)

        hook_text = scorer._extract_hook_text(transcription, 0.0)
        assert len(hook_text) > 0
        assert isinstance(hook_text, str)

    def test_score_clip(self):
        """Test scoring a complete clip."""
        transcription = MockTranscription(
            "What's the secret? This amazing trick will change everything!"
        )
        clip = MockClip(0.0, 30.0)

        scorer = VVSAScorer(llm_client=None)  # No LLM for testing
        hook_score = scorer.score_clip(transcription, start_time=0.0)

        assert isinstance(hook_score, HookScore)
        assert 0 <= hook_score.overall_score <= 10
        assert hook_score.text_score > 0


class TestVariationGenerator:
    """Test clip variation generation."""

    def test_generator_creation(self):
        """Test creating a variation generator."""
        gen = VariationGenerator()
        assert gen is not None
        assert len(gen.durations) > 0
        assert len(gen.aspect_ratios) > 0

    def test_generate_variations(self):
        """Test generating variations for a clip."""
        gen = VariationGenerator(
            temporal_shifts=[-1.0, 0, 1.0],
            durations=[15, 30],
            aspect_ratios=["9:16", "1:1"],
            max_variations=10
        )

        clip = MockClip(30.0, 60.0)
        variations = gen.generate_variations(
            clip=clip,
            base_id="test_clip",
            source_duration=120.0
        )

        assert len(variations) > 0
        assert all(isinstance(v, ClipVariation) for v in variations)
        assert all(0 <= v.adjusted_start <= 120.0 for v in variations)

    def test_aspect_ratio_parsing(self):
        """Test parsing aspect ratio strings."""
        gen = VariationGenerator()

        width, height = gen.parse_aspect_ratio("9:16")
        assert width == 9
        assert height == 16

        width, height = gen.parse_aspect_ratio("16:9")
        assert width == 16
        assert height == 9

    def test_crop_box_calculation(self):
        """Test calculating crop boxes."""
        gen = VariationGenerator()

        # 16:9 source to 9:16 target (portrait)
        crop = gen.calculate_crop_box(1920, 1080, "9:16")
        assert crop["width"] > 0
        assert crop["height"] > 0
        assert crop["x"] >= 0
        assert crop["y"] >= 0


class TestTitleGenerator:
    """Test title and tag generation."""

    def test_generator_creation(self):
        """Test creating a title generator."""
        gen = TitleGenerator(llm_client=None)
        assert gen is not None

    def test_template_titles(self):
        """Test template-based title generation."""
        gen = TitleGenerator(llm_client=None, num_variants=2)

        titles = gen._generate_template_titles(
            transcript="This is an amazing discovery that will change everything",
            duration=30,
            hook_score=7.0
        )

        assert len(titles) == 2
        assert all(t.text for t in titles)
        assert all(len(t.text) <= 60 for t in titles)

    def test_fallback_tags(self):
        """Test fallback tag generation."""
        gen = TitleGenerator(llm_client=None)

        tags = gen._generate_fallback_tags(
            "This amazing discovery will revolutionize everything"
        )

        assert len(tags) > 0
        assert "fyp" in tags or "viral" in tags

    def test_title_quality_scoring(self):
        """Test title quality scoring."""
        gen = TitleGenerator(llm_client=None)

        # Good title
        good_score = gen.score_title_quality(
            "The Secret to 10x Growth Nobody Talks About"
        )
        assert good_score > 5.0

        # Poor title (all caps)
        poor_score = gen.score_title_quality("CLICK HERE NOW!!!")
        assert poor_score < good_score


class TestCaptionHandler:
    """Test caption generation."""

    def test_handler_creation(self):
        """Test creating a caption handler."""
        handler = CaptionHandler()
        assert handler is not None

    def test_extract_words(self):
        """Test extracting words from transcription."""
        transcription = MockTranscription("Hello world this is a test")
        handler = CaptionHandler()

        words = handler._extract_words_in_range(transcription, 0.0, 10.0)
        assert len(words) > 0
        assert all("text" in w for w in words)

    def test_srt_generation(self, tmp_path):
        """Test SRT file generation."""
        transcription = MockTranscription("Test transcript with words")
        handler = CaptionHandler()

        output_path = str(tmp_path / "test.srt")
        result = handler.generate_srt(transcription, 0.0, 5.0, output_path)

        assert os.path.exists(result)
        with open(result, 'r') as f:
            content = f.read()
            assert len(content) > 0

    def test_time_formatting(self):
        """Test SRT and VTT time formatting."""
        handler = CaptionHandler()

        srt_time = handler._format_srt_time(65.5)
        assert "01:05" in srt_time

        vtt_time = handler._format_vtt_time(65.5)
        assert "01:05" in vtt_time


class TestClipExporter:
    """Test clip export functionality."""

    def test_exporter_creation(self):
        """Test creating a clip exporter."""
        exporter = ClipExporter()
        assert exporter is not None
        assert exporter.video_codec == "libx264"

    def test_aspect_crop_calculation(self):
        """Test aspect ratio crop calculation."""
        exporter = ClipExporter()

        crop = exporter.calculate_aspect_crop(1920, 1080, "9:16")
        assert crop["width"] > 0
        assert crop["height"] > 0


class TestManifestWriter:
    """Test manifest writing and management."""

    def test_writer_creation(self, tmp_path):
        """Test creating a manifest writer."""
        writer = ManifestWriter(str(tmp_path))
        assert writer is not None

    def test_write_manifest(self, tmp_path):
        """Test writing a manifest file."""
        writer = ManifestWriter(str(tmp_path))

        clips_data = [
            {
                "clip_id": "test_1",
                "video_path": "test.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
                "hook_score": {"overall": 7.5}
            },
            {
                "clip_id": "test_2",
                "video_path": "test.mp4",
                "start_time": 30.0,
                "end_time": 60.0,
                "hook_score": {"overall": 8.2}
            }
        ]

        manifest_path = str(tmp_path / "test_manifest.jsonl")
        result = writer.write_manifest(clips_data, manifest_path)

        assert os.path.exists(result)

        # Read back
        loaded = writer.read_manifest(result)
        assert len(loaded) == 2

    def test_deduplication(self, tmp_path):
        """Test clip deduplication."""
        writer = ManifestWriter(str(tmp_path))

        clips_data = [
            {
                "clip_id": "test_1",
                "video_path": "test.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            },
            {
                "clip_id": "test_1",
                "video_path": "test.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            },  # Duplicate
        ]

        unique = writer.deduplicate_clips(clips_data)
        assert len(unique) == 1

    def test_summary_generation(self, tmp_path):
        """Test generating summary statistics."""
        writer = ManifestWriter(str(tmp_path))

        clips_data = [
            {
                "clip_id": f"test_{i}",
                "duration": 30,
                "aspect_ratio": "9:16",
                "hook_score": {"overall": 5.0 + i}
            }
            for i in range(10)
        ]

        manifest_path = str(tmp_path / "test_manifest.jsonl")
        writer.write_manifest(clips_data, manifest_path)

        summary = writer.generate_summary(manifest_path)
        assert summary["total_clips"] == 10
        assert summary["hook_score_avg"] > 0


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_end_to_end_dry_run(self, tmp_path):
        """Test complete workflow without actual video processing."""
        # Create config
        config = Config()
        config.set("export.output_dir", str(tmp_path))

        # Create mock data
        transcription = MockTranscription(
            "This is a secret hack that will revolutionize everything! "
            "You won't believe what happens next in this amazing discovery."
        )

        clip = MockClip(0.0, 60.0)

        # Score clip
        scorer = VVSAScorer(llm_client=None)
        hook_score = scorer.score_clip(transcription, start_time=0.0)
        assert hook_score.overall_score > 0

        # Generate variations
        gen = VariationGenerator(max_variations=5)
        variations = gen.generate_variations(
            clip=clip,
            base_id="test_clip",
            source_duration=120.0
        )
        assert len(variations) > 0

        # Generate titles
        title_gen = TitleGenerator(llm_client=None)
        titles = title_gen._generate_template_titles(
            transcript=transcription.text,
            duration=30,
            hook_score=hook_score.overall_score
        )
        assert len(titles) > 0

        # Generate captions
        caption_handler = CaptionHandler()
        caption_paths = caption_handler.create_caption_bundle(
            transcription=transcription,
            start_time=0.0,
            end_time=30.0,
            output_dir=str(tmp_path / "captions"),
            base_name="test_clip"
        )
        assert "srt" in caption_paths
        assert os.path.exists(caption_paths["srt"])

        # Write manifest
        manifest_writer = ManifestWriter(str(tmp_path))
        manifest_entry = manifest_writer.create_clip_entry(
            variation=variations[0],
            video_path="test.mp4",
            source_video="source.mp4",
            hook_score=hook_score,
            title_variants=titles,
            captions=caption_paths,
            thumbnail_path=None
        )

        manifest_path = str(tmp_path / "manifest.jsonl")
        manifest_writer.write_manifest([manifest_entry], manifest_path)

        assert os.path.exists(manifest_path)


# Run tests with: pytest tests/test_adlab_smoke.py -v
