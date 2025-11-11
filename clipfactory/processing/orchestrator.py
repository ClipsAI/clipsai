"""
Complete Pipeline Orchestrator
Runs all 5 phases sequentially
"""
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import json

# Import all processors
from .premiere.xml_generator import export_clips_to_premiere
from .matrix.canvas_renderer import render_all_styles
from .matrix.face_tracker import FaceTracker
from .variations.temporal_generator import create_variations_for_clip
from .ai.title_generator import TitleGenerator, CaptionFormatter

logger = logging.getLogger(__name__)


class ClipFactoryOrchestrator:
    """
    Orchestrates the complete clip factory pipeline.

    Phases:
    1. Council Deliberation (import from existing adlab)
    2. Premiere Integration (XML export + re-upload)
    3. Matrix Processing (face tracking + canvas)
    4. Variation Generation (temporal + music + captions)
    5. Distribution (posting helper)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize orchestrator with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.title_generator = TitleGenerator(
            anthropic_key=config.get('anthropic_api_key'),
            openai_key=config.get('openai_api_key')
        )
        self.caption_formatter = CaptionFormatter()
        self.face_tracker = FaceTracker()

    async def run_complete_pipeline(
        self,
        video_path: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from video upload to distribution.

        Args:
            video_path: Path to uploaded video
            output_dir: Output directory for all files

        Returns:
            Dict with pipeline results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "video_path": video_path,
            "output_dir": str(output_dir),
            "phases": {}
        }

        try:
            # PHASE 1: Council Deliberation
            logger.info("Phase 1: Council Deliberation")
            clips = await self.phase1_council_deliberation(video_path)
            results["phases"]["phase1"] = {
                "status": "complete",
                "clips_found": len(clips)
            }

            # PHASE 2: Premiere Integration
            logger.info("Phase 2: Premiere Integration")
            xml_path = await self.phase2_premiere_export(
                clips,
                video_path,
                output_dir / "premiere"
            )
            results["phases"]["phase2"] = {
                "status": "complete",
                "xml_path": str(xml_path)
            }

            # Wait for Wes to edit and re-upload
            # This is a manual step - orchestrator pauses here
            logger.info("Waiting for edited clips re-upload...")

            # PHASE 3: Matrix Processing
            logger.info("Phase 3: Matrix Processing")
            processed_clips = await self.phase3_matrix_processing(
                clips,
                video_path,
                output_dir / "matrix"
            )
            results["phases"]["phase3"] = {
                "status": "complete",
                "processed_clips": len(processed_clips)
            }

            # PHASE 4: Variation Generation
            logger.info("Phase 4: Variation Generation")
            variations = await self.phase4_generate_variations(
                processed_clips,
                output_dir / "variations"
            )
            results["phases"]["phase4"] = {
                "status": "complete",
                "variations_generated": len(variations)
            }

            # PHASE 5: Distribution Setup
            logger.info("Phase 5: Distribution Setup")
            distribution = await self.phase5_distribution_setup(
                variations,
                output_dir / "distribution"
            )
            results["phases"]["phase5"] = {
                "status": "complete",
                "distribution": distribution
            }

            # Write final manifest
            manifest_path = output_dir / "pipeline_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Pipeline complete! Manifest: {manifest_path}")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results["error"] = str(e)
            raise

    async def phase1_council_deliberation(
        self,
        video_path: str
    ) -> List[Dict[str, Any]]:
        """
        Phase 1: Run council deliberation.

        Integrates with adlab council voting system:
        1. Transcribe video
        2. Find candidate clips (TextTiling)
        3. VVSA + Council scoring
        4. Select top 500 clips

        Returns
        -------
        List[Dict[str, Any]]
            Selected clips with scores
        """
        logger.info("Running council deliberation...")

        try:
            # Import adlab components
            from clipsai import ClipFinder, Transcriber
            from adlab.config import Config
            from adlab.vvsa import create_hybrid_scorer

            # Load config
            config = Config()

            # Step 1: Transcribe video
            logger.info("Step 1: Transcribing video...")
            transcriber = Transcriber()
            transcription = transcriber.transcribe(
                audio_file_path=video_path,
                model="large-v3"
            )
            logger.info(f"  Transcription complete: {len(transcription.words)} words")

            # Step 2: Find candidate clips using TextTiling
            logger.info("Step 2: Finding candidate clips...")
            clip_finder = ClipFinder()
            base_clips = clip_finder.find_clips(
                transcription=transcription,
                min_clips=config.get("processing.target_clips", 300),
                max_clips=1000,  # Find many candidates for council to vote on
                min_clip_duration=config.get("processing.min_clip_duration", 10),
                max_clip_duration=config.get("processing.max_clip_duration", 90)
            )
            logger.info(f"  Found {len(base_clips)} candidate clips")

            # Step 3: Extract hook text for each clip
            logger.info("Step 3: Extracting hook transcripts...")
            clips_with_hooks = []
            for clip in base_clips:
                # Extract first 3 seconds of text
                hook_text = self._extract_hook_text(transcription, clip.start_time, duration=3.0)
                clips_with_hooks.append((clip, hook_text))

            # Step 4: VVSA + Council scoring
            logger.info("Step 4: Council voting (VVSA + Multi-model consensus)...")
            hybrid_scorer = create_hybrid_scorer(config)

            selected_clips = hybrid_scorer.score_and_vote(
                clips=clips_with_hooks,
                transcription=transcription,
                vvsa_threshold=config.get("vvsa.min_score", 6.0),
                council_top_n=config.get("processing.max_clips", 500)
            )

            logger.info(f"  Council selected {len(selected_clips)} clips")

            # Step 5: Convert to output format
            result_clips = []
            for i, item in enumerate(selected_clips):
                # Handle both formats (with or without council vote)
                if len(item) == 3:
                    clip, vvsa_score, council_vote = item
                    consensus_score = council_vote.consensus_score if council_vote else vvsa_score.overall_score
                else:
                    clip, hook_text, vvsa_score = item
                    consensus_score = vvsa_score.overall_score

                result_clips.append({
                    "clip_id": f"clip_{i:03d}",
                    "start_time": clip.start_time,
                    "end_time": clip.end_time,
                    "duration": clip.end_time - clip.start_time,
                    "transcript": transcription.get_text(
                        start_time=clip.start_time,
                        end_time=clip.end_time
                    ),
                    "hook_score": consensus_score,
                    "vvsa_score": vvsa_score.overall_score if hasattr(vvsa_score, 'overall_score') else vvsa_score,
                    "council_consensus": consensus_score
                })

            logger.info(f"Council deliberation complete: {len(result_clips)} clips selected")
            if result_clips:
                logger.info(f"  Score range: {result_clips[0]['hook_score']:.2f} - {result_clips[-1]['hook_score']:.2f}")

            return result_clips

        except Exception as e:
            logger.error(f"Council deliberation failed: {e}")
            logger.exception(e)
            # Return empty list on failure
            return []

    def _extract_hook_text(self, transcription, start_time: float, duration: float = 3.0) -> str:
        """
        Extract transcript text for the hook period (first N seconds).

        Parameters
        ----------
        transcription : Transcription
            Full transcription
        start_time : float
            Clip start time
        duration : float
            Hook duration in seconds

        Returns
        -------
        str
            Hook transcript text
        """
        try:
            hook_end = start_time + duration
            words = []

            char_info = transcription.get_char_info()
            current_word = []

            for char_data in char_info:
                char_start = char_data.get("start_time")
                char_text = char_data.get("char", "")

                if char_start is None:
                    current_word.append(char_text)
                    continue

                if char_start < start_time:
                    continue
                if char_start >= hook_end:
                    break

                if char_text == " ":
                    if current_word:
                        words.append("".join(current_word))
                        current_word = []
                else:
                    current_word.append(char_text)

            if current_word:
                words.append("".join(current_word))

            return " ".join(words)

        except Exception as e:
            logger.warning(f"Could not extract hook text: {e}")
            # Fallback: use first 50 chars of full transcript
            try:
                full_text = transcription.get_text(start_time=start_time, end_time=start_time + duration)
                return full_text[:50]
            except:
                return ""

    async def phase2_premiere_export(
        self,
        clips: List[Dict[str, Any]],
        source_video: str,
        output_dir: Path
    ) -> Path:
        """
        Phase 2: Export clips to Premiere XML.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        xml_path = output_dir / "clips_export.xml"

        export_clips_to_premiere(clips, source_video, str(xml_path))

        logger.info(f"Premiere XML exported: {xml_path}")
        return xml_path

    async def phase3_matrix_processing(
        self,
        clips: List[Dict[str, Any]],
        source_video: str,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: Matrix processing - face tracking + canvas rendering.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_clips = []

        for clip in clips:
            clip_id = clip["clip_id"]

            # TODO: Extract clip from source video first
            # clip_path = await self.extract_clip(source_video, clip)

            # For now, assume clip exists
            clip_path = f"/tmp/{clip_id}.mp4"  # Placeholder

            # Face tracking
            tracking_data = self.face_tracker.track_faces_in_video(clip_path)

            # Canvas rendering (3 styles)
            rendered = render_all_styles(
                clip_path,
                str(output_dir),
                clip_id
            )

            processed_clips.append({
                **clip,
                "rendered_paths": rendered,
                "tracking_data": len(tracking_data)
            })

        return processed_clips

    async def phase4_generate_variations(
        self,
        clips: List[Dict[str, Any]],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Phase 4: Generate all variations.

        For each clip:
        - Create 3 temporal variations
        - Apply 3 reframe styles
        - Generate titles
        - Generate captions
        - Apply music (selection done via UI)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_variations = []

        for clip in clips:
            # Get temporal variations
            variations = create_variations_for_clip(
                clip["clip_id"],
                clip["start_time"],
                clip["end_time"],
                video_duration=7200.0,  # 2 hours
                reframe_styles=['original', 'flipped', 'blurry_bg']
            )

            # Generate titles for each
            for variation in variations:
                # Generate title variants
                title_variants = await self.title_generator.generate_from_transcript(
                    clip["transcript"],
                    clip["hook_score"],
                    variation["duration"],
                    num_variants=5
                )

                # Format captions
                caption = self.caption_formatter.format_caption(
                    clip["transcript"][:50]
                )

                variation["title_variants"] = title_variants
                variation["caption"] = caption

                all_variations.append(variation)

        logger.info(f"Generated {len(all_variations)} total variations")
        return all_variations

    async def phase5_distribution_setup(
        self,
        variations: List[Dict[str, Any]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Phase 5: Set up distribution.

        - Organize variations by account
        - Create posting schedule
        - Generate calendar events
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        distribution = {
            "total_variations": len(variations),
            "accounts": [],
            "schedule": []
        }

        # TODO: Organize by account
        # TODO: Create calendar events
        # TODO: Generate posting helper data

        return distribution


async def main():
    """Example usage."""
    config = {
        "anthropic_api_key": "sk-ant-...",
        "openai_api_key": "sk-...",
    }

    orchestrator = ClipFactoryOrchestrator(config)

    results = await orchestrator.run_complete_pipeline(
        video_path="/path/to/video.mp4",
        output_dir="./output"
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
