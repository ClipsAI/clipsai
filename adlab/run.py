"""
Main CLI orchestration for AdLab viral clip factory.

Usage:
    python -m adlab.run process video.mp4 --config config.yaml
    python -m adlab.run batch videos/*.mp4 --output ./output
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.logging import RichHandler

# ClipsAI imports
from clipsai import Transcriber, ClipFinder

# AdLab imports
from .config import Config
from .vvsa import create_scorer
from .variations import create_variation_generator
from .titles import create_title_generator
from .captions import create_caption_handler
from .export import create_exporter
from .manifest import create_manifest_writer

# Setup
app = typer.Typer(help="AdLab Viral Clip Factory CLI")
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("adlab")


@app.command()
def process(
    video_path: str = typer.Argument(..., help="Path to input video file"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    target_clips: int = typer.Option(300, "--target", "-t", help="Target number of clips"),
    max_clips: int = typer.Option(500, "--max", "-m", help="Maximum number of clips"),
    min_score: float = typer.Option(6.0, "--min-score", help="Minimum VVSA score threshold"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze only, don't export videos"),
):
    """
    Process a single video and generate viral clips.

    This command:
    1. Transcribes the video using WhisperX
    2. Finds candidate clips using TextTiling
    3. Scores each clip's hook using VVSA
    4. Generates variations (temporal/duration/aspect)
    5. Creates titles and captions
    6. Exports video files and manifest
    """
    console.print(f"\n[bold blue]AdLab Viral Clip Factory[/bold blue]")
    console.print(f"Processing: {video_path}\n")

    # Load config
    config = Config(config_path)
    if output_dir:
        config.set("export.output_dir", output_dir)
    config.set("processing.target_clips", target_clips)
    config.set("processing.max_clips", max_clips)

    try:
        config.validate()
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)

    # Validate input
    if not os.path.exists(video_path):
        console.print(f"[red]Video file not found: {video_path}[/red]")
        raise typer.Exit(1)

    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing components...", total=None)

        transcriber = Transcriber(
            model_size=config.get("transcription.model_size"),
            device=config.get("transcription.device"),
        )

        clip_finder = ClipFinder(
            min_clip_duration=config.get("processing.min_clip_duration"),
            max_clip_duration=config.get("processing.max_clip_duration"),
        )

        scorer = create_scorer(config)
        variation_gen = create_variation_generator(config)
        title_gen = create_title_generator(config)
        caption_handler = create_caption_handler(config)
        exporter = create_exporter(config)
        manifest_writer = create_manifest_writer(config)

        progress.update(task, description="[green]Components initialized")

    # Step 1: Transcription
    console.print("\n[bold]Step 1: Transcription[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Transcribing with WhisperX...", total=None)
        transcription = transcriber.transcribe(
            video_path,
            iso6391_lang_code=config.get("transcription.language")
        )
        progress.update(task, description=f"[green]Transcription complete ({transcription.end_time:.1f}s)")

    console.print(f"  Duration: {transcription.end_time:.1f}s")
    console.print(f"  Language: {transcription.language}")

    # Step 2: Find clips
    console.print("\n[bold]Step 2: Finding clips[/bold]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Finding clips with TextTiling...", total=None)
        base_clips = clip_finder.find_clips(transcription)
        progress.update(task, description=f"[green]Found {len(base_clips)} base clips")

    console.print(f"  Found: {len(base_clips)} base clips")

    # Step 3: Score clips
    console.print("\n[bold]Step 3: Hook scoring (VVSA)[/bold]")
    scored_clips = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Scoring hooks...", total=len(base_clips))

        for i, clip in enumerate(base_clips):
            hook_score = scorer.score_clip(
                transcription=transcription,
                video_path=video_path,
                start_time=clip.start_time
            )
            scored_clips.append((clip, hook_score))
            progress.update(task, advance=1)

    # Filter by minimum score
    scored_clips = [(c, s) for c, s in scored_clips if s.overall_score >= min_score]
    console.print(f"  Scored: {len(scored_clips)} clips passed threshold (>= {min_score})")

    if not scored_clips:
        console.print("[red]No clips passed the score threshold![/red]")
        raise typer.Exit(1)

    # Sort by score
    scored_clips = sorted(scored_clips, key=lambda x: x[1].overall_score, reverse=True)

    # Step 4: Generate variations
    console.print("\n[bold]Step 4: Generating variations[/bold]")
    all_variations = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Creating variations...", total=len(scored_clips))

        for i, (clip, hook_score) in enumerate(scored_clips):
            base_id = f"clip_{i:04d}"

            variations = variation_gen.generate_smart_variations(
                clip=clip,
                base_id=base_id,
                source_duration=transcription.end_time,
                hook_score=hook_score.overall_score
            )

            for var in variations:
                all_variations.append((var, clip, hook_score))

            progress.update(task, advance=1)

    console.print(f"  Generated: {len(all_variations)} total variations")

    # Optimize to target count
    if len(all_variations) > max_clips:
        console.print(f"  Optimizing to {max_clips} variations...")
        # Sort by hook score and take top N
        all_variations = sorted(
            all_variations,
            key=lambda x: x[2].overall_score,
            reverse=True
        )[:max_clips]

    console.print(f"  Final count: {len(all_variations)} variations")

    # Step 5: Generate titles and captions
    console.print("\n[bold]Step 5: Titles and captions[/bold]")
    clip_metadata = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating titles...", total=len(all_variations))

        output_dir = config.get("export.output_dir")

        for variation, clip, hook_score in all_variations:
            # Extract transcript for this variation
            transcript_text = _extract_transcript_text(
                transcription,
                variation.adjusted_start,
                variation.adjusted_end
            )

            # Generate titles
            title_variants = title_gen.generate_titles(
                transcript=transcript_text,
                duration=int(variation.duration),
                hook_score=hook_score.overall_score
            )

            # Generate captions
            caption_paths = caption_handler.create_caption_bundle(
                transcription=transcription,
                start_time=variation.adjusted_start,
                end_time=variation.adjusted_end,
                output_dir=os.path.join(output_dir, "captions"),
                base_name=variation.variation_id
            )

            clip_metadata.append({
                "variation": variation,
                "clip": clip,
                "hook_score": hook_score,
                "title_variants": title_variants,
                "caption_paths": caption_paths,
                "transcript": transcript_text,
            })

            progress.update(task, advance=1)

    console.print(f"  Generated titles and captions for {len(clip_metadata)} clips")

    # Step 6: Export videos
    if dry_run:
        console.print("\n[yellow]Dry run mode: Skipping video export[/yellow]")
    else:
        console.print("\n[bold]Step 6: Exporting videos[/bold]")

        # Get video info
        video_info = exporter.get_video_info(video_path)
        console.print(f"  Source: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps")

        output_dir = config.get("export.output_dir")
        os.makedirs(output_dir, exist_ok=True)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting clips...", total=len(clip_metadata))

            for meta in clip_metadata:
                variation = meta["variation"]
                caption_paths = meta["caption_paths"]

                # Create export task
                export_task = exporter.create_export_task(
                    variation=variation,
                    source_path=video_path,
                    output_dir=os.path.join(output_dir, "videos"),
                    source_info=video_info,
                    subtitle_path=caption_paths.get("srt")
                )

                # Export video
                try:
                    video_output = exporter.export_clip(**export_task)
                    meta["video_path"] = video_output

                    # Generate thumbnail
                    thumbnail_path = os.path.join(
                        output_dir,
                        "thumbnails",
                        f"{variation.variation_id}.jpg"
                    )
                    exporter.generate_thumbnail(
                        video_path=video_output,
                        output_path=thumbnail_path,
                        timestamp=1.0
                    )
                    meta["thumbnail_path"] = thumbnail_path

                except Exception as e:
                    logger.error(f"Export failed for {variation.variation_id}: {e}")
                    meta["video_path"] = None
                    meta["thumbnail_path"] = None

                progress.update(task, advance=1)

    # Step 7: Write manifest
    console.print("\n[bold]Step 7: Writing manifest[/bold]")

    manifest_entries = []
    for meta in clip_metadata:
        if dry_run or meta.get("video_path"):
            entry = manifest_writer.create_clip_entry(
                variation=meta["variation"],
                video_path=meta.get("video_path", ""),
                source_video=video_path,
                hook_score=meta["hook_score"],
                title_variants=meta["title_variants"],
                captions=meta["caption_paths"],
                thumbnail_path=meta.get("thumbnail_path"),
                metadata={"transcript": meta["transcript"]}
            )
            manifest_entries.append(entry)

    output_dir = config.get("export.output_dir")
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    manifest_writer.write_manifest(manifest_entries, manifest_path)

    # Write summary
    summary_path = manifest_writer.write_summary(manifest_path)

    # Export CSV
    csv_path = manifest_writer.export_csv(manifest_path)

    console.print(f"\n[bold green]Success![/bold green]")
    console.print(f"  Generated: {len(manifest_entries)} clips")
    console.print(f"  Manifest: {manifest_path}")
    console.print(f"  Summary: {summary_path}")
    console.print(f"  CSV: {csv_path}")

    if not dry_run:
        console.print(f"  Videos: {os.path.join(output_dir, 'videos')}")
        console.print(f"  Thumbnails: {os.path.join(output_dir, 'thumbnails')}")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Review manifest.jsonl for clip metadata")
    console.print("  2. Upload clips to your platform")
    console.print("  3. Test A/B title variants")
    console.print("  4. Track performance and iterate!")


@app.command()
def batch(
    pattern: str = typer.Argument(..., help="Glob pattern for video files (e.g., 'videos/*.mp4')"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    output_dir: Optional[str] = typer.Option("./output", "--output", "-o", help="Output directory"),
):
    """
    Process multiple videos in batch mode.
    """
    console.print("[bold blue]AdLab Batch Processing[/bold blue]\n")

    # Find videos
    import glob
    videos = glob.glob(pattern)

    if not videos:
        console.print(f"[red]No videos found matching: {pattern}[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(videos)} videos to process\n")

    for i, video in enumerate(videos, 1):
        console.print(f"\n[bold]Processing {i}/{len(videos)}: {video}[/bold]")

        # Create output subdir for each video
        video_name = Path(video).stem
        video_output = os.path.join(output_dir, video_name)

        try:
            # Process video
            ctx = typer.Context(process)
            ctx.invoke(
                process,
                video_path=video,
                config_path=config_path,
                output_dir=video_output,
            )
        except Exception as e:
            console.print(f"[red]Error processing {video}: {e}[/red]")
            continue

    console.print(f"\n[bold green]Batch complete![/bold green]")
    console.print(f"Processed {len(videos)} videos")


@app.command()
def init_config(
    output_path: str = typer.Option("config.yaml", "--output", "-o", help="Output path"),
):
    """
    Initialize a new config.yaml file with default settings.
    """
    from .config import create_example_config

    if os.path.exists(output_path):
        if not typer.confirm(f"{output_path} already exists. Overwrite?"):
            raise typer.Exit(0)

    create_example_config(output_path)
    console.print(f"[green]Config created: {output_path}[/green]")
    console.print("\nNext steps:")
    console.print("  1. Set your ANTHROPIC_API_KEY in the config")
    console.print("  2. Adjust settings as needed")
    console.print("  3. Run: python -m adlab.run process your_video.mp4")


def _extract_transcript_text(transcription, start_time: float, end_time: float) -> str:
    """Helper to extract transcript text for a time range."""
    words = []
    current_word = []

    char_info = transcription.get_char_info()

    for char_data in char_info:
        char_start = char_data.get("start_time")
        char = char_data.get("char", "")

        if char_start is None:
            current_word.append(char)
            continue

        if char_start < start_time:
            continue
        if char_start >= end_time:
            break

        if char == " ":
            if current_word:
                words.append("".join(current_word))
                current_word = []
        else:
            current_word.append(char)

    if current_word:
        words.append("".join(current_word))

    return " ".join(words)


if __name__ == "__main__":
    app()
