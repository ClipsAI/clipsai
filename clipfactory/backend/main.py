"""
Clip Factory Backend - FastAPI Application
Main entry point for the API server
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager
import os
import shutil
from pathlib import Path
import logging
import secrets
from uuid import UUID

# Import database and CRUD operations
from . import crud
from .database import get_db, startup_event, shutdown_event, health_check as db_health_check

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    logger.info("Starting Clip Factory API...")
    await startup_event()
    logger.info("Clip Factory API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Clip Factory API...")
    await shutdown_event()
    logger.info("Clip Factory API shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Clip Factory API",
    description="Viral clip generation and processing system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Security: Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
MAX_VIDEO_SIZE = 50 * 1024 * 1024 * 1024  # 50GB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

# ============================================================================
# SECURITY HELPER FUNCTIONS
# ============================================================================

def sanitize_filename(filename: str) -> str:
    """
    Remove path traversal sequences and dangerous characters from filename.
    Returns only the base filename without any path components.
    """
    # Get just the filename, no directory components
    filename = os.path.basename(filename)
    # Remove path separators (defense in depth)
    filename = filename.replace('\\', '').replace('/', '')
    # Remove null bytes
    filename = filename.replace('\0', '')
    # Remove leading dots to prevent hidden files
    filename = filename.lstrip('.')
    # If filename is empty after sanitization, use a default
    if not filename:
        filename = "upload"
    return filename

def safe_path_join(base_dir: Path, *parts: str) -> Path:
    """
    Safely join path components and verify the result is within base_dir.
    Raises HTTPException if path traversal is detected.
    """
    # Sanitize each part
    sanitized_parts = [sanitize_filename(part) for part in parts]

    # Join paths
    target_path = base_dir.joinpath(*sanitized_parts)

    # Resolve to absolute path and verify it's within base_dir
    try:
        resolved_target = target_path.resolve()
        resolved_base = base_dir.resolve()
        resolved_target.relative_to(resolved_base)
        return target_path
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path detected")

def validate_file_extension(filename: str, allowed_extensions: set) -> bool:
    """Validate that file has an allowed extension."""
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions

def generate_safe_video_id() -> str:
    """Generate a cryptographically secure random video ID."""
    return f"vid_{secrets.token_hex(16)}"

def generate_safe_clip_id() -> str:
    """Generate a cryptographically secure random clip ID."""
    return f"clip_{secrets.token_hex(12)}"

# ============================================================================
# DATA MODELS
# ============================================================================

class VideoUploadResponse(BaseModel):
    video_id: str
    filename: str
    size: int
    duration: Optional[float] = None
    status: str

class ClipMetadata(BaseModel):
    clip_id: str
    start_time: float
    end_time: float
    duration: float
    hook_score: Optional[float] = None
    transcript: Optional[str] = None

class VariationRequest(BaseModel):
    clip_id: str
    temporal_variations: List[str]  # ['base', '+4s', '+35s']
    reframe_styles: List[str]  # ['original', 'flipped', 'blurry_bg']
    title_style: str  # 'TT3' or 'AdLab'
    music_id: Optional[str] = None

class TitleGenerationRequest(BaseModel):
    transcript: str
    hook_score: float
    duration: float
    num_variants: int = 5

# ============================================================================
# PHASE 1: COUNCIL DELIBERATION
# ============================================================================

async def run_council_deliberation_task(video_path: Path, video_id: UUID):
    """
    Background task to run council deliberation on uploaded video.

    This is a wrapper that creates its own DB session for the background task.
    """
    from .database import async_session_maker

    async with async_session_maker() as db_session:
        try:
            from clipfactory.processing.orchestrator import ClipFactoryOrchestrator

            logger.info(f"Starting council deliberation for video {video_id}")

            # Update video status to processing
            await crud.update_video_status(db_session, video_id, "processing")

            # Create orchestrator with API keys from environment
            config = {
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "google_api_key": os.getenv("GOOGLE_API_KEY"),
            }
            orchestrator = ClipFactoryOrchestrator(config)

            # Run council deliberation
            clips = await orchestrator.phase1_council_deliberation(str(video_path))

            logger.info(f"Council selected {len(clips)} clips for video {video_id}")

            # Store clips in database
            for clip_data in clips:
                await crud.create_clip(
                    db=db_session,
                    video_id=video_id,
                    start_time=clip_data["start_time"],
                    end_time=clip_data["end_time"],
                    transcript=clip_data.get("transcript", ""),
                    hook_score=clip_data.get("hook_score"),
                    metadata={
                        "vvsa_score": clip_data.get("vvsa_score"),
                        "council_consensus": clip_data.get("council_consensus"),
                        "duration": clip_data.get("duration")
                    }
                )

            # Commit all changes
            await db_session.commit()

            # Update video status to completed
            await crud.update_video_status(db_session, video_id, "completed")
            await db_session.commit()

            logger.info(f"Council deliberation complete for video {video_id}")

        except Exception as e:
            logger.error(f"Council deliberation failed for video {video_id}: {e}")
            logger.exception(e)
            # Rollback on error
            await db_session.rollback()
            # Update video status to failed
            try:
                await crud.update_video_status(db_session, video_id, "failed", error=str(e))
                await db_session.commit()
            except:
                pass


@app.post("/api/phase1/upload", response_model=VideoUploadResponse)
async def upload_video_for_council(
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Upload video for council deliberation.
    Returns video_id for tracking.
    """
    try:
        # Validate file extension
        if not validate_file_extension(video.filename, ALLOWED_VIDEO_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
            )

        # Generate secure video ID and sanitize filename
        video_id = generate_safe_video_id()
        safe_filename = sanitize_filename(video.filename)
        file_ext = Path(safe_filename).suffix.lower()

        # Create safe path (use video_id as filename to prevent collisions)
        video_path = safe_path_join(UPLOAD_DIR, f"{video_id}{file_ext}")

        # Read file content for size validation
        content = await video.read()
        if len(content) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 50GB)")

        # Save uploaded file
        with open(video_path, "wb") as buffer:
            buffer.write(content)

        file_size = len(content)

        # Store video metadata in database
        db_video = await crud.create_video(
            db=db,
            filename=safe_filename,
            file_path=str(video_path),
            file_size=file_size,
            metadata={"original_filename": video.filename}
        )

        # Trigger council deliberation in background
        if background_tasks:
            background_tasks.add_task(
                run_council_deliberation_task,
                video_path,
                db_video.id
            )
            logger.info(f"Queued council deliberation for video {db_video.id}")

        return VideoUploadResponse(
            video_id=str(db_video.id),
            filename=safe_filename,
            size=file_size,
            status=db_video.status
        )

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/phase1/status/{video_id}")
async def get_council_status(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get status of council deliberation."""
    try:
        # Convert video_id to UUID
        video_uuid = UUID(video_id)

        # Get video from database
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get clips for this video
        clips = await crud.get_clips_by_video(db, video_uuid)

        # Get statistics
        stats = await crud.get_clip_statistics_by_video(db, video_uuid)

        return {
            "video_id": video_id,
            "status": video.status,
            "clips_found": len(clips),
            "clips_with_scores": stats.get("clips_with_scores", 0),
            "avg_hook_score": stats.get("avg_hook_score"),
            "max_hook_score": stats.get("max_hook_score"),
            "progress": 1.0 if video.status == "completed" else 0.5 if clips else 0.0
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/phase1/clips/{video_id}")
async def get_council_clips(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get clips selected by council."""
    try:
        # Convert video_id to UUID
        video_uuid = UUID(video_id)

        # Get video from database
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get clips for this video
        clips = await crud.get_clips_by_video(db, video_uuid)

        # Format clips for response
        clips_data = [
            {
                "clip_id": str(clip.id),
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "hook_score": clip.hook_score,
                "transcript": clip.transcript,
                "status": clip.status,
                "created_at": clip.created_at.isoformat() if clip.created_at else None
            }
            for clip in clips
        ]

        return {
            "video_id": video_id,
            "clips": clips_data,
            "total": len(clips_data)
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except Exception as e:
        logger.error(f"Clips fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 2: PREMIERE INTEGRATION
# ============================================================================

@app.post("/api/phase2/export-xml/{video_id}")
async def export_premiere_xml(video_id: str, db: AsyncSession = Depends(get_db)):
    """
    Export clips to Premiere Pro XML format.
    Returns download link for XML file.
    """
    try:
        # Convert video_id to UUID
        video_uuid = UUID(video_id)

        # Get video from database
        video = await crud.get_video_by_id(db, video_uuid)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Get clips for this video
        clips = await crud.get_clips_by_video(db, video_uuid)
        if not clips:
            raise HTTPException(status_code=400, detail="No clips found for this video")

        # Update video status
        await crud.update_video_status(db, video_uuid, "exporting")

        # Generate Premiere Pro XML with clip data
        xml_content = generate_premiere_xml(video_id, clips)

        # Create safe path for XML file
        xml_path = safe_path_join(OUTPUT_DIR, f"{video_id}_premiere.xml")

        with open(xml_path, "w") as f:
            f.write(xml_content)

        # Update video status
        await crud.update_video_status(db, video_uuid, "exported")

        return {
            "video_id": video_id,
            "xml_path": str(xml_path),
            "download_url": f"/api/download/xml/{video_id}",
            "clips_count": len(clips)
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")
    except Exception as e:
        logger.error(f"XML export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/xml/{video_id}")
async def download_xml(video_id: str):
    """Download Premiere XML file."""
    xml_path = OUTPUT_DIR / f"{video_id}_premiere.xml"

    if not xml_path.exists():
        raise HTTPException(status_code=404, detail="XML file not found")

    return FileResponse(
        path=xml_path,
        media_type="application/xml",
        filename=f"{video_id}_premiere.xml"
    )

@app.post("/api/phase2/reupload")
async def reupload_edited_clips(
    video_id: str,
    clips: List[UploadFile] = File(...)
):
    """
    Re-upload clips edited by Wes in Premiere.
    Tracks which clips were approved.
    """
    try:
        uploaded_clips = []

        # Create safe video directory
        video_dir = safe_path_join(OUTPUT_DIR, video_id, "edited")
        video_dir.mkdir(parents=True, exist_ok=True)

        for clip in clips:
            # Validate file extension
            if not validate_file_extension(clip.filename, ALLOWED_VIDEO_EXTENSIONS):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}"
                )

            # Generate secure clip ID and sanitize filename
            clip_id = generate_safe_clip_id()
            safe_filename = sanitize_filename(clip.filename)
            file_ext = Path(safe_filename).suffix.lower()

            # Create safe path
            clip_path = video_dir / f"{clip_id}{file_ext}"

            # Read and validate file size
            content = await clip.read()
            if len(content) > MAX_VIDEO_SIZE:
                raise HTTPException(status_code=413, detail="File too large")

            # Save clip
            with open(clip_path, "wb") as buffer:
                buffer.write(content)

            uploaded_clips.append({
                "clip_id": clip_id,
                "filename": safe_filename,
                "path": str(clip_path)
            })

        # TODO: Update database with approved clips

        return {
            "video_id": video_id,
            "clips_uploaded": len(uploaded_clips),
            "clips": uploaded_clips
        }

    except Exception as e:
        logger.error(f"Reupload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 3: MATRIX PROCESSING
# ============================================================================

@app.post("/api/phase3/process/{clip_id}")
async def process_matrix(
    clip_id: str,
    background_tasks: BackgroundTasks
):
    """
    Process clip through matrix:
    - Face tracking
    - Canvas rendering (3 styles)
    - Watermark overlay
    - Title card overlay
    """
    try:
        # TODO: Trigger matrix processing
        # background_tasks.add_task(run_matrix_processing, clip_id)

        return {
            "clip_id": clip_id,
            "status": "processing",
            "message": "Matrix processing started"
        }

    except Exception as e:
        logger.error(f"Matrix processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 4: VARIATION GENERATION
# ============================================================================

@app.post("/api/phase4/generate-variations")
async def generate_variations(request: VariationRequest):
    """
    Generate all variations for a clip:
    - Temporal variations (base, +4s, +35s)
    - Reframe styles (original, flipped, blurry_bg)
    - Music swapping
    - Caption generation
    """
    try:
        # TODO: Generate variations
        variations = []

        for temporal in request.temporal_variations:
            for reframe in request.reframe_styles:
                variation_id = f"{request.clip_id}_{temporal}_{reframe}"
                variations.append({
                    "variation_id": variation_id,
                    "temporal": temporal,
                    "reframe": reframe,
                    "status": "pending"
                })

        return {
            "clip_id": request.clip_id,
            "variations": variations,
            "total": len(variations)
        }

    except Exception as e:
        logger.error(f"Variation generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/phase4/generate-titles")
async def generate_titles(request: TitleGenerationRequest):
    """Generate title variants using Claude/GPT."""
    try:
        # TODO: Call Claude API for title generation
        titles = [
            {
                "variant_id": "A",
                "text": "Sample Title A",
                "hook_style": "curiosity",
                "predicted_ctr": 0.085
            },
            {
                "variant_id": "B",
                "text": "Sample Title B",
                "hook_style": "revelation",
                "predicted_ctr": 0.092
            }
        ]

        return {
            "titles": titles,
            "count": len(titles)
        }

    except Exception as e:
        logger.error(f"Title generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PHASE 5: DISTRIBUTION
# ============================================================================

@app.post("/api/phase5/screenshot-to-title")
async def screenshot_to_title(
    screenshot: UploadFile = File(...),
    account_type: str = "fan"
):
    """
    Upload screenshot, generate posting title.
    Different styles per account type.
    """
    try:
        # Validate file extension
        if not validate_file_extension(screenshot.filename, ALLOWED_IMAGE_EXTENSIONS):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
            )

        # Generate secure filename
        safe_filename = sanitize_filename(screenshot.filename)
        file_ext = Path(safe_filename).suffix.lower()
        temp_filename = f"temp_{secrets.token_hex(8)}{file_ext}"

        # Create safe path
        screenshot_path = safe_path_join(UPLOAD_DIR, temp_filename)

        # Read and validate file size
        content = await screenshot.read()
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

        # Save screenshot temporarily
        with open(screenshot_path, "wb") as buffer:
            buffer.write(content)

        # TODO: Analyze screenshot with Claude Vision
        # TODO: Generate title based on account type

        title = "Sample generated title based on screenshot"

        return {
            "title": title,
            "account_type": account_type,
            "screenshot_analyzed": True
        }

    except Exception as e:
        logger.error(f"Screenshot analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# DATABASE MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/api/videos")
async def list_videos(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List all videos with optional status filter."""
    try:
        videos = await crud.get_videos(db, skip=skip, limit=limit, status=status)

        videos_data = [
            {
                "id": str(video.id),
                "filename": video.filename,
                "file_size": video.file_size,
                "duration": video.duration,
                "resolution": video.resolution,
                "fps": video.fps,
                "status": video.status,
                "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None
            }
            for video in videos
        ]

        return {
            "videos": videos_data,
            "count": len(videos_data)
        }

    except Exception as e:
        logger.error(f"List videos error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/statistics")
async def get_statistics(db: AsyncSession = Depends(get_db)):
    """Get overall system statistics."""
    try:
        stats = await crud.get_video_statistics(db)
        return stats

    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    db_status = await db_health_check()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "database": db_status
    }

@app.get("/api/music/list")
async def list_music(db: AsyncSession = Depends(get_db)):
    """List available music tracks."""
    try:
        tracks = await crud.get_all_music_tracks(db, is_available=True)

        tracks_data = [
            {
                "id": str(track.id),
                "name": track.name,
                "vibe": track.vibe,
                "context_description": track.context_description,
                "color": track.color,
                "bpm": track.bpm,
                "duration": track.duration,
                "times_used": track.times_used
            }
            for track in tracks
        ]

        return {
            "tracks": tracks_data,
            "total": len(tracks_data)
        }

    except Exception as e:
        logger.error(f"Music list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_premiere_xml(video_id: str, clips: list = None) -> str:
    """Generate Premiere Pro XML structure."""
    # Generate clip sequences
    clip_sequences = ""
    if clips:
        for i, clip in enumerate(clips, 1):
            clip_sequences += f"""
            <clip id="clip{i}">
                <name>Clip {i} - Hook Score: {clip.hook_score or 'N/A'}</name>
                <start>{clip.start_time}</start>
                <end>{clip.end_time}</end>
                <duration>{clip.duration}</duration>
            </clip>"""

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<xmeml version="4">
    <project>
        <name>ClipFactory_{video_id}</name>
        <children>{clip_sequences}
        </children>
    </project>
</xmeml>
"""
    return xml

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
