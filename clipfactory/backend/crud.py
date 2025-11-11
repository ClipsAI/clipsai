"""
CRUD Operations for Clip Factory
Async database operations using SQLAlchemy 2.0
"""
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

from .models import (
    Video, Clip, Variation, TitleVariant,
    MusicTrack, Account, Post, CalendarEvent,
    PerformanceTracking
)

logger = logging.getLogger(__name__)


# ============================================================================
# VIDEO OPERATIONS
# ============================================================================

async def create_video(
    db: AsyncSession,
    filename: str,
    file_path: str,
    file_size: Optional[int] = None,
    duration: Optional[float] = None,
    resolution: Optional[str] = None,
    fps: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Video:
    """Create a new video record."""
    video = Video(
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        duration=duration,
        resolution=resolution,
        fps=fps,
        status='uploaded',
        metadata=metadata
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)
    logger.info(f"Created video: {video.id}")
    return video


async def get_video_by_id(db: AsyncSession, video_id: UUID) -> Optional[Video]:
    """Get video by ID."""
    result = await db.execute(
        select(Video).where(Video.id == video_id)
    )
    return result.scalar_one_or_none()


async def get_videos(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None
) -> List[Video]:
    """Get list of videos with optional status filter."""
    query = select(Video)

    if status:
        query = query.where(Video.status == status)

    query = query.offset(skip).limit(limit).order_by(Video.uploaded_at.desc())

    result = await db.execute(query)
    return result.scalars().all()


async def update_video_status(
    db: AsyncSession,
    video_id: UUID,
    status: str,
    error: Optional[str] = None
) -> Optional[Video]:
    """Update video status and optionally store error message."""
    video = await get_video_by_id(db, video_id)
    if video:
        video.status = status
        if error:
            # Store error in metadata
            if not video.metadata:
                video.metadata = {}
            video.metadata["error"] = error
        await db.commit()
        await db.refresh(video)
        logger.info(f"Updated video {video_id} status to {status}")
    return video


async def update_video_metadata(
    db: AsyncSession,
    video_id: UUID,
    metadata: Dict[str, Any]
) -> Optional[Video]:
    """Update video metadata."""
    video = await get_video_by_id(db, video_id)
    if video:
        video.metadata = metadata
        await db.commit()
        await db.refresh(video)
        logger.info(f"Updated video {video_id} metadata")
    return video


async def delete_video(db: AsyncSession, video_id: UUID) -> bool:
    """Delete video and all related clips (cascades)."""
    video = await get_video_by_id(db, video_id)
    if video:
        await db.delete(video)
        await db.commit()
        logger.info(f"Deleted video {video_id}")
        return True
    return False


# ============================================================================
# CLIP OPERATIONS
# ============================================================================

async def create_clip(
    db: AsyncSession,
    video_id: UUID,
    start_time: float,
    end_time: float,
    transcript: Optional[str] = None,
    hook_score: Optional[float] = None,
    hook_score_data: Optional[Dict[str, Any]] = None,
    status: str = 'pending',
    metadata: Optional[Dict[str, Any]] = None
) -> Clip:
    """Create a new clip record."""
    clip = Clip(
        video_id=video_id,
        start_time=start_time,
        end_time=end_time,
        transcript=transcript,
        hook_score=hook_score,
        hook_score_data=hook_score_data,
        status=status,
        metadata=metadata
    )
    db.add(clip)
    await db.commit()
    await db.refresh(clip)
    logger.info(f"Created clip: {clip.id}")
    return clip


async def get_clip_by_id(db: AsyncSession, clip_id: UUID) -> Optional[Clip]:
    """Get clip by ID with video relationship."""
    result = await db.execute(
        select(Clip)
        .options(joinedload(Clip.video))
        .where(Clip.id == clip_id)
    )
    return result.scalar_one_or_none()


async def get_clips_by_video(
    db: AsyncSession,
    video_id: UUID,
    status: Optional[str] = None
) -> List[Clip]:
    """Get all clips for a video."""
    query = select(Clip).where(Clip.video_id == video_id)

    if status:
        query = query.where(Clip.status == status)

    query = query.order_by(Clip.hook_score.desc().nullslast(), Clip.created_at)

    result = await db.execute(query)
    return result.scalars().all()


async def get_top_clips(
    db: AsyncSession,
    limit: int = 10,
    min_hook_score: Optional[float] = None
) -> List[Clip]:
    """Get top clips by hook score."""
    query = select(Clip).where(Clip.hook_score.isnot(None))

    if min_hook_score:
        query = query.where(Clip.hook_score >= min_hook_score)

    query = query.order_by(Clip.hook_score.desc()).limit(limit)

    result = await db.execute(query)
    return result.scalars().all()


async def update_clip_status(
    db: AsyncSession,
    clip_id: UUID,
    status: str
) -> Optional[Clip]:
    """Update clip status."""
    clip = await get_clip_by_id(db, clip_id)
    if clip:
        clip.status = status
        await db.commit()
        await db.refresh(clip)
        logger.info(f"Updated clip {clip_id} status to {status}")
    return clip


async def update_clip_hook_score(
    db: AsyncSession,
    clip_id: UUID,
    hook_score: float,
    hook_score_data: Optional[Dict[str, Any]] = None
) -> Optional[Clip]:
    """Update clip hook score."""
    clip = await get_clip_by_id(db, clip_id)
    if clip:
        clip.hook_score = hook_score
        if hook_score_data:
            clip.hook_score_data = hook_score_data
        await db.commit()
        await db.refresh(clip)
        logger.info(f"Updated clip {clip_id} hook score to {hook_score}")
    return clip


# ============================================================================
# VARIATION OPERATIONS
# ============================================================================

async def create_variation(
    db: AsyncSession,
    clip_id: UUID,
    variation_type: str,
    reframe_style: str,
    start_time: float,
    end_time: float,
    title_style: Optional[str] = None,
    duration: Optional[float] = None,
    music_track_id: Optional[UUID] = None,
    video_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Variation:
    """Create a new variation record."""
    variation = Variation(
        clip_id=clip_id,
        variation_type=variation_type,
        reframe_style=reframe_style,
        title_style=title_style,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        music_track_id=music_track_id,
        video_path=video_path,
        metadata=metadata
    )
    db.add(variation)
    await db.commit()
    await db.refresh(variation)
    logger.info(f"Created variation: {variation.id}")
    return variation


async def get_variation_by_id(db: AsyncSession, variation_id: UUID) -> Optional[Variation]:
    """Get variation by ID with relationships."""
    result = await db.execute(
        select(Variation)
        .options(joinedload(Variation.clip))
        .where(Variation.id == variation_id)
    )
    return result.scalar_one_or_none()


async def get_variations_by_clip(
    db: AsyncSession,
    clip_id: UUID
) -> List[Variation]:
    """Get all variations for a clip."""
    result = await db.execute(
        select(Variation)
        .where(Variation.clip_id == clip_id)
        .order_by(Variation.created_at)
    )
    return result.scalars().all()


async def get_variations_by_video(
    db: AsyncSession,
    video_id: UUID
) -> List[Variation]:
    """Get all variations for a video (through clips)."""
    result = await db.execute(
        select(Variation)
        .join(Clip)
        .where(Clip.video_id == video_id)
        .order_by(Variation.created_at)
    )
    return result.scalars().all()


async def update_variation_paths(
    db: AsyncSession,
    variation_id: UUID,
    video_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    captions_path: Optional[str] = None
) -> Optional[Variation]:
    """Update variation file paths."""
    variation = await get_variation_by_id(db, variation_id)
    if variation:
        if video_path:
            variation.video_path = video_path
        if thumbnail_path:
            variation.thumbnail_path = thumbnail_path
        if captions_path:
            variation.captions_path = captions_path
        await db.commit()
        await db.refresh(variation)
        logger.info(f"Updated variation {variation_id} paths")
    return variation


# ============================================================================
# TITLE VARIANT OPERATIONS
# ============================================================================

async def create_title_variant(
    db: AsyncSession,
    variation_id: UUID,
    variant_id: str,
    title_text: str,
    hook_style: Optional[str] = None,
    target_audience: Optional[str] = None,
    predicted_ctr: Optional[float] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> TitleVariant:
    """Create a new title variant."""
    title_variant = TitleVariant(
        variation_id=variation_id,
        variant_id=variant_id,
        title_text=title_text,
        hook_style=hook_style,
        target_audience=target_audience,
        predicted_ctr=predicted_ctr,
        tags=tags,
        metadata=metadata
    )
    db.add(title_variant)
    await db.commit()
    await db.refresh(title_variant)
    logger.info(f"Created title variant: {title_variant.id}")
    return title_variant


async def get_title_variants_by_variation(
    db: AsyncSession,
    variation_id: UUID
) -> List[TitleVariant]:
    """Get all title variants for a variation."""
    result = await db.execute(
        select(TitleVariant)
        .where(TitleVariant.variation_id == variation_id)
        .order_by(TitleVariant.predicted_ctr.desc().nullslast())
    )
    return result.scalars().all()


async def create_multiple_title_variants(
    db: AsyncSession,
    variation_id: UUID,
    titles: List[Dict[str, Any]]
) -> List[TitleVariant]:
    """Create multiple title variants at once."""
    variants = []
    for title_data in titles:
        variant = TitleVariant(
            variation_id=variation_id,
            **title_data
        )
        db.add(variant)
        variants.append(variant)

    await db.commit()
    for variant in variants:
        await db.refresh(variant)

    logger.info(f"Created {len(variants)} title variants for variation {variation_id}")
    return variants


# ============================================================================
# MUSIC TRACK OPERATIONS
# ============================================================================

async def create_music_track(
    db: AsyncSession,
    name: str,
    file_path: str,
    vibe: Optional[str] = None,
    context_description: Optional[str] = None,
    color: Optional[str] = None,
    bpm: Optional[int] = None,
    duration: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> MusicTrack:
    """Create a new music track."""
    track = MusicTrack(
        name=name,
        file_path=file_path,
        vibe=vibe,
        context_description=context_description,
        color=color,
        bpm=bpm,
        duration=duration,
        metadata=metadata
    )
    db.add(track)
    await db.commit()
    await db.refresh(track)
    logger.info(f"Created music track: {track.id}")
    return track


async def get_all_music_tracks(
    db: AsyncSession,
    is_available: Optional[bool] = None
) -> List[MusicTrack]:
    """Get all music tracks."""
    query = select(MusicTrack)

    if is_available is not None:
        query = query.where(MusicTrack.is_available == is_available)

    query = query.order_by(MusicTrack.name)

    result = await db.execute(query)
    return result.scalars().all()


async def get_music_track_by_id(db: AsyncSession, track_id: UUID) -> Optional[MusicTrack]:
    """Get music track by ID."""
    result = await db.execute(
        select(MusicTrack).where(MusicTrack.id == track_id)
    )
    return result.scalar_one_or_none()


async def increment_music_track_usage(db: AsyncSession, track_id: UUID) -> Optional[MusicTrack]:
    """Increment times_used counter for a music track."""
    track = await get_music_track_by_id(db, track_id)
    if track:
        track.times_used += 1
        await db.commit()
        await db.refresh(track)
    return track


# ============================================================================
# ACCOUNT OPERATIONS
# ============================================================================

async def create_account(
    db: AsyncSession,
    platform: str,
    username: str,
    account_type: str,
    posting_strategy: Optional[str] = None,
    is_active: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> Account:
    """Create a new account."""
    account = Account(
        platform=platform,
        username=username,
        account_type=account_type,
        posting_strategy=posting_strategy,
        is_active=is_active,
        metadata=metadata
    )
    db.add(account)
    await db.commit()
    await db.refresh(account)
    logger.info(f"Created account: {account.id}")
    return account


async def get_all_accounts(
    db: AsyncSession,
    platform: Optional[str] = None,
    is_active: Optional[bool] = None
) -> List[Account]:
    """Get all accounts with optional filters."""
    query = select(Account)

    if platform:
        query = query.where(Account.platform == platform)
    if is_active is not None:
        query = query.where(Account.is_active == is_active)

    query = query.order_by(Account.created_at)

    result = await db.execute(query)
    return result.scalars().all()


async def get_account_by_id(db: AsyncSession, account_id: UUID) -> Optional[Account]:
    """Get account by ID."""
    result = await db.execute(
        select(Account).where(Account.id == account_id)
    )
    return result.scalar_one_or_none()


# ============================================================================
# POST OPERATIONS
# ============================================================================

async def create_post(
    db: AsyncSession,
    variation_id: Optional[UUID] = None,
    account_id: Optional[UUID] = None,
    title_variant_id: Optional[UUID] = None,
    scheduled_for: Optional[Any] = None,
    status: str = 'scheduled',
    metadata: Optional[Dict[str, Any]] = None
) -> Post:
    """Create a new post."""
    post = Post(
        variation_id=variation_id,
        account_id=account_id,
        title_variant_id=title_variant_id,
        scheduled_for=scheduled_for,
        status=status,
        metadata=metadata
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)
    logger.info(f"Created post: {post.id}")
    return post


async def get_posts_by_account(
    db: AsyncSession,
    account_id: UUID,
    status: Optional[str] = None
) -> List[Post]:
    """Get all posts for an account."""
    query = select(Post).where(Post.account_id == account_id)

    if status:
        query = query.where(Post.status == status)

    query = query.order_by(Post.scheduled_for.desc().nullslast())

    result = await db.execute(query)
    return result.scalars().all()


async def get_scheduled_posts(
    db: AsyncSession,
    limit: int = 100
) -> List[Post]:
    """Get scheduled posts."""
    result = await db.execute(
        select(Post)
        .where(Post.status == 'scheduled')
        .order_by(Post.scheduled_for)
        .limit(limit)
    )
    return result.scalars().all()


async def update_post_status(
    db: AsyncSession,
    post_id: UUID,
    status: str,
    post_url: Optional[str] = None,
    posted_at: Optional[Any] = None
) -> Optional[Post]:
    """Update post status."""
    result = await db.execute(
        select(Post).where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()

    if post:
        post.status = status
        if post_url:
            post.post_url = post_url
        if posted_at:
            post.posted_at = posted_at
        await db.commit()
        await db.refresh(post)
        logger.info(f"Updated post {post_id} status to {status}")
    return post


# ============================================================================
# PERFORMANCE TRACKING OPERATIONS
# ============================================================================

async def create_performance_tracking(
    db: AsyncSession,
    post_id: UUID,
    views: Optional[int] = None,
    likes: Optional[int] = None,
    comments: Optional[int] = None,
    shares: Optional[int] = None,
    avg_view_duration: Optional[float] = None,
    completion_rate: Optional[float] = None,
    engagement_rate: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> PerformanceTracking:
    """Create a new performance tracking record."""
    tracking = PerformanceTracking(
        post_id=post_id,
        views=views,
        likes=likes,
        comments=comments,
        shares=shares,
        avg_view_duration=avg_view_duration,
        completion_rate=completion_rate,
        engagement_rate=engagement_rate,
        metadata=metadata
    )
    db.add(tracking)
    await db.commit()
    await db.refresh(tracking)
    logger.info(f"Created performance tracking: {tracking.id}")
    return tracking


async def get_performance_by_post(
    db: AsyncSession,
    post_id: UUID
) -> List[PerformanceTracking]:
    """Get all performance tracking records for a post."""
    result = await db.execute(
        select(PerformanceTracking)
        .where(PerformanceTracking.post_id == post_id)
        .order_by(PerformanceTracking.tracked_at.desc())
    )
    return result.scalars().all()


# ============================================================================
# ANALYTICS OPERATIONS
# ============================================================================

async def get_video_statistics(db: AsyncSession) -> Dict[str, Any]:
    """Get overall video statistics."""
    total_videos = await db.execute(select(func.count(Video.id)))
    total_clips = await db.execute(select(func.count(Clip.id)))
    total_variations = await db.execute(select(func.count(Variation.id)))

    return {
        "total_videos": total_videos.scalar(),
        "total_clips": total_clips.scalar(),
        "total_variations": total_variations.scalar()
    }


async def get_clip_statistics_by_video(db: AsyncSession, video_id: UUID) -> Dict[str, Any]:
    """Get clip statistics for a video."""
    clips = await get_clips_by_video(db, video_id)

    if not clips:
        return {"total_clips": 0}

    hook_scores = [c.hook_score for c in clips if c.hook_score is not None]

    return {
        "total_clips": len(clips),
        "clips_with_scores": len(hook_scores),
        "avg_hook_score": sum(hook_scores) / len(hook_scores) if hook_scores else None,
        "max_hook_score": max(hook_scores) if hook_scores else None,
        "min_hook_score": min(hook_scores) if hook_scores else None,
    }
