"""
SQLAlchemy ORM Models for Clip Factory
Matches schema from clipfactory/database/schemas/schema.sql
"""
from sqlalchemy import (
    Column, String, BigInteger, Float, Integer, Boolean,
    Text, TIMESTAMP, ForeignKey, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIME
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func
import uuid


class Base(DeclarativeBase):
    """Base class for all models"""
    pass


class Video(Base):
    """Source videos uploaded for processing"""
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(BigInteger, nullable=True)
    duration = Column(Float, nullable=True)
    resolution = Column(Text, nullable=True)
    fps = Column(Float, nullable=True)
    uploaded_at = Column(TIMESTAMP, server_default=func.now())
    status = Column(Text, default='uploaded')
    metadata = Column(JSONB, nullable=True)

    # Relationships
    clips = relationship("Clip", back_populates="video", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_videos_status', 'status'),
    )

    def __repr__(self):
        return f"<Video(id={self.id}, filename={self.filename}, status={self.status})>"


class Clip(Base):
    """Clips identified by AI council from source videos"""
    __tablename__ = "clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey('videos.id', ondelete='CASCADE'), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    # duration is a generated column in SQL, but we can compute it in Python
    transcript = Column(Text, nullable=True)
    hook_score = Column(Float, nullable=True)
    hook_score_data = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    status = Column(Text, default='pending')
    metadata = Column(JSONB, nullable=True)

    # Relationships
    video = relationship("Video", back_populates="clips")
    variations = relationship("Variation", back_populates="clip", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_clips_video_id', 'video_id'),
        Index('idx_clips_hook_score', 'hook_score'),
    )

    @property
    def duration(self):
        """Computed duration property"""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None

    def __repr__(self):
        return f"<Clip(id={self.id}, video_id={self.video_id}, duration={self.duration})>"


class Variation(Base):
    """Variations generated from clips (temporal + reframe combinations)"""
    __tablename__ = "variations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clip_id = Column(UUID(as_uuid=True), ForeignKey('clips.id', ondelete='CASCADE'), nullable=False)
    variation_type = Column(Text, nullable=False)  # base, +4s, +35s
    reframe_style = Column(Text, nullable=False)   # original, flipped, blurry_bg
    title_style = Column(Text, nullable=True)       # TT3, AdLab
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=True)
    frame_offset = Column(Float, nullable=True)
    music_track_id = Column(UUID(as_uuid=True), nullable=True)
    video_path = Column(Text, nullable=True)
    thumbnail_path = Column(Text, nullable=True)
    captions_path = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    metadata = Column(JSONB, nullable=True)

    # Relationships
    clip = relationship("Clip", back_populates="variations")
    title_variants = relationship("TitleVariant", back_populates="variation", cascade="all, delete-orphan")
    posts = relationship("Post", back_populates="variation")

    # Indexes
    __table_args__ = (
        Index('idx_variations_clip_id', 'clip_id'),
    )

    def __repr__(self):
        return f"<Variation(id={self.id}, type={self.variation_type}, reframe={self.reframe_style})>"


class TitleVariant(Base):
    """Title variants for A/B testing"""
    __tablename__ = "title_variants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    variation_id = Column(UUID(as_uuid=True), ForeignKey('variations.id', ondelete='CASCADE'), nullable=False)
    variant_id = Column(Text, nullable=True)  # A, B, C, D, E
    title_text = Column(Text, nullable=False)
    hook_style = Column(Text, nullable=True)
    target_audience = Column(Text, nullable=True)
    predicted_ctr = Column(Float, nullable=True)
    tags = Column(ARRAY(Text), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    metadata = Column(JSONB, nullable=True)

    # Relationships
    variation = relationship("Variation", back_populates="title_variants")
    posts = relationship("Post", back_populates="title_variant")

    def __repr__(self):
        return f"<TitleVariant(id={self.id}, variant={self.variant_id}, text={self.title_text[:30]})>"


class MusicTrack(Base):
    """Music tracks available for swapping (40 total)"""
    __tablename__ = "music_tracks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False)
    vibe = Column(Text, nullable=True)
    context_description = Column(Text, nullable=True)
    color = Column(Text, nullable=True)  # for UI
    bpm = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    is_available = Column(Boolean, default=True)
    times_used = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, server_default=func.now())
    metadata = Column(JSONB, nullable=True)

    def __repr__(self):
        return f"<MusicTrack(id={self.id}, name={self.name}, vibe={self.vibe})>"


class Account(Base):
    """Social media accounts for multi-account posting"""
    __tablename__ = "accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform = Column(Text, nullable=False)  # tiktok, instagram, youtube
    username = Column(Text, nullable=False)
    account_type = Column(Text, nullable=False)  # fan, brand, watermark
    posting_strategy = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    metadata = Column(JSONB, nullable=True)

    # Relationships
    posts = relationship("Post", back_populates="account")

    def __repr__(self):
        return f"<Account(id={self.id}, platform={self.platform}, username={self.username})>"


class Post(Base):
    """Posted or scheduled clips"""
    __tablename__ = "posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    variation_id = Column(UUID(as_uuid=True), ForeignKey('variations.id'), nullable=True)
    account_id = Column(UUID(as_uuid=True), ForeignKey('accounts.id'), nullable=True)
    title_variant_id = Column(UUID(as_uuid=True), ForeignKey('title_variants.id'), nullable=True)
    post_url = Column(Text, nullable=True)
    posted_at = Column(TIMESTAMP, nullable=True)
    scheduled_for = Column(TIMESTAMP, nullable=True)
    status = Column(Text, default='scheduled')  # scheduled, posted, failed
    performance_data = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    metadata = Column(JSONB, nullable=True)

    # Relationships
    variation = relationship("Variation", back_populates="posts")
    account = relationship("Account", back_populates="posts")
    title_variant = relationship("TitleVariant", back_populates="posts")
    calendar_events = relationship("CalendarEvent", back_populates="post", cascade="all, delete-orphan")
    performance_tracking = relationship("PerformanceTracking", back_populates="post", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_posts_account_id', 'account_id'),
        Index('idx_posts_scheduled_for', 'scheduled_for'),
    )

    def __repr__(self):
        return f"<Post(id={self.id}, status={self.status}, scheduled_for={self.scheduled_for})>"


class CalendarEvent(Base):
    """Calendar events for posting schedule"""
    __tablename__ = "calendar_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(UUID(as_uuid=True), ForeignKey('posts.id'), nullable=True)
    event_start = Column(TIMESTAMP, nullable=False)
    event_end = Column(TIMESTAMP, nullable=True)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    calendar_service = Column(Text, nullable=True)  # google, icloud
    external_event_id = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    metadata = Column(JSONB, nullable=True)

    # Relationships
    post = relationship("Post", back_populates="calendar_events")

    def __repr__(self):
        return f"<CalendarEvent(id={self.id}, title={self.title}, start={self.event_start})>"


class PerformanceTracking(Base):
    """Performance metrics for posted clips (A/B testing, analytics)"""
    __tablename__ = "performance_tracking"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(UUID(as_uuid=True), ForeignKey('posts.id'), nullable=True)
    tracked_at = Column(TIMESTAMP, server_default=func.now())
    views = Column(Integer, nullable=True)
    likes = Column(Integer, nullable=True)
    comments = Column(Integer, nullable=True)
    shares = Column(Integer, nullable=True)
    avg_view_duration = Column(Float, nullable=True)
    completion_rate = Column(Float, nullable=True)
    engagement_rate = Column(Float, nullable=True)
    metadata = Column(JSONB, nullable=True)

    # Relationships
    post = relationship("Post", back_populates="performance_tracking")

    # Indexes
    __table_args__ = (
        Index('idx_performance_post_id', 'post_id'),
    )

    def __repr__(self):
        return f"<PerformanceTracking(id={self.id}, post_id={self.post_id}, views={self.views})>"
