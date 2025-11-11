# Database Schema Documentation

Complete database schema for Clip Factory system.

## Table of Contents

- [Overview](#overview)
- [Entity-Relationship Diagram](#entity-relationship-diagram)
- [Tables](#tables)
- [Relationships](#relationships)
- [Indexes](#indexes)
- [Functions](#functions)
- [Query Patterns](#query-patterns)
- [Migration Strategy](#migration-strategy)

## Overview

The Clip Factory database is built on **PostgreSQL 15+** and uses:
- UUID primary keys for distributed systems compatibility
- JSONB for flexible metadata storage
- Generated columns for computed values
- Stored procedures for analytics
- Strategic indexes for performance

**Database**: `clipfactory`
**Total Tables**: 9
**Schema Version**: 1.0.0

## Entity-Relationship Diagram

```
┌──────────────┐
│   videos     │
│──────────────│
│ id (PK)      │
│ filename     │
│ file_path    │
│ duration     │
│ status       │
└──────┬───────┘
       │
       │ 1:N
       │
┌──────▼───────┐
│    clips     │
│──────────────│
│ id (PK)      │
│ video_id(FK) │
│ start_time   │
│ end_time     │
│ transcript   │
│ hook_score   │
└──────┬───────┘
       │
       │ 1:N
       │
┌──────▼───────────┐       ┌──────────────────┐
│   variations     │       │  music_tracks    │
│──────────────────│       │──────────────────│
│ id (PK)          │       │ id (PK)          │
│ clip_id (FK)     │◄──────│ name             │
│ variation_type   │       │ vibe             │
│ reframe_style    │       │ bpm              │
│ music_track_id(FK)│──────▶│ duration         │
└──────┬───────────┘       └──────────────────┘
       │
       │ 1:N
       │
┌──────▼───────────┐
│ title_variants   │
│──────────────────│
│ id (PK)          │
│ variation_id(FK) │
│ title_text       │
│ hook_style       │
│ predicted_ctr    │
└──────────────────┘

┌──────────────┐       ┌──────▼───────┐       ┌──────────────────┐
│  accounts    │       │    posts     │       │calendar_events   │
│──────────────│       │──────────────│       │──────────────────│
│ id (PK)      │◄──────│ id (PK)      │◄──────│ id (PK)          │
│ platform     │       │ variation_id │       │ post_id (FK)     │
│ username     │       │ account_id   │       │ event_start      │
│ account_type │       │ title_var_id │       │ external_event_id│
└──────────────┘       │ posted_at    │       └──────────────────┘
                       │ status       │
                       └──────┬───────┘
                              │
                              │ 1:N
                              │
                       ┌──────▼────────────┐
                       │performance_tracking│
                       │────────────────────│
                       │ id (PK)            │
                       │ post_id (FK)       │
                       │ views              │
                       │ likes              │
                       │ engagement_rate    │
                       └────────────────────┘
```

## Tables

### 1. videos

Source videos uploaded for processing.

**Purpose**: Track uploaded videos and their processing status.

```sql
CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size BIGINT,
    duration FLOAT,
    resolution TEXT,
    fps FLOAT,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    status TEXT DEFAULT 'uploaded',
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `filename` | TEXT | No | Original filename |
| `file_path` | TEXT | No | Absolute path to video file |
| `file_size` | BIGINT | Yes | File size in bytes |
| `duration` | FLOAT | Yes | Duration in seconds |
| `resolution` | TEXT | Yes | e.g., "1920x1080" |
| `fps` | FLOAT | Yes | Frames per second |
| `uploaded_at` | TIMESTAMP | No | Upload timestamp |
| `status` | TEXT | No | 'uploaded', 'processing', 'completed', 'failed' |
| `metadata` | JSONB | Yes | Additional metadata (codec, bitrate, etc.) |

**Example Row**:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "podcast_ep_123.mp4",
  "file_path": "/uploads/vid_abc123_podcast_ep_123.mp4",
  "file_size": 2147483648,
  "duration": 7320.5,
  "resolution": "1920x1080",
  "fps": 30.0,
  "uploaded_at": "2025-11-10T10:00:00Z",
  "status": "completed",
  "metadata": {
    "codec": "h264",
    "bitrate": 5000000,
    "audio_codec": "aac"
  }
}
```

---

### 2. clips

Clips identified by AI council from source videos.

**Purpose**: Store clip boundaries and hook scores.

```sql
CREATE TABLE clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    transcript TEXT,
    hook_score FLOAT,
    hook_score_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    status TEXT DEFAULT 'pending',
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `video_id` | UUID | No | Foreign key to videos |
| `start_time` | FLOAT | No | Start time in seconds |
| `end_time` | FLOAT | No | End time in seconds |
| `duration` | FLOAT | No | Computed duration (generated column) |
| `transcript` | TEXT | Yes | Clip transcript |
| `hook_score` | FLOAT | Yes | Overall hook score (0-10) |
| `hook_score_data` | JSONB | Yes | VVSA breakdown (virality, value, specificity, actionability) |
| `created_at` | TIMESTAMP | No | When clip was identified |
| `status` | TEXT | No | 'pending', 'approved', 'rejected' |
| `metadata` | JSONB | Yes | Additional data |

**Example Row**:

```json
{
  "id": "650e8400-e29b-41d4-a716-446655440001",
  "video_id": "550e8400-e29b-41d4-a716-446655440000",
  "start_time": 123.5,
  "end_time": 175.2,
  "duration": 51.7,
  "transcript": "This is the most important thing to understand...",
  "hook_score": 8.5,
  "hook_score_data": {
    "virality": 8.5,
    "value": 8.0,
    "specificity": 9.0,
    "actionability": 8.0
  },
  "created_at": "2025-11-10T10:15:00Z",
  "status": "approved"
}
```

---

### 3. variations

All variations generated from base clips.

**Purpose**: Store 9 variations per clip (3 temporal × 3 reframe).

```sql
CREATE TABLE variations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clip_id UUID REFERENCES clips(id) ON DELETE CASCADE,
    variation_type TEXT NOT NULL,
    reframe_style TEXT NOT NULL,
    title_style TEXT,
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT,
    frame_offset FLOAT,
    music_track_id UUID,
    video_path TEXT,
    thumbnail_path TEXT,
    captions_path TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `clip_id` | UUID | No | Foreign key to clips |
| `variation_type` | TEXT | No | 'base', '+4s', '+35s' |
| `reframe_style` | TEXT | No | 'original', 'flipped', 'blurry_bg' |
| `title_style` | TEXT | Yes | 'TT3', 'AdLab' |
| `start_time` | FLOAT | No | Adjusted start time |
| `end_time` | FLOAT | No | Adjusted end time |
| `duration` | FLOAT | Yes | Variation duration |
| `frame_offset` | FLOAT | Yes | Random frame offset (0-2s) |
| `music_track_id` | UUID | Yes | Foreign key to music_tracks |
| `video_path` | TEXT | Yes | Path to rendered video |
| `thumbnail_path` | TEXT | Yes | Path to thumbnail |
| `captions_path` | TEXT | Yes | Path to caption file |
| `created_at` | TIMESTAMP | No | Creation timestamp |
| `metadata` | JSONB | Yes | Additional data |

**Example Row**:

```json
{
  "id": "750e8400-e29b-41d4-a716-446655440002",
  "clip_id": "650e8400-e29b-41d4-a716-446655440001",
  "variation_type": "+4s",
  "reframe_style": "flipped",
  "title_style": "TT3",
  "start_time": 119.5,
  "end_time": 179.2,
  "duration": 59.7,
  "frame_offset": 1.2,
  "music_track_id": "850e8400-e29b-41d4-a716-446655440010",
  "video_path": "/output/var_750e8400.mp4",
  "thumbnail_path": "/output/var_750e8400_thumb.jpg",
  "captions_path": "/output/var_750e8400_captions.srt",
  "created_at": "2025-11-10T11:00:00Z"
}
```

---

### 4. title_variants

A/B testing title variants for each variation.

**Purpose**: Store multiple title options for performance testing.

```sql
CREATE TABLE title_variants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variation_id UUID REFERENCES variations(id) ON DELETE CASCADE,
    variant_id TEXT,
    title_text TEXT NOT NULL,
    hook_style TEXT,
    target_audience TEXT,
    predicted_ctr FLOAT,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `variation_id` | UUID | No | Foreign key to variations |
| `variant_id` | TEXT | Yes | A, B, C, D, E (for A/B testing) |
| `title_text` | TEXT | No | The actual title |
| `hook_style` | TEXT | Yes | 'curiosity', 'revelation', 'emotional', etc. |
| `target_audience` | TEXT | Yes | Target demographic |
| `predicted_ctr` | FLOAT | Yes | Predicted click-through rate |
| `tags` | TEXT[] | Yes | Array of hashtags/tags |
| `created_at` | TIMESTAMP | No | Creation timestamp |
| `metadata` | JSONB | Yes | Additional data |

**Example Row**:

```json
{
  "id": "950e8400-e29b-41d4-a716-446655440003",
  "variation_id": "750e8400-e29b-41d4-a716-446655440002",
  "variant_id": "A",
  "title_text": "This CHANGES Everything 🤯",
  "hook_style": "curiosity",
  "target_audience": "gen_z",
  "predicted_ctr": 0.085,
  "tags": ["mindblown", "viral", "mustwatch"],
  "created_at": "2025-11-10T11:05:00Z"
}
```

---

### 5. music_tracks

Library of 40 music tracks for background audio.

**Purpose**: Manage music track library and usage.

```sql
CREATE TABLE music_tracks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    vibe TEXT,
    context_description TEXT,
    color TEXT,
    bpm INTEGER,
    duration FLOAT,
    is_available BOOLEAN DEFAULT true,
    times_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `name` | TEXT | No | Track name |
| `file_path` | TEXT | No | Path to audio file |
| `vibe` | TEXT | Yes | 'High energy', 'Relaxed', 'Inspiring', etc. |
| `context_description` | TEXT | Yes | When to use this track |
| `color` | TEXT | Yes | Hex color for UI (#FF5722) |
| `bpm` | INTEGER | Yes | Beats per minute |
| `duration` | FLOAT | Yes | Track duration in seconds |
| `is_available` | BOOLEAN | No | Is track available for use |
| `times_used` | INTEGER | No | Usage counter |
| `created_at` | TIMESTAMP | No | Creation timestamp |
| `metadata` | JSONB | Yes | Additional data |

**Example Row**:

```json
{
  "id": "850e8400-e29b-41d4-a716-446655440010",
  "name": "Energetic Beat 1",
  "file_path": "/music/energetic_1.mp3",
  "vibe": "High energy",
  "context_description": "Use for intense training moments",
  "color": "#FF5722",
  "bpm": 140,
  "duration": 180.0,
  "is_available": true,
  "times_used": 42,
  "created_at": "2025-10-01T00:00:00Z"
}
```

---

### 6. accounts

Social media accounts for multi-account posting.

**Purpose**: Manage multiple accounts across platforms.

```sql
CREATE TABLE accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    platform TEXT NOT NULL,
    username TEXT NOT NULL,
    account_type TEXT NOT NULL,
    posting_strategy TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `platform` | TEXT | No | 'tiktok', 'instagram', 'youtube' |
| `username` | TEXT | No | Account username |
| `account_type` | TEXT | No | 'fan', 'brand', 'watermark' |
| `posting_strategy` | TEXT | Yes | Description of strategy |
| `is_active` | BOOLEAN | No | Is account active |
| `created_at` | TIMESTAMP | No | Creation timestamp |
| `metadata` | JSONB | Yes | Additional data (API tokens, etc.) |

---

### 7. posts

Posted or scheduled clips.

**Purpose**: Track all posts across platforms.

```sql
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    variation_id UUID REFERENCES variations(id),
    account_id UUID REFERENCES accounts(id),
    title_variant_id UUID REFERENCES title_variants(id),
    post_url TEXT,
    posted_at TIMESTAMP,
    scheduled_for TIMESTAMP,
    status TEXT DEFAULT 'scheduled',
    performance_data JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `variation_id` | UUID | Yes | Foreign key to variations |
| `account_id` | UUID | Yes | Foreign key to accounts |
| `title_variant_id` | UUID | Yes | Which title was used |
| `post_url` | TEXT | Yes | URL of published post |
| `posted_at` | TIMESTAMP | Yes | When actually posted |
| `scheduled_for` | TIMESTAMP | Yes | When scheduled to post |
| `status` | TEXT | No | 'scheduled', 'posted', 'failed' |
| `performance_data` | JSONB | Yes | Real-time performance metrics |
| `created_at` | TIMESTAMP | No | Creation timestamp |
| `metadata` | JSONB | Yes | Additional data |

---

### 8. calendar_events

Calendar events for posting schedule.

**Purpose**: Integration with Google Calendar, iCloud Calendar.

```sql
CREATE TABLE calendar_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id),
    event_start TIMESTAMP NOT NULL,
    event_end TIMESTAMP,
    title TEXT,
    description TEXT,
    calendar_service TEXT,
    external_event_id TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB
);
```

---

### 9. performance_tracking

Performance metrics for posted clips.

**Purpose**: Track engagement and analytics over time.

```sql
CREATE TABLE performance_tracking (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    post_id UUID REFERENCES posts(id),
    tracked_at TIMESTAMP DEFAULT NOW(),
    views INTEGER,
    likes INTEGER,
    comments INTEGER,
    shares INTEGER,
    avg_view_duration FLOAT,
    completion_rate FLOAT,
    engagement_rate FLOAT,
    metadata JSONB
);
```

**Fields**:

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `id` | UUID | No | Primary key |
| `post_id` | UUID | Yes | Foreign key to posts |
| `tracked_at` | TIMESTAMP | No | When metrics were captured |
| `views` | INTEGER | Yes | View count |
| `likes` | INTEGER | Yes | Like count |
| `comments` | INTEGER | Yes | Comment count |
| `shares` | INTEGER | Yes | Share count |
| `avg_view_duration` | FLOAT | Yes | Average watch time in seconds |
| `completion_rate` | FLOAT | Yes | Percentage who watched to end |
| `engagement_rate` | FLOAT | Yes | (likes + comments + shares) / views |
| `metadata` | JSONB | Yes | Platform-specific metrics |

---

## Relationships

### Foreign Key Constraints

```sql
-- clips references videos
ALTER TABLE clips
  ADD CONSTRAINT fk_clips_video
  FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE;

-- variations references clips
ALTER TABLE variations
  ADD CONSTRAINT fk_variations_clip
  FOREIGN KEY (clip_id) REFERENCES clips(id) ON DELETE CASCADE;

-- title_variants references variations
ALTER TABLE title_variants
  ADD CONSTRAINT fk_title_variants_variation
  FOREIGN KEY (variation_id) REFERENCES variations(id) ON DELETE CASCADE;

-- posts references variations, accounts, title_variants
ALTER TABLE posts
  ADD CONSTRAINT fk_posts_variation
  FOREIGN KEY (variation_id) REFERENCES variations(id);

ALTER TABLE posts
  ADD CONSTRAINT fk_posts_account
  FOREIGN KEY (account_id) REFERENCES accounts(id);

ALTER TABLE posts
  ADD CONSTRAINT fk_posts_title_variant
  FOREIGN KEY (title_variant_id) REFERENCES title_variants(id);

-- performance_tracking references posts
ALTER TABLE performance_tracking
  ADD CONSTRAINT fk_performance_tracking_post
  FOREIGN KEY (post_id) REFERENCES posts(id);

-- calendar_events references posts
ALTER TABLE calendar_events
  ADD CONSTRAINT fk_calendar_events_post
  FOREIGN KEY (post_id) REFERENCES posts(id);
```

**Cascade Behavior**:
- Deleting a video deletes all its clips (CASCADE)
- Deleting a clip deletes all its variations (CASCADE)
- Deleting a variation deletes all its title variants (CASCADE)
- Posts, performance tracking, and calendar events remain on delete (orphaned)

---

## Indexes

Strategic indexes for query performance:

```sql
-- Videos
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_uploaded_at ON videos(uploaded_at DESC);

-- Clips
CREATE INDEX idx_clips_video_id ON clips(video_id);
CREATE INDEX idx_clips_hook_score ON clips(hook_score DESC);
CREATE INDEX idx_clips_status ON clips(status);

-- Variations
CREATE INDEX idx_variations_clip_id ON variations(clip_id);
CREATE INDEX idx_variations_created_at ON variations(created_at DESC);

-- Posts
CREATE INDEX idx_posts_account_id ON posts(account_id);
CREATE INDEX idx_posts_scheduled_for ON posts(scheduled_for);
CREATE INDEX idx_posts_status ON posts(status);
CREATE INDEX idx_posts_posted_at ON posts(posted_at DESC);

-- Performance Tracking
CREATE INDEX idx_performance_post_id ON performance_tracking(post_id);
CREATE INDEX idx_performance_tracked_at ON performance_tracking(tracked_at DESC);

-- JSONB GIN indexes for metadata searches
CREATE INDEX idx_videos_metadata ON videos USING GIN(metadata);
CREATE INDEX idx_clips_hook_score_data ON clips USING GIN(hook_score_data);
```

---

## Functions

### 1. get_top_performing_variations

Get top performing variations by engagement.

```sql
CREATE OR REPLACE FUNCTION get_top_performing_variations(limit_count INT DEFAULT 10)
RETURNS TABLE (
    variation_id UUID,
    avg_views FLOAT,
    avg_engagement FLOAT,
    total_posts BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        v.id as variation_id,
        AVG(pt.views)::FLOAT as avg_views,
        AVG(pt.engagement_rate)::FLOAT as avg_engagement,
        COUNT(p.id) as total_posts
    FROM variations v
    JOIN posts p ON p.variation_id = v.id
    JOIN performance_tracking pt ON pt.post_id = p.id
    GROUP BY v.id
    ORDER BY avg_engagement DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
```

**Usage**:

```sql
SELECT * FROM get_top_performing_variations(10);
```

---

### 2. get_account_performance

Get performance metrics for a specific account.

```sql
CREATE OR REPLACE FUNCTION get_account_performance(account_uuid UUID)
RETURNS TABLE (
    total_posts BIGINT,
    avg_views FLOAT,
    avg_engagement FLOAT,
    best_posting_time TIME
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(p.id) as total_posts,
        AVG(pt.views)::FLOAT as avg_views,
        AVG(pt.engagement_rate)::FLOAT as avg_engagement,
        (
            SELECT EXTRACT(HOUR FROM posted_at)::TIME
            FROM posts p2
            JOIN performance_tracking pt2 ON pt2.post_id = p2.id
            WHERE p2.account_id = account_uuid
            GROUP BY EXTRACT(HOUR FROM posted_at)
            ORDER BY AVG(pt2.engagement_rate) DESC
            LIMIT 1
        ) as best_posting_time
    FROM posts p
    JOIN performance_tracking pt ON pt.post_id = p.id
    WHERE p.account_id = account_uuid;
END;
$$ LANGUAGE plpgsql;
```

**Usage**:

```sql
SELECT * FROM get_account_performance('550e8400-e29b-41d4-a716-446655440100');
```

---

## Query Patterns

### Get all clips for a video with high hook scores

```sql
SELECT
    c.id,
    c.start_time,
    c.end_time,
    c.duration,
    c.hook_score,
    c.transcript
FROM clips c
WHERE c.video_id = '550e8400-e29b-41d4-a716-446655440000'
  AND c.hook_score > 7.5
  AND c.status = 'approved'
ORDER BY c.hook_score DESC
LIMIT 50;
```

### Get all variations for a clip

```sql
SELECT
    v.id,
    v.variation_type,
    v.reframe_style,
    v.video_path,
    m.name as music_track_name
FROM variations v
LEFT JOIN music_tracks m ON v.music_track_id = m.id
WHERE v.clip_id = '650e8400-e29b-41d4-a716-446655440001'
ORDER BY v.variation_type, v.reframe_style;
```

### Get scheduled posts for today

```sql
SELECT
    p.id,
    p.scheduled_for,
    a.username,
    a.platform,
    tv.title_text,
    v.video_path
FROM posts p
JOIN accounts a ON p.account_id = a.id
JOIN title_variants tv ON p.title_variant_id = tv.id
JOIN variations var ON p.variation_id = var.id
WHERE p.status = 'scheduled'
  AND DATE(p.scheduled_for) = CURRENT_DATE
ORDER BY p.scheduled_for;
```

### A/B test results for titles

```sql
SELECT
    tv.variant_id,
    tv.title_text,
    tv.predicted_ctr,
    COUNT(p.id) as times_used,
    AVG(pt.views) as avg_views,
    AVG(pt.engagement_rate) as avg_engagement
FROM title_variants tv
JOIN posts p ON p.title_variant_id = tv.id
JOIN performance_tracking pt ON pt.post_id = p.id
WHERE tv.variation_id = '750e8400-e29b-41d4-a716-446655440002'
GROUP BY tv.id, tv.variant_id, tv.title_text, tv.predicted_ctr
ORDER BY avg_engagement DESC;
```

### Most used music tracks

```sql
SELECT
    m.name,
    m.vibe,
    m.times_used,
    COUNT(v.id) as variation_count,
    AVG(pt.engagement_rate) as avg_engagement
FROM music_tracks m
LEFT JOIN variations v ON v.music_track_id = m.id
LEFT JOIN posts p ON p.variation_id = v.id
LEFT JOIN performance_tracking pt ON pt.post_id = p.id
WHERE m.is_available = true
GROUP BY m.id, m.name, m.vibe, m.times_used
ORDER BY m.times_used DESC
LIMIT 10;
```

---

## Migration Strategy

### Initial Setup

```sql
-- Run the schema file
psql -U postgres -d clipfactory -f database/schemas/schema.sql
```

### Future Migrations

Use migration tools like:
- **Alembic** (Python)
- **Flyway** (Java/cross-platform)
- **Manual SQL scripts** with version tracking

**Migration Naming Convention**:
```
V001__initial_schema.sql
V002__add_title_variants_table.sql
V003__add_performance_indexes.sql
```

### Backup Strategy

```bash
# Full database backup
pg_dump -U postgres clipfactory > backup_$(date +%Y%m%d).sql

# Restore
psql -U postgres -d clipfactory < backup_20251110.sql
```

### Data Seeding

Seed music tracks:

```sql
-- See schema.sql for initial 5 tracks
-- Add 35 more to reach 40 total
INSERT INTO music_tracks (name, file_path, vibe, context_description, color, bpm) VALUES
('Energetic Beat 2', '/music/energetic_2.mp3', 'High energy', 'Workout intensity', '#FF5722', 145),
-- ... 34 more
```

---

## Performance Optimization

### Vacuum and Analyze

```sql
-- Regular maintenance
VACUUM ANALYZE videos;
VACUUM ANALYZE clips;
VACUUM ANALYZE variations;
VACUUM ANALYZE posts;
VACUUM ANALYZE performance_tracking;
```

### Partition Large Tables

For performance_tracking (grows quickly):

```sql
-- Partition by month
CREATE TABLE performance_tracking_2025_11 PARTITION OF performance_tracking
FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
```

### Connection Pooling

Use **PgBouncer** or **connection pooling in application**:

```python
# SQLAlchemy example
from sqlalchemy import create_engine
engine = create_engine(
    'postgresql://user:pass@localhost/clipfactory',
    pool_size=20,
    max_overflow=40
)
```

---

## Security Considerations

1. **Never store API keys in database** - Use environment variables
2. **Encrypt sensitive metadata** - Use `pgcrypto` extension
3. **Use connection encryption** - SSL/TLS for database connections
4. **Limit user permissions** - Principle of least privilege
5. **Audit logging** - Track sensitive operations

---

## Monitoring Queries

### Database size

```sql
SELECT pg_size_pretty(pg_database_size('clipfactory'));
```

### Table sizes

```sql
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Active connections

```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'clipfactory';
```

---

**Schema Version**: 1.0.0
**Last Updated**: 2025-11-10

For schema questions, see [ARCHITECTURE.md](ARCHITECTURE.md) or open a GitHub issue.
