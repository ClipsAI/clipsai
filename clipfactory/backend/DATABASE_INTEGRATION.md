# Database Integration Documentation

## Overview

The ClipsAI backend now has a complete database integration layer using PostgreSQL and SQLAlchemy 2.0 with async support.

## Architecture

### Components

1. **models.py** - SQLAlchemy ORM models
   - Video
   - Clip
   - Variation
   - TitleVariant
   - MusicTrack
   - Account
   - Post
   - CalendarEvent
   - PerformanceTracking

2. **database.py** - Connection management
   - Async engine with connection pooling
   - Session factory
   - Database initialization
   - Health check
   - Startup/shutdown hooks

3. **crud.py** - CRUD operations
   - 25+ database operations
   - Async/await pattern
   - Proper error handling
   - Transaction management

4. **main.py** - API endpoints
   - Integrated with database
   - Dependency injection
   - Background tasks support

## Database Schema

The schema is defined in `/home/user/clipsai/clipfactory/database/schemas/schema.sql` and includes:

- **Videos**: Source videos uploaded for processing
- **Clips**: AI-selected clips from videos with hook scores
- **Variations**: 9 variations per clip (3 temporal × 3 reframe)
- **Title Variants**: A/B testing variants (A, B, C, D, E)
- **Music Tracks**: 40 music tracks for swapping
- **Accounts**: Multi-platform social media accounts
- **Posts**: Scheduled and posted clips
- **Calendar Events**: Posting schedule integration
- **Performance Tracking**: Analytics and A/B test results

## Setup

### 1. Environment Configuration

Copy the example environment file:

```bash
cd /home/user/clipsai/clipfactory
cp .env.example .env
```

Edit `.env` and set:

```bash
# Database credentials
POSTGRES_PASSWORD=your_secure_password_here
DATABASE_URL=postgresql+asyncpg://clipfactory:your_secure_password_here@postgres:5432/clipfactory

# Optional: Database pool configuration
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
DB_ECHO=false
```

### 2. Start Database

Using Docker Compose:

```bash
cd /home/user/clipsai/clipfactory
docker-compose up -d postgres
```

Or start PostgreSQL manually:

```bash
# Install PostgreSQL 15
# Create database and user
psql -U postgres
CREATE DATABASE clipfactory;
CREATE USER clipfactory WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE clipfactory TO clipfactory;
```

### 3. Test Database Connection

Run the test script:

```bash
cd /home/user/clipsai/clipfactory/backend
python test_db.py
```

This will:
- Initialize the database
- Create tables
- Run sample CRUD operations
- Verify connectivity

### 4. Start Backend

```bash
cd /home/user/clipsai/clipfactory/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
cd /home/user/clipsai/clipfactory
docker-compose up backend
```

## API Endpoints

### Phase 1: Council Deliberation

**POST /api/phase1/upload**
- Upload video for processing
- Stores metadata in database
- Returns video_id (UUID)

**GET /api/phase1/status/{video_id}**
- Query video processing status
- Returns clip count and statistics

**GET /api/phase1/clips/{video_id}**
- Get all clips for a video
- Sorted by hook score (highest first)

### Phase 2: Premiere Integration

**POST /api/phase2/export-xml/{video_id}**
- Generate Premiere Pro XML
- Updates video status to "exported"
- Returns download URL

**GET /api/download/xml/{video_id}**
- Download Premiere XML file

### Database Management

**GET /api/videos**
- List all videos
- Query params: skip, limit, status

**GET /api/statistics**
- System-wide statistics
- Total videos, clips, variations

**GET /api/music/list**
- List all music tracks
- Includes usage counts

**GET /api/health**
- Health check endpoint
- Includes database connection status

## CRUD Operations

### Videos
- `create_video()` - Create new video record
- `get_video_by_id()` - Get video by UUID
- `get_videos()` - List videos with filters
- `update_video_status()` - Update processing status
- `update_video_metadata()` - Update metadata
- `delete_video()` - Delete video (cascades to clips)

### Clips
- `create_clip()` - Create clip with hook score
- `get_clip_by_id()` - Get clip by UUID
- `get_clips_by_video()` - Get all clips for video
- `get_top_clips()` - Get highest scoring clips
- `update_clip_status()` - Update clip status
- `update_clip_hook_score()` - Update hook score

### Variations
- `create_variation()` - Create variation
- `get_variation_by_id()` - Get variation by UUID
- `get_variations_by_clip()` - Get all variations for clip
- `get_variations_by_video()` - Get all variations for video
- `update_variation_paths()` - Update file paths

### Title Variants
- `create_title_variant()` - Create title variant
- `get_title_variants_by_variation()` - Get all variants
- `create_multiple_title_variants()` - Bulk create

### Music Tracks
- `create_music_track()` - Add new track
- `get_all_music_tracks()` - List all tracks
- `get_music_track_by_id()` - Get track by UUID
- `increment_music_track_usage()` - Track usage

### Accounts
- `create_account()` - Add social media account
- `get_all_accounts()` - List accounts
- `get_account_by_id()` - Get account by UUID

### Posts
- `create_post()` - Schedule/record post
- `get_posts_by_account()` - Get account posts
- `get_scheduled_posts()` - Get scheduled posts
- `update_post_status()` - Update post status

### Performance Tracking
- `create_performance_tracking()` - Record metrics
- `get_performance_by_post()` - Get post analytics

### Statistics
- `get_video_statistics()` - System statistics
- `get_clip_statistics_by_video()` - Video statistics

## Database Schema Features

### Cascade Deletes
- Deleting a video cascades to all clips
- Deleting a clip cascades to all variations
- Deleting a variation cascades to title variants

### Indexes
- Videos: status
- Clips: video_id, hook_score (DESC)
- Variations: clip_id
- Posts: account_id, scheduled_for
- Performance: post_id

### Computed Columns
- Clip duration = end_time - start_time (stored)

### Analytics Functions
- `get_top_performing_variations()` - PostgreSQL function
- `get_account_performance()` - PostgreSQL function

## Connection Pooling

The database uses SQLAlchemy's QueuePool with:
- Pool size: 5 connections (default)
- Max overflow: 10 additional connections
- Pool timeout: 30 seconds
- Pool recycle: 3600 seconds (1 hour)
- Pre-ping: Enabled for connection health

## Error Handling

All CRUD operations include:
- Try/catch blocks
- Automatic session rollback on error
- Logging of errors
- Proper HTTP exception raising in API endpoints

## Transaction Management

The `get_db()` dependency:
- Creates a new session per request
- Auto-commits on success
- Auto-rolls back on error
- Closes session after request

## Testing

Run the test suite:

```bash
cd /home/user/clipsai/clipfactory/backend
python test_db.py
```

Expected output:
```
✓ Database initialized successfully
✓ Created video: [UUID]
✓ Retrieved video: test_video.mp4
✓ Created 2 clips
✓ Found 2 clips
✓ Created 2 music tracks
✓ Found 2 music tracks
✓ Statistics: ...
✓ All tests passed!
```

## Migration Strategy

For schema changes:

1. Update `schema.sql`
2. Update `models.py`
3. For production, use Alembic:

```bash
# Generate migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head
```

## Performance Tips

1. Use indexes for frequently queried fields
2. Use `selectinload()` for relationships to avoid N+1 queries
3. Use connection pooling (already configured)
4. Monitor slow queries with `DB_ECHO=true`
5. Use pagination for large result sets

## Security Notes

1. Never commit `.env` file
2. Use strong database passwords
3. Validate all user input
4. Use parameterized queries (SQLAlchemy handles this)
5. Enable SSL for production database connections

## Troubleshooting

### Connection Refused
```
ERROR: Database connection refused
```
Solution: Ensure PostgreSQL is running: `docker-compose up -d postgres`

### Import Errors
```
ERROR: No module named 'backend'
```
Solution: Run from backend directory or use: `python -m backend.main`

### Migration Conflicts
```
ERROR: Table already exists
```
Solution: Database tables already exist. Drop and recreate if needed.

### Pool Timeout
```
ERROR: QueuePool limit exceeded
```
Solution: Increase `DB_POOL_SIZE` or `DB_MAX_OVERFLOW` in `.env`

## Next Steps

1. Add Alembic for database migrations
2. Implement remaining background tasks
3. Add database backup strategy
4. Set up database replication for production
5. Add database monitoring (Prometheus/Grafana)
6. Implement connection pooling metrics

## Support

For issues or questions:
- Check logs: `docker-compose logs backend`
- Run health check: `curl http://localhost:8000/api/health`
- Test database: `python test_db.py`
