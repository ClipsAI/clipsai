# Clip Factory Setup Guide

Complete setup instructions for the viral clip factory system.

## Prerequisites

- Docker & Docker Compose
- FFmpeg installed locally (for development)
- Python 3.11+
- Node.js 18+
- PostgreSQL 15+ (or use Docker)
- Redis (or use Docker)

## Quick Start (Docker)

### 1. Clone and Configure

```bash
# Navigate to clipfactory directory
cd clipfactory

# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

### 2. Start Services

```bash
# Start all services with Docker Compose
docker-compose up -d

# Check status
docker-compose ps
```

### 3. Initialize Database

```bash
# Database will auto-initialize from schema.sql
# Verify it's working:
docker-compose exec postgres psql -U clipfactory -d clipfactory -c "\dt"
```

### 4. Access Services

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Manual Setup (Development)

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://clipfactory:clipfactory_password@localhost:5432/clipfactory"
export REDIS_URL="redis://localhost:6379/0"
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# Run migrations (create database first)
createdb clipfactory
psql clipfactory < ../database/schemas/schema.sql

# Start backend
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Set environment
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

### Celery Worker

```bash
cd backend

# Start Celery worker
celery -A tasks worker --loglevel=info
```

## Configuration

### API Keys Required

1. **Anthropic Claude**
   - Get key from: https://console.anthropic.com/
   - Used for: Title generation, hook scoring
   - Set: `ANTHROPIC_API_KEY`

2. **OpenAI GPT**
   - Get key from: https://platform.openai.com/
   - Used for: Title variants (GPT-5 when available)
   - Set: `OPENAI_API_KEY`

3. **Google Gemini** (Optional)
   - Get key from: https://ai.google.dev/
   - Used for: Cheap caption formatting
   - Set: `GOOGLE_GEMINI_API_KEY`

### Music Tracks

Add your 40 music tracks:

```bash
# Create music directory
mkdir -p backend/static/music

# Add your tracks (MP3 format recommended)
cp /path/to/your/tracks/*.mp3 backend/static/music/

# Update database
psql clipfactory < scripts/seed_music.sql
```

### Accounts Setup

Configure your social media accounts:

```sql
-- Add accounts to database
INSERT INTO accounts (platform, username, account_type, posting_strategy) VALUES
('tiktok', '@barcleartv', 'fan', 'casual_reactions'),
('instagram', '@cleanbarclips', 'brand', 'professional'),
('youtube', '@barclearshorts', 'brand', 'educational');
```

## Usage Workflow

### Phase 1: Upload Video

1. Open frontend: http://localhost:3000
2. Drag & drop 2-3 hour video
3. Wait for council deliberation (10-20 minutes)
4. Review 500 selected clips

### Phase 2: Premiere Editing

1. Click "Export to Premiere XML"
2. Download XML file
3. Import into Premiere Pro
4. Edit clips (add human touch for TOS)
5. Export edited clips
6. Re-upload via "Re-upload Edited Clips"

### Phase 3: Matrix Processing

1. System auto-processes each clip
2. Applies face tracking
3. Renders 3 canvas styles (original, flipped, blurry_bg)
4. Adds watermarks and title cards

### Phase 4: Variation Generation

1. Open "Variation Generator" UI
2. Voice dictate title ideas
3. Select music from 40 tracks
4. Choose title style (TT³ or AdLab)
5. Generate 9 variations (3 temporal × 3 reframe)
6. Captions auto-generated

### Phase 5: Distribution

1. Download all variations
2. Use "Posting Helper"
3. Upload screenshot of variation
4. Select account type
5. Get AI-generated title
6. Copy and schedule post

## Database Management

### Backup

```bash
# Backup database
docker-compose exec postgres pg_dump -U clipfactory clipfactory > backup.sql

# Restore
docker-compose exec -T postgres psql -U clipfactory clipfactory < backup.sql
```

### View Analytics

```bash
# Connect to database
docker-compose exec postgres psql -U clipfactory clipfactory

# Get top performing variations
SELECT * FROM get_top_performing_variations(10);

# Get account performance
SELECT * FROM get_account_performance('account-uuid-here');
```

## Monitoring

### Logs

```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend
docker-compose logs -f celery_worker

# View last 100 lines
docker-compose logs --tail=100 backend
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/api/health

# Check database
docker-compose exec postgres pg_isready

# Check Redis
docker-compose exec redis redis-cli ping
```

## Troubleshooting

### FFmpeg Not Found

```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Database Connection Issues

```bash
# Check database is running
docker-compose ps postgres

# Restart database
docker-compose restart postgres

# Check logs
docker-compose logs postgres
```

### Celery Tasks Not Running

```bash
# Check Redis is running
docker-compose ps redis

# Restart Celery worker
docker-compose restart celery_worker

# Check worker logs
docker-compose logs celery_worker
```

### Frontend Can't Connect to Backend

```bash
# Verify backend is running
curl http://localhost:8000/api/health

# Check CORS settings in backend/main.py

# Verify environment variable
cat frontend/.env.local
```

## Production Deployment

### 1. Build Images

```bash
# Build all images
docker-compose build

# Push to registry
docker tag clipfactory_backend your-registry/clipfactory-backend
docker push your-registry/clipfactory-backend
```

### 2. Environment Variables

Set production environment variables:

```bash
export DATABASE_URL="your-production-db-url"
export REDIS_URL="your-production-redis-url"
export ANTHROPIC_API_KEY="your-production-key"
```

### 3. Deploy

```bash
# Using docker-compose on production server
docker-compose -f docker-compose.prod.yml up -d

# Or use Kubernetes/AWS ECS/etc.
```

### 4. Set Up Monitoring

```bash
# Add Sentry for error tracking
export SENTRY_DSN="your-sentry-dsn"

# Set up Prometheus
docker-compose up -d prometheus grafana
```

## Performance Optimization

### Video Processing

```bash
# Use GPU for faster processing
docker-compose -f docker-compose.gpu.yml up -d

# Adjust worker concurrency
celery -A tasks worker --concurrency=8
```

### Database

```bash
# Add indexes if needed
psql clipfactory < scripts/add_indexes.sql

# Vacuum database
docker-compose exec postgres vacuumdb -U clipfactory clipfactory
```

## Support

For issues or questions:

1. Check logs: `docker-compose logs -f`
2. Review API docs: http://localhost:8000/docs
3. Check database: `docker-compose exec postgres psql -U clipfactory`

## Next Steps

1. Configure your 40 music tracks
2. Add your social media accounts
3. Upload your first video
4. Test the complete workflow
5. Set up posting calendar
6. Start tracking performance

Happy clipping! 🎬🚀
