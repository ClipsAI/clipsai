# Clip Factory - Complete Viral Clip Generation System

## Architecture Overview

```
clipfactory/
├── backend/                 # FastAPI backend server
│   ├── api/                # API routes
│   ├── services/           # Business logic
│   ├── models/             # Data models
│   └── utils/              # Utilities
├── frontend/               # React/Next.js frontend
│   ├── components/         # UI components
│   ├── pages/              # Page components
│   ├── hooks/              # Custom hooks
│   └── styles/             # CSS/styling
├── processing/             # Video processing pipeline
│   ├── council/            # Phase 1: AI council
│   ├── premiere/           # Phase 2: XML export
│   ├── matrix/             # Phase 3: Reframing
│   ├── variations/         # Phase 4: Variations
│   └── distribution/       # Phase 5: Posting
├── shared/                 # Shared utilities
│   ├── config/             # Configuration
│   ├── constants/          # Constants
│   └── types/              # Type definitions
├── database/               # Database schemas & migrations
│   ├── schemas/            # SQL schemas
│   └── migrations/         # Database migrations
└── tests/                  # Test suites
```

## Tech Stack

**Backend:**
- FastAPI (Python)
- Celery (async task queue)
- Redis (caching/queue)
- PostgreSQL (database)

**Frontend:**
- Next.js 14 (React framework)
- TypeScript
- Tailwind CSS
- shadcn/ui components

**Video Processing:**
- FFmpeg (video manipulation)
- ClipsAI (segmentation)
- OpenCV (face tracking)
- Whisper (transcription)

**AI/ML:**
- Anthropic Claude (title generation)
- OpenAI GPT (title variants)
- Google Gemini (cheap formatting)
- Kimi K2 (caption formatting)

## Installation

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install

# Start services
docker-compose up -d
```

## Workflow Phases

### Phase 1: Council Deliberation
- Upload 2-3 hour video
- AI council selects 500 moments
- Export horizontal clips

### Phase 2: Premiere Integration
- Export clips to XML format
- Wes edits in Premiere Pro
- Re-upload edited clips

### Phase 3: Matrix Processing
- Face tracking & auto-reframing
- Canvas rendering (3 styles)
- Apply watermarks
- Add title cards

### Phase 4: Variation Generation
- Create temporal variations (base, +4s, +35s)
- Generate 3 reframe styles each = 9 variations
- Add random frame offset
- Music selection & swapping
- Caption generation

### Phase 5: Distribution
- Screenshot → title generation
- Calendar scheduling
- Multi-account posting
- Performance tracking

## Running the System

```bash
# Start backend
cd backend
uvicorn main:app --reload

# Start frontend
cd frontend
npm run dev

# Process video
curl -X POST http://localhost:8000/api/process \
  -F "video=@video.mp4"
```
