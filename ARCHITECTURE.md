# ClipsAI Architecture

This document describes the architecture of ClipsAI, including both the core library and the Clip Factory system.

## Table of Contents

- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Core Components](#core-components)
- [Clip Factory System](#clip-factory-system)
- [Data Flow](#data-flow)
- [Technology Stack](#technology-stack)
- [Design Decisions](#design-decisions)
- [Performance Considerations](#performance-considerations)

## System Overview

ClipsAI consists of two major systems:

1. **ClipsAI Core Library**: A Python library for automatic video clip generation
2. **Clip Factory**: An enterprise-grade web application for viral clip production at scale

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ClipsAI Ecosystem                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────┐         ┌─────────────────────────┐   │
│  │   ClipsAI Core     │         │     Clip Factory        │   │
│  │   (Python Lib)     │◄────────┤   (Web Application)     │   │
│  └────────────────────┘         └─────────────────────────┘   │
│          │                                   │                  │
│          ├─ Transcription                    ├─ Backend API    │
│          ├─ Clip Finding                     ├─ Frontend UI    │
│          ├─ Video Resizing                   ├─ Processing     │
│          └─ Speaker Diarization              ├─ Database       │
│                                               └─ Distribution   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Diagram

### ClipsAI Core Library

```
┌──────────────────────────────────────────────────────────────┐
│                      ClipsAI Core Library                     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐      ┌──────────────┐    ┌──────────────┐ │
│  │             │      │              │    │              │ │
│  │ Transcriber │─────▶│  ClipFinder  │───▶│   Resizer    │ │
│  │             │      │              │    │              │ │
│  └──────┬──────┘      └──────┬───────┘    └──────┬───────┘ │
│         │                    │                    │         │
│         │                    │                    │         │
│  ┌──────▼──────┐      ┌──────▼───────┐    ┌──────▼───────┐ │
│  │  WhisperX   │      │  Segmenter   │    │  Pyannote    │ │
│  │ Integration │      │    Logic     │    │  Diarizer    │ │
│  └─────────────┘      └──────────────┘    └──────────────┘ │
│                                                               │
│  Supporting Modules:                                         │
│  ├─ Media Utilities (FFmpeg wrapper)                        │
│  ├─ Face Detection (MediaPipe, FaceNet)                     │
│  ├─ Scene Detection (PySceneDetect)                         │
│  └─ Crop Calculator (Reframing logic)                       │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Clip Factory System

```
┌────────────────────────────────────────────────────────────────────┐
│                        Clip Factory System                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐     ┌──────────────────┐                    │
│  │  Frontend (Next)  │────▶│  Backend (FastAPI)│                    │
│  │  - Upload UI      │◄────│  - REST API      │                    │
│  │  - Dashboard      │     │  - File Handling │                    │
│  │  - Preview        │     │  - Task Queue    │                    │
│  └──────────────────┘     └────────┬─────────┘                    │
│                                     │                               │
│                                     │                               │
│  ┌──────────────────────────────────▼──────────────────────────┐  │
│  │              Processing Pipeline (5 Phases)                  │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                               │  │
│  │  Phase 1: Council    ─▶  Phase 2: Premiere  ─▶  Phase 3:    │  │
│  │  Deliberation            Integration          Matrix         │  │
│  │  [AI Council]            [XML Export]          [Reframing]   │  │
│  │                                                               │  │
│  │  Phase 4: Variations ─▶  Phase 5: Distribution               │  │
│  │  [9 per clip]            [Multi-platform]                    │  │
│  │                                                               │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                     PostgreSQL Database                       │  │
│  │  - Videos, Clips, Variations                                  │  │
│  │  - Titles, Accounts, Posts                                    │  │
│  │  - Performance Tracking                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ClipsAI Core Library

The foundation library that handles automatic video-to-clips conversion.

#### Transcriber

- **Purpose**: Convert audio to timestamped text transcription
- **Technology**: WhisperX (wrapper around OpenAI Whisper)
- **Input**: Video/audio file path
- **Output**: Word-level transcription with timestamps
- **Key Features**:
  - Word-level alignment
  - Multiple language support
  - GPU acceleration
  - Batch processing

#### ClipFinder

- **Purpose**: Identify clip-worthy segments from transcription
- **Technology**: NLP + ML segmentation
- **Input**: Transcription object
- **Output**: List of Clip objects with start/end times
- **Algorithm**:
  1. Sentence segmentation
  2. Semantic coherence scoring
  3. Length optimization (30-90s default)
  4. Hook score calculation
  5. Overlap prevention

#### Resizer

- **Purpose**: Reframe videos from 16:9 to 9:16 (or other ratios)
- **Technology**: Pyannote diarization + Face tracking
- **Input**: Video path, auth token, aspect ratio
- **Output**: Crops object with reframing coordinates
- **Process**:
  1. Speaker diarization (identify speakers)
  2. Face detection per frame
  3. Dynamic crop calculation
  4. Smooth transitions between speakers
  5. Center-weighted reframing

#### Media Utilities

- FFmpeg integration for video manipulation
- File validation and conversion
- Codec handling
- Format normalization

### 2. Clip Factory Components

#### A. Backend (FastAPI)

**Structure:**
```
backend/
├── main.py              # FastAPI app entry point
├── routes/              # API route handlers
│   ├── phase1.py        # Council deliberation
│   ├── phase2.py        # Premiere integration
│   ├── phase3.py        # Matrix processing
│   ├── phase4.py        # Variations
│   └── phase5.py        # Distribution
├── services/            # Business logic
│   ├── video_service.py
│   ├── clip_service.py
│   └── title_service.py
├── models/              # Pydantic models
├── database/            # Database layer
└── utils/               # Utilities
```

**Responsibilities:**
- REST API endpoints (13 total)
- File upload/download handling
- Background task management (Celery)
- Database operations
- Integration with ClipsAI Core

#### B. Frontend (Next.js 14)

**Structure:**
```
frontend/
├── app/                 # Next.js app directory
│   ├── page.tsx         # Home page
│   ├── upload/          # Upload interface
│   ├── dashboard/       # Clip management
│   └── preview/         # Video preview
├── components/          # React components
│   ├── UploadInterface.tsx
│   ├── VariationGenerator.tsx
│   └── PostingHelper.tsx
├── hooks/               # Custom React hooks
├── lib/                 # Utilities
└── styles/              # Tailwind CSS
```

**Features:**
- Drag-and-drop video upload
- Real-time processing status
- Video preview with timeline
- Variation generation UI
- Title A/B testing interface
- Posting calendar

#### C. Processing Pipeline

Five-phase processing system for viral clip generation:

**Phase 1: Council Deliberation**
- Upload 2-3 hour source video
- AI council analyzes content
- Selects 500 potential moments
- Exports horizontal clips

**Phase 2: Premiere Integration**
- Export clips to Premiere Pro XML
- Manual editing by creator
- Re-upload approved clips
- Tracking edited vs original

**Phase 3: Matrix Processing**
- Face tracking across frames
- Canvas rendering (3 styles)
- Watermark overlay
- Title card generation

**Phase 4: Variation Generation**
- Temporal variations: base, +4s, +35s
- Reframe styles: original, flipped, blurry_bg
- Creates 9 variations per clip (3x3)
- Random frame offsets
- Music track selection
- Caption generation

**Phase 5: Distribution**
- Screenshot-based title generation
- Calendar scheduling
- Multi-account posting
- Performance tracking

#### D. Database Layer

PostgreSQL database with 9 main tables:

- **videos**: Source video metadata
- **clips**: AI-selected segments
- **variations**: Generated clip variations
- **title_variants**: A/B testing titles
- **music_tracks**: 40 background tracks
- **accounts**: Social media accounts
- **posts**: Published/scheduled content
- **calendar_events**: Posting schedule
- **performance_tracking**: Analytics data

See [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for detailed schema.

## Data Flow

### Core Library Flow

```
┌─────────┐     ┌──────────────┐     ┌────────────┐     ┌─────────┐
│  Video  │────▶│ Transcriber  │────▶│ ClipFinder │────▶│  Clips  │
│  File   │     │  (WhisperX)  │     │   (NLP)    │     │  List   │
└─────────┘     └──────────────┘     └────────────┘     └─────────┘
                                                               │
                                                               ▼
┌─────────┐     ┌──────────────┐                      ┌────────────┐
│Reframed │◀────│   Resizer    │◀─────────────────────│   Clip     │
│  Video  │     │  (Pyannote)  │                      │   Object   │
└─────────┘     └──────────────┘                      └────────────┘
```

### Clip Factory Flow

```
┌──────────────┐
│ Source Video │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Phase 1: Council │ ─── 500 moments identified
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Phase 2: Premiere│ ─── Manual editing, 50-100 clips approved
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Phase 3: Matrix  │ ─── 3 canvas styles per clip
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│Phase 4: Variations│ ─── 9 variations per clip (3 temporal × 3 reframe)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│Phase 5: Distribute│ ─── Multi-platform posting with tracking
└──────────────────┘
```

### Request Flow (API)

```
Client (Browser)
      │
      │ HTTP Request
      ▼
┌─────────────┐
│   FastAPI   │
│   Backend   │
└─────┬───────┘
      │
      ├──▶ Synchronous: Immediate response
      │    (upload, status check)
      │
      └──▶ Asynchronous: Background task
           (video processing, AI inference)
                    │
                    ▼
           ┌────────────────┐
           │ Celery Worker  │
           └────────┬───────┘
                    │
                    ├──▶ ClipsAI Core
                    ├──▶ AI APIs (Claude, GPT)
                    └──▶ Database
```

## Technology Stack

### Core Library

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Transcription | WhisperX, Whisper | Speech-to-text with alignment |
| Diarization | Pyannote Audio | Speaker identification |
| Face Detection | MediaPipe, FaceNet | Face tracking |
| Scene Detection | PySceneDetect | Scene boundary detection |
| Video Processing | FFmpeg, OpenCV | Video manipulation |
| ML Framework | PyTorch | Neural network inference |
| NLP | NLTK, Sentence Transformers | Text analysis |

### Clip Factory

#### Backend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Web Framework | FastAPI | REST API |
| Task Queue | Celery | Async processing |
| Message Broker | Redis | Task queue backend |
| Database | PostgreSQL 15+ | Data persistence |
| ORM | SQLAlchemy (optional) | Database abstraction |
| Validation | Pydantic | Request/response models |

#### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | Next.js 14 | React framework |
| Language | TypeScript | Type safety |
| Styling | Tailwind CSS | Utility-first CSS |
| Components | shadcn/ui, Radix UI | UI components |
| State Management | React hooks, SWR | Client state |
| HTTP Client | Axios | API requests |
| Animation | Framer Motion | UI animations |

#### AI/ML Services

| Service | Purpose |
|---------|---------|
| Anthropic Claude 3.5 Haiku | Title generation, cheap/fast |
| OpenAI GPT-4 | High-quality title variants |
| Claude Vision | Screenshot analysis |

#### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker | Service isolation |
| Orchestration | Docker Compose | Multi-service management |
| Reverse Proxy | Nginx (optional) | Load balancing |
| Storage | Local filesystem | Video/clip storage |

## Design Decisions

### 1. WhisperX over Whisper

**Decision**: Use WhisperX instead of base Whisper

**Rationale**:
- Word-level timestamps (required for precise clipping)
- Better alignment accuracy
- Forced phoneme alignment
- Active development and maintenance

**Trade-offs**:
- Additional dependency
- Slightly slower than base Whisper
- More complex installation

### 2. FastAPI over Flask/Django

**Decision**: Use FastAPI for Clip Factory backend

**Rationale**:
- Native async support (crucial for video processing)
- Automatic OpenAPI documentation
- Pydantic validation built-in
- High performance (comparable to Node.js)
- Type hints everywhere
- Modern Python features

**Trade-offs**:
- Smaller ecosystem than Django
- Less mature than Flask

### 3. Next.js 14 over Create React App

**Decision**: Use Next.js 14 for frontend

**Rationale**:
- App directory (modern React patterns)
- Built-in API routes (simplified architecture)
- Server components for better performance
- Image optimization out of the box
- Great developer experience

**Trade-offs**:
- More opinionated than CRA
- Learning curve for Next.js specifics

### 4. PostgreSQL over MongoDB

**Decision**: Use PostgreSQL for database

**Rationale**:
- ACID compliance (important for financial/performance data)
- Rich query capabilities (analytics)
- JSONB support (flexible metadata)
- Proven reliability
- Strong consistency

**Trade-offs**:
- Requires schema management
- Less flexible than NoSQL

### 5. Multi-Phase Processing Pipeline

**Decision**: Split processing into 5 distinct phases

**Rationale**:
- Human-in-the-loop editing (Phase 2)
- Quality control at each stage
- Resource optimization (process only approved clips)
- Clear separation of concerns
- Easier debugging and monitoring

**Trade-offs**:
- More complex workflow
- Longer total processing time
- More state management

### 6. Variation Matrix (3x3=9)

**Decision**: Generate 9 variations per clip

**Rationale**:
- Temporal diversity (base, +4s, +35s)
- Visual diversity (3 reframe styles)
- A/B testing opportunities
- Platform-specific optimization
- Maximizes content from each clip

**Trade-offs**:
- 9x storage requirements
- Longer processing time
- More complexity in tracking

## Performance Considerations

### Transcription Performance

- **GPU Acceleration**: WhisperX supports CUDA
- **Batch Processing**: Process multiple videos in parallel
- **Model Selection**: Trade-off between accuracy and speed
  - Tiny: Fast, less accurate
  - Base: Balanced
  - Large: Slow, most accurate

**Bottleneck**: GPU memory for large models

### Video Processing Performance

- **FFmpeg Optimization**:
  - Hardware acceleration (NVENC, VideoToolbox)
  - Efficient codecs (H.264, H.265)
  - Proper preset selection

- **Face Tracking**:
  - Frame sampling (process every Nth frame)
  - Resolution downscaling
  - Model optimization

**Bottleneck**: I/O for large video files

### Database Performance

- **Indexes**: Strategic indexing on frequently queried fields
- **Connection Pooling**: Reuse database connections
- **Batch Operations**: Bulk inserts for variations
- **JSONB Indexing**: GIN indexes on metadata fields

**Bottleneck**: Disk I/O for analytics queries

### API Performance

- **Async I/O**: FastAPI async for all endpoints
- **Background Tasks**: Celery for long-running operations
- **Caching**: Redis for frequently accessed data
- **CDN**: Static assets and thumbnails

**Bottleneck**: Video upload bandwidth

### Scaling Strategy

**Horizontal Scaling**:
- Stateless backend (easy to replicate)
- Load balancer (Nginx)
- Multiple Celery workers
- Read replicas for database

**Vertical Scaling**:
- GPU for faster transcription
- More CPU cores for video encoding
- SSD for faster I/O
- More RAM for caching

### Monitoring & Optimization

- **Metrics**:
  - Processing time per phase
  - Task queue length
  - API response times
  - Database query performance

- **Profiling**:
  - Python cProfile for CPU bottlenecks
  - line_profiler for line-by-line analysis
  - memory_profiler for memory usage

- **Logging**:
  - Structured logging (JSON)
  - Centralized log aggregation
  - Error tracking (Sentry)

## Security Architecture

See [SECURITY.md](SECURITY.md) for detailed security considerations.

## Future Architecture Considerations

### Microservices Migration

Current: Monolithic backend
Future: Separate services for:
- Video processing
- AI/ML inference
- API gateway
- Database service

### Cloud Storage

Current: Local filesystem
Future: S3/Cloud Storage for:
- Scalability
- Durability
- CDN integration

### Real-time Updates

Current: Polling for status
Future: WebSocket for:
- Live progress updates
- Real-time notifications
- Collaborative editing

### Machine Learning Pipeline

Future: Custom ML models for:
- Better clip selection
- Virality prediction
- Automatic A/B testing
- Performance forecasting

---

**Last Updated**: 2025-11-10

For questions about architecture decisions, open a GitHub Discussion.
