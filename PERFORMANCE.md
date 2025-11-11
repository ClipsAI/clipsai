# Performance Guide

Optimization strategies and performance characteristics for ClipsAI and Clip Factory.

## Table of Contents

- [Overview](#overview)
- [Performance Characteristics](#performance-characteristics)
- [Bottlenecks](#bottlenecks)
- [Optimization Strategies](#optimization-strategies)
- [Model Caching](#model-caching)
- [Scaling Considerations](#scaling-considerations)
- [Benchmarking](#benchmarking)
- [Tuning Guide](#tuning-guide)
- [Monitoring](#monitoring)

## Overview

ClipsAI involves computationally intensive operations:
- Video transcription (WhisperX)
- Speaker diarization (Pyannote)
- Face detection and tracking
- Video encoding/decoding
- AI inference (Claude, GPT)

Performance optimization is critical for:
- Faster processing times
- Better user experience
- Cost efficiency
- Scalability

## Performance Characteristics

### Processing Times (Approximate)

Based on a typical 2-hour (7200s) podcast video:

| Operation | Time | Notes |
|-----------|------|-------|
| **Upload** | 2-5 min | Depends on bandwidth |
| **Transcription** | 15-30 min | GPU: 15 min, CPU: 30 min |
| **Diarization** | 10-20 min | GPU accelerated |
| **Clip Finding** | 1-2 min | Fast (NLP only) |
| **Face Tracking** | 5-10 min/clip | Per clip, parallelizable |
| **Video Rendering** | 2-5 min/variation | Per variation |
| **Title Generation** | 5-10 sec | API call latency |

**Total Time (Phase 1)**: ~30-60 minutes for 2-hour video
**Total Time (Full Pipeline)**: 2-4 hours for 50 clips with 9 variations each

### Resource Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 16GB
- GPU: Not required (slower)
- Disk: 100GB

**Recommended**:
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA with 8GB+ VRAM (Tesla T4, RTX 3060+)
- Disk: 500GB SSD

**Optimal**:
- CPU: 16+ cores
- RAM: 64GB+
- GPU: NVIDIA A100 (40GB VRAM)
- Disk: 1TB NVMe SSD

## Bottlenecks

### 1. Video Transcription

**Bottleneck**: WhisperX model inference

**Impact**: 20-40% of total processing time

**Factors**:
- Video length
- Audio quality
- Model size (tiny/base/small/medium/large)
- GPU availability

**Symptoms**:
- CPU/GPU at 100%
- High memory usage
- Slow progress

### 2. Speaker Diarization

**Bottleneck**: Pyannote model inference

**Impact**: 15-30% of total processing time

**Factors**:
- Number of speakers
- Audio overlap
- GPU availability

**Symptoms**:
- GPU memory pressure
- Slow speaker segmentation

### 3. Video I/O

**Bottleneck**: FFmpeg encoding/decoding

**Impact**: 10-20% of total processing time

**Factors**:
- Video resolution
- Codec efficiency
- Disk I/O speed
- CPU cores

**Symptoms**:
- High disk I/O wait
- CPU bottlenecked on encoding

### 4. Face Tracking

**Bottleneck**: Per-frame face detection

**Impact**: 5-10 min per clip

**Factors**:
- Video resolution
- Frame rate
- Face detection model
- GPU availability

**Symptoms**:
- Slow frame-by-frame processing
- GPU at 100%

### 5. Database Operations

**Bottleneck**: Large result sets, complex queries

**Impact**: <5% of total processing time

**Factors**:
- Number of variations (9 per clip × 500 clips = 4500 rows)
- Query complexity
- Index usage

**Symptoms**:
- Slow API responses
- High database CPU

## Optimization Strategies

### Transcription Optimization

#### Use GPU Acceleration

```python
from clipsai import Transcriber

# GPU-accelerated transcription
transcriber = Transcriber(device="cuda")  # Uses GPU
transcription = transcriber.transcribe(
    audio_file_path="/path/to/video.mp4",
    compute_type="float16"  # Faster than float32
)
```

**Speedup**: 2-3x faster than CPU

#### Choose Appropriate Model Size

```python
# Fast but less accurate
transcriber = Transcriber(model="tiny")  # ~5x faster

# Balanced (recommended)
transcriber = Transcriber(model="base")  # Default

# Most accurate but slower
transcriber = Transcriber(model="large")  # 3-4x slower
```

#### Batch Processing

Process multiple videos in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def process_video(video_path):
    transcriber = Transcriber(device="cuda")
    return transcriber.transcribe(video_path)

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_video, path) for path in video_paths]
    results = [f.result() for f in futures]
```

### Diarization Optimization

#### Use GPU

```python
from clipsai import resize

crops = resize(
    video_file_path="/path/to/video.mp4",
    pyannote_auth_token="token",
    aspect_ratio=(9, 16),
    device="cuda"  # GPU acceleration
)
```

#### Cache Diarization Results

```python
import pickle

def get_or_compute_diarization(video_path, cache_path):
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Compute
    result = diarize(video_path)

    # Cache
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result
```

### Video Processing Optimization

#### Hardware Acceleration

Use hardware-accelerated encoding:

```python
import ffmpeg

# NVIDIA NVENC
stream = ffmpeg.input('input.mp4')
stream = ffmpeg.output(stream, 'output.mp4', vcodec='h264_nvenc', preset='fast')
ffmpeg.run(stream)

# Intel QuickSync
stream = ffmpeg.output(stream, 'output.mp4', vcodec='h264_qsv')

# macOS VideoToolbox
stream = ffmpeg.output(stream, 'output.mp4', vcodec='h264_videotoolbox')
```

**Speedup**: 3-5x faster encoding

#### Optimize Codec Settings

```python
# Fast encoding preset
ffmpeg.output(
    stream,
    'output.mp4',
    vcodec='libx264',
    preset='ultrafast',  # Fastest (larger files)
    crf=23  # Quality (18-28, lower = better)
)

# Balanced
ffmpeg.output(
    stream,
    'output.mp4',
    vcodec='libx264',
    preset='medium',  # Default
    crf=23
)
```

#### Reduce Resolution During Processing

```python
# Resize video before processing
stream = ffmpeg.input('input.mp4')
stream = ffmpeg.filter(stream, 'scale', width=1280, height=-1)  # 720p
stream = ffmpeg.output(stream, 'processed.mp4')
```

### Face Tracking Optimization

#### Sample Frames

Don't process every frame:

```python
# Process every Nth frame
fps = 30
sample_rate = 5  # Process every 5th frame

for i, frame in enumerate(video_frames):
    if i % sample_rate == 0:
        faces = detect_faces(frame)
        # Interpolate between keyframes
```

**Speedup**: 5x faster (with sample_rate=5)

#### Use Lighter Models

```python
# Fast but less accurate
detector = FaceDetector(model="mtcnn_fast")

# Balanced
detector = FaceDetector(model="mtcnn")

# Most accurate
detector = FaceDetector(model="retinaface")
```

#### Parallelize Clip Processing

```python
from multiprocessing import Pool

def process_clip(clip):
    return track_faces(clip)

with Pool(processes=4) as pool:
    results = pool.map(process_clip, clips)
```

### Database Optimization

#### Use Indexes

Already defined in schema:

```sql
CREATE INDEX idx_clips_hook_score ON clips(hook_score DESC);
CREATE INDEX idx_variations_clip_id ON variations(clip_id);
```

#### Batch Inserts

```python
# Bad: Insert one at a time
for variation in variations:
    db.execute("INSERT INTO variations (...) VALUES (...)", variation)

# Good: Batch insert
values = [(v.clip_id, v.type, v.style) for v in variations]
db.executemany("INSERT INTO variations (...) VALUES (...)", values)
```

**Speedup**: 10-100x faster

#### Connection Pooling

```python
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://user:pass@localhost/clipfactory',
    pool_size=20,  # Number of connections
    max_overflow=40,  # Extra connections when needed
    pool_pre_ping=True  # Verify connections
)
```

#### Query Optimization

```python
# Bad: N+1 query problem
clips = db.query(Clip).all()
for clip in clips:
    variations = db.query(Variation).filter_by(clip_id=clip.id).all()

# Good: Join query
clips = db.query(Clip).options(joinedload(Clip.variations)).all()
```

### API Performance

#### Async Endpoints

```python
from fastapi import FastAPI

app = FastAPI()

# Synchronous (blocks)
@app.get("/slow")
def slow_endpoint():
    result = expensive_operation()
    return result

# Asynchronous (non-blocking)
@app.get("/fast")
async def fast_endpoint():
    result = await expensive_operation_async()
    return result
```

#### Background Tasks

```python
from fastapi import BackgroundTasks

@app.post("/process")
async def process_video(background_tasks: BackgroundTasks):
    video_id = save_video()

    # Run in background
    background_tasks.add_task(process_video_task, video_id)

    return {"video_id": video_id, "status": "processing"}
```

#### Caching

```python
from functools import lru_cache
import redis

# In-memory cache
@lru_cache(maxsize=128)
def get_music_tracks():
    return db.query(MusicTrack).all()

# Redis cache
redis_client = redis.Redis(host='localhost', port=6379)

def get_cached_clips(video_id):
    cache_key = f"clips:{video_id}"

    # Try cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Compute
    clips = db.query(Clip).filter_by(video_id=video_id).all()

    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, json.dumps(clips))

    return clips
```

## Model Caching

### PyTorch Model Caching

Models are automatically cached after first load:

```python
import torch

# First load: Downloads and caches
model = torch.hub.load('repo', 'model')

# Subsequent loads: Uses cache
model = torch.hub.load('repo', 'model')
```

**Cache Location**:
- Linux: `~/.cache/torch/hub/`
- macOS: `~/Library/Caches/torch/hub/`
- Windows: `%LOCALAPPDATA%\torch\hub\`

### WhisperX Model Caching

```python
from clipsai import Transcriber

# First use: Downloads model (~1-5GB depending on size)
transcriber = Transcriber(model="base")

# Cached at: ~/.cache/whisperx/
```

**Pre-download Models**:

```bash
# Download all models ahead of time
python -c "from clipsai import Transcriber; \
  Transcriber(model='tiny'); \
  Transcriber(model='base'); \
  Transcriber(model='small');"
```

### Pyannote Model Caching

```bash
# Cache location
~/.cache/torch/pyannote/
```

### Shared Model Cache

For multi-user systems, use shared cache:

```bash
export TORCH_HOME=/shared/cache/torch
export HF_HOME=/shared/cache/huggingface
```

## Scaling Considerations

### Horizontal Scaling

#### Load Balancer

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

#### Stateless Backend

Ensure backend is stateless:
- Store files in shared storage (S3, NFS)
- Use Redis for session state
- Database for persistent state

#### Multiple Workers

```bash
# Gunicorn with multiple workers
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

#### Celery Distributed Tasks

```python
# Scale Celery workers across machines
celery -A tasks worker --concurrency=8 --loglevel=info
```

### Vertical Scaling

#### More CPU Cores

Benefits:
- Faster FFmpeg encoding
- More concurrent requests
- Parallel clip processing

#### More RAM

Benefits:
- Larger model cache
- More database connections
- More concurrent uploads

#### Better GPU

Benefits:
- Faster transcription (2-5x)
- Faster diarization (2-3x)
- Faster face detection (3-5x)

GPU Recommendations:
- Budget: NVIDIA Tesla T4 (16GB)
- Recommended: NVIDIA RTX 3090 (24GB)
- Optimal: NVIDIA A100 (40GB/80GB)

#### SSD Storage

Benefits:
- Faster video I/O (3-5x)
- Faster database queries
- Faster model loading

Upgrade path:
- HDD → SATA SSD: 3x faster
- SATA SSD → NVMe SSD: 2x faster

## Benchmarking

### Transcription Benchmark

```python
import time
from clipsai import Transcriber

def benchmark_transcription(video_path, model="base", device="cuda"):
    transcriber = Transcriber(model=model, device=device)

    start = time.time()
    transcription = transcriber.transcribe(video_path)
    elapsed = time.time() - start

    duration = transcription.duration
    rtf = elapsed / duration  # Real-time factor

    print(f"Model: {model}, Device: {device}")
    print(f"Video Duration: {duration:.1f}s")
    print(f"Processing Time: {elapsed:.1f}s")
    print(f"Real-Time Factor: {rtf:.2f}x")
    print(f"Speed: {1/rtf:.2f}x faster than real-time")

# Run benchmarks
benchmark_transcription("video.mp4", model="tiny", device="cuda")
benchmark_transcription("video.mp4", model="base", device="cuda")
benchmark_transcription("video.mp4", model="base", device="cpu")
```

**Example Results** (1-hour video):

| Model | Device | Time | RTF | Speed |
|-------|--------|------|-----|-------|
| tiny | GPU | 5 min | 0.08x | 12x faster |
| base | GPU | 8 min | 0.13x | 7.5x faster |
| base | CPU | 25 min | 0.42x | 2.4x faster |
| large | GPU | 20 min | 0.33x | 3x faster |

### Database Benchmark

```python
import time

def benchmark_query(query_func, iterations=100):
    start = time.time()
    for _ in range(iterations):
        query_func()
    elapsed = time.time() - start

    avg_time = elapsed / iterations
    qps = iterations / elapsed

    print(f"Total Time: {elapsed:.2f}s")
    print(f"Avg Query Time: {avg_time*1000:.1f}ms")
    print(f"Queries Per Second: {qps:.1f}")

# Benchmark
benchmark_query(lambda: db.query(Clip).filter(Clip.hook_score > 7.5).all())
```

### End-to-End Benchmark

```bash
#!/bin/bash

VIDEO_PATH="/path/to/2hour_podcast.mp4"
START=$(date +%s)

# Phase 1: Upload and council
curl -X POST http://localhost:8000/api/phase1/upload \
  -F "video=@$VIDEO_PATH" | jq -r '.video_id' > video_id.txt

VIDEO_ID=$(cat video_id.txt)

# Wait for processing
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/phase1/status/$VIDEO_ID" | jq -r '.status')
  if [ "$STATUS" == "completed" ]; then
    break
  fi
  sleep 10
done

END=$(date +%s)
ELAPSED=$((END - START))

echo "Total Processing Time: $ELAPSED seconds ($(($ELAPSED/60)) minutes)"
```

## Tuning Guide

### WhisperX Tuning

```python
from clipsai import Transcriber

transcriber = Transcriber(
    model="base",          # tiny/base/small/medium/large
    device="cuda",         # cuda/cpu
    compute_type="float16", # float16/float32 (float16 faster on GPU)
    batch_size=16,         # Larger = faster but more memory
    language="en",         # Specify language for better performance
)
```

### FFmpeg Tuning

```bash
# Fast encoding
ffmpeg -i input.mp4 -c:v libx264 -preset ultrafast output.mp4

# Balanced
ffmpeg -i input.mp4 -c:v libx264 -preset medium output.mp4

# High quality (slow)
ffmpeg -i input.mp4 -c:v libx264 -preset slow output.mp4

# Hardware acceleration (NVIDIA)
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset fast output.mp4
```

### PostgreSQL Tuning

```sql
-- postgresql.conf

# Memory settings
shared_buffers = 4GB          # 25% of RAM
effective_cache_size = 12GB   # 75% of RAM
maintenance_work_mem = 1GB
work_mem = 64MB

# Connection settings
max_connections = 200

# Query planner
random_page_cost = 1.1        # For SSD (default 4.0 for HDD)

# Write-ahead log
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

Apply changes:

```bash
sudo systemctl restart postgresql
```

## Monitoring

### System Monitoring

```bash
# CPU usage
htop

# GPU usage
nvidia-smi -l 1

# Disk I/O
iostat -x 1

# Network
iftop
```

### Application Monitoring

```python
import logging
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        logging.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

@monitor_performance
def process_video(video_path):
    # Processing logic
    pass
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge
from fastapi import FastAPI

app = FastAPI()

# Metrics
video_uploads = Counter('video_uploads_total', 'Total video uploads')
processing_time = Histogram('video_processing_seconds', 'Video processing time')
active_tasks = Gauge('active_tasks', 'Currently active tasks')

@app.post("/upload")
async def upload():
    video_uploads.inc()
    with processing_time.time():
        # Process video
        pass
```

### Grafana Dashboards

Create dashboards for:
- Request rate (requests/sec)
- Response time (p50, p95, p99)
- Error rate
- CPU/Memory usage
- GPU utilization
- Queue length

---

## Performance Checklist

### Development
- [ ] Use GPU for transcription and diarization
- [ ] Choose appropriate model sizes
- [ ] Enable model caching
- [ ] Profile code to identify bottlenecks

### Production
- [ ] Enable hardware acceleration (NVENC, QuickSync)
- [ ] Configure connection pooling
- [ ] Set up Redis caching
- [ ] Enable Celery for async tasks
- [ ] Configure load balancing
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Optimize database queries
- [ ] Use CDN for static assets

---

**Last Updated**: 2025-11-10

For performance questions, open a GitHub issue with the `performance` label.
