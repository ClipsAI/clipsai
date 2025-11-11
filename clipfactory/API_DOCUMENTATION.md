# Clip Factory API Documentation

Complete API reference for the Clip Factory backend system.

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [API Endpoints](#api-endpoints)
  - [Phase 1: Council Deliberation](#phase-1-council-deliberation)
  - [Phase 2: Premiere Integration](#phase-2-premiere-integration)
  - [Phase 3: Matrix Processing](#phase-3-matrix-processing)
  - [Phase 4: Variation Generation](#phase-4-variation-generation)
  - [Phase 5: Distribution](#phase-5-distribution)
  - [Utility Endpoints](#utility-endpoints)

## Overview

The Clip Factory API provides a complete REST interface for viral clip generation and distribution. The API follows RESTful principles and returns JSON responses.

**API Version**: 1.0.0

**Content Types**:
- `application/json` - Standard JSON responses
- `multipart/form-data` - File uploads
- `application/xml` - Premiere Pro XML exports

## Base URL

**Development**: `http://localhost:8000`
**Production**: `https://api.clipfactory.io` (when deployed)

All endpoints are prefixed with `/api`:

```
http://localhost:8000/api/health
```

## Authentication

**Current**: No authentication (development)

**Future Implementation**:
- API Key authentication via header: `X-API-Key: your_api_key`
- JWT tokens for session management
- OAuth 2.0 for third-party integrations

Example authenticated request (future):

```bash
curl -H "X-API-Key: your_api_key" \
     http://localhost:8000/api/phase1/clips/video_id
```

## Error Handling

All errors follow a consistent JSON format:

```json
{
  "detail": "Error message describing what went wrong",
  "error_code": "ERROR_CODE",
  "timestamp": "2025-11-10T12:34:56Z"
}
```

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Common Error Codes

| Error Code | Description |
|------------|-------------|
| `FILE_TOO_LARGE` | Uploaded file exceeds size limit |
| `INVALID_FORMAT` | File format not supported |
| `VIDEO_NOT_FOUND` | Video ID doesn't exist |
| `PROCESSING_FAILED` | Video processing error |
| `TASK_TIMEOUT` | Processing took too long |

## Rate Limiting

**Current**: No rate limiting (development)

**Future**:
- 100 requests per minute per API key
- 1000 requests per hour per API key
- Upload size limit: 5GB per request

Rate limit headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699635600
```

## API Endpoints

---

## Phase 1: Council Deliberation

Council deliberation identifies clip-worthy moments from long-form videos.

### 1. Upload Video for Council

Upload a video for AI council analysis.

**Endpoint**: `POST /api/phase1/upload`

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video` | File | Yes | Video file (MP4, MOV, AVI) |

**Request Example**:

```bash
curl -X POST http://localhost:8000/api/phase1/upload \
  -F "video=@/path/to/video.mp4"
```

**Response** (200 OK):

```json
{
  "video_id": "vid_a1b2c3d4e5f6g7h8",
  "filename": "podcast_episode_123.mp4",
  "size": 2147483648,
  "duration": 7320.5,
  "status": "uploaded"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | string | Unique identifier for the video |
| `filename` | string | Original filename |
| `size` | integer | File size in bytes |
| `duration` | float | Video duration in seconds (null if not yet analyzed) |
| `status` | string | Processing status: `uploaded`, `processing`, `completed`, `failed` |

**Possible Errors**:
- `400`: Invalid file format
- `413`: File too large
- `500`: Upload failed

---

### 2. Get Council Status

Check the status of council deliberation.

**Endpoint**: `GET /api/phase1/status/{video_id}`

**Path Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_id` | string | Video identifier from upload |

**Request Example**:

```bash
curl http://localhost:8000/api/phase1/status/vid_a1b2c3d4e5f6g7h8
```

**Response** (200 OK):

```json
{
  "video_id": "vid_a1b2c3d4e5f6g7h8",
  "status": "processing",
  "clips_found": 342,
  "progress": 0.68,
  "estimated_completion": "2025-11-10T14:30:00Z"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `video_id` | string | Video identifier |
| `status` | string | `processing`, `completed`, `failed` |
| `clips_found` | integer | Number of clips identified so far |
| `progress` | float | Progress from 0.0 to 1.0 |
| `estimated_completion` | string | ISO 8601 timestamp (optional) |

---

### 3. Get Council Clips

Retrieve all clips identified by the council.

**Endpoint**: `GET /api/phase1/clips/{video_id}`

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `min_score` | float | No | 0.0 | Minimum hook score |
| `limit` | integer | No | 100 | Max clips to return |
| `offset` | integer | No | 0 | Pagination offset |

**Request Example**:

```bash
curl "http://localhost:8000/api/phase1/clips/vid_a1b2c3d4e5f6g7h8?min_score=7.0&limit=50"
```

**Response** (200 OK):

```json
{
  "video_id": "vid_a1b2c3d4e5f6g7h8",
  "clips": [
    {
      "clip_id": "clip_1a2b3c4d",
      "start_time": 123.5,
      "end_time": 175.2,
      "duration": 51.7,
      "hook_score": 8.5,
      "transcript": "This is the most important thing to understand about...",
      "hook_score_data": {
        "virality": 8.5,
        "value": 8.0,
        "specificity": 9.0,
        "actionability": 8.0
      }
    }
  ],
  "total": 500,
  "limit": 50,
  "offset": 0
}
```

---

## Phase 2: Premiere Integration

Export clips to Premiere Pro for manual editing.

### 4. Export Premiere XML

Generate Premiere Pro XML file for video editing.

**Endpoint**: `POST /api/phase2/export-xml/{video_id}`

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `clip_ids` | string[] | No | all | Comma-separated clip IDs to include |

**Request Example**:

```bash
curl -X POST "http://localhost:8000/api/phase2/export-xml/vid_a1b2c3d4e5f6g7h8?clip_ids=clip_1a2b,clip_2c3d"
```

**Response** (200 OK):

```json
{
  "video_id": "vid_a1b2c3d4e5f6g7h8",
  "xml_path": "/output/vid_a1b2c3d4e5f6g7h8_premiere.xml",
  "download_url": "/api/download/xml/vid_a1b2c3d4e5f6g7h8",
  "clips_included": 50,
  "generated_at": "2025-11-10T13:45:00Z"
}
```

---

### 5. Download XML File

Download the generated Premiere Pro XML file.

**Endpoint**: `GET /api/download/xml/{video_id}`

**Request Example**:

```bash
curl -O http://localhost:8000/api/download/xml/vid_a1b2c3d4e5f6g7h8
```

**Response**: XML file download

**Content-Type**: `application/xml`

**Possible Errors**:
- `404`: XML file not found (not yet generated)

---

### 6. Re-upload Edited Clips

Upload clips after Premiere Pro editing.

**Endpoint**: `POST /api/phase2/reupload`

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video_id` | string | Yes | Original video ID |
| `clips` | File[] | Yes | Array of edited clip files |

**Request Example**:

```bash
curl -X POST http://localhost:8000/api/phase2/reupload \
  -F "video_id=vid_a1b2c3d4e5f6g7h8" \
  -F "clips=@clip1.mp4" \
  -F "clips=@clip2.mp4" \
  -F "clips=@clip3.mp4"
```

**Response** (200 OK):

```json
{
  "video_id": "vid_a1b2c3d4e5f6g7h8",
  "clips_uploaded": 3,
  "clips": [
    {
      "clip_id": "clip_edited_1a2b",
      "filename": "clip1.mp4",
      "path": "/output/vid_a1b2c3d4e5f6g7h8/edited/clip_edited_1a2b_clip1.mp4"
    }
  ]
}
```

---

## Phase 3: Matrix Processing

Apply face tracking, reframing, and overlays.

### 7. Process Matrix

Process a clip through the matrix pipeline.

**Endpoint**: `POST /api/phase3/process/{clip_id}`

**Request Body** (JSON):

```json
{
  "canvas_styles": ["original", "flipped", "blurry_bg"],
  "watermark": true,
  "title_card": true
}
```

**Request Example**:

```bash
curl -X POST http://localhost:8000/api/phase3/process/clip_edited_1a2b \
  -H "Content-Type: application/json" \
  -d '{
    "canvas_styles": ["original", "flipped"],
    "watermark": true,
    "title_card": true
  }'
```

**Response** (202 Accepted):

```json
{
  "clip_id": "clip_edited_1a2b",
  "status": "processing",
  "message": "Matrix processing started",
  "task_id": "task_9f8e7d6c"
}
```

**Note**: This is an async operation. Use task_id to check status.

---

## Phase 4: Variation Generation

Generate multiple variations of each clip.

### 8. Generate Variations

Create all variations for a clip.

**Endpoint**: `POST /api/phase4/generate-variations`

**Request Body** (JSON):

```json
{
  "clip_id": "clip_edited_1a2b",
  "temporal_variations": ["base", "+4s", "+35s"],
  "reframe_styles": ["original", "flipped", "blurry_bg"],
  "title_style": "TT3",
  "music_id": "music_123abc"
}
```

**Request Example**:

```bash
curl -X POST http://localhost:8000/api/phase4/generate-variations \
  -H "Content-Type: application/json" \
  -d @variation_request.json
```

**Response** (200 OK):

```json
{
  "clip_id": "clip_edited_1a2b",
  "variations": [
    {
      "variation_id": "var_base_original",
      "temporal": "base",
      "reframe": "original",
      "status": "pending"
    },
    {
      "variation_id": "var_base_flipped",
      "temporal": "base",
      "reframe": "flipped",
      "status": "pending"
    }
  ],
  "total": 9
}
```

---

### 9. Generate Titles

Generate title variants for A/B testing.

**Endpoint**: `POST /api/phase4/generate-titles`

**Request Body** (JSON):

```json
{
  "transcript": "This is the most important thing...",
  "hook_score": 8.5,
  "duration": 51.7,
  "num_variants": 5
}
```

**Request Example**:

```bash
curl -X POST http://localhost:8000/api/phase4/generate-titles \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "This changes everything about how we think...",
    "hook_score": 9.0,
    "duration": 45.0,
    "num_variants": 5
  }'
```

**Response** (200 OK):

```json
{
  "titles": [
    {
      "variant_id": "A",
      "text": "This CHANGES Everything 🤯",
      "hook_style": "curiosity",
      "predicted_ctr": 0.085,
      "length": 25
    },
    {
      "variant_id": "B",
      "text": "You've Been Doing This WRONG 💀",
      "hook_style": "revelation",
      "predicted_ctr": 0.092,
      "length": 32
    },
    {
      "variant_id": "C",
      "text": "The SECRET Nobody Tells You 🔥",
      "hook_style": "emotional",
      "predicted_ctr": 0.088,
      "length": 31
    }
  ],
  "count": 5
}
```

---

## Phase 5: Distribution

Screenshot-to-title and posting management.

### 10. Screenshot to Title

Upload a screenshot and generate posting title.

**Endpoint**: `POST /api/phase5/screenshot-to-title`

**Content-Type**: `multipart/form-data`

**Request Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `screenshot` | File | Yes | - | Screenshot image (PNG, JPG) |
| `account_type` | string | No | "fan" | Account type: `fan`, `brand`, `watermark` |

**Request Example**:

```bash
curl -X POST http://localhost:8000/api/phase5/screenshot-to-title \
  -F "screenshot=@screenshot.png" \
  -F "account_type=fan"
```

**Response** (200 OK):

```json
{
  "title": "bro was STRUGGLING 💀",
  "account_type": "fan",
  "screenshot_analyzed": true,
  "tone": "casual",
  "engagement_score": 0.85
}
```

**Account Type Styles**:

| Type | Style | Example |
|------|-------|---------|
| `fan` | Casual, excited | "bro was STRUGGLING 💀" |
| `brand` | Professional | "Elite Training Techniques Revealed" |
| `watermark` | Commentary | "the way he crushed this tho 🔥" |

---

## Utility Endpoints

### 11. Health Check

Check API health and version.

**Endpoint**: `GET /api/health`

**Request Example**:

```bash
curl http://localhost:8000/api/health
```

**Response** (200 OK):

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-10T13:45:30Z",
  "uptime": 86400
}
```

---

### 12. List Music Tracks

Get available music tracks for variations.

**Endpoint**: `GET /api/music/list`

**Query Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vibe` | string | No | all | Filter by vibe |
| `limit` | integer | No | 40 | Max tracks to return |

**Request Example**:

```bash
curl "http://localhost:8000/api/music/list?vibe=energetic"
```

**Response** (200 OK):

```json
{
  "tracks": [
    {
      "id": "music_123abc",
      "name": "Energetic Beat 1",
      "vibe": "High energy",
      "context_description": "Use for intense training moments",
      "color": "#FF5722",
      "bpm": 140,
      "duration": 180.0,
      "times_used": 15
    }
  ],
  "total": 40
}
```

---

### 13. Get API Documentation

Get OpenAPI/Swagger documentation.

**Endpoint**: `GET /docs`

**Description**: Interactive API documentation (Swagger UI)

**URL**: `http://localhost:8000/docs`

**Alternative**: ReDoc format at `/redoc`

---

## Webhooks

**Future Feature**: Receive notifications when processing completes.

**Endpoint**: Configure via API settings (not yet implemented)

**Events**:
- `video.processed` - Council completed
- `variations.generated` - All variations ready
- `post.published` - Clip posted to platform

Example webhook payload:

```json
{
  "event": "variations.generated",
  "video_id": "vid_a1b2c3d4e5f6g7h8",
  "clip_id": "clip_edited_1a2b",
  "variations_count": 9,
  "timestamp": "2025-11-10T15:30:00Z"
}
```

---

## Data Models

### VideoUploadResponse

```typescript
{
  video_id: string;
  filename: string;
  size: number;
  duration?: number;
  status: "uploaded" | "processing" | "completed" | "failed";
}
```

### ClipMetadata

```typescript
{
  clip_id: string;
  start_time: number;
  end_time: number;
  duration: number;
  hook_score?: number;
  transcript?: string;
}
```

### VariationRequest

```typescript
{
  clip_id: string;
  temporal_variations: ("base" | "+4s" | "+35s")[];
  reframe_styles: ("original" | "flipped" | "blurry_bg")[];
  title_style: "TT3" | "AdLab";
  music_id?: string;
}
```

### TitleGenerationRequest

```typescript
{
  transcript: string;
  hook_score: number;
  duration: number;
  num_variants?: number; // default: 5
}
```

---

## Examples

### Complete Workflow Example

```bash
#!/bin/bash

# 1. Upload video
RESPONSE=$(curl -X POST http://localhost:8000/api/phase1/upload \
  -F "video=@podcast_ep_123.mp4")

VIDEO_ID=$(echo $RESPONSE | jq -r '.video_id')
echo "Uploaded video: $VIDEO_ID"

# 2. Wait for council to complete
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/phase1/status/$VIDEO_ID" | jq -r '.status')
  if [ "$STATUS" == "completed" ]; then
    break
  fi
  echo "Processing... $STATUS"
  sleep 10
done

# 3. Get clips
curl "http://localhost:8000/api/phase1/clips/$VIDEO_ID?min_score=7.5" | jq '.clips[0:10]'

# 4. Export to Premiere
curl -X POST "http://localhost:8000/api/phase2/export-xml/$VIDEO_ID" \
  -o "premiere_project.xml"

echo "Download XML and edit in Premiere Pro"
echo "Then re-upload edited clips..."

# 5. Generate variations (after re-upload)
CLIP_ID="clip_edited_1a2b"
curl -X POST http://localhost:8000/api/phase4/generate-variations \
  -H "Content-Type: application/json" \
  -d "{
    \"clip_id\": \"$CLIP_ID\",
    \"temporal_variations\": [\"base\", \"+4s\", \"+35s\"],
    \"reframe_styles\": [\"original\", \"flipped\", \"blurry_bg\"],
    \"title_style\": \"TT3\"
  }"

# 6. Generate titles
curl -X POST http://localhost:8000/api/phase4/generate-titles \
  -H "Content-Type: application/json" \
  -d "{
    \"transcript\": \"This is revolutionary...\",
    \"hook_score\": 8.5,
    \"duration\": 45.0,
    \"num_variants\": 5
  }" | jq '.titles'
```

---

## SDKs and Client Libraries

**Future**: Official SDKs planned for:
- Python
- JavaScript/TypeScript
- Go

Example Python SDK usage (future):

```python
from clipfactory import ClipFactoryClient

client = ClipFactoryClient(api_key="your_key")

# Upload and process
video = client.upload_video("podcast.mp4")
client.wait_for_processing(video.id)

# Get clips
clips = client.get_clips(video.id, min_score=7.5)

# Generate variations
for clip in clips[:10]:
    variations = client.generate_variations(
        clip.id,
        temporal=["base", "+4s", "+35s"],
        reframe=["original", "flipped"]
    )
```

---

## Changelog

### Version 1.0.0 (2025-11-10)

- Initial API release
- 13 endpoints across 5 phases
- Support for full clip factory pipeline

---

## Support

- **Documentation**: [Full Docs](../README.md)
- **Issues**: [GitHub Issues](https://github.com/ClipsAI/clipsai/issues)
- **Email**: support@clipsai.com

For API questions, tag issues with `api` label.
