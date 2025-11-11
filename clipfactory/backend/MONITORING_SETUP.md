# ClipFactory Backend - Monitoring & Error Handling Setup

This guide explains how to integrate the new monitoring, logging, and error handling system into the ClipFactory backend.

## Quick Start

### 1. Add Required Dependencies

Make sure these are in `requirements.txt`:

```text
sentry-sdk==1.40.0
python-json-logger==2.0.7  # Optional, for structured JSON logs
```

Install:
```bash
pip install sentry-sdk python-json-logger
```

### 2. Configure Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required for Sentry error tracking
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Environment configuration
ENVIRONMENT=development  # or 'production', 'staging'
LOG_LEVEL=INFO  # or 'DEBUG', 'WARNING', 'ERROR'

# Optional Sentry configuration
SENTRY_TRACES_SAMPLE_RATE=1.0  # 1.0 = 100% of traces
SENTRY_PROFILES_SAMPLE_RATE=1.0

# Performance monitoring
SLOW_REQUEST_THRESHOLD_MS=1000  # Log requests slower than this
```

### 3. Initialize in main.py

Add this to the top of `main.py` (after imports, before FastAPI app creation):

```python
# Import monitoring initialization
from init_monitoring import init_monitoring
from middleware import setup_middleware

# Initialize monitoring systems (Sentry, logging)
init_monitoring()
```

Then after creating the FastAPI app, add middleware:

```python
# Initialize FastAPI app
app = FastAPI(
    title="Clip Factory API",
    description="Viral clip generation and processing system",
    version="1.0.0",
    lifespan=lifespan  # Your existing lifespan function
)

# Setup middleware (request logging, error handling, performance monitoring)
setup_middleware(app)
```

### 4. Update Exception Handlers

Replace generic exception handlers with safe error responses:

#### Before (Unsafe - exposes internal errors to clients):
```python
@app.post("/api/phase1/upload")
async def upload_video(video: UploadFile = File(...)):
    try:
        # ... processing code ...
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))  # ❌ Exposes error to client
```

#### After (Safe - returns sanitized error with tracking ID):
```python
from clipsai.utils.error_handling import handle_specific_exceptions

@app.post("/api/phase1/upload")
async def upload_video(video: UploadFile = File(...)):
    error_mappings = {
        ValueError: {"status": 400, "message": "Invalid video format"},
        FileNotFoundError: {"status": 404, "message": "Video not found"},
        PermissionError: {"status": 403, "message": "Access denied"},
    }

    try:
        # ... processing code ...
        return {"status": "success"}
    except Exception as e:
        status, error_response = handle_specific_exceptions(
            e,
            error_mappings,
            context={"video_id": video_id}
        )
        return JSONResponse(status_code=status, content=error_response)
```

## What You Get

### 1. Automatic Request/Response Logging

Every API request is automatically logged with:
- Request ID (for tracing)
- Method, path, query params
- Client IP
- Response status code
- Duration in milliseconds
- Sanitized headers (sensitive data redacted)

Example log:
```
2025-01-10 12:34:56 | INFO | Request completed: POST /api/phase1/upload - 200
  request_id: 550e8400-e29b-41d4-a716-446655440000
  duration_ms: 1234.56
  client_ip: 192.168.1.1
```

### 2. Error Tracking with Sentry

All errors are automatically captured in Sentry with:
- Full stack traces
- Request context
- User context
- Custom tags

Errors get unique error IDs that users can reference when contacting support.

### 3. Safe Error Responses

Errors returned to clients never expose:
- Stack traces
- Internal file paths
- Database connection strings
- API keys or secrets

Instead, clients get:
```json
{
  "error": "Internal server error",
  "error_id": "ERR_20250110_abc123",
  "message": "An unexpected error occurred. Please contact support with this error ID.",
  "timestamp": "2025-01-10T12:34:56.789Z"
}
```

### 4. Performance Monitoring

Slow requests (>1000ms by default) are automatically logged:
```
2025-01-10 12:34:56 | WARNING | Slow request detected: POST /api/phase3/process/clip_123
  duration_ms: 2500.00
  threshold_ms: 1000
```

### 5. Structured Logging

All logs include structured context:
```python
from clipsai.utils.error_handling import log_operation

log_operation(
    "video_transcription",
    "started",
    context={"video_id": "vid_123", "user_id": "user_456"}
)
# ... do work ...
log_operation(
    "video_transcription",
    "completed",
    context={"video_id": "vid_123"},
    duration_ms=1500
)
```

## Advanced Usage

### Using ErrorContext for Automatic Logging

```python
from clipsai.utils.error_handling import ErrorContext

@app.post("/api/phase1/upload")
async def upload_video(video: UploadFile = File(...)):
    video_id = generate_video_id()

    with ErrorContext("video_upload", context={"video_id": video_id}):
        # Any errors here are automatically logged with full context
        # The operation is automatically tracked (started/completed/failed)
        save_video(video, video_id)
        start_processing(video_id)

    return {"video_id": video_id, "status": "uploaded"}
```

### Custom Error Mappings per Endpoint

```python
# For video processing
VIDEO_ERROR_MAPPINGS = {
    ValueError: {"status": 400, "message": "Invalid video format"},
    FileNotFoundError: {"status": 404, "message": "Video not found"},
    PermissionError: {"status": 403, "message": "Access denied"},
    TimeoutError: {"status": 408, "message": "Processing timeout"},
}

# For user operations
USER_ERROR_MAPPINGS = {
    ValueError: {"status": 400, "message": "Invalid user data"},
    KeyError: {"status": 404, "message": "User not found"},
    PermissionError: {"status": 403, "message": "Unauthorized access"},
}
```

### Sanitizing Error Messages

```python
from clipsai.utils.error_handling import sanitize_error_message

error_msg = "Connection failed: postgresql://user:password@db.example.com/mydb"
safe_msg = sanitize_error_message(error_msg)
# Output: "Connection failed: postgresql://[REDACTED]"
```

## Logging Best Practices

### 1. Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages (operations completed, state changes)
- **WARNING**: Something unexpected but handled (fallback used, deprecated feature)
- **ERROR**: Error that prevented an operation from completing
- **CRITICAL**: System-level failure

### 2. Include Context

```python
# ❌ Bad - no context
logger.error("Processing failed")

# ✅ Good - includes context
logger.error(
    "Video processing failed",
    exc_info=True,  # Include stack trace
    extra={
        "video_id": video_id,
        "user_id": user_id,
        "duration_attempted": 45.2,
        "error_stage": "transcription"
    }
)
```

### 3. Don't Log Sensitive Data

```python
# ❌ Bad - logs sensitive data
logger.info(f"User logged in: {username}, password: {password}")

# ✅ Good - no sensitive data
logger.info(f"User logged in", extra={"user_id": user_id})
```

## Testing

### Test Error Handling Locally

```python
# Start the server
python -m uvicorn clipfactory.backend.main:app --reload

# Test error response
curl -X POST http://localhost:8000/api/test-error
```

### View Logs

Development mode (pretty console output):
```bash
export ENVIRONMENT=development
python -m uvicorn clipfactory.backend.main:app --reload
```

Production mode (JSON logs):
```bash
export ENVIRONMENT=production
python -m uvicorn clipfactory.backend.main:app --reload
tail -f logs/clipfactory.log
```

## Monitoring in Production

### Sentry Dashboard

1. Go to https://sentry.io
2. View real-time errors
3. Filter by environment, user, endpoint
4. Track error trends over time

### Key Metrics to Monitor

- Error rate per endpoint
- Average response time
- Slow requests (>1s)
- 5xx error rate
- Most common errors

### Setting Up Alerts

Configure Sentry alerts for:
- Error rate > threshold
- New error types
- Performance degradation
- Specific error patterns

## Troubleshooting

### Sentry Not Working

Check:
1. `SENTRY_DSN` is set correctly
2. `sentry-sdk` is installed
3. Network connectivity to Sentry
4. Check logs for initialization errors

### Logs Not Appearing

Check:
1. `LOG_LEVEL` is set appropriately
2. Log directory exists and is writable
3. Logging initialized before other imports

### Performance Issues

If logging impacts performance:
1. Reduce `SENTRY_TRACES_SAMPLE_RATE` (e.g., 0.1 for 10%)
2. Increase `SLOW_REQUEST_THRESHOLD_MS`
3. Use async logging handlers

## File Structure

```
clipfactory/backend/
├── main.py                     # Main FastAPI app
├── init_monitoring.py          # Monitoring initialization
├── middleware.py               # Request/response middleware
├── logging_config.py           # Logging configuration
└── MONITORING_SETUP.md         # This file

clipsai/utils/
└── error_handling.py           # Error handling utilities
```

## Next Steps

1. ✅ Install dependencies
2. ✅ Set environment variables
3. ✅ Initialize monitoring in main.py
4. ✅ Update exception handlers
5. ✅ Test locally
6. ✅ Deploy to staging
7. ✅ Monitor Sentry dashboard
8. ✅ Configure alerts
9. ✅ Deploy to production

## Support

For questions or issues:
- Check Sentry documentation: https://docs.sentry.io/
- Review error handling utilities: `clipsai/utils/error_handling.py`
- Check logs: `logs/clipfactory.log`
