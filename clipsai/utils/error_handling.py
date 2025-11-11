"""
Error handling utilities for ClipsAI.
Provides consistent error handling patterns, safe API responses, and Sentry integration.
"""
import logging
import uuid
import traceback
from typing import Dict, Any, Optional, Type, Union
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_error_id() -> str:
    """
    Generate a unique error ID for tracking.

    Returns
    -------
    str
        Unique error ID in format: ERR_timestamp_uuid
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"ERR_{timestamp}_{short_uuid}"


def safe_api_error(
    exc: Exception,
    error_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    include_type: bool = False
) -> Dict[str, Any]:
    """
    Convert exception to safe API response.

    Logs the full exception with traceback and Sentry integration,
    but returns a sanitized error message safe for clients.

    Parameters
    ----------
    exc : Exception
        The exception to handle
    error_id : str, optional
        Custom error ID. If None, generates a new one.
    context : dict, optional
        Additional context to log (e.g., user_id, video_id)
    include_type : bool
        Whether to include exception type in response (default: False)

    Returns
    -------
    dict
        Safe error response with structure:
        {
            "error": "Internal server error",
            "error_id": "ERR_20250110_abc123",
            "message": "An unexpected error occurred. Please contact support.",
            "timestamp": "2025-01-10T12:34:56.789Z"
        }

    Examples
    --------
    >>> try:
    ...     process_video(video_id)
    ... except Exception as e:
    ...     error_response = safe_api_error(e, context={"video_id": video_id})
    ...     return JSONResponse(status_code=500, content=error_response)
    """
    if error_id is None:
        error_id = generate_error_id()

    # Log full error with traceback
    log_context = {
        "error_id": error_id,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    if context:
        log_context.update(context)

    logger.error(
        f"Error {error_id}: {type(exc).__name__}",
        exc_info=True,
        extra=log_context
    )

    # Capture in Sentry if available
    try:
        import sentry_sdk
        sentry_sdk.set_context("error_context", log_context)
        sentry_sdk.capture_exception(exc)
    except ImportError:
        pass  # Sentry not installed

    # Build safe response
    response = {
        "error": "Internal server error",
        "error_id": error_id,
        "message": "An unexpected error occurred. Please contact support with this error ID.",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    if include_type:
        response["error_type"] = type(exc).__name__

    return response


def handle_specific_exceptions(
    exc: Exception,
    error_mappings: Dict[Type[Exception], Dict[str, Any]],
    default_status: int = 500,
    context: Optional[Dict[str, Any]] = None
) -> tuple[int, Dict[str, Any]]:
    """
    Handle exceptions with specific mappings to HTTP status codes and messages.

    Parameters
    ----------
    exc : Exception
        The exception to handle
    error_mappings : dict
        Mapping of exception types to response config:
        {
            ValueError: {"status": 400, "message": "Invalid input"},
            FileNotFoundError: {"status": 404, "message": "Resource not found"}
        }
    default_status : int
        Default HTTP status code if exception type not mapped
    context : dict, optional
        Additional context to log

    Returns
    -------
    tuple[int, dict]
        (status_code, error_response)

    Examples
    --------
    >>> error_mappings = {
    ...     ValueError: {"status": 400, "message": "Invalid video format"},
    ...     FileNotFoundError: {"status": 404, "message": "Video not found"}
    ... }
    >>> try:
    ...     load_video(video_id)
    ... except Exception as e:
    ...     status, response = handle_specific_exceptions(e, error_mappings)
    ...     return JSONResponse(status_code=status, content=response)
    """
    error_id = generate_error_id()

    # Check if this is a known exception type
    for exc_type, config in error_mappings.items():
        if isinstance(exc, exc_type):
            status_code = config.get("status", default_status)
            message = config.get("message", str(exc))

            # Log with appropriate level
            if status_code >= 500:
                logger.error(
                    f"Error {error_id}: {type(exc).__name__}",
                    exc_info=True,
                    extra={"error_id": error_id, "context": context}
                )
            else:
                logger.warning(
                    f"Client error {error_id}: {type(exc).__name__}: {str(exc)}",
                    extra={"error_id": error_id, "context": context}
                )

            return status_code, {
                "error": message,
                "error_id": error_id,
                "details": str(exc) if status_code < 500 else None,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    # Unknown exception - use safe_api_error
    return default_status, safe_api_error(exc, error_id=error_id, context=context)


def log_operation(
    operation: str,
    status: str,
    context: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None
) -> None:
    """
    Log structured operation events.

    Parameters
    ----------
    operation : str
        Operation name (e.g., "video_upload", "clip_processing")
    status : str
        Operation status ("started", "completed", "failed")
    context : dict, optional
        Operation context (user_id, video_id, etc.)
    duration_ms : float, optional
        Operation duration in milliseconds

    Examples
    --------
    >>> log_operation("video_transcription", "started", {"video_id": "vid_123"})
    >>> # ... do work ...
    >>> log_operation("video_transcription", "completed", {"video_id": "vid_123"}, duration_ms=1500)
    """
    log_data = {
        "operation": operation,
        "status": status,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    if context:
        log_data.update(context)

    if duration_ms is not None:
        log_data["duration_ms"] = duration_ms

    if status == "failed":
        logger.error(f"Operation failed: {operation}", extra=log_data)
    elif status == "completed":
        logger.info(f"Operation completed: {operation}", extra=log_data)
    else:
        logger.info(f"Operation {status}: {operation}", extra=log_data)


class ErrorContext:
    """
    Context manager for operation error handling with automatic logging.

    Examples
    --------
    >>> with ErrorContext("video_processing", context={"video_id": "vid_123"}):
    ...     process_video("vid_123")
    """

    def __init__(
        self,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ):
        """
        Parameters
        ----------
        operation : str
            Operation name
        context : dict, optional
            Operation context
        reraise : bool
            Whether to reraise exceptions after logging (default: True)
        """
        self.operation = operation
        self.context = context or {}
        self.reraise = reraise
        self.start_time = None
        self.error_id = None

    def __enter__(self):
        self.start_time = datetime.now()
        log_operation(self.operation, "started", self.context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success
            duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000
            log_operation(self.operation, "completed", self.context, duration_ms)
            return False

        # Error occurred
        self.error_id = generate_error_id()
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        error_context = {
            **self.context,
            "error_id": self.error_id,
            "duration_ms": duration_ms
        }

        logger.error(
            f"Operation failed: {self.operation} - {type(exc_val).__name__}",
            exc_info=True,
            extra=error_context
        )

        # Capture in Sentry if available
        try:
            import sentry_sdk
            sentry_sdk.set_context("operation_context", error_context)
            sentry_sdk.capture_exception(exc_val)
        except ImportError:
            pass

        log_operation(self.operation, "failed", error_context, duration_ms)

        # Don't suppress exception if reraise=True
        return not self.reraise


def sanitize_error_message(message: str, replacements: Optional[Dict[str, str]] = None) -> str:
    """
    Sanitize error message to remove sensitive information.

    Parameters
    ----------
    message : str
        Error message to sanitize
    replacements : dict, optional
        Additional string replacements to apply

    Returns
    -------
    str
        Sanitized message

    Examples
    --------
    >>> sanitize_error_message("Failed to connect to db at postgresql://user:pass@host")
    'Failed to connect to db at [REDACTED]'
    """
    import re

    # Remove common sensitive patterns
    patterns = [
        (r'postgresql://[^@]+@[^\s]+', 'postgresql://[REDACTED]'),
        (r'mysql://[^@]+@[^\s]+', 'mysql://[REDACTED]'),
        (r'mongodb://[^@]+@[^\s]+', 'mongodb://[REDACTED]'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+', 'api_key=[REDACTED]'),
        (r'token["\']?\s*[:=]\s*["\']?[\w.-]+', 'token=[REDACTED]'),
        (r'password["\']?\s*[:=]\s*["\']?[^\s"\']+', 'password=[REDACTED]'),
        (r'/home/[\w/]+', '[PATH]'),
        (r'[a-zA-Z]:\\[\w\\]+', '[PATH]'),
    ]

    sanitized = message
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    if replacements:
        for old, new in replacements.items():
            sanitized = sanitized.replace(old, new)

    return sanitized
