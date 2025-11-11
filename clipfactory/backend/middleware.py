"""
FastAPI middleware for ClipFactory backend.

Provides:
- Request/response logging
- Error handling
- Performance monitoring
- Request ID tracking
"""
import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import sys
from pathlib import Path

# Add clipsai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clipsai.utils.error_handling import safe_api_error, generate_error_id

logger = logging.getLogger("clipfactory.http")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses.

    Logs:
    - Request method, path, client IP
    - Response status code
    - Request duration
    - Error details (if any)

    Redacts sensitive headers (Authorization, Cookie, etc.)
    """

    SENSITIVE_HEADERS = {
        "authorization",
        "cookie",
        "x-api-key",
        "x-auth-token",
    }

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Start timer
        start_time = time.time()

        # Log request
        self._log_request(request, request_id)

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            self._log_response(request, response, request_id, duration_ms)

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": self._get_client_ip(request),
                    "duration_ms": duration_ms,
                    "error": str(e),
                }
            )

            # Return safe error response
            error_id = generate_error_id()
            error_response = safe_api_error(e, error_id=error_id, context={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            })

            return JSONResponse(
                status_code=500,
                content=error_response,
                headers={"X-Request-ID": request_id, "X-Error-ID": error_id}
            )

    def _log_request(self, request: Request, request_id: str):
        """Log incoming request."""
        headers = self._sanitize_headers(dict(request.headers))

        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", "unknown"),
                "headers": headers,
            }
        )

    def _log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        duration_ms: float
    ):
        """Log response."""
        log_level = logging.INFO if response.status_code < 400 else logging.WARNING

        logger.log(
            log_level,
            f"Request completed: {request.method} {request.url.path} - {response.status_code}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "client_ip": self._get_client_ip(request),
            }
        )

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, handling proxies."""
        # Check for forwarded IP (behind proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fallback to direct client
        if request.client:
            return request.client.host

        return "unknown"

    def _sanitize_headers(self, headers: dict) -> dict:
        """Redact sensitive headers for logging."""
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in self.SENSITIVE_HEADERS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor API performance.

    Tracks:
    - Slow requests (> threshold)
    - Request count per endpoint
    - Average response time
    """

    def __init__(self, app: ASGIApp, slow_request_threshold_ms: float = 1000):
        super().__init__(app)
        self.slow_request_threshold_ms = slow_request_threshold_ms

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        response = await call_next(request)

        duration_ms = (time.time() - start_time) * 1000

        # Log slow requests
        if duration_ms > self.slow_request_threshold_ms:
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path}",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": round(duration_ms, 2),
                    "threshold_ms": self.slow_request_threshold_ms,
                }
            )

        # Add performance header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.

    Catches any unhandled exceptions and returns safe error responses.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Generate error ID
            error_id = generate_error_id()

            # Get request ID if available
            request_id = getattr(request.state, "request_id", "unknown")

            # Log error
            logger.error(
                f"Unhandled exception: {type(e).__name__}",
                exc_info=True,
                extra={
                    "error_id": error_id,
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                }
            )

            # Send to Sentry
            try:
                import sentry_sdk
                sentry_sdk.capture_exception(e)
            except ImportError:
                pass

            # Return safe error response
            error_response = safe_api_error(e, error_id=error_id)

            return JSONResponse(
                status_code=500,
                content=error_response,
                headers={
                    "X-Request-ID": request_id,
                    "X-Error-ID": error_id
                }
            )


def setup_middleware(app):
    """
    Add all middleware to the FastAPI app.

    Call this during app initialization.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance
    """
    # Add middleware in reverse order of execution
    # (middleware added first is executed last)

    # Error handling (innermost - catches everything)
    app.add_middleware(ErrorHandlingMiddleware)

    # Performance monitoring
    slow_threshold = float(os.getenv("SLOW_REQUEST_THRESHOLD_MS", "1000"))
    app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold_ms=slow_threshold)

    # Request logging (outermost - logs everything)
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("Middleware configured successfully")


# For importing
import os
