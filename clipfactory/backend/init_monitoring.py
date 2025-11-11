"""
Initialize monitoring, logging, and error tracking for ClipFactory backend.

This module sets up:
- Sentry for error tracking
- Structured logging
- Request/response logging middleware
"""
import os
import logging
import sys
from pathlib import Path

# Add clipsai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logging_config import setup_logging

logger = logging.getLogger(__name__)


def init_sentry():
    """
    Initialize Sentry SDK for error tracking.

    Reads configuration from environment variables:
    - SENTRY_DSN: Sentry project DSN
    - ENVIRONMENT: deployment environment (development/production)
    - SENTRY_TRACES_SAMPLE_RATE: traces sampling rate (default: 1.0)
    - SENTRY_PROFILES_SAMPLE_RATE: profiling sampling rate (default: 1.0)
    """
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_dsn = os.getenv("SENTRY_DSN")

        if not sentry_dsn:
            logger.warning("SENTRY_DSN not set - Sentry error tracking disabled")
            return False

        # Logging integration - capture ERROR and above
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Capture INFO and above as breadcrumbs
            event_level=logging.ERROR  # Send ERROR and above as events
        )

        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=os.getenv("ENVIRONMENT", "development"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "1.0")),
            enable_tracing=True,
            integrations=[
                FastApiIntegration(),
                StarletteIntegration(),
                SqlalchemyIntegration(),
                sentry_logging,
            ],
            # Set traces_sample_rate lower in production if needed
            # traces_sample_rate=0.1 if os.getenv("ENVIRONMENT") == "production" else 1.0,

            # Performance monitoring
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send PII by default

            # Custom options
            before_send=before_send_filter,
        )

        logger.info(f"Sentry initialized successfully for environment: {os.getenv('ENVIRONMENT', 'development')}")
        return True

    except ImportError:
        logger.error("sentry-sdk not installed. Install with: pip install sentry-sdk")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}", exc_info=True)
        return False


def before_send_filter(event, hint):
    """
    Filter and enrich Sentry events before sending.

    This function allows you to:
    - Filter out certain types of errors
    - Add custom tags or context
    - Sanitize sensitive data
    """
    # Skip HTTPException errors with status < 500 (client errors)
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        if exc_type.__name__ == 'HTTPException':
            # Only report 5xx errors to Sentry
            if hasattr(exc_value, 'status_code') and exc_value.status_code < 500:
                return None

    # Add custom tags
    event.setdefault('tags', {})
    event['tags']['service'] = 'clipfactory-backend'

    return event


def init_monitoring():
    """
    Initialize all monitoring systems.

    Call this once at application startup.
    """
    # Setup logging first
    setup_logging(
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

    # Initialize Sentry
    sentry_enabled = init_sentry()

    logger.info("Monitoring initialized", extra={
        "sentry_enabled": sentry_enabled,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "log_level": os.getenv("LOG_LEVEL", "INFO")
    })

    return sentry_enabled
