"""
Logging configuration for ClipFactory Backend.

Provides environment-aware logging:
- Development: Pretty console output with colors
- Production: Structured JSON logs for aggregation

Usage:
    from logging_config import setup_logging
    setup_logging()
"""
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    environment: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Parameters
    ----------
    environment : str, optional
        Environment name ("development", "production", "test").
        Defaults to ENVIRONMENT env var or "development".
    log_level : str, optional
        Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        Defaults to LOG_LEVEL env var or "INFO".
    log_file : str, optional
        Path to log file. If None and in production, uses "./logs/clipfactory.log"

    Examples
    --------
    >>> # Development mode with pretty console output
    >>> setup_logging(environment="development", log_level="DEBUG")

    >>> # Production mode with JSON logs
    >>> setup_logging(environment="production", log_level="INFO", log_file="/var/log/clipfactory.log")
    """
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    # Normalize environment name
    environment = environment.lower()

    if environment == "production":
        _setup_production_logging(log_level, log_file)
    elif environment == "test":
        _setup_test_logging(log_level)
    else:
        _setup_development_logging(log_level)

    # Set up module-specific log levels
    _configure_third_party_loggers()

    logging.info(f"Logging configured for {environment} environment at level {log_level}")


def _setup_development_logging(log_level: str) -> None:
    """Set up development logging with pretty console output."""
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "logging.Formatter",
                "format": "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s | %(name)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "colored",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(logging_config)


def _setup_production_logging(log_level: str, log_file: Optional[str] = None) -> None:
    """Set up production logging with JSON format for log aggregation."""
    if log_file is None:
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "clipfactory.log")

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d",
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "json",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file"],
        },
    }

    try:
        # Try to use JSON logger if available
        import pythonjsonlogger  # noqa
        logging.config.dictConfig(logging_config)
    except ImportError:
        # Fallback to simple format if pythonjsonlogger not installed
        logging.config.dictConfig({
            **logging_config,
            "handlers": {
                "console": logging_config["handlers"]["console"],
                "file": {
                    **logging_config["handlers"]["file"],
                    "formatter": "simple",
                }
            }
        })
        logging.warning("python-json-logger not installed. Using simple format. Install with: pip install python-json-logger")


def _setup_test_logging(log_level: str) -> None:
    """Set up test logging with minimal output."""
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(levelname)s | %(name)s - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "simple",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
    }

    logging.config.dictConfig(logging_config)


def _configure_third_party_loggers() -> None:
    """Configure log levels for third-party libraries."""
    # Reduce noise from verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)

    # Keep our loggers at INFO or higher
    logging.getLogger("clipfactory").setLevel(logging.INFO)
    logging.getLogger("clipsai").setLevel(logging.INFO)
    logging.getLogger("adlab").setLevel(logging.INFO)


def get_request_logger() -> logging.Logger:
    """
    Get logger for HTTP requests.

    Returns
    -------
    logging.Logger
        Logger configured for request/response logging
    """
    return logging.getLogger("clipfactory.http")


def get_processing_logger() -> logging.Logger:
    """
    Get logger for video processing operations.

    Returns
    -------
    logging.Logger
        Logger configured for processing operations
    """
    return logging.getLogger("clipfactory.processing")


def get_error_logger() -> logging.Logger:
    """
    Get logger for error tracking.

    Returns
    -------
    logging.Logger
        Logger configured for error tracking
    """
    return logging.getLogger("clipfactory.errors")


class RedactingFormatter(logging.Formatter):
    """
    Custom formatter that redacts sensitive information from logs.

    Redacts:
    - Passwords
    - API keys
    - Tokens
    - Database connection strings
    """

    REDACT_PATTERNS = [
        ("password", "[REDACTED_PASSWORD]"),
        ("api_key", "[REDACTED_API_KEY]"),
        ("apikey", "[REDACTED_API_KEY]"),
        ("token", "[REDACTED_TOKEN]"),
        ("secret", "[REDACTED_SECRET]"),
        ("authorization", "[REDACTED_AUTH]"),
    ]

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with redaction."""
        original = super().format(record)

        # Redact sensitive patterns
        redacted = original
        for pattern, replacement in self.REDACT_PATTERNS:
            import re
            # Match pattern=value or pattern: value or pattern="value"
            regex = rf'{pattern}["\']?\s*[:=]\s*["\']?([^\s"\']+)'
            redacted = re.sub(regex, f"{pattern}={replacement}", redacted, flags=re.IGNORECASE)

        return redacted


# Context filter for structured logging
class ContextFilter(logging.Filter):
    """
    Add contextual information to log records.

    Usage:
        logger.addFilter(ContextFilter())
        # Set context for current request/operation
        import contextvars
        request_id = contextvars.ContextVar('request_id', default=None)
        request_id.set('req_123')
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context variables to the record."""
        # Try to get context variables if available
        try:
            import contextvars
            request_id = contextvars.ContextVar('request_id', default=None)
            user_id = contextvars.ContextVar('user_id', default=None)
            video_id = contextvars.ContextVar('video_id', default=None)

            record.request_id = request_id.get()
            record.user_id = user_id.get()
            record.video_id = video_id.get()
        except Exception:
            pass

        return True


if __name__ == "__main__":
    # Test logging setup
    print("Testing development logging:")
    setup_logging(environment="development", log_level="DEBUG")

    logger = logging.getLogger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    print("\n\nTesting production logging:")
    setup_logging(environment="production", log_level="INFO")

    logger = logging.getLogger(__name__)
    logger.info("Production log message")
    logger.error("Production error message")
