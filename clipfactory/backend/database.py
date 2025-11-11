"""
Database Connection Management for Clip Factory
Async SQLAlchemy 2.0 style with connection pooling
"""
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text, create_engine
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
import logging
from pathlib import Path

from .models import Base

logger = logging.getLogger(__name__)

# Database configuration from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://clipfactory:clipfactory@localhost:5432/clipfactory"
)

# For synchronous operations (schema creation)
SYNC_DATABASE_URL = DATABASE_URL.replace("+asyncpg", "")

# Connection pool settings
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "false").lower() == "true",
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    pool_pre_ping=True,  # Enable connection health checks
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Export for use in background tasks
async_session_maker = AsyncSessionLocal


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection for database sessions.
    Usage in FastAPI:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context():
    """
    Context manager for database sessions.
    Usage:
        async with get_db_context() as db:
            result = await db.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database context error: {e}")
            raise
        finally:
            await session.close()


async def init_db():
    """
    Initialize database on startup.
    Creates tables if they don't exist.
    """
    try:
        logger.info("Initializing database...")

        # Create tables using async engine
        async with engine.begin() as conn:
            # Check if tables exist
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'videos'
                );
            """))
            tables_exist = result.scalar()

            if not tables_exist:
                logger.info("Creating database tables...")
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")

                # Load and execute schema.sql for additional features (functions, seed data)
                await load_schema_sql(conn)
            else:
                logger.info("Database tables already exist")

        logger.info("Database initialization complete")

    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise


async def load_schema_sql(conn):
    """
    Load and execute schema.sql for database functions and seed data.
    """
    try:
        schema_path = Path(__file__).parent.parent / "database" / "schemas" / "schema.sql"

        if schema_path.exists():
            logger.info(f"Loading schema from {schema_path}")

            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            # Split by statement (simple split on semicolon)
            # Skip CREATE TABLE statements as they're handled by SQLAlchemy
            statements = []
            for statement in schema_sql.split(';'):
                statement = statement.strip()
                if statement and not statement.upper().startswith('CREATE TABLE'):
                    # Only execute functions, indexes, and inserts
                    if any(keyword in statement.upper() for keyword in [
                        'CREATE OR REPLACE FUNCTION',
                        'CREATE INDEX',
                        'INSERT INTO',
                        'COMMENT ON'
                    ]):
                        statements.append(statement)

            for statement in statements:
                try:
                    await conn.execute(text(statement))
                except Exception as e:
                    # Log but don't fail - some statements might already exist
                    logger.warning(f"Schema statement warning: {e}")

            logger.info("Schema SQL loaded successfully")
        else:
            logger.warning(f"Schema file not found at {schema_path}")

    except Exception as e:
        logger.error(f"Error loading schema SQL: {e}")
        # Don't raise - this is non-critical


async def health_check() -> dict:
    """
    Check database connection health.
    Returns status information.
    """
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            result.scalar()

            # Get connection pool stats
            pool = engine.pool

            return {
                "status": "healthy",
                "database": "connected",
                "pool_size": pool.size(),
                "pool_checked_in": pool.checkedin(),
                "pool_checked_out": pool.checkedout(),
                "pool_overflow": pool.overflow(),
            }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }


async def close_db():
    """
    Close database connections.
    Call on application shutdown.
    """
    try:
        logger.info("Closing database connections...")
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")


def create_database_if_not_exists():
    """
    Create the database if it doesn't exist.
    This runs synchronously before the async app starts.
    """
    try:
        # Extract database name from URL
        db_name = DATABASE_URL.split('/')[-1].split('?')[0]
        base_url = DATABASE_URL.rsplit('/', 1)[0]

        # Connect to postgres database to create our database
        admin_url = f"{base_url.replace('+asyncpg', '')}/postgres"

        logger.info(f"Checking if database '{db_name}' exists...")

        engine_admin = create_engine(admin_url, isolation_level="AUTOCOMMIT")

        with engine_admin.connect() as conn:
            # Check if database exists
            result = conn.execute(text(
                f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"
            ))
            exists = result.scalar()

            if not exists:
                logger.info(f"Creating database '{db_name}'...")
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info(f"Database '{db_name}' created successfully")
            else:
                logger.info(f"Database '{db_name}' already exists")

        engine_admin.dispose()

    except Exception as e:
        logger.error(f"Error creating database: {e}")
        logger.warning("Continuing anyway - database might already exist")


# Startup hook for FastAPI
async def startup_event():
    """
    Run on application startup.
    """
    logger.info("Running database startup tasks...")
    create_database_if_not_exists()
    await init_db()
    logger.info("Database startup complete")


# Shutdown hook for FastAPI
async def shutdown_event():
    """
    Run on application shutdown.
    """
    logger.info("Running database shutdown tasks...")
    await close_db()
    logger.info("Database shutdown complete")
