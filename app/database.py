import os
import time
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import declarative_base


import os


DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

engine = None
SessionLocal = None
Base = declarative_base()

# -------------------------------
# Configuration
# -------------------------------

def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL must be set.")
    
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)

    return url

# Get the database URL
SQLALCHEMY_DATABASE_URL = get_database_url()

# -------------------------------
# Engine Configuration
# -------------------------------
def create_database_engine(url: Optional[str] = None) -> Engine:
    """
    Create and configure the SQLAlchemy engine.
    """
    database_url = url or SQLALCHEMY_DATABASE_URL
    
    # Determine if we're in development or production
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    
    engine_args = {
        "echo": os.getenv("SQL_ECHO", "false").lower() == "true",  # Log SQL statements
        "pool_pre_ping": True,  # Verify connections before using
        "pool_recycle": 300,  # Recycle connections after 5 minutes
        "pool_size": 20,  # Maximum number of connections in pool
        "max_overflow": 40,  # Maximum overflow connections
        "pool_timeout": 30,  # Timeout for getting a connection from pool
        "connect_args": {
            "connect_timeout": 10,  # Connection timeout in seconds
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
    }
    
    # Use connection pool
    engine_args["poolclass"] = QueuePool
    
    # Add more strict settings for production
    if is_production:
        engine_args.update({
            "echo": False,  # Never echo SQL in production
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,
            "pool_recycle": 1800,  # 30 minutes
        })
    
    # For SQLite (if you ever need it)
    if "sqlite" in database_url:
        # SQLite specific settings
        engine_args.pop("pool_size", None)
        engine_args.pop("max_overflow", None)
        engine_args.pop("pool_recycle", None)
        engine_args["connect_args"] = {"check_same_thread": False}
    
    engine = create_engine(database_url, **engine_args)
    
    # Add event listeners for better debugging
    if engine_args["echo"]:
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault('query_start_time', []).append(time.time())
            print(f"🚀 SQL: {statement}")
            if parameters:
                print(f"📝 Params: {parameters}")
        
        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = time.time() - conn.info['query_start_time'].pop(-1)
            print(f"⏱️  Execution time: {total:.3f}s")
    
    return engine

# Create the engine
engine = create_database_engine()

# -------------------------------
# Session Factory
# -------------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,  # Better for web apps
)

# Scoped session for thread safety (useful for background tasks)
ScopedSession = scoped_session(SessionLocal)

# -------------------------------
# FastAPI Dependency
# -------------------------------
def get_db() -> Generator[Session, None, None]:
    """
    Dependency function that yields a database session.
    
    Usage in FastAPI:
        db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------
# Context Manager for Manual Session Handling
# -------------------------------
@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with get_db_context() as db:
            # use db session
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

# -------------------------------
# Database Health Check
# -------------------------------
def check_database_connection() -> bool:
    """
    Check if database is accessible.
    Returns True if successful, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"Database connection error: {e}")
        return False

# -------------------------------
# Database Initialization
# -------------------------------
def init_db() -> None:
    """
    Initialize database by creating all tables.
    
    WARNING: Only use in development!
    For production, use Alembic migrations.
    """
    print("Initializing database...")
    
    # Check connection first
    if not check_database_connection():
        raise RuntimeError("Cannot connect to database")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")

def drop_db() -> None:
    """
    Drop all database tables.
    
    WARNING: Destructive operation! Only for development/testing.
    """
    if os.getenv("ENVIRONMENT") == "production":
        raise RuntimeError("Cannot drop database in production!")
    
    Base.metadata.drop_all(bind=engine)
    print("Database tables dropped.")

  