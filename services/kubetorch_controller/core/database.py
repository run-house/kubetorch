"""
SQLAlchemy database setup and models for Kubetorch Controller.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import Column, create_engine, DateTime, event, Integer, String, Text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

# SQLite database path - should be mounted to a PV for persistence
# Default to local "./data" for local devlopment, "/data" for in-cluster
_default_db_path = (
    "./data/kubetorch.db" if not os.path.exists("/data") else "/data/kubetorch.db"
)
DB_PATH = os.getenv("KUBETORCH_DB_PATH", _default_db_path)


class Base(DeclarativeBase):
    pass


class Pool(Base):
    """Model for compute pools.

    A Pool is a logical group of pods that calls can be directed to.
    Registered via /pool endpoint.
    """

    __tablename__ = "pools"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Name - Unique identifier of the pool (does not need to match K8s resource name)
    name = Column(String, nullable=False, unique=True, index=True)

    # Namespace where the pool resources live
    namespace = Column(String, nullable=False)

    # Specifier - How we track pods in the pool
    # JSON: {"type": "label_selector", "selector": {"app": "workers", "team": "ml"}}
    specifier = Column(Text, nullable=False)

    # Service (Optional) - How we make calls to the pool
    # JSON containing either:
    #   null - auto-create service
    #   {"url": "..."} - user-provided URL (e.g. Knative)
    #   {"selector": {...}} - custom selector for routing (e.g. Ray head node)
    #   {"name": "..."} - custom service name
    service_config = Column(Text, nullable=True)

    # Dockerfile (Optional) - Instructions to rebuild workers on deployment
    dockerfile = Column(Text, nullable=True)

    # Module (Optional) - Application deployed on the pool
    # JSON module spec:
    #   {"type": "fn|cls|cmd|app", "pointers": {...}, "dispatch": "regular|spmd|load_balanced", "procs": 1}
    module = Column(Text, nullable=True)

    # Metadata - username, etc.
    pool_metadata = Column(Text, nullable=True)

    # Service configuration fields (used for K8s Service creation)
    server_port = Column(Integer, nullable=True, default=32300)

    # K8s resource info (for teardown to know what to delete)
    resource_kind = Column(
        String, nullable=True
    )  # e.g., "Deployment", "StatefulSet", "PyTorchJob"
    resource_name = Column(
        String, nullable=True
    )  # Name of the K8s resource (defaults to pool name)

    # Labels and annotations (JSON) - for K8s Service creation and querying
    labels = Column(Text, nullable=True)
    annotations = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_deployed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "namespace": self.namespace,
            "specifier": json.loads(self.specifier) if self.specifier else None,
            "service_config": json.loads(self.service_config)
            if self.service_config
            else None,
            "dockerfile": self.dockerfile,
            "module": json.loads(self.module) if self.module else None,
            "pool_metadata": json.loads(self.pool_metadata)
            if self.pool_metadata
            else None,
            "server_port": self.server_port,
            "resource_kind": self.resource_kind,
            "resource_name": self.resource_name,
            "labels": json.loads(self.labels) if self.labels else None,
            "annotations": json.loads(self.annotations) if self.annotations else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_deployed_at": self.last_deployed_at.isoformat()
            if self.last_deployed_at
            else None,
        }


# Engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        # Ensure directory exists
        db_dir = os.path.dirname(DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        _engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={
                "check_same_thread": False,  # Needed for SQLite with multiple threads
                "timeout": 60,  # Wait up to 60s for locks instead of failing immediately
            },
            echo=False,
            pool_size=1,  # Single connection to avoid lock contention
            max_overflow=0,  # No additional connections
        )

        # https://sqlite.org/wal.html
        # Enable WAL mode for better concurrent read/write performance
        def set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            # Faster writes, still safe with WAL
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=60000")  # 60s timeout in milliseconds
            cursor.close()

        event.listen(_engine, "connect", set_sqlite_pragma)
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


def init_db():
    """Initialize the database and create tables.

    Handles race conditions when multiple Uvicorn workers start simultaneously
    and try to create the same tables. The "table already exists" error is
    safely ignored since it means another worker already created the tables.
    """
    engine = get_engine()
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"SQLite database initialized at path: '{DB_PATH}'")
    except OperationalError as e:
        if "already exists" in str(e):
            # Another worker already created the tables - this is fine
            logger.info(f"SQLite database already initialized at path: '{DB_PATH}'")
        else:
            raise


def get_db() -> Session:
    """Get a database session. Use as context manager or manually close."""
    SessionLocal = get_session_factory()
    return SessionLocal()
