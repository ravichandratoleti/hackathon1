import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

BASE_DIR = Path(__file__).resolve().parents[1]
default_path = BASE_DIR / "data" / "veritas.db"
is_vercel = os.getenv("VERCEL", "").lower() in {"1", "true", "yes"}
is_production = os.getenv("ENVIRONMENT", "").lower() in {"prod", "production"}
is_serverless_production = is_vercel or is_production

database_url = os.getenv("DATABASE_URL", "").strip()
if not database_url:
    if is_serverless_production:
        raise RuntimeError(
            "DATABASE_URL must be configured in production; refusing to fall back to SQLite."
        )
    default_path.parent.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{default_path.as_posix()}"

if database_url.startswith("postgres://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgres://")
elif database_url.startswith("postgresql://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql://")
elif database_url.startswith("postgresql+psycopg2://"):
    database_url = "postgresql+psycopg://" + database_url.removeprefix("postgresql+psycopg2://")

if is_serverless_production and not database_url.startswith("postgresql+psycopg://"):
    raise RuntimeError(
        "Production DATABASE_URL must use PostgreSQL with the Psycopg 3 driver."
    )

connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
engine_options = {"connect_args": connect_args, "future": True, "pool_pre_ping": True}
if is_vercel:
    # Keep each warm serverless instance to one checked connection; Neon is expected
    # to be configured with its pooled connection string.
    engine_options.update(pool_size=1, max_overflow=0, pool_recycle=300)

engine = create_engine(database_url, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
