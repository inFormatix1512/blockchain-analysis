from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class IngestSettings:
    mempool_snapshot_interval: int = int(os.environ.get("MEMPOOL_SNAPSHOT_INTERVAL", 60))
    block_ingest_interval: int = int(os.environ.get("BLOCK_INGEST_INTERVAL", 300))
    ingest_start_height: int = int(os.environ.get("INGEST_START_HEIGHT", 0))
    ingest_end_height: int = int(os.environ.get("INGEST_END_HEIGHT", 0))
    pg_host: str = os.environ.get("PGHOST", "localhost")
    pg_user: str = os.environ.get("PGUSER", "postgres")
    pg_password: str = os.environ.get("PGPASSWORD", "postgres")
    pg_database: str = os.environ.get("PGDATABASE", "blockchain")
    loop_sleep_seconds: int = int(os.environ.get("INGEST_LOOP_SLEEP", 10))
