import logging
import psycopg2

from .settings import IngestSettings

logger = logging.getLogger(__name__)


def get_max_block_in_db(settings: IngestSettings) -> int:
    """Ottiene il blocco massimo processato nel database."""
    try:
        conn = psycopg2.connect(
            host=settings.pg_host,
            user=settings.pg_user,
            password=settings.pg_password,
            database=settings.pg_database,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(block_height), 0) FROM tx_basic;")
        max_block = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return max_block
    except Exception as exc:
        logger.error("Errore nel recupero max block: %s", exc)
        return 0
