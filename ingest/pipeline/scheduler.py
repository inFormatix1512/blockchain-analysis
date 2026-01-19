#!/usr/bin/env python3
"""
Scheduler principale per l'ingestione dati dalla blockchain Bitcoin.
Avvia cicli periodici di snapshot del mempool e ingestione dei nuovi blocchi.
"""

from __future__ import annotations

import logging
import time

from .db_utils import get_max_block_in_db
from .settings import IngestSettings
from tasks.mempool_snapshot import take_mempool_snapshot
from tasks.block_ingest import ingest_new_blocks


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run(settings: IngestSettings | None = None) -> None:
    settings = settings or IngestSettings()

    logger.info("Avvio del servizio di ingestione dati...")
    logger.info("Modalità continua: ingestione nuovi blocchi all'infinito.")
    logger.info("Il nodo potrebbe essere in sincronizzazione iniziale, attendere...")

    last_mempool_snapshot = 0.0
    last_block_ingest = 0.0

    while True:
        current_time = time.time()

        # Snapshot del mempool
        if current_time - last_mempool_snapshot >= settings.mempool_snapshot_interval:
            try:
                take_mempool_snapshot()
                last_mempool_snapshot = current_time
                logger.info("Snapshot del mempool completato")
            except Exception as exc:
                logger.error("Errore durante snapshot mempool: %s", exc)

        # Ingestione nuovi blocchi
        if current_time - last_block_ingest >= settings.block_ingest_interval:
            try:
                ingest_new_blocks(
                    start_height=settings.ingest_start_height,
                    end_height=settings.ingest_end_height,
                )
                last_block_ingest = current_time
                max_block = get_max_block_in_db(settings)
                progress = (
                    (max_block * 100.0 / settings.ingest_end_height)
                    if settings.ingest_end_height > 0
                    else 0
                )
                logger.info(
                    "Progresso: %s/%s blocchi (%.2f%%)",
                    max_block,
                    settings.ingest_end_height,
                    progress,
                )
            except Exception as exc:
                logger.error("Errore durante ingestione blocchi: %s", exc)

        time.sleep(settings.loop_sleep_seconds)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
