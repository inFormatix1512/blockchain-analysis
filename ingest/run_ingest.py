#!/usr/bin/env python3
"""
Script principale per l'ingestione dati dalla blockchain Bitcoin.
Avvia cicli periodici di snapshot del mempool e ingestione dei nuovi blocchi.
"""

import time
import os
import logging
from mempool_snapshot import take_mempool_snapshot
from block_ingest import ingest_new_blocks

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Intervalli di default (in secondi)
MEMPOOL_SNAPSHOT_INTERVAL = int(os.environ.get('MEMPOOL_SNAPSHOT_INTERVAL', 60))  # 1 minuto
BLOCK_INGEST_INTERVAL = int(os.environ.get('BLOCK_INGEST_INTERVAL', 300))  # 5 minuti

def main():
    logger.info("Avvio del servizio di ingestione dati...")

    last_mempool_snapshot = 0
    last_block_ingest = 0

    while True:
        current_time = time.time()

        # Snapshot del mempool
        if current_time - last_mempool_snapshot >= MEMPOOL_SNAPSHOT_INTERVAL:
            try:
                take_mempool_snapshot()
                last_mempool_snapshot = current_time
                logger.info("Snapshot del mempool completato")
            except Exception as e:
                logger.error(f"Errore durante snapshot mempool: {e}")

        # Ingestione nuovi blocchi
        if current_time - last_block_ingest >= BLOCK_INGEST_INTERVAL:
            try:
                ingest_new_blocks()
                last_block_ingest = current_time
                logger.info("Ingestione blocchi completata")
            except Exception as e:
                logger.error(f"Errore durante ingestione blocchi: {e}")

        time.sleep(10)  # Controllo ogni 10 secondi

if __name__ == "__main__":
    main()