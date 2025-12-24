#!/usr/bin/env python3
"""
Script principale per l'ingestione dati dalla blockchain Bitcoin.
Avvia cicli periodici di snapshot del mempool e ingestione dei nuovi blocchi.
"""

import time
import os
import sys
import logging
import psycopg2
from mempool_snapshot import take_mempool_snapshot
from block_ingest import ingest_new_blocks

# Configurazione logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Intervalli di default (in secondi)
MEMPOOL_SNAPSHOT_INTERVAL = int(os.environ.get('MEMPOOL_SNAPSHOT_INTERVAL', 60))  # 1 minuto
BLOCK_INGEST_INTERVAL = int(os.environ.get('BLOCK_INGEST_INTERVAL', 300))  # 5 minuti
INGEST_START_HEIGHT = int(os.environ.get('INGEST_START_HEIGHT', 0))  # Altezza minima da ingestire (opzionale)
INGEST_END_HEIGHT = int(os.environ.get('INGEST_END_HEIGHT', 0))  # Altezza massima da ingestire (opzionale)

# Database config
PGHOST = os.environ.get('PGHOST', 'localhost')
PGUSER = os.environ.get('PGUSER', 'postgres')
PGPASSWORD = os.environ.get('PGPASSWORD', 'postgres')
PGDATABASE = os.environ.get('PGDATABASE', 'blockchain')

def get_max_block_in_db():
    """Ottiene il blocco massimo processato nel database."""
    try:
        conn = psycopg2.connect(
            host=PGHOST,
            user=PGUSER,
            password=PGPASSWORD,
            database=PGDATABASE
        )
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(block_height), 0) FROM tx_basic;")
        max_block = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return max_block
    except Exception as e:
        logger.error(f"Errore nel recupero max block: {e}")
        return 0

def main():
    logger.info("Avvio del servizio di ingestione dati...")
    logger.info("Modalità continua: ingestione nuovi blocchi all'infinito.")
    logger.info("Il nodo potrebbe essere in sincronizzazione iniziale, attendere...")

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
                # Passiamo start_height e end_height se definiti
                ingest_new_blocks(start_height=INGEST_START_HEIGHT, end_height=INGEST_END_HEIGHT)
                last_block_ingest = current_time
                max_block = get_max_block_in_db()
                progress = (max_block * 100.0 / INGEST_END_HEIGHT) if INGEST_END_HEIGHT > 0 else 0
                logger.info(f"Progresso: {max_block}/{INGEST_END_HEIGHT} blocchi ({progress:.2f}%)")
            except Exception as e:
                logger.error(f"Errore durante ingestione blocchi: {e}")

        time.sleep(10)  # Controllo ogni 10 secondi

if __name__ == "__main__":
    main()