#!/usr/bin/env python3
"""Capture periodic Bitcoin mempool snapshots."""

import json
import logging
import os
import time
from datetime import datetime
from decimal import Decimal

import psycopg2
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RPC_USER = os.environ.get('RPC_USER', 'yourrpcuser')
RPC_PASSWORD = os.environ.get('RPC_PASSWORD', 'yourrpcpassword')
RPC_HOST = os.environ.get('RPC_HOST', 'localhost')
RPC_PORT = os.environ.get('RPC_PORT', '8332')
RPC_URL = f"http://{RPC_HOST}:{RPC_PORT}"

PGHOST = os.environ.get('PGHOST', 'localhost')
PGUSER = os.environ.get('PGUSER', 'postgres')
PGPASSWORD = os.environ.get('PGPASSWORD', 'postgres')
PGDATABASE = os.environ.get('PGDATABASE', 'blockchain')

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def rpc_call(method, params=None, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Call the Bitcoin RPC endpoint with retry logic."""
    payload = {
        'jsonrpc': '2.0',
        'id': 'mempool',
        'method': method,
        'params': params or [],
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                RPC_URL,
                auth=(RPC_USER, RPC_PASSWORD),
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            if result.get('error') is not None:
                raise RuntimeError(f"RPC error: {result['error']}")
            return result['result']
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            logger.warning("RPC call %s failed (attempt %s/%s): %s", method, attempt + 1, max_retries, exc)
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError(f"RPC call {method} failed after {max_retries} attempts")


def take_mempool_snapshot():
    """Fetch mempool details and store them in the database."""
    conn = None
    try:
        logger.info("Fetching mempool information")
        mempool_info = rpc_call('getmempoolinfo')
        mempool_raw = rpc_call('getrawmempool', [True])

        total_tx = mempool_info.get('size', 0)
        total_size = mempool_info.get('bytes', 0)
        total_fee = sum(
            Decimal(str(tx_data.get('fee', 0))) * Decimal('1e8')
            for tx_data in mempool_raw.values()
            if isinstance(tx_data, dict) and tx_data.get('fee') is not None
        )
        total_fee_int = int(total_fee)

        logger.info(
            "Mempool snapshot: %s transactions, %s bytes, %s satoshi in fees",
            total_tx,
            total_size,
            total_fee_int,
        )

        conn = psycopg2.connect(host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mempool_snapshot (ts, total_tx, total_size, total_fee, raw)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.utcnow(),
                        total_tx,
                        total_size,
                        total_fee_int,
                        json.dumps(mempool_raw, separators=(',', ':')),
                    ),
                )

        logger.info("Mempool snapshot saved successfully")
    except Exception as exc:
        logger.error("Error taking mempool snapshot: %s", exc)
        raise
    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")


if __name__ == '__main__':
    take_mempool_snapshot()