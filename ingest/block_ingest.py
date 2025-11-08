#!/usr/bin/env python3
"""Ingest new Bitcoin blocks and compute basic heuristics."""

import json
import logging
import os
import time
from collections import Counter
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
MAX_BLOCKS_PER_RUN = int(os.environ.get('MAX_BLOCKS_PER_RUN', '10'))


def rpc_call(method, params=None, max_retries=MAX_RETRIES, retry_delay=RETRY_DELAY):
    """Call the Bitcoin RPC endpoint with simple retry logic."""
    payload = {
        'jsonrpc': '2.0',
        'id': 'ingest',
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


def get_last_processed_height(conn):
    """Return the latest block height stored in the database."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(block_height) FROM tx_basic")
            row = cur.fetchone()
            return row[0] or 0
    except psycopg2.Error as exc:
        logger.error("Error querying last processed height: %s", exc)
        return 0


def calculate_heuristics(tx):
    """Compute simple heuristics for a transaction."""
    heuristics = {
        'is_rbf': False,
        'coinjoin_score': 0.0,
        'equal_output': False,
        'likely_change_index': None,
    }

    vin = tx.get('vin', [])
    heuristics['is_rbf'] = any(inp.get('sequence', 0xFFFFFFFF) < 0xFFFFFFFE for inp in vin)

    outputs = [float(out.get('value', 0)) for out in tx.get('vout', []) if out.get('value') is not None]
    if not outputs:
        return heuristics

    output_counts = Counter(outputs)
    max_count = max(output_counts.values())
    heuristics['coinjoin_score'] = max_count / len(outputs)
    heuristics['equal_output'] = max_count > 1

    if len(outputs) >= 2:
        sorted_outputs = sorted(outputs, reverse=True)
        second_output = sorted_outputs[1]
        if len(outputs) >= 3 and second_output in sorted_outputs[2:]:
            heuristics['likely_change_index'] = None
        else:
            try:
                heuristics['likely_change_index'] = outputs.index(second_output)
            except ValueError:
                heuristics['likely_change_index'] = None

    return heuristics


def calculate_tx_fee(tx, conn):
    """Estimate the transaction fee in satoshi; return None if not determinable."""
    if any('coinbase' in inp for inp in tx.get('vin', [])):
        return 0

    sats_per_btc = Decimal('1e8')
    output_total = sum(Decimal(str(out.get('value', 0))) for out in tx.get('vout', [])) * sats_per_btc
    input_total = Decimal('0')

    try:
        with conn.cursor() as cur:
            for vin in tx.get('vin', []):
                prev_txid = vin.get('txid')
                vout_index = vin.get('vout')
                if prev_txid is None or vout_index is None:
                    return None

                cur.execute("SELECT raw FROM tx_basic WHERE txid = %s", (prev_txid,))
                row = cur.fetchone()
                if not row:
                    return None

                try:
                    prev_tx = json.loads(row[0])
                except json.JSONDecodeError:
                    logger.debug("Unable to decode stored raw transaction %s", prev_txid)
                    return None

                prev_vouts = prev_tx.get('vout', [])
                if vout_index >= len(prev_vouts):
                    return None

                input_total += Decimal(str(prev_vouts[vout_index].get('value', 0))) * sats_per_btc
    except psycopg2.Error as exc:
        logger.warning("Database error while computing fee for %s: %s", tx.get('txid', 'unknown'), exc)
        return None

    fee = input_total - output_total
    if fee < 0:
        logger.debug("Negative fee computed for %s, skipping", tx.get('txid', 'unknown'))
        return None

    return int(fee)


def ingest_new_blocks():
    """Fetch new blocks from the node and persist their transactions."""
    conn = None
    try:
        conn = psycopg2.connect(host=PGHOST, user=PGUSER, password=PGPASSWORD, dbname=PGDATABASE)
        conn.autocommit = False
    except psycopg2.Error as exc:
        logger.error("Failed to connect to database: %s", exc)
        return

    try:
        last_height = get_last_processed_height(conn)
        current_height = rpc_call('getblockcount')

        if current_height <= last_height:
            logger.info("No new blocks to process (last=%s)", last_height)
            return

        blocks_to_process = min(MAX_BLOCKS_PER_RUN, current_height - last_height)
        start_height = last_height + 1
        end_height = last_height + blocks_to_process
        logger.info("Processing blocks from %s to %s", start_height, end_height)

        with conn.cursor() as cur:
            for height in range(start_height, end_height + 1):
                try:
                    block_hash = rpc_call('getblockhash', [height])
                    block = rpc_call('getblock', [block_hash, 2])
                    txs = block.get('tx', [])
                    logger.info("Processing block %s with %s transactions", height, len(txs))

                    block_time = datetime.fromtimestamp(block.get('time', 0))

                    for tx in txs:
                        txid = tx.get('txid')
                        if not txid:
                            continue

                        cur.execute("SELECT 1 FROM tx_basic WHERE txid = %s", (txid,))
                        if cur.fetchone():
                            continue

                        size = tx.get('size', 0)
                        vsize = tx.get('vsize', size)
                        weight = tx.get('weight', size * 4)

                        fee = calculate_tx_fee(tx, conn)
                        feerate = (fee / vsize) if (fee is not None and vsize > 0) else None

                        inputs_count = len(tx.get('vin', []))
                        outputs_count = len(tx.get('vout', []))

                        script_types = {}
                        for out in tx.get('vout', []):
                            script_pub_key = out.get('scriptPubKey', {})
                            script_type = script_pub_key.get('type', 'unknown')
                            script_types[script_type] = script_types.get(script_type, 0) + 1

                        cur.execute(
                            """
                            INSERT INTO tx_basic (
                                txid, block_height, ts, size, vsize, weight,
                                fee, feerate, inputs_count, outputs_count, script_types, raw
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                txid,
                                height,
                                block_time,
                                size,
                                vsize,
                                weight,
                                int(fee) if fee is not None else None,
                                feerate,
                                inputs_count,
                                outputs_count,
                                json.dumps(script_types),
                                json.dumps(tx, separators=(',', ':')),
                            ),
                        )

                        heuristics = calculate_heuristics(tx)
                        cur.execute(
                            """
                            INSERT INTO tx_heuristics (
                                txid, is_rbf, coinjoin_score, equal_output, likely_change_index, notes
                            )
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                txid,
                                heuristics['is_rbf'],
                                heuristics['coinjoin_score'],
                                heuristics['equal_output'],
                                heuristics['likely_change_index'],
                                '',
                            ),
                        )

                    conn.commit()
                    logger.info("Block %s committed successfully", height)
                except Exception as exc:
                    conn.rollback()
                    logger.error("Error processing block %s: %s", height, exc)
                    time.sleep(1)
    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed")


if __name__ == '__main__':
    ingest_new_blocks()