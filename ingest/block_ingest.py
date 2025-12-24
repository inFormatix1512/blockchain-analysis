#!/usr/bin/env python3
"""
Bitcoin block ingestion module.

Fetches new blocks from Bitcoin Core and persists transactions
with computed heuristics to PostgreSQL.
"""

import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_values

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from common import Config, DatabaseManager, BitcoinRPC
from common.rpc import RPCError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BlockIngestor:
    """
    Handles Bitcoin block ingestion and transaction processing.
    
    This class manages the complete pipeline of fetching blocks from
    Bitcoin Core, computing heuristics, and storing data in PostgreSQL.
    """
    
    SATS_PER_BTC = Decimal('100000000')
    START_BLOCK_HEIGHT = 600000  # Approx Oct 2019
    
    def __init__(self):
        """Initialize the block ingestor with configuration."""
        self._config = Config()
        self._rpc = BitcoinRPC()
        self.MAX_BLOCKS_PER_RUN = int(os.getenv('MAX_BLOCKS_PER_RUN', 0))
        self.WORKER_ID = os.getenv('WORKER_ID', f"worker-{os.getpid()}")

    def ensure_schema(self, conn):
        """Ensure the coordination table exists."""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ingest_status (
                        block_height INTEGER PRIMARY KEY,
                        worker_id VARCHAR(50),
                        status VARCHAR(20) DEFAULT 'processing',
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ,
                        error_msg TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_ingest_status_status ON ingest_status(status);
                """)
            conn.commit()
        except psycopg2.Error as exc:
            logger.error("Error creating schema: %s", exc)
            conn.rollback()

    def claim_next_block(self, conn) -> Optional[int]:
        """
        Atomically claims the next available block for processing.
        """
        try:
            while True:
                with conn.cursor() as cur:
                    # Check if table is empty to seed it
                    cur.execute("SELECT COUNT(*) FROM ingest_status")
                    count = cur.fetchone()[0]
                    
                    next_height = None
                    
                    if count == 0:
                        # Seed with START_BLOCK_HEIGHT
                        next_height = self.START_BLOCK_HEIGHT
                    else:
                        # Find the first gap or the next block after max
                        cur.execute("""
                            SELECT MAX(block_height) + 1 FROM ingest_status
                        """)
                        row = cur.fetchone()
                        if row and row[0]:
                            next_height = row[0]
                        else:
                            next_height = self.START_BLOCK_HEIGHT

                    # Check against blockchain tip
                    blockchain_info = self._rpc.get_blockchain_info()
                    chain_height = blockchain_info.get('blocks', 0)
                    
                    if next_height > chain_height:
                        return None

                    # Fast Forward: Check if already in tx_basic (Legacy data)
                    cur.execute("SELECT 1 FROM tx_basic WHERE block_height = %s LIMIT 1", (next_height,))
                    if cur.fetchone():
                        logger.info(f"Block {next_height} already in DB. Marking as completed.")
                        cur.execute("""
                            INSERT INTO ingest_status (block_height, worker_id, status, completed_at)
                            VALUES (%s, 'legacy', 'completed', NOW())
                            ON CONFLICT (block_height) DO UPDATE SET status='completed'
                        """, (next_height,))
                        conn.commit()
                        continue # Loop to find next

                    # Try to claim
                    cur.execute("""
                        INSERT INTO ingest_status (block_height, worker_id, status)
                        VALUES (%s, %s, 'processing')
                        ON CONFLICT (block_height) DO NOTHING
                        RETURNING block_height
                    """, (next_height, self.WORKER_ID))
                    
                    row = cur.fetchone()
                    if row:
                        conn.commit()
                        logger.info(f"Worker {self.WORKER_ID} claimed block {next_height}")
                        return row[0]
                    else:
                        # Conflict, someone else took it
                        conn.rollback()
                        # Loop to try next
                        continue
                        
        except psycopg2.Error as exc:
            logger.error("Error claiming block: %s", exc)
            conn.rollback()
            return None
        except RPCError as exc:
            logger.error("RPC Error checking tip: %s", exc)
            return None

    def update_block_status(self, conn, height: int, status: str, error: str = None):
        """Update the status of a block in the coordination table."""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ingest_status 
                    SET status = %s, 
                        completed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE NULL END,
                        error_msg = %s
                    WHERE block_height = %s
                """, (status, status, error, height))
            conn.commit()
        except psycopg2.Error as exc:
            logger.error("Error updating block status: %s", exc)
            conn.rollback()

    def get_last_processed_height(self, conn) -> int:
        """
        Get the latest block height stored in the database.
        
        Args:
            conn: Database connection.
            
        Returns:
            The maximum block height or 0 if no blocks processed.
        """
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(block_height) FROM tx_basic")
                row = cur.fetchone()
                return row[0] or 0
        except psycopg2.Error as exc:
            logger.error("Error querying last processed height: %s", exc)
            return 0
    
    def calculate_heuristics(self, tx: Dict) -> Dict[str, Any]:
        """
        Compute privacy heuristics for a transaction.
        
        Analyzes the transaction structure to identify potential
        CoinJoin transactions, RBF usage, and likely change outputs.
        
        Args:
            tx: Transaction data dictionary.
            
        Returns:
            Dictionary with computed heuristics:
            - is_rbf: Whether RBF is signaled
            - coinjoin_score: Ratio of equal outputs (0.0-1.0)
            - equal_output: Whether multiple equal outputs exist
            - likely_change_index: Index of likely change output
        """
        heuristics = {
            'is_rbf': False,
            'coinjoin_score': 0.0,
            'equal_output': False,
            'likely_change_index': None,
        }
        
        # Check RBF signaling
        vin = tx.get('vin', [])
        heuristics['is_rbf'] = any(
            inp.get('sequence', 0xFFFFFFFF) < 0xFFFFFFFE 
            for inp in vin
        )
        
        # Analyze outputs for CoinJoin patterns
        outputs = [
            float(out.get('value', 0)) 
            for out in tx.get('vout', []) 
            if out.get('value') is not None
        ]
        
        if not outputs:
            return heuristics
        
        output_counts = Counter(outputs)
        max_count = max(output_counts.values())
        heuristics['coinjoin_score'] = max_count / len(outputs)
        heuristics['equal_output'] = max_count > 1
        
        # Identify likely change output using round number heuristic
        if len(outputs) >= 2:
            sorted_outputs = sorted(outputs, reverse=True)
            second_output = sorted_outputs[1]
            
            # Avoid marking if second value appears multiple times (likely payment)
            if len(outputs) >= 3 and second_output in sorted_outputs[2:]:
                heuristics['likely_change_index'] = None
            else:
                try:
                    heuristics['likely_change_index'] = outputs.index(second_output)
                except ValueError:
                    heuristics['likely_change_index'] = None
        
        return heuristics
    
    
    def _parse_raw_tx(self, raw_data: Any, txid: str) -> Optional[Dict]:
        """
        Parse raw transaction data from database.
        
        Args:
            raw_data: Raw transaction data (dict or JSON string).
            txid: Transaction ID for logging.
            
        Returns:
            Parsed transaction dictionary or None on failure.
        """
        if isinstance(raw_data, dict):
            return raw_data
        
        if isinstance(raw_data, str):
            try:
                return json.loads(raw_data)
            except json.JSONDecodeError:
                logger.debug("Unable to decode stored raw transaction %s", txid)
                return None
        
        logger.debug("Unexpected type for raw transaction %s: %s", txid, type(raw_data))
        return None
    
    def _extract_script_types(self, tx: Dict) -> Dict[str, int]:
        """Extract and count script types from transaction outputs."""
        script_types = {}
        for out in tx.get('vout', []):
            script_pub_key = out.get('scriptPubKey', {})
            script_type = script_pub_key.get('type', 'unknown')
            script_types[script_type] = script_types.get(script_type, 0) + 1
        return script_types
    
    
    def _batch_get_prev_txs(self, txs: List[Dict], cur) -> Dict[str, Dict]:
        """Fetch all previous transactions needed for fee calculation in a single batch."""
        needed_txids = set()
        for tx in txs:
            if any('coinbase' in inp for inp in tx.get('vin', [])):
                continue
            for inp in tx.get('vin', []):
                if 'txid' in inp:
                    needed_txids.add(inp['txid'])
        
        if not needed_txids:
            return {}
            
        prev_tx_map = {}
        needed_list = list(needed_txids)
        batch_size = 1000
        
        for i in range(0, len(needed_list), batch_size):
            batch = needed_list[i:i + batch_size]
            cur.execute("SELECT txid, raw FROM tx_basic WHERE txid = ANY(%s)", (batch,))
            for row in cur.fetchall():
                txid, raw = row
                parsed = self._parse_raw_tx(raw, txid)
                if parsed:
                    prev_tx_map[txid] = parsed
        return prev_tx_map

    def _calculate_fee_from_map(self, tx: Dict, prev_tx_map: Dict[str, Dict]) -> Optional[int]:
        """Calculate fee using in-memory map of previous transactions."""
        vin = tx.get('vin', [])
        if any('coinbase' in inp for inp in vin):
            return 0
            
        output_total = sum(Decimal(str(out.get('value', 0))) for out in tx.get('vout', [])) * self.SATS_PER_BTC
        input_total = Decimal('0')
        
        for inp in vin:
            prev_txid = inp.get('txid')
            vout_index = inp.get('vout')
            
            if prev_txid not in prev_tx_map:
                return None
                
            prev_tx = prev_tx_map[prev_txid]
            prev_vouts = prev_tx.get('vout', [])
            
            if vout_index >= len(prev_vouts):
                return None
                
            input_total += Decimal(str(prev_vouts[vout_index].get('value', 0))) * self.SATS_PER_BTC
            
        fee = input_total - output_total
        return int(fee) if fee >= 0 else None

    def process_block(self, height: int, conn) -> int:
        """
        Fetch and process a single block using batch insertion.
        
        Args:
            height: Block height to process.
            conn: Database connection.
            
        Returns:
            Number of transactions processed.
        """
        block_hash = self._rpc.get_block_hash(height)
        block = self._rpc.get_block(block_hash, verbosity=2)
        
        # Check if block data is pruned
        if block.get('tx') is None:
            logger.warning("Block %d data is pruned, skipping", height)
            return 0
        
        txs = block.get('tx', [])
        block_time = datetime.fromtimestamp(block.get('time', 0))
        
        logger.info("Processing block %d with %d transactions", height, len(txs))
        
        with conn.cursor() as cur:
            # 1. Batch fetch previous transactions for fee calculation
            prev_tx_map = self._batch_get_prev_txs(txs, cur)
            
            basic_rows = []
            heuristic_rows = []
            
            for tx in txs:
                txid = tx.get('txid')
                if not txid:
                    continue
                    
                # Extract transaction metrics
                size = tx.get('size', 0)
                vsize = tx.get('vsize', size)
                weight = tx.get('weight', size * 4)
                
                fee = self._calculate_fee_from_map(tx, prev_tx_map)
                feerate = (fee / vsize) if (fee is not None and vsize > 0) else None
                
                inputs_count = len(tx.get('vin', []))
                outputs_count = len(tx.get('vout', []))
                script_types = self._extract_script_types(tx)
                
                basic_rows.append((
                    txid, height, block_time, size, vsize, weight,
                    fee, feerate, inputs_count, outputs_count,
                    json.dumps(script_types), None
                ))
                
                # Heuristics
                heuristics = self.calculate_heuristics(tx)
                heuristic_rows.append((
                    txid,
                    heuristics['is_rbf'],
                    heuristics['coinjoin_score'],
                    heuristics['equal_output'],
                    heuristics['likely_change_index'],
                    ''
                ))
            
            # 2. Batch Insert
            if basic_rows:
                execute_values(cur, """
                    INSERT INTO tx_basic (
                        txid, block_height, ts, size, vsize, weight,
                        fee, feerate, inputs_count, outputs_count, script_types, raw
                    ) VALUES %s
                    ON CONFLICT (txid) DO NOTHING
                """, basic_rows)
                
            if heuristic_rows:
                execute_values(cur, """
                    INSERT INTO tx_heuristics (
                        txid, is_rbf, coinjoin_score, equal_output, likely_change_index, notes
                    ) VALUES %s
                    ON CONFLICT (txid) DO NOTHING
                """, heuristic_rows)
        
        conn.commit()
        logger.info("Block %d committed: %d transactions processed", height, len(txs))
        
        return len(txs)
    
    def ingest(self, start_height_override: int = 0, end_height_override: int = 0) -> Tuple[int, int]:
        """
        Main ingestion loop - fetch and process new blocks.
        
        Args:
            start_height_override: Force start from this height.
            end_height_override: Force stop at this height.

        Returns:
            Tuple of (blocks_processed, transactions_processed).
        """
        db = DatabaseManager()
        
        try:
            conn = db.connect()
        except Exception as exc:
            logger.error("Failed to connect to database: %s", exc)
            return 0, 0
        
        try:
            self.ensure_schema(conn)

            # Dynamic Mode (Default if no start height provided)
            if start_height_override == 0:
                blocks_processed = 0
                total_transactions = 0
                logger.info(f"Starting dynamic ingestion for worker {self.WORKER_ID}")
                
                while True:
                    if self.MAX_BLOCKS_PER_RUN > 0 and blocks_processed >= self.MAX_BLOCKS_PER_RUN:
                        logger.info("Reached MAX_BLOCKS_PER_RUN limit")
                        break
                        
                    height = self.claim_next_block(conn)
                    if not height:
                        logger.info("No more blocks to claim. Stopping run.")
                        break
                        
                    try:
                        tx_count = self.process_block(height, conn)
                        self.update_block_status(conn, height, 'completed')
                        blocks_processed += 1
                        total_transactions += tx_count
                    except Exception as exc:
                        logger.error("Error processing block %d: %s", height, exc)
                        self.update_block_status(conn, height, 'failed', str(exc))
                        time.sleep(1)
                        
                return blocks_processed, total_transactions

            # Static Mode (Legacy)
            last_height = self.get_last_processed_height(conn)
            
            # Get blockchain info
            blockchain_info = self._rpc.get_blockchain_info()
            current_height = blockchain_info.get('blocks', 0)
            prune_height = blockchain_info.get('pruneheight', 0)
            
            # Determine start height
            start_height = last_height + 1
            
            # Override start height if provided (useful for parallel workers)
            if start_height_override > 0:
                start_height = max(start_height, start_height_override)
            
            # Start from 2021 if DB is empty (unless pruned higher)
            if last_height == 0 and start_height_override == 0:
                target_start = self.START_BLOCK_HEIGHT - 1
                if target_start >= prune_height:
                    start_height = target_start + 1
                    logger.info("Database empty. Starting analysis from Jan 2021 (Block %d)", start_height)
            
            # Start from prune height if needed
            if start_height < prune_height:
                logger.info(
                    "Start block %d is below prune height %d, starting from prune height",
                    start_height, prune_height
                )
                start_height = prune_height + 1

            # Determine end height
            end_height = current_height
            if self.MAX_BLOCKS_PER_RUN > 0:
                end_height = min(end_height, start_height + self.MAX_BLOCKS_PER_RUN - 1)
            
            # Override end height if provided
            if end_height_override > 0:
                end_height = min(end_height, end_height_override)
            
            if start_height > end_height:
                logger.info("No new blocks to process (last=%d, current=%d)", last_height, current_height)
                return 0, 0
            
            logger.info("Starting ingestion from block %d to %d", start_height, end_height)
            
            blocks_processed = 0
            total_transactions = 0
            
            for height in range(start_height, end_height + 1):
                try:
                    tx_count = self.process_block(height, conn)
                    blocks_processed += 1
                    total_transactions += tx_count
                except RPCError as exc:
                    logger.error("RPC error processing block %d: %s", height, exc)
                    conn.rollback()
                    time.sleep(1)
                except psycopg2.Error as exc:
                    logger.error("Database error processing block %d: %s", height, exc)
                    conn.rollback()
                    time.sleep(1)
            
            return blocks_processed, total_transactions
            
        finally:
            self._rpc.close()
            db.close()
            logger.info("Connections closed")


def ingest_new_blocks(start_height: int = 0, end_height: int = 0) -> Tuple[int, int]:
    """
    Entry point for block ingestion.
    
    Args:
        start_height: Optional start height override.
        end_height: Optional end height override.

    Returns:
        Tuple of (blocks_processed, transactions_processed).
    """
    ingestor = BlockIngestor()
    # Monkey-patching or modifying ingest method would be cleaner, but for now we pass via env vars or modify ingest
    # Actually, let's modify the ingest method to accept arguments
    return ingestor.ingest(start_height_override=start_height, end_height_override=end_height)


if __name__ == '__main__':
    blocks, txs = ingest_new_blocks()
    print(f"Processed {blocks} blocks with {txs} transactions")
