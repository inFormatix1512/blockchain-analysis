#!/usr/bin/env python3
"""
Bitcoin block ingestion module.

Fetches new blocks from Bitcoin Core and persists transactions
with computed heuristics to PostgreSQL.
"""

import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import psycopg2

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
    
    def __init__(self):
        """Initialize the block ingestor with configuration."""
        self._config = Config()
        self._rpc = BitcoinRPC()
    
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
    
    def calculate_tx_fee(self, tx: Dict, conn) -> Optional[int]:
        """
        Calculate transaction fee in satoshis.
        
        Computes fee by summing inputs and subtracting outputs.
        Requires previous transactions to be in the database.
        
        Args:
            tx: Transaction data dictionary.
            conn: Database connection.
            
        Returns:
            Fee in satoshis, 0 for coinbase, or None if cannot be determined.
        """
        vin = tx.get('vin', [])
        
        # Coinbase transactions have no fee
        if any('coinbase' in inp for inp in vin):
            return 0
        
        # Sum outputs
        output_total = sum(
            Decimal(str(out.get('value', 0))) 
            for out in tx.get('vout', [])
        ) * self.SATS_PER_BTC
        
        # Sum inputs by looking up previous transactions
        input_total = Decimal('0')
        
        try:
            with conn.cursor() as cur:
                for inp in vin:
                    prev_txid = inp.get('txid')
                    vout_index = inp.get('vout')
                    
                    if prev_txid is None or vout_index is None:
                        return None
                    
                    cur.execute("SELECT raw FROM tx_basic WHERE txid = %s", (prev_txid,))
                    row = cur.fetchone()
                    
                    if not row:
                        return None
                    
                    prev_tx = self._parse_raw_tx(row[0], prev_txid)
                    if prev_tx is None:
                        return None
                    
                    prev_vouts = prev_tx.get('vout', [])
                    if vout_index >= len(prev_vouts):
                        return None
                    
                    input_total += Decimal(str(prev_vouts[vout_index].get('value', 0))) * self.SATS_PER_BTC
                    
        except psycopg2.Error as exc:
            logger.warning(
                "Database error computing fee for %s: %s",
                tx.get('txid', 'unknown'), exc
            )
            return None
        
        fee = input_total - output_total
        
        if fee < 0:
            logger.debug("Negative fee computed for %s, skipping", tx.get('txid', 'unknown'))
            return None
        
        return int(fee)
    
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
    
    def process_transaction(
        self, 
        tx: Dict, 
        height: int, 
        block_time: datetime,
        cur,
        conn
    ) -> bool:
        """
        Process and store a single transaction.
        
        Args:
            tx: Transaction data.
            height: Block height.
            block_time: Block timestamp.
            cur: Database cursor.
            conn: Database connection (for fee calculation).
            
        Returns:
            True if transaction was processed, False if skipped.
        """
        txid = tx.get('txid')
        if not txid:
            return False
        
        # Skip if already processed
        cur.execute("SELECT 1 FROM tx_basic WHERE txid = %s", (txid,))
        if cur.fetchone():
            return False
        
        # Extract transaction metrics
        size = tx.get('size', 0)
        vsize = tx.get('vsize', size)
        weight = tx.get('weight', size * 4)
        
        fee = self.calculate_tx_fee(tx, conn)
        feerate = (fee / vsize) if (fee is not None and vsize > 0) else None
        
        inputs_count = len(tx.get('vin', []))
        outputs_count = len(tx.get('vout', []))
        script_types = self._extract_script_types(tx)
        
        # Insert transaction basic data
        cur.execute(
            """
            INSERT INTO tx_basic (
                txid, block_height, ts, size, vsize, weight,
                fee, feerate, inputs_count, outputs_count, script_types, raw
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                txid, height, block_time, size, vsize, weight,
                int(fee) if fee is not None else None,
                feerate, inputs_count, outputs_count,
                json.dumps(script_types), json.dumps(tx, separators=(',', ':')),
            ),
        )
        
        # Insert heuristics
        heuristics = self.calculate_heuristics(tx)
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
        
        return True
    
    def process_block(self, height: int, conn) -> int:
        """
        Fetch and process a single block.
        
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
        
        processed = 0
        with conn.cursor() as cur:
            for tx in txs:
                if self.process_transaction(tx, height, block_time, cur, conn):
                    processed += 1
        
        conn.commit()
        logger.info("Block %d committed: %d transactions processed", height, processed)
        
        return processed
    
    def ingest(self) -> Tuple[int, int]:
        """
        Main ingestion loop - fetch and process new blocks.
        
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
            last_height = self.get_last_processed_height(conn)
            
            # Get blockchain info
            blockchain_info = self._rpc.get_blockchain_info()
            current_height = blockchain_info.get('blocks', 0)
            prune_height = blockchain_info.get('pruneheight', 0)
            
            # Start from prune height if needed
            if last_height < prune_height:
                logger.info(
                    "Last processed block %d is below prune height %d, starting from prune height",
                    last_height, prune_height
                )
                last_height = prune_height
            
            if current_height <= last_height:
                logger.info(
                    "No new blocks to process (last=%d, current=%d)",
                    last_height, current_height
                )
                return 0, 0
            
            # Determine blocks to process
            max_blocks = self._config.ingest.max_blocks_per_run
            blocks_to_process = min(max_blocks, current_height - last_height)
            start_height = last_height + 1
            end_height = last_height + blocks_to_process
            
            logger.info("Processing blocks from %d to %d", start_height, end_height)
            
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


def ingest_new_blocks() -> Tuple[int, int]:
    """
    Entry point for block ingestion.
    
    Returns:
        Tuple of (blocks_processed, transactions_processed).
    """
    ingestor = BlockIngestor()
    return ingestor.ingest()


if __name__ == '__main__':
    blocks, txs = ingest_new_blocks()
    print(f"Processed {blocks} blocks with {txs} transactions")
