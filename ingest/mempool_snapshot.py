#!/usr/bin/env python3
"""
Bitcoin mempool snapshot capture module.

Periodically captures and stores mempool state for analysis
of fee market dynamics and transaction propagation.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from common import Config, DatabaseManager, BitcoinRPC

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MempoolSnapshot:
    """
    Handles Bitcoin mempool snapshot capture and storage.
    
    Captures mempool state including transaction count, size,
    and total fees for historical analysis.
    """
    
    SATS_PER_BTC = Decimal('100000000')
    
    def __init__(self):
        """Initialize with configuration."""
        self._config = Config()
        self._rpc = BitcoinRPC()
    
    def _calculate_total_fee(self, mempool_raw: Dict) -> int:
        """
        Calculate total fees in the mempool.
        
        Args:
            mempool_raw: Raw mempool data from getrawmempool.
            
        Returns:
            Total fees in satoshis.
        """
        total_fee = sum(
            Decimal(str(tx_data.get('fee', 0))) * self.SATS_PER_BTC
            for tx_data in mempool_raw.values()
            if isinstance(tx_data, dict) and tx_data.get('fee') is not None
        )
        return int(total_fee)
    
    def capture(self) -> Tuple[int, int, int]:
        """
        Capture current mempool state and store in database.
        
        Returns:
            Tuple of (total_tx, total_size, total_fee_satoshis).
            
        Raises:
            Exception: On capture or storage failure.
        """
        logger.info("Fetching mempool information")
        
        # Get mempool data from Bitcoin Core
        mempool_info = self._rpc.get_mempool_info()
        mempool_raw = self._rpc.get_raw_mempool(verbose=True)
        
        total_tx = mempool_info.get('size', 0)
        total_size = mempool_info.get('bytes', 0)
        total_fee = self._calculate_total_fee(mempool_raw)
        
        logger.info(
            "Mempool snapshot: %d transactions, %d bytes, %d satoshi in fees",
            total_tx, total_size, total_fee
        )
        
        # Store snapshot
        with DatabaseManager() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mempool_snapshot (ts, total_tx, total_size, total_fee, raw)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        datetime.now(timezone.utc),
                        total_tx,
                        total_size,
                        total_fee,
                        json.dumps(mempool_raw, separators=(',', ':')),
                    ),
                )
            conn.commit()
        
        logger.info("Mempool snapshot saved successfully")
        
        return total_tx, total_size, total_fee
    
    def close(self) -> None:
        """Close RPC connection."""
        self._rpc.close()


def take_mempool_snapshot() -> Tuple[int, int, int]:
    """
    Entry point for mempool snapshot capture.
    
    Returns:
        Tuple of (total_tx, total_size, total_fee_satoshis).
    """
    snapshot = MempoolSnapshot()
    try:
        return snapshot.capture()
    finally:
        snapshot.close()


if __name__ == '__main__':
    try:
        tx_count, size, fee = take_mempool_snapshot()
        print(f"Captured: {tx_count} txs, {size} bytes, {fee} sat fees")
    except Exception as exc:
        logger.error("Error taking mempool snapshot: %s", exc)
        sys.exit(1)
