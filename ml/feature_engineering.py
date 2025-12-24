import logging
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports if not present
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from common import get_db_connection

logger = logging.getLogger(__name__)

def load_transaction_data(min_samples: int = 1000, limit: int = 100000) -> pd.DataFrame:
    """
    Load transaction data from database.
    
    Args:
        min_samples: Minimum samples required for training.
        limit: Maximum number of records to load.
        
    Returns:
        DataFrame with transaction data.
    """
    logger.info("Loading transaction data from database (limit=%d)...", limit)
    
    query = f"""
    SELECT 
        b.txid,
        b.block_height,
        EXTRACT(EPOCH FROM b.ts) as timestamp,
        b.size,
        b.vsize,
        b.weight,
        b.fee,
        COALESCE(
            b.feerate,
            CASE 
                WHEN b.fee IS NOT NULL AND b.vsize IS NOT NULL AND b.vsize > 0
                    THEN (b.fee::numeric / b.vsize)
                ELSE NULL
            END
        ) as feerate,
        b.inputs_count,
        b.outputs_count,
        h.is_rbf,
        h.coinjoin_score,
        h.equal_output,
        h.likely_change_index
    FROM tx_basic b
    JOIN tx_heuristics h ON b.txid = h.txid
    ORDER BY b.block_height DESC
    LIMIT {limit}
    """
    
    with get_db_connection() as conn:
        # Use chunksize to show progress and manage memory
        chunk_size = 10000
        chunks = []
        total_loaded = 0
        
        logger.info(f"Executing query (fetching {limit} rows in chunks of {chunk_size})...")
        
        # Note: params argument in read_sql is used to pass parameters safely
        # We need to format the limit directly or use params, but read_sql params handling depends on driver
        # simpler here to keep f-string for limit as it is an int
        
        for chunk in pd.read_sql(query, conn, chunksize=chunk_size):
            chunks.append(chunk)
            total_loaded += len(chunk)
            logger.info(f"   ...loaded {total_loaded} rows so far")
            
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.DataFrame()
    
    logger.info("Total loaded %d records", len(df))
    
    if len(df) < min_samples:
        logger.warning(
            "Insufficient data: %d < %d. Some models may not work properly.",
            len(df), min_samples
        )
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for ML models.
    
    Args:
        df: Raw transaction DataFrame.
        
    Returns:
        DataFrame with engineered features.
    """
    logger.info("Engineering features...")
    
    df = df.copy()
    
    # Ratio features
    df['fee_per_input'] = df['fee'] / df['inputs_count']
    df['fee_per_output'] = df['fee'] / df['outputs_count']
    df['io_ratio'] = df['inputs_count'] / df['outputs_count']
    df['size_per_input'] = df['vsize'] / df['inputs_count']
    df['size_per_output'] = df['vsize'] / df['outputs_count']
    df['weight_to_size_ratio'] = df['weight'] / df['size']
    
    # Temporal features
    timestamps = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = timestamps.dt.hour
    df['day_of_week'] = timestamps.dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Binary features
    df['is_rbf_int'] = df['is_rbf'].astype(int)
    df['equal_output_int'] = df['equal_output'].astype(int)
    df['has_change'] = df['likely_change_index'].notna().astype(int)
    
    # Clean up invalid values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df
