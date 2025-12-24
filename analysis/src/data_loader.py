import pandas as pd
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from common import get_db_connection

def load_all_data(limit=None):
    """Carica i dati necessari per l'analisi dal database."""
    print("\n" + "="*80)
    print("CARICAMENTO DATI DAL DATABASE (FULL DATASET)")
    print("="*80)
    
    # Query ottimizzata: NO TXID (risparmio RAM), solo colonne numeriche/booleane
    query_tx = """
        SELECT 
            t.block_height,
            EXTRACT(EPOCH FROM t.ts) as ts,
            t.size,
            t.vsize,
            t.weight,
            t.fee,
            t.inputs_count,
            t.outputs_count,
            h.is_rbf,
            h.coinjoin_score,
            h.equal_output
        FROM tx_basic t
        LEFT JOIN tx_heuristics h ON t.txid = h.txid
    """
    
    if limit:
        query_tx += f" LIMIT {limit}"
    
    print(f"\nEsecuzione query su intero dataset (può richiedere minuti)...")
    with get_db_connection() as conn:
        # Use chunksize to show progress and manage memory
        chunk_size = 500000  # 500k righe per chunk
        chunks = []
        total_loaded = 0
        
        # Passiamo params vuoto se limit è None, altrimenti gestito nella f-string sopra per semplicità
        for chunk in pd.read_sql(query_tx, conn, chunksize=chunk_size):
            # Ottimizzazione tipi per risparmiare RAM
            chunk['block_height'] = chunk['block_height'].astype('int32')
            chunk['size'] = chunk['size'].astype('int32')
            chunk['vsize'] = chunk['vsize'].astype('int32')
            chunk['weight'] = chunk['weight'].astype('int32')
            chunk['inputs_count'] = chunk['inputs_count'].astype('int16')
            chunk['outputs_count'] = chunk['outputs_count'].astype('int16')
            # is_rbf e equal_output sono già bool o object, convertiamo in bool/int
            chunk['is_rbf'] = chunk['is_rbf'].fillna(False).astype(bool)
            chunk['equal_output'] = chunk['equal_output'].fillna(False).astype(bool)
            chunk['coinjoin_score'] = chunk['coinjoin_score'].astype('float32')
            
            chunks.append(chunk)
            total_loaded += len(chunk)
            print(f"   ...caricate {total_loaded:,} righe")
            
        if chunks:
            df_tx = pd.concat(chunks, ignore_index=True)
        else:
            df_tx = pd.DataFrame()
            
    print(f"[OK] {len(df_tx):,} transazioni caricate in memoria.")
    
    # Query mempool snapshot (se disponibile)
    print("\nCaricamento mempool snapshots...")
    try:
        query_mempool = """
            SELECT 
                id, ts, total_tx, total_size, total_fee
            FROM mempool_snapshot
            ORDER BY ts DESC
            LIMIT 1000
        """
        with get_db_connection() as conn:
            df_mempool = pd.read_sql(query_mempool, conn)
        print(f"[OK] {len(df_mempool):,} snapshot caricati")
    except Exception as e:
        df_mempool = pd.DataFrame()
        print(f"[WARN] Nessun snapshot mempool disponibile: {e}")
    
    return df_tx, df_mempool
