#!/usr/bin/env python3
"""
Analisi esplorativa rapida dei dati blockchain.
Alternativa a Jupyter Notebook per visualizzare i dati raccolti.
"""

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configurazione stile
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def connect_db():
    """Connessione al database."""
    return psycopg2.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password='postgres',
        dbname='blockchain'
    )

def analyze_transactions():
    """Analisi delle transazioni."""
    print("\n" + "="*70)
    print("📊 ANALISI TRANSAZIONI BLOCKCHAIN")
    print("="*70 + "\n")
    
    conn = connect_db()
    
    # Carica dati transazioni
    query = """
        SELECT 
            t.txid,
            t.block_height,
            t.ts,
            t.size,
            t.vsize,
            t.weight,
            t.fee,
            t.feerate,
            t.inputs_count,
            t.outputs_count,
            h.is_rbf,
            h.coinjoin_score,
            h.equal_output
        FROM tx_basic t
        LEFT JOIN tx_heuristics h ON t.txid = h.txid
        ORDER BY t.block_height
    """
    
    df = pd.read_sql(query, conn)
    
    print(f"📈 Dataset: {len(df)} transazioni caricate\n")
    
    # Statistiche descrittive
    print("📊 STATISTICHE DESCRITTIVE")
    print("-" * 70)
    print(df[['block_height', 'size', 'inputs_count', 'outputs_count', 'fee']].describe())
    
    # Analisi per blocco
    print("\n\n🧱 ANALISI PER BLOCCO")
    print("-" * 70)
    blocks = df.groupby('block_height').agg({
        'txid': 'count',
        'size': 'mean',
        'fee': 'sum',
        'inputs_count': 'mean',
        'outputs_count': 'mean'
    }).round(2)
    blocks.columns = ['TX Count', 'Avg Size', 'Total Fees', 'Avg Inputs', 'Avg Outputs']
    print(blocks.tail(10))
    
    # Euristiche
    print("\n\n🔍 ANALISI EURISTICHE")
    print("-" * 70)
    print(f"Transazioni con RBF: {df['is_rbf'].sum()} ({df['is_rbf'].sum()/len(df)*100:.1f}%)")
    print(f"Transazioni con output uguali: {df['equal_output'].sum()} ({df['equal_output'].sum()/len(df)*100:.1f}%)")
    print(f"CoinJoin score medio: {df['coinjoin_score'].mean():.3f}")
    print(f"CoinJoin score max: {df['coinjoin_score'].max():.3f}")
    
    # Distribuzione input/output
    print("\n\n📈 DISTRIBUZIONE INPUT/OUTPUT")
    print("-" * 70)
    io_dist = df.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
    io_dist = io_dist.sort_values('count', ascending=False).head(10)
    print(io_dist.to_string(index=False))
    
    conn.close()
    
    return df

def analyze_mempool():
    """Analisi snapshot mempool."""
    print("\n\n" + "="*70)
    print("💾 ANALISI MEMPOOL SNAPSHOTS")
    print("="*70 + "\n")
    
    conn = connect_db()
    
    query = """
        SELECT 
            id,
            ts,
            total_tx,
            total_size,
            total_fee
        FROM mempool_snapshot
        ORDER BY ts DESC
        LIMIT 20
    """
    
    df = pd.read_sql(query, conn)
    
    print(f"📸 {len(df)} snapshot caricati\n")
    
    if len(df) > 0:
        print("📊 STATISTICHE MEMPOOL")
        print("-" * 70)
        print(df[['total_tx', 'total_size', 'total_fee']].describe())
        
        print("\n\n📋 ULTIMI 10 SNAPSHOT")
        print("-" * 70)
        print(df[['ts', 'total_tx', 'total_size', 'total_fee']].head(10).to_string(index=False))
    else:
        print("⚠️  Nessun snapshot disponibile")
    
    conn.close()
    
    return df

def generate_summary():
    """Genera riepilogo generale."""
    print("\n\n" + "="*70)
    print("📋 RIEPILOGO GENERALE")
    print("="*70 + "\n")
    
    conn = connect_db()
    cur = conn.cursor()
    
    # Conta record
    cur.execute('SELECT COUNT(*) FROM tx_basic')
    tx_count = cur.fetchone()[0]
    
    cur.execute('SELECT COUNT(*) FROM mempool_snapshot')
    snap_count = cur.fetchone()[0]
    
    cur.execute('SELECT MIN(block_height), MAX(block_height) FROM tx_basic')
    min_block, max_block = cur.fetchone()
    
    cur.execute('SELECT MIN(ts), MAX(ts) FROM tx_basic')
    first_ts, last_ts = cur.fetchone()
    
    print(f"📊 Transazioni totali: {tx_count:,}")
    print(f"📸 Snapshot mempool: {snap_count:,}")
    print(f"🧱 Range blocchi: {min_block} → {max_block}")
    print(f"⏰ Periodo temporale: {first_ts} → {last_ts}")
    
    # Spazio occupato
    cur.execute("""
        SELECT 
            pg_size_pretty(pg_total_relation_size('tx_basic')) as tx_size,
            pg_size_pretty(pg_total_relation_size('tx_heuristics')) as heur_size,
            pg_size_pretty(pg_total_relation_size('mempool_snapshot')) as snap_size
    """)
    sizes = cur.fetchone()
    
    print(f"\n💾 SPAZIO DATABASE")
    print("-" * 70)
    print(f"Tabella tx_basic: {sizes[0]}")
    print(f"Tabella tx_heuristics: {sizes[1]}")
    print(f"Tabella mempool_snapshot: {sizes[2]}")
    
    conn.close()

def main():
    """Esegue l'analisi completa."""
    # Fix encoding per Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n" + "="*70)
    print(" " * 15 + "🔬 ANALISI ESPLORATIVA DATI BLOCKCHAIN")
    print("="*70)
    print(f"⏰ Eseguita il: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    try:
        # Analisi transazioni
        df_tx = analyze_transactions()
        
        # Analisi mempool
        df_mempool = analyze_mempool()
        
        # Riepilogo
        generate_summary()
        
        print("\n" + "="*70)
        print("✅ ANALISI COMPLETATA CON SUCCESSO")
        print("="*70 + "\n")
        
        # Suggerimenti
        print("💡 PROSSIMI PASSI:")
        print("   • Usa questi DataFrames per ulteriori analisi")
        print("   • Crea visualizzazioni con matplotlib/seaborn")
        print("   • Applica modelli ML dai dati raccolti")
        print("   • Esporta i dati: df.to_csv('export.csv')")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}\n")
        return False
    
    return True

if __name__ == "__main__":
    main()
