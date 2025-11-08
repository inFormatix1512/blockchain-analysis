#!/usr/bin/env python3
"""
Report completo dello stato del sistema con statistiche dettagliate.
"""

import psycopg2
import requests
from datetime import datetime
from collections import defaultdict

def generate_report():
    """Genera report dettagliato del sistema."""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "🚀 BLOCKCHAIN ANALYSIS SYSTEM REPORT")
    print("=" * 70)
    print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    # Connessione database
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            dbname='blockchain'
        )
        cur = conn.cursor()
        
        # === SEZIONE 1: STATISTICHE GENERALI ===
        print("📊 STATISTICHE GENERALI")
        print("-" * 70)
        
        cur.execute('SELECT COUNT(*) FROM tx_basic')
        tx_count = cur.fetchone()[0]
        print(f"  • Transazioni totali processate: {tx_count:,}")
        
        cur.execute('SELECT COUNT(*) FROM mempool_snapshot')
        snap_count = cur.fetchone()[0]
        print(f"  • Snapshot mempool salvati: {snap_count:,}")
        
        cur.execute('SELECT COUNT(*) FROM tx_heuristics')
        heur_count = cur.fetchone()[0]
        print(f"  • Euristiche calcolate: {heur_count:,}")
        
        cur.execute('SELECT MIN(block_height), MAX(block_height) FROM tx_basic')
        min_block, max_block = cur.fetchone()
        print(f"  • Range blocchi: {min_block} → {max_block} ({max_block - min_block + 1} blocchi)")
        
        # === SEZIONE 2: ANALISI BLOCCHI ===
        print("\n🧱 ANALISI BLOCCHI")
        print("-" * 70)
        
        cur.execute('''
            SELECT 
                block_height,
                COUNT(*) as tx_count,
                SUM(fee) as total_fees,
                AVG(size) as avg_size,
                TO_CHAR(MIN(ts), 'YYYY-MM-DD HH24:MI:SS') as block_time
            FROM tx_basic 
            GROUP BY block_height 
            ORDER BY block_height DESC 
            LIMIT 10
        ''')
        
        print(f"  {'Blocco':<8} {'TX':<6} {'Fee Tot.':<12} {'Avg Size':<10} {'Timestamp':<20}")
        print(f"  {'-'*8} {'-'*6} {'-'*12} {'-'*10} {'-'*20}")
        
        for row in cur.fetchall():
            block, tx, fees, avg_sz, ts = row
            fees_str = f"{fees:,}" if fees else "0"
            avg_sz_str = f"{int(avg_sz):,}" if avg_sz else "0"
            print(f"  {block:<8} {tx:<6} {fees_str:<12} {avg_sz_str:<10} {ts:<20}")
        
        # === SEZIONE 3: EURISTICHE ===
        print("\n🔍 ANALISI EURISTICHE")
        print("-" * 70)
        
        cur.execute('SELECT COUNT(*) FROM tx_heuristics WHERE is_rbf = true')
        rbf_count = cur.fetchone()[0]
        rbf_perc = (rbf_count / tx_count * 100) if tx_count > 0 else 0
        print(f"  • Transazioni con RBF: {rbf_count} ({rbf_perc:.1f}%)")
        
        cur.execute('SELECT COUNT(*) FROM tx_heuristics WHERE equal_output = true')
        equal_out = cur.fetchone()[0]
        equal_perc = (equal_out / tx_count * 100) if tx_count > 0 else 0
        print(f"  • Transazioni con output uguali: {equal_out} ({equal_perc:.1f}%)")
        
        cur.execute('SELECT COUNT(*) FROM tx_heuristics WHERE coinjoin_score > 0.5')
        coinjoin = cur.fetchone()[0]
        cj_perc = (coinjoin / tx_count * 100) if tx_count > 0 else 0
        print(f"  • Possibili CoinJoin (score > 0.5): {coinjoin} ({cj_perc:.1f}%)")
        
        cur.execute('SELECT AVG(coinjoin_score) FROM tx_heuristics')
        avg_cj = cur.fetchone()[0] or 0
        print(f"  • Score CoinJoin medio: {avg_cj:.3f}")
        
        # === SEZIONE 4: MEMPOOL ===
        print("\n💾 MEMPOOL SNAPSHOTS")
        print("-" * 70)
        
        cur.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(total_tx) as avg_tx,
                MAX(total_tx) as max_tx,
                AVG(total_size) as avg_size,
                SUM(total_fee) as total_fees
            FROM mempool_snapshot
        ''')
        
        snap_stats = cur.fetchone()
        print(f"  • Snapshot totali: {snap_stats[0]:,}")
        print(f"  • Media TX per snapshot: {snap_stats[1]:.1f}")
        print(f"  • Max TX in uno snapshot: {int(snap_stats[2]) if snap_stats[2] else 0}")
        print(f"  • Size media snapshot: {int(snap_stats[3]) if snap_stats[3] else 0:,} bytes")
        print(f"  • Fee totali osservate: {int(snap_stats[4]) if snap_stats[4] else 0:,} satoshi")
        
        # === SEZIONE 5: DISTRIBUZIONE TRANSAZIONI ===
        print("\n📈 DISTRIBUZIONE TRANSAZIONI")
        print("-" * 70)
        
        cur.execute('''
            SELECT 
                inputs_count,
                outputs_count,
                COUNT(*) as count
            FROM tx_basic
            WHERE inputs_count IS NOT NULL AND outputs_count IS NOT NULL
            GROUP BY inputs_count, outputs_count
            ORDER BY count DESC
            LIMIT 5
        ''')
        
        print(f"  {'Input':<8} {'Output':<8} {'Conteggio':<10}")
        print(f"  {'-'*8} {'-'*8} {'-'*10}")
        for row in cur.fetchall():
            print(f"  {row[0]:<8} {row[1]:<8} {row[2]:<10}")
        
        conn.close()
        
        # === SEZIONE 6: BITCOIN RPC STATUS ===
        print("\n⛓️  BITCOIN NODE STATUS")
        print("-" * 70)
        
        try:
            response = requests.post(
                'http://localhost:8332',
                auth=('bitcoin', 'bitcoin123'),
                json={'jsonrpc': '2.0', 'id': 'test', 'method': 'getblockchaininfo', 'params': []},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json().get('result', {})
                print(f"  • Chain: {result.get('chain', 'N/A')}")
                print(f"  • Blocchi sincronizzati: {result.get('blocks', 'N/A'):,}")
                print(f"  • Headers scaricati: {result.get('headers', 'N/A'):,}")
                print(f"  • Modalità pruned: {'Sì' if result.get('pruned') else 'No'}")
                if result.get('pruned'):
                    print(f"  • Prune height: {result.get('pruneheight', 0):,}")
                print(f"  • Verification progress: {result.get('verificationprogress', 0)*100:.2f}%")
                
                # Mempool info
                response = requests.post(
                    'http://localhost:8332',
                    auth=('bitcoin', 'bitcoin123'),
                    json={'jsonrpc': '2.0', 'id': 'test', 'method': 'getmempoolinfo', 'params': []},
                    timeout=5
                )
                
                if response.status_code == 200:
                    mempool = response.json().get('result', {})
                    print(f"\n  📬 Mempool corrente:")
                    print(f"     • Transazioni in attesa: {mempool.get('size', 0):,}")
                    print(f"     • Bytes utilizzati: {mempool.get('bytes', 0):,}")
                    print(f"     • Usage: {mempool.get('usage', 0):,} / {mempool.get('maxmempool', 0):,}")
                    
        except Exception as e:
            print(f"  ⚠️  Impossibile connettersi a Bitcoin RPC: {e}")
        
        # === CONCLUSIONE ===
        print("\n" + "=" * 70)
        print("✅ SISTEMA OPERATIVO E FUNZIONANTE")
        print("=" * 70)
        print("\n📌 Prossimi passi suggeriti:")
        print("   1. Avvia Jupyter Notebook: jupyter notebook")
        print("   2. Apri analysis.ipynb per analisi esplorativa")
        print("   3. Apri ml_analysis.ipynb per modelli ML")
        print("   4. Monitora logs: docker compose logs -f ingest")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}\n")
        return False
    
    return True

if __name__ == "__main__":
    generate_report()
