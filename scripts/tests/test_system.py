#!/usr/bin/env python3
"""
Script di test per verificare il funzionamento del sistema di ingestione.
Mostra statistiche sul database e verifica la connettività.
"""

import psycopg2
import requests
from datetime import datetime

def test_database():
    """Testa la connessione al database e mostra statistiche."""
    print("=" * 60)
    print("🔍 TEST DATABASE POSTGRESQL")
    print("=" * 60)
    
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            user='postgres',
            password='postgres',
            dbname='blockchain'
        )
        cur = conn.cursor()
        
        # Statistiche transazioni
        cur.execute('SELECT COUNT(*) FROM tx_basic')
        tx_count = cur.fetchone()[0]
        print(f"✅ Transazioni totali: {tx_count}")
        
        # Statistiche snapshot
        cur.execute('SELECT COUNT(*) FROM mempool_snapshot')
        snap_count = cur.fetchone()[0]
        print(f"✅ Snapshot mempool: {snap_count}")
        
        # Statistiche euristiche
        cur.execute('SELECT COUNT(*) FROM tx_heuristics')
        heur_count = cur.fetchone()[0]
        print(f"✅ Euristiche calcolate: {heur_count}")
        
        # Blocchi processati
        cur.execute('SELECT MIN(block_height), MAX(block_height) FROM tx_basic')
        min_block, max_block = cur.fetchone()
        if min_block and max_block:
            print(f"✅ Range blocchi: {min_block} - {max_block}")
        
        # Ultimi 5 blocchi
        print("\n📊 Ultimi 5 blocchi processati:")
        cur.execute('''
            SELECT block_height, COUNT(*) as tx_count, 
                   TO_CHAR(MIN(ts), 'YYYY-MM-DD HH24:MI:SS') as first_tx
            FROM tx_basic 
            GROUP BY block_height 
            ORDER BY block_height DESC 
            LIMIT 5
        ''')
        for row in cur.fetchall():
            print(f"   Blocco {row[0]}: {row[1]} tx, prima tx: {row[2]}")
        
        # Euristiche RBF
        cur.execute('SELECT COUNT(*) FROM tx_heuristics WHERE is_rbf = true')
        rbf_count = cur.fetchone()[0]
        print(f"\n🔍 Transazioni con RBF: {rbf_count}")
        
        # Euristiche CoinJoin
        cur.execute('SELECT COUNT(*) FROM tx_heuristics WHERE coinjoin_score > 0.5')
        coinjoin_count = cur.fetchone()[0]
        print(f"🔍 Possibili CoinJoin (score > 0.5): {coinjoin_count}")
        
        conn.close()
        print("\n✅ Test database completato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore database: {e}")
        return False

def test_bitcoin_rpc():
    """Testa la connessione al nodo Bitcoin."""
    print("\n" + "=" * 60)
    print("🔍 TEST BITCOIN RPC")
    print("=" * 60)
    
    try:
        response = requests.post(
            'http://localhost:8332',
            auth=('bitcoin', 'bitcoin123'),
            json={
                'jsonrpc': '2.0',
                'id': 'test',
                'method': 'getblockchaininfo',
                'params': []
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json().get('result', {})
            print(f"✅ Connessione RPC attiva")
            print(f"✅ Chain: {result.get('chain', 'N/A')}")
            print(f"✅ Blocchi: {result.get('blocks', 'N/A')}")
            print(f"✅ Headers: {result.get('headers', 'N/A')}")
            print(f"✅ Pruned: {result.get('pruned', 'N/A')}")
            if result.get('pruned'):
                print(f"✅ Prune height: {result.get('pruneheight', 'N/A')}")
            return True
        else:
            print(f"❌ Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Errore RPC: {e}")
        return False

def test_mempool():
    """Testa il mempool snapshot."""
    print("\n" + "=" * 60)
    print("🔍 TEST MEMPOOL")
    print("=" * 60)
    
    try:
        response = requests.post(
            'http://localhost:8332',
            auth=('bitcoin', 'bitcoin123'),
            json={
                'jsonrpc': '2.0',
                'id': 'test',
                'method': 'getmempoolinfo',
                'params': []
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json().get('result', {})
            print(f"✅ Mempool size: {result.get('size', 0)} transazioni")
            print(f"✅ Mempool bytes: {result.get('bytes', 0):,}")
            print(f"✅ Usage: {result.get('usage', 0):,} bytes")
            print(f"✅ Max mempool: {result.get('maxmempool', 0):,} bytes")
            return True
        else:
            print(f"❌ Status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Errore mempool: {e}")
        return False

def main():
    """Esegue tutti i test."""
    print("\n🚀 AVVIO TEST SISTEMA BLOCKCHAIN ANALYSIS")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = []
    results.append(("Database", test_database()))
    results.append(("Bitcoin RPC", test_bitcoin_rpc()))
    results.append(("Mempool", test_mempool()))
    
    print("\n" + "=" * 60)
    print("📋 RIEPILOGO TEST")
    print("=" * 60)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 TUTTI I TEST SUPERATI - SISTEMA OPERATIVO!")
    else:
        print("⚠️  ALCUNI TEST FALLITI - VERIFICA I LOG")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
