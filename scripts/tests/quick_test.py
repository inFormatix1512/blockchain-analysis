#!/usr/bin/env python3
"""Test rapido del sistema."""
import psycopg2

try:
    conn = psycopg2.connect(host='localhost', port=5432, user='postgres', password='postgres', dbname='blockchain')
    cur = conn.cursor()
    
    cur.execute('SELECT COUNT(*) FROM tx_basic')
    print(f'Transazioni: {cur.fetchone()[0]}')
    
    cur.execute('SELECT COUNT(*) FROM mempool_snapshot')
    print(f'Snapshot: {cur.fetchone()[0]}')
    
    cur.execute('SELECT MAX(block_height) FROM tx_basic')
    print(f'Ultimo blocco: {cur.fetchone()[0]}')
    
    conn.close()
    print('✅ Sistema funzionante!')
except Exception as e:
    print(f'❌ Errore: {e}')
