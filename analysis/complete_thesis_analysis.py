#!/usr/bin/env python3
"""
ANALISI COMPLETA BLOCKCHAIN - TESI TRIENNALE
Università degli Studi di Catania - Ingegneria Informatica
Autore: Luca Impellizzeri

Script consolidato che combina:
- Analisi esplorativa dati (analyze_data.py)
- Generazione risultati sperimentali (experimental_results.py)
- Report per tesi (generate_thesis_report.py)

Output:
- Statistiche descrittive complete
- Grafici per la tesi (salvati in results/)
- Report tabellare per LaTeX/Word
"""

import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Configurazione
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'postgres',
    'dbname': 'blockchain'
}

OUTPUT_DIR = 'results/thesis_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stile grafici per tesi
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# CONNESSIONE DATABASE
# ============================================================================

def connect_db():
    """Stabilisce connessione al database PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("[OK] Connessione database stabilita")
        return conn
    except Exception as e:
        print(f"[ERRORE] Connessione database fallita: {e}")
        sys.exit(1)

# ============================================================================
# CARICAMENTO DATI
# ============================================================================

def load_all_data(conn):
    """Carica tutti i dati necessari per l'analisi dal database."""
    print("\n" + "="*80)
    print("CARICAMENTO DATI DAL DATABASE")
    print("="*80)
    
    # Query principale transazioni con euristiche
    query_tx = """
        SELECT 
            t.txid,
            t.block_height,
            t.ts,
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
        ORDER BY t.block_height
    """
    
    print("\nCaricamento transazioni in corso...")
    df_tx = pd.read_sql(query_tx, conn)
    print(f"[OK] {len(df_tx):,} transazioni caricate")
    
    # Query mempool snapshot (se disponibile)
    print("\nCaricamento mempool snapshots...")
    try:
        query_mempool = """
            SELECT 
                id, ts, total_tx, total_size, total_fee
            FROM mempool_snapshot
            ORDER BY ts DESC
        """
        df_mempool = pd.read_sql(query_mempool, conn)
        print(f"[OK] {len(df_mempool):,} snapshot caricati")
    except:
        df_mempool = pd.DataFrame()
        print("[WARN] Nessun snapshot mempool disponibile")
    
    return df_tx, df_mempool

# ============================================================================
# STATISTICHE DESCRITTIVE
# ============================================================================

def print_dataset_overview(df_tx):
    """Stampa statistiche generali del dataset."""
    print("\n" + "="*80)
    print("OVERVIEW DATASET")
    print("="*80)
    
    print(f"\n  Transazioni totali:  {len(df_tx):,}")
    print(f"  Blocco iniziale:     {df_tx['block_height'].min()}")
    print(f"  Blocco finale:       {df_tx['block_height'].max()}")
    print(f"  Blocchi unici:       {df_tx['block_height'].nunique():,}")
    print(f"  Periodo temporale:   {df_tx['ts'].min()} - {df_tx['ts'].max()}")
    
    # Media transazioni per blocco
    avg_tx_per_block = len(df_tx) / df_tx['block_height'].nunique()
    print(f"  Media tx/blocco:     {avg_tx_per_block:.2f}")

def print_io_distribution(df_tx):
    """Analisi distribuzione input/output."""
    print("\n" + "="*80)
    print("DISTRIBUZIONE INPUT/OUTPUT")
    print("="*80)
    
    # Top 15 pattern
    io_dist = df_tx.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
    io_dist['percentage'] = io_dist['count'] / len(df_tx) * 100
    io_dist = io_dist.sort_values('count', ascending=False).head(15)
    
    print("\nTop 15 pattern più comuni:")
    print("-" * 60)
    print(f"{'Input':<8} {'Output':<8} {'Transazioni':<15} {'%':<10}")
    print("-" * 60)
    for _, row in io_dist.iterrows():
        print(f"{row['inputs_count']:<8} {row['outputs_count']:<8} {row['count']:>12,}   {row['percentage']:>6.2f}%")

def print_heuristics_analysis(df_tx):
    """Analisi euristiche (RBF, CoinJoin, Equal Output)."""
    print("\n" + "="*80)
    print("ANALISI EURISTICHE")
    print("="*80)
    
    total = len(df_tx)
    rbf_count = df_tx['is_rbf'].sum()
    equal_out_count = df_tx['equal_output'].sum()
    
    print(f"\nReplace-By-Fee (RBF):")
    print(f"  Transazioni: {rbf_count:,} ({rbf_count/total*100:.2f}%)")
    
    print(f"\nEqual Output (CoinJoin indicator):")
    print(f"  Transazioni: {equal_out_count:,} ({equal_out_count/total*100:.2f}%)")
    
    print(f"\nCoinJoin Score:")
    print(f"  Media: {df_tx['coinjoin_score'].mean():.4f}")
    print(f"  Mediana: {df_tx['coinjoin_score'].median():.4f}")
    print(f"  Max: {df_tx['coinjoin_score'].max():.4f}")
    print(f"  Score > 0.7: {(df_tx['coinjoin_score'] > 0.7).sum():,} transazioni")
    print(f"  Score = 1.0: {(df_tx['coinjoin_score'] == 1.0).sum():,} transazioni")

# ============================================================================
# GENERAZIONE GRAFICI
# ============================================================================

def plot_io_distribution(df_tx):
    """Genera grafico distribuzione input/output."""
    print("\nGenerazione grafico distribuzione I/O...")
    
    # Top 10 pattern
    io_dist = df_tx.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
    io_dist['label'] = io_dist['inputs_count'].astype(str) + '→' + io_dist['outputs_count'].astype(str)
    io_dist = io_dist.sort_values('count', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(io_dist['label'], io_dist['count'], color='steelblue', edgecolor='black', alpha=0.7)
    
    # Percentuali sopra le barre
    for bar, count in zip(bars, io_dist['count']):
        height = bar.get_height()
        percentage = count / len(df_tx) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Pattern Input → Output', fontsize=13, fontweight='bold')
    ax.set_ylabel('Numero Transazioni', fontsize=13, fontweight='bold')
    ax.set_title('Distribuzione Pattern Input/Output (Top 10)', fontsize=15, fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    filepath = os.path.join(OUTPUT_DIR, 'fig1_io_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_heuristics_pie(df_tx):
    """Genera grafico a torta per euristiche."""
    print("\nGenerazione grafico euristiche...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # RBF
    rbf_data = [df_tx['is_rbf'].sum(), len(df_tx) - df_tx['is_rbf'].sum()]
    colors1 = ['#ff6b6b', '#95e1d3']
    explode1 = (0.1, 0)
    
    ax1.pie(rbf_data, labels=['RBF Enabled', 'RBF Disabled'], 
            autopct='%1.1f%%', startangle=90, colors=colors1, explode=explode1,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title('Replace-By-Fee (RBF) Adoption', fontsize=14, fontweight='bold')
    
    # Equal Output
    equal_data = [df_tx['equal_output'].sum(), len(df_tx) - df_tx['equal_output'].sum()]
    colors2 = ['#feca57', '#48dbfb']
    explode2 = (0.1, 0)
    
    ax2.pie(equal_data, labels=['Equal Outputs', 'Mixed Outputs'],
            autopct='%1.1f%%', startangle=90, colors=colors2, explode=explode2,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Equal Output Pattern (CoinJoin Indicator)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig2_heuristics_pie.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_coinjoin_distribution(df_tx):
    """Genera istogramma distribuzione CoinJoin Score."""
    print("\nGenerazione grafico CoinJoin score...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Istogramma
    n, bins, patches = ax.hist(df_tx['coinjoin_score'], bins=50, 
                                color='purple', alpha=0.7, edgecolor='black')
    
    # Colora diversamente le barre con score > 0.7
    for i, patch in enumerate(patches):
        if bins[i] > 0.7:
            patch.set_facecolor('red')
            patch.set_alpha(0.9)
    
    # Linea mediana
    median = df_tx['coinjoin_score'].median()
    ax.axvline(median, color='green', linestyle='--', linewidth=2, 
               label=f'Mediana: {median:.3f}')
    
    # Linea soglia CoinJoin
    ax.axvline(0.7, color='red', linestyle='--', linewidth=2,
               label='Soglia CoinJoin: 0.7')
    
    ax.set_xlabel('CoinJoin Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('Numero Transazioni', fontsize=13, fontweight='bold')
    ax.set_title('Distribuzione CoinJoin Score', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig3_coinjoin_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_tx_size_distribution(df_tx):
    """Genera istogramma distribuzione dimensione transazioni."""
    print("\nGenerazione grafico dimensione transazioni...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Size distribution
    ax1.hist(df_tx['size'], bins=100, color='teal', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Size (bytes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Numero Transazioni', fontsize=12, fontweight='bold')
    ax1.set_title('Distribuzione Dimensione Transazioni', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, df_tx['size'].quantile(0.95))  # Zoom su 95% dei dati
    
    # VSize distribution
    ax2.hist(df_tx['vsize'], bins=100, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('VSize (vbytes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Numero Transazioni', fontsize=12, fontweight='bold')
    ax2.set_title('Distribuzione Virtual Size (SegWit)', fontsize=13, fontweight='bold')
    ax2.set_xlim(0, df_tx['vsize'].quantile(0.95))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig4_tx_size_distribution.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

def plot_blocks_timeline(df_tx):
    """Genera scatter plot timeline blocchi analizzati."""
    print("\nGenerazione timeline blocchi...")
    
    # Raggruppa per blocco
    blocks = df_tx.groupby('block_height').size().reset_index(name='tx_count')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.scatter(blocks['block_height'], blocks['tx_count'], 
               alpha=0.5, s=10, color='darkblue')
    
    ax.set_xlabel('Block Height', fontsize=13, fontweight='bold')
    ax.set_ylabel('Transazioni per Blocco', fontsize=13, fontweight='bold')
    ax.set_title('Distribuzione Temporale Blocchi Analizzati', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Annotazione
    ax.text(0.02, 0.98, f'Blocchi totali: {len(blocks):,}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig5_blocks_timeline.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Salvato: {filepath}")

# ============================================================================
# ESPORTAZIONE TABELLE
# ============================================================================

def export_tables(df_tx):
    """Esporta tabelle in formato CSV per LaTeX/Word."""
    print("\nEsportazione tabelle...")
    
    # Tabella 1: Top 15 I/O patterns
    io_dist = df_tx.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
    io_dist['percentage'] = (io_dist['count'] / len(df_tx) * 100).round(2)
    io_dist = io_dist.sort_values('count', ascending=False).head(15)
    io_dist.columns = ['Input', 'Output', 'Transazioni', 'Percentuale (%)']
    
    filepath1 = os.path.join(OUTPUT_DIR, 'table1_io_patterns.csv')
    io_dist.to_csv(filepath1, index=False, sep=';')
    print(f"[OK] Salvato: {filepath1}")
    
    # Tabella 2: Statistiche euristiche
    heuristics_stats = pd.DataFrame({
        'Euristica': ['RBF Enabled', 'Equal Output', 'CoinJoin Score > 0.7', 'CoinJoin Score = 1.0'],
        'Count': [
            df_tx['is_rbf'].sum(),
            df_tx['equal_output'].sum(),
            (df_tx['coinjoin_score'] > 0.7).sum(),
            (df_tx['coinjoin_score'] == 1.0).sum()
        ],
        'Percentuale (%)': [
            (df_tx['is_rbf'].sum() / len(df_tx) * 100).round(2),
            (df_tx['equal_output'].sum() / len(df_tx) * 100).round(2),
            ((df_tx['coinjoin_score'] > 0.7).sum() / len(df_tx) * 100).round(2),
            ((df_tx['coinjoin_score'] == 1.0).sum() / len(df_tx) * 100).round(2)
        ]
    })
    
    filepath2 = os.path.join(OUTPUT_DIR, 'table2_heuristics_stats.csv')
    heuristics_stats.to_csv(filepath2, index=False, sep=';')
    print(f"[OK] Salvato: {filepath2}")
    
    # Tabella 3: Statistiche descrittive generali
    general_stats = pd.DataFrame({
        'Metrica': [
            'Transazioni Totali',
            'Blocchi Unici',
            'Blocco Min',
            'Blocco Max',
            'Media TX/Blocco',
            'Media Input/TX',
            'Media Output/TX',
            'Media Size (bytes)',
            'Media VSize (vbytes)'
        ],
        'Valore': [
            f"{len(df_tx):,}",
            f"{df_tx['block_height'].nunique():,}",
            f"{df_tx['block_height'].min()}",
            f"{df_tx['block_height'].max()}",
            f"{len(df_tx) / df_tx['block_height'].nunique():.2f}",
            f"{df_tx['inputs_count'].mean():.2f}",
            f"{df_tx['outputs_count'].mean():.2f}",
            f"{df_tx['size'].mean():.2f}",
            f"{df_tx['vsize'].mean():.2f}"
        ]
    })
    
    filepath3 = os.path.join(OUTPUT_DIR, 'table3_general_stats.csv')
    general_stats.to_csv(filepath3, index=False, sep=';')
    print(f"[OK] Salvato: {filepath3}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue l'analisi completa e genera tutti gli output."""
    
    # Header
    print("\n" + "="*80)
    print(" " * 15 + "ANALISI COMPLETA BLOCKCHAIN - TESI")
    print("="*80)
    print(f"Avviata il: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Connessione DB
    conn = connect_db()
    
    # Caricamento dati
    df_tx, df_mempool = load_all_data(conn)
    
    if len(df_tx) == 0:
        print("\n[ERRORE] Nessun dato disponibile nel database.")
        sys.exit(1)
    
    # Statistiche descrittive
    print_dataset_overview(df_tx)
    print_io_distribution(df_tx)
    print_heuristics_analysis(df_tx)
    
    # Generazione grafici
    print("\n" + "="*80)
    print("GENERAZIONE GRAFICI")
    print("="*80)
    
    plot_io_distribution(df_tx)
    plot_heuristics_pie(df_tx)
    plot_coinjoin_distribution(df_tx)
    plot_tx_size_distribution(df_tx)
    plot_blocks_timeline(df_tx)
    
    # Esportazione tabelle
    print("\n" + "="*80)
    print("ESPORTAZIONE TABELLE")
    print("="*80)
    
    export_tables(df_tx)
    
    # Chiusura
    conn.close()
    
    print("\n" + "="*80)
    print("ANALISI COMPLETATA CON SUCCESSO")
    print("="*80)
    print(f"\nTutti i file salvati in: {OUTPUT_DIR}/")
    print("\nFile generati:")
    print("  - 5 grafici PNG (300 DPI)")
    print("  - 3 tabelle CSV (formato Excel/LaTeX)")
    print("\nProssimi passi:")
    print("  - Inserire i grafici nella tesi")
    print("  - Importare le tabelle CSV in Excel/Word/LaTeX")
    print("  - Consultare REPORT_RISULTATI_SPERIMENTALI_FINALE.md\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARN] Analisi interrotta dall'utente")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERRORE] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
