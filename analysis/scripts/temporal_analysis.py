#!/usr/bin/env python3
"""
ANALISI TEMPORALE - TESI TRIENNALE
Analisi dei trend storici basata su range di blocchi.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add repo root to path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from analysis.src.data_loader import load_all_data
from datetime import datetime

# Create a unique output directory based on timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = f'analysis/results/thesis_temporal_{timestamp}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_temporal_trends(df, bin_size=50000):
    """
    Analizza i dati raggruppandoli per range di blocchi.
    
    Args:
        df (pd.DataFrame): DataFrame con i dati delle transazioni.
        bin_size (int): Dimensione del range di blocchi (default 50k).
    """
    print(f"\nAvvio analisi temporale (Bin size: {bin_size} blocchi)...")
    
    # Creazione dei bin
    df['block_range'] = (df['block_height'] // bin_size) * bin_size
    
    # Raggruppamento e calcolo metriche
    agg_dict = {
        'fee': 'mean',
        'size': 'mean',
        'coinjoin_score': lambda x: (x > 0.5).mean() * 100, # Percentuale CoinJoin
        'is_rbf': lambda x: x.mean() * 100, # Percentuale RBF
        'inputs_count': 'mean',
        'outputs_count': 'mean'
    }
    
    if 'txid' in df.columns:
        agg_dict['txid'] = 'count'
    else:
        agg_dict['block_height'] = 'count'

    grouped = df.groupby('block_range').agg(agg_dict)
    
    # Rename count column
    if 'txid' in df.columns:
        grouped = grouped.rename(columns={'txid': 'tx_count'})
    else:
        grouped = grouped.rename(columns={'block_height': 'tx_count'})

    grouped.index.name = 'Start Block Height'
    grouped = grouped.reset_index()
    
    # Creazione etichette leggibili per i range
    grouped['range_label'] = grouped['Start Block Height'].apply(
        lambda x: f"{int(x/1000)}k-{int((x+bin_size)/1000)}k"
    )
    
    print("\nRisultati aggregati per range:")
    print(grouped[['range_label', 'tx_count', 'coinjoin_score', 'fee']].to_string())
    
    return grouped

def plot_trends(grouped_df):
    """Genera grafici dei trend temporali."""
    sns.set_theme(style="whitegrid")
    
    metrics = [
        ('tx_count', 'Numero di Transazioni', 'Volume Transazionale per Range di Blocchi'),
        ('coinjoin_score', '% Transazioni CoinJoin', 'Adozione CoinJoin nel Tempo'),
        ('fee', 'Fee Media (satoshi)', 'Trend delle Commissioni Medie'),
        ('size', 'Dimensione Media (byte)', 'Dimensione Media Transazioni'),
        ('is_rbf', '% Transazioni RBF', 'Adozione Replace-By-Fee (RBF)')
    ]
    
    for col, ylabel, title in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='range_label', y=col, data=grouped_df, palette="viridis")
        plt.title(title, fontsize=16)
        plt.xlabel('Range di Blocchi', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = f"{OUTPUT_DIR}/trend_{col}.png"
        plt.savefig(filename, dpi=300)
        print(f"Grafico salvato: {filename}")
        plt.close()

def main():
    # Carica i dati
    df, _ = load_all_data(limit=None)
    
    if df.empty:
        print("Nessun dato trovato.")
        return

    # Analisi
    grouped_stats = analyze_temporal_trends(df, bin_size=50000)
    
    # Plotting
    plot_trends(grouped_stats)
    
    # Export CSV
    csv_path = f"{OUTPUT_DIR}/temporal_stats.csv"
    grouped_stats.to_csv(csv_path, index=False)
    print(f"\nDati tabellari esportati in: {csv_path}")

if __name__ == "__main__":
    main()
