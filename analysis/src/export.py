import pandas as pd
import os

def export_tables(df_tx, output_dir):
    """Esporta tabelle in formato CSV per LaTeX/Word."""
    print("\nEsportazione tabelle...")
    
    # Tabella 1: Top 15 I/O patterns
    io_dist = df_tx.groupby(['inputs_count', 'outputs_count']).size().reset_index(name='count')
    io_dist['percentage'] = (io_dist['count'] / len(df_tx) * 100).round(2)
    io_dist = io_dist.sort_values('count', ascending=False).head(15)
    io_dist.columns = ['Input', 'Output', 'Transazioni', 'Percentuale (%)']
    
    filepath1 = os.path.join(output_dir, 'table1_io_patterns.csv')
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
    
    filepath2 = os.path.join(output_dir, 'table2_heuristics_stats.csv')
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
    
    filepath3 = os.path.join(output_dir, 'table3_general_stats.csv')
    general_stats.to_csv(filepath3, index=False, sep=';')
    print(f"[OK] Salvato: {filepath3}")
