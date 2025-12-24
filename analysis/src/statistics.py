import pandas as pd

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
