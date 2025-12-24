#!/usr/bin/env python3
"""
ANALISI COMPLETA BLOCKCHAIN - TESI TRIENNALE
Università degli Studi di Catania - Ingegneria Informatica
Autore: Luca Impellizzeri

Script consolidato che combina:
- Analisi esplorativa dati
- Generazione risultati sperimentali
- Report per tesi

Output:
- Statistiche descrittive complete
- Grafici per la tesi (salvati in results/)
- Report tabellare per LaTeX/Word
"""

import os
import sys
import warnings
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0].rsplit('/', 2)[0])

from analysis.src.data_loader import load_all_data
from analysis.src.statistics import (
    print_dataset_overview,
    print_io_distribution,
    print_heuristics_analysis
)
from analysis.src.plots import (
    setup_plotting_style,
    plot_io_distribution,
    plot_heuristics_pie,
    plot_coinjoin_distribution,
    plot_tx_size_distribution,
    plot_blocks_timeline
)
from analysis.src.export import export_tables

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'results/thesis_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Esegue l'analisi completa e genera tutti gli output."""
    
    # Setup stile grafici
    setup_plotting_style()
    
    # Header
    print("\n" + "="*80)
    print(" " * 15 + "ANALISI COMPLETA BLOCKCHAIN - TESI")
    print("="*80)
    print(f"Avviata il: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Caricamento dati (FULL DATASET)
    # Limit=None carica tutto. Rimuoviamo il limite per l'analisi finale.
    df_tx, df_mempool = load_all_data(limit=None)
    
    if df_tx.empty:
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
    
    plot_io_distribution(df_tx, OUTPUT_DIR)
    plot_heuristics_pie(df_tx, OUTPUT_DIR)
    plot_coinjoin_distribution(df_tx, OUTPUT_DIR)
    plot_tx_size_distribution(df_tx, OUTPUT_DIR)
    # plot_blocks_timeline(df_tx, OUTPUT_DIR) # Richiede dati temporali completi, potrebbe fallire con LIMIT
    
    # Esportazione tabelle
    print("\n" + "="*80)
    print("ESPORTAZIONE TABELLE")
    print("="*80)
    
    export_tables(df_tx, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANALISI COMPLETATA CON SUCCESSO")
    print("="*80)
    print(f"\nTutti i file salvati in: {OUTPUT_DIR}/")
    print("\nFile generati:")
    print("  - 4 grafici PNG (300 DPI)")
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
