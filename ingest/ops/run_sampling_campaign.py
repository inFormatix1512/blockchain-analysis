#!/usr/bin/env python3
"""
Script di orchestrazione per il campionamento storico della blockchain.
Esegue l'ingestione di 2016 blocchi (1 epoca di difficoltà) per ogni anno dal 2011 al 2023.
"""

import os
import time
import subprocess
import sys

# Configurazione dei range di campionamento (Luglio di ogni anno)
# 2016 blocchi = ~2 settimane
SAMPLING_RANGES = [
    # Anno, Start Block, End Block
    (2011, 134000, 136016),
    (2012, 187000, 189016),
    (2013, 244000, 246016),
    (2014, 308000, 310016),
    (2015, 363000, 365016),
    (2016, 419000, 421016),
    (2017, 474000, 476016),
    (2018, 530000, 532016),
    (2019, 583000, 585016),
    (2020, 637000, 639016),
    (2021, 690000, 692016),
    (2022, 743000, 745016),
    (2023, 797000, 799016)
]

ENV_FILE = '.env'

DEFAULT_SAMPLING_END_YEAR = 2023


def get_sampling_end_year() -> int:
    """Anno massimo da campionare (inclusivo).

    Override via env var SAMPLING_END_YEAR (es. 2020 per fermarsi al 2020).
    """
    value = os.getenv("SAMPLING_END_YEAR", str(DEFAULT_SAMPLING_END_YEAR)).strip()
    try:
        return int(value)
    except ValueError:
        print(
            f"Warning: SAMPLING_END_YEAR='{value}' non valido; uso {DEFAULT_SAMPLING_END_YEAR}"
        )
        return DEFAULT_SAMPLING_END_YEAR

def update_env_file(start_height, end_height):
    """Aggiorna il file .env con i nuovi parametri di ingestione."""
    with open(ENV_FILE, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if line.startswith('INGEST_START_HEIGHT='):
            new_lines.append(f'INGEST_START_HEIGHT={start_height}\n')
        elif line.startswith('INGEST_END_HEIGHT='):
            new_lines.append(f'INGEST_END_HEIGHT={end_height}\n')
        else:
            new_lines.append(line)
            
    with open(ENV_FILE, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated .env: START={start_height}, END={end_height}")

def run_command(cmd):
    """Esegue un comando shell."""
    subprocess.run(cmd, shell=True, check=True)

def get_max_block_in_range(start, end):
    """Recupera l'ultimo blocco processato nel DB all'interno di un range specifico."""
    try:
        # Query specifica per il range: cerca il MAX block_height SOLO tra start ed end
        # Usiamo docker-compose exec per evitare problemi con i nomi dei container
        cmd = f'docker-compose exec -T postgres psql -U postgres -d blockchain -t -c "SELECT COALESCE(MAX(block_height), 0) FROM tx_basic WHERE block_height >= {start} AND block_height <= {end};"'
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        if not result or result == 'None':
            return 0
        return int(result)
    except Exception as e:
        print(f"Error checking DB for range {start}-{end}: {e}")
        return 0

def main():
    sampling_end_year = get_sampling_end_year()
    ranges = [r for r in SAMPLING_RANGES if r[0] <= sampling_end_year]
    if not ranges:
        print(
            f"Nessun range da eseguire: SAMPLING_END_YEAR={sampling_end_year} è prima del primo anno disponibile ({SAMPLING_RANGES[0][0]})."
        )
        return

    print(f"=== AVVIO CAMPIONAMENTO STORICO ({ranges[0][0]}-{ranges[-1][0]}) ===")
    
    for year, start, end in ranges:
        print(f"\n>>> Iniziando campionamento per l'anno {year} (Blocchi {start}-{end})")
        
        # Check if already completed or partially done
        current_max = get_max_block_in_range(start, end)
        if current_max >= end:
            print(f"Anno {year} già completato! Skipping...")
            continue
            
        # Resume from last block if possible
        effective_start = max(start, current_max + 1) if current_max > 0 else start
        print(f"Resuming from block {effective_start}")
        
        # 1. Aggiorna configurazione
        update_env_file(effective_start, end)
        
        # 2. Riavvia ingest container
        print("Riavvio container ingest...")
        try:
            run_command("docker-compose stop ingest")
            run_command("docker-compose rm -f ingest")
            time.sleep(2)
            run_command("docker-compose up -d ingest")
        except Exception as e:
            print(f"Warning: Error restarting ingest: {e}")
            # Try one more time with force remove
            time.sleep(5)
            try:
                run_command("docker-compose rm -f ingest")
                run_command("docker-compose up -d ingest")
            except Exception as e2:
                print(f"Critical Error restarting ingest: {e2}")
        
        # 3. Monitoraggio
        print("Monitoraggio avanzamento...")
        while True:
            # Controlliamo il progresso SPECIFICO per questo range
            current_in_range = get_max_block_in_range(start, end)
            
            # Se non abbiamo ancora iniziato (0), usiamo start come base per il calcolo
            effective_current = max(current_in_range, start) if current_in_range > 0 else start
            
            done = effective_current - start
            total = end - start
            pct = (done / total) * 100
            
            # Usiamo print invece di \r per avere un log leggibile su file
            print(f"Progress Anno {year}: {effective_current}/{end} ({pct:.1f}%)")
            
            if current_in_range >= end:
                print(f"\nAnno {year} completato!")
                break
            
            time.sleep(10)
            
        # Pausa di sicurezza tra i cicli
        time.sleep(5)

    print("\n=== CAMPIONAMENTO COMPLETATO ===")
    print("Ora puoi eseguire l'analisi comparativa completa.")

if __name__ == "__main__":
    main()
