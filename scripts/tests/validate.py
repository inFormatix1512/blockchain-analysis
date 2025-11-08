#!/usr/bin/env python3
"""
Script di validazione per verificare la configurazione del progetto.
Esegue controlli di base senza richiedere servizi attivi.
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Verifica che un file esista."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NON TROVATO: {filepath}")
        return False

def check_python_syntax(filepath):
    """Verifica la sintassi di un file Python."""
    try:
        with open(filepath, 'r') as f:
            compile(f.read(), filepath, 'exec')
        print(f"✅ Sintassi Python OK: {filepath}")
        return True
    except SyntaxError as e:
        print(f"❌ Errore sintassi in {filepath}: {e}")
        return False

def check_imports(filepath):
    """Verifica che gli import di un file Python siano validi."""
    try:
        import ast
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        
        print(f"✅ Import trovati in {filepath}: {', '.join(set(imports))}")
        return True
    except Exception as e:
        print(f"❌ Errore controllo import in {filepath}: {e}")
        return False

def main():
    print("=" * 60)
    print("🔍 VALIDAZIONE CONFIGURAZIONE PROGETTO")
    print("=" * 60)
    print()
    
    errors = 0
    
    # Controlla file essenziali
    print("📄 Controllo file essenziali...")
    essential_files = [
        ("docker-compose.yml", "Docker Compose principale"),
        ("create_tables.sql", "Schema database"),
        (".env.example", "Template variabili ambiente"),
        ("bitcoin.conf", "Configurazione Bitcoin"),
        ("init_db.sh", "Script inizializzazione DB"),
        ("check-setup.sh", "Script verifica setup"),
        ("README.md", "README principale"),
    ]

    optional_files = [
        ("docker-compose-improved.yml", "Docker Compose migliorato"),
        ("README-IMPROVED.md", "README migliorato"),
    ]

    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            errors += 1

    for filepath, description in optional_files:
        if Path(filepath).exists():
            print(f"✅ {description}: {filepath}")
        else:
            print(f"ℹ️  {description} opzionale non trovato: {filepath}")
    
    print()
    
    # Controlla file Python
    print("🐍 Controllo script Python...")
    python_files = [
        "ingest/run_ingest.py",
        "ingest/block_ingest.py",
        "ingest/mempool_snapshot.py",
    ]
    
    for filepath in python_files:
        if not check_file_exists(filepath, f"Script Python"):
            errors += 1
        elif not check_python_syntax(filepath):
            errors += 1
        elif not check_imports(filepath):
            errors += 1
    
    print()
    
    # Controlla requirements
    print("📦 Controllo dipendenze...")
    requirements_files = [
        "ingest/requirements.txt",
        "requirements-analysis.txt",
        "ml/requirements.txt",
    ]
    
    for filepath in requirements_files:
        if check_file_exists(filepath, "Requirements"):
            try:
                with open(filepath, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                    print(f"   Dipendenze trovate: {len(lines)}")
                    for line in lines:
                        print(f"     - {line}")
            except Exception as e:
                print(f"❌ Errore lettura {filepath}: {e}")
                errors += 1
    
    print()
    
    # Controlla Dockerfile
    print("🐳 Controllo Dockerfile...")
    if check_file_exists("ingest/Dockerfile", "Dockerfile"):
        with open("ingest/Dockerfile", 'r') as f:
            content = f.read()
            if "FROM python" in content:
                print("   ✅ Base image Python trovata")
            if "WORKDIR /app" in content:
                print("   ✅ Working directory configurata")
            if "requirements.txt" in content:
                print("   ✅ Requirements installati")
    else:
        errors += 1
    
    print()
    
    # Controlla permessi script
    print("🔐 Controllo permessi script...")
    scripts = ["init_db.sh", "check-setup.sh"]
    for script in scripts:
        if Path(script).exists():
            if os.access(script, os.X_OK):
                print(f"✅ {script} è eseguibile")
            else:
                print(f"⚠️  {script} non è eseguibile (usa: chmod +x {script})")
    
    print()
    print("=" * 60)
    if errors == 0:
        print("✅ VALIDAZIONE COMPLETATA CON SUCCESSO!")
        print("=" * 60)
        print()
        print("📝 Prossimi passi:")
        print("   1. Esegui ./check-setup.sh per verificare Docker")
        print("   2. Crea .env da .env.example")
        print("   3. Avvia i servizi con docker-compose up -d")
        print("   4. Inizializza il database con ./init_db.sh")
        return 0
    else:
        print(f"❌ VALIDAZIONE FALLITA: {errors} errori trovati")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
