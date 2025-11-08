# Script PowerShell per setup completo del sistema ML

Write-Host "🚀 Setup Sistema ML per Blockchain Analysis" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host ""

# Verifica Python
Write-Host "📋 Verifica Python..."
try {
    $pythonVersion = python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python non trovato!" -ForegroundColor Red
    Write-Host "   Installa Python 3.13+ da python.org" -ForegroundColor Yellow
    exit 1
}

# Crea directory necessarie
Write-Host ""
Write-Host "📁 Creazione directory..."
$directories = @(
    "ml\models",
    "ml\results",
    "analysis\output",
    "analysis\plots"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "✅ Creata: $dir" -ForegroundColor Green
    } else {
        Write-Host "✓ Esiste: $dir" -ForegroundColor Gray
    }
}

# Verifica virtual environment
Write-Host ""
Write-Host "🔧 Verifica Virtual Environment..."
if (Test-Path .venv) {
    Write-Host "✅ Virtual environment trovato" -ForegroundColor Green
    Write-Host "   Attivalo: .venv\Scripts\Activate.ps1"
} else {
    Write-Host "⚠️  Virtual environment non trovato" -ForegroundColor Yellow
    Write-Host "   Creo nuovo virtual environment..."
    python -m venv .venv
    Write-Host "✅ Virtual environment creato" -ForegroundColor Green
    Write-Host "   Attivalo: .venv\Scripts\Activate.ps1"
}

# Installa dipendenze ML
Write-Host ""
Write-Host "📦 Installazione dipendenze ML..."

$requirementsFiles = @(
    @{Path="ml\requirements.txt"; Name="ML"},
    @{Path="requirements-analysis.txt"; Name="Analysis"}
)

foreach ($req in $requirementsFiles) {
    if (Test-Path $req.Path) {
        Write-Host "Installing $($req.Name) dependencies from $($req.Path)..."
        pip install -r $req.Path --quiet
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $($req.Name) dipendenze installate" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Errore installazione $($req.Name)" -ForegroundColor Yellow
        }
    }
}

# Verifica installazione pacchetti critici
Write-Host ""
Write-Host "🔍 Verifica pacchetti critici..."
$criticalPackages = @("pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "psycopg2-binary")

foreach ($pkg in $criticalPackages) {
    try {
        python -c "import $($pkg.Replace('-', '_'))" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $pkg" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $pkg non trovato" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  $pkg - errore verifica" -ForegroundColor Yellow
    }
}

# Test rapido ML
Write-Host ""
Write-Host "🧪 Test configurazione ML..."
$testScript = @"
import sys
try:
    import pandas as pd
    import numpy as np
    import sklearn
    print(f'✅ Pandas {pd.__version__}')
    print(f'✅ NumPy {np.__version__}')
    print(f'✅ Scikit-learn {sklearn.__version__}')
    sys.exit(0)
except Exception as e:
    print(f'❌ Errore: {e}')
    sys.exit(1)
"@

$testScript | python -
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Setup ML completato con successo!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Verifica manuale richiesta" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "✅ Setup ML completato!" -ForegroundColor Green
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Cyan
Write-Host "1. Attiva venv: .venv\Scripts\Activate.ps1"
Write-Host "2. Training: python ml\train_models.py"
Write-Host "3. Analisi: python analysis\analyze_data.py"
Write-Host "4. Jupyter: jupyter notebook analysis\analysis.ipynb"
Write-Host "=" * 50 -ForegroundColor Cyan
