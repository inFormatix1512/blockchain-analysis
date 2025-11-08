# Script PowerShell per verificare la configurazione prima di avviare i servizi

Write-Host "🔍 Verifica configurazione progetto..." -ForegroundColor Cyan
Write-Host ""

# Controlla se .env esiste
if (-not (Test-Path .env)) {
    Write-Host "⚠️  File .env non trovato!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Creazione .env da .env.example..."
    
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Host "✅ File .env creato" -ForegroundColor Green
        Write-Host ""
        Write-Host "⚠️  IMPORTANTE: Modifica .env con password sicure prima di procedere!" -ForegroundColor Yellow
        Write-Host "   notepad .env"
        Write-Host ""
    } else {
        Write-Host "❌ Errore: .env.example non trovato!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "✅ File .env trovato" -ForegroundColor Green
}

# Controlla docker-compose.yml
Write-Host "📋 Verifica docker-compose.yml..."
if (Test-Path docker-compose.yml) {
    Write-Host "✅ docker-compose.yml trovato" -ForegroundColor Green
} else {
    Write-Host "❌ Errore: docker-compose.yml non trovato!" -ForegroundColor Red
    exit 1
}

# Controlla create_tables.sql
Write-Host "📋 Verifica schema database..."
if (Test-Path config\create_tables.sql) {
    Write-Host "✅ config\create_tables.sql trovato" -ForegroundColor Green
} else {
    Write-Host "❌ Errore: config\create_tables.sql non trovato!" -ForegroundColor Red
    exit 1
}

# Verifica Docker
Write-Host ""
Write-Host "🐳 Verifica Docker..."
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker non trovato! Installa Docker Desktop." -ForegroundColor Red
    exit 1
}

# Verifica Docker Compose
try {
    $composeVersion = docker compose version
    Write-Host "✅ Docker Compose: $composeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose non disponibile!" -ForegroundColor Red
    exit 1
}

# Verifica Python
Write-Host ""
Write-Host "🐍 Verifica Python..."
try {
    $pythonVersion = python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Python non trovato nel PATH" -ForegroundColor Yellow
}

# Verifica requirements
Write-Host ""
Write-Host "📦 Verifica file requirements..."
$reqFiles = @(
    "requirements-analysis.txt",
    "ingest\requirements.txt",
    "ml\requirements.txt"
)

foreach ($req in $reqFiles) {
    if (Test-Path $req) {
        Write-Host "✅ $req" -ForegroundColor Green
    } else {
        Write-Host "⚠️  $req non trovato" -ForegroundColor Yellow
    }
}

# Riepilogo
Write-Host ""
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "✅ Verifica completata!" -ForegroundColor Green
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Cyan
Write-Host "1. Se necessario, modifica .env con credenziali sicure"
Write-Host "2. Avvia i servizi: docker compose up -d"
Write-Host "3. Inizializza DB: docker compose exec postgres psql -U postgres -d blockchain -f /create_tables.sql"
Write-Host "4. Testa: python scripts\tests\quick_test.py"
Write-Host "=" * 50 -ForegroundColor Cyan
