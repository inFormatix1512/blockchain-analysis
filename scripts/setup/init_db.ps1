# Script PowerShell per inizializzare il database PostgreSQL

param(
    [string]$Host = "localhost",
    [int]$Port = 5432,
    [string]$User = "postgres",
    [string]$Database = "blockchain"
)

Write-Host "🗄️  Inizializzazione Database PostgreSQL" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host ""

# Verifica che il container postgres sia in esecuzione
Write-Host "📋 Verifica container PostgreSQL..."
$pgContainer = docker compose ps --filter "service=postgres" --format json | ConvertFrom-Json

if (-not $pgContainer) {
    Write-Host "❌ Container PostgreSQL non trovato!" -ForegroundColor Red
    Write-Host "   Avvia prima i servizi: docker compose up -d" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Container PostgreSQL attivo" -ForegroundColor Green
Write-Host ""

# Verifica schema SQL
$sqlFile = "config\create_tables.sql"
if (-not (Test-Path $sqlFile)) {
    Write-Host "❌ File $sqlFile non trovato!" -ForegroundColor Red
    exit 1
}

Write-Host "📄 File SQL: $sqlFile"
Write-Host ""

# Esegui lo script SQL
Write-Host "🔧 Esecuzione script SQL..."
Write-Host ""

try {
    # Copia il file nel container e eseguilo
    docker compose exec -T postgres psql -U $User -d $Database -f /create_tables.sql
    
    Write-Host ""
    Write-Host "✅ Database inizializzato con successo!" -ForegroundColor Green
    Write-Host ""
    
    # Verifica tabelle create
    Write-Host "📊 Verifica tabelle create..."
    $tables = docker compose exec -T postgres psql -U $User -d $Database -c "\dt" 2>$null
    
    if ($tables) {
        Write-Host $tables
    }
    
} catch {
    Write-Host "❌ Errore durante l'inizializzazione: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "✅ Inizializzazione completata!" -ForegroundColor Green
Write-Host ""
Write-Host "Prossimi passi:" -ForegroundColor Cyan
Write-Host "1. Verifica: docker compose exec postgres psql -U postgres -d blockchain -c 'SELECT COUNT(*) FROM tx_basic;'"
Write-Host "2. Testa: python scripts\tests\test_system.py"
Write-Host "=" * 50 -ForegroundColor Cyan
