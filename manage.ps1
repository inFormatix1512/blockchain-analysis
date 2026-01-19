param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "Setup", "Deploy", "Status", "Logs", "Ssh", "Clean",
        "Start", "Stop", "Restart",
        "Recover", "Sample",
        "AnalyzeYearly", "CompareEras", "Thesis", "Temporal"
    )]
    [string]$Command,

    [string]$Worker = "ingest_worker_1"
)

# Assicura esecuzione dal root del progetto
Set-Location -Path $PSScriptRoot

# --- CONFIGURAZIONE ---
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match "^([^#=]+)=(.*)") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            Set-Variable -Name $name -Value $value -Scope Script -ErrorAction SilentlyContinue
        }
    }
}

if (-not $SERVER_USER) { $SERVER_USER = "user" }
if (-not $SERVER_HOST) { $SERVER_HOST = "example.com" }
$RemotePath = "~/blockchain-analysis"

# --- FUNZIONI ---

function Run-Setup {
    Write-Host "Setup Ambiente Locale" -ForegroundColor Cyan
    
    if (-not (Test-Path .env)) {
        if (Test-Path .env.example) {
            Copy-Item .env.example .env
            Write-Host "OK: Creato .env da .env.example" -ForegroundColor Green
        } else {
            New-Item .env -ItemType File | Out-Null
            Write-Host "Warning: Creato .env vuoto" -ForegroundColor Yellow
        }
    }

    $envContent = ""
    if (Test-Path .env) { $envContent = Get-Content .env -Raw }
    
    if ($envContent -notmatch "SERVER_HOST") {
        Write-Host "Warning: Configurazione server mancante nel .env" -ForegroundColor Yellow
        $hostInput = Read-Host "Inserisci l'host del server"
        $userInput = Read-Host "Inserisci l'utente SSH"
        
        if ($hostInput -and $userInput) {
            Add-Content .env "`nSERVER_HOST=$hostInput"
            Add-Content .env "SERVER_USER=$userInput"
            Write-Host "OK: Configurazione salvata" -ForegroundColor Green
        }
    }

    $dirs = @("bitcoin-data", "pgdata", "analysis/results", "ml/models")
    foreach ($d in $dirs) {
        if (-not (Test-Path $d)) {
            New-Item -ItemType Directory -Path $d -Force | Out-Null
        }
    }
    Write-Host "Setup completato!" -ForegroundColor Green
}

function Run-Deploy {
    Write-Host "Avvio Deploy su $SERVER_HOST..." -ForegroundColor Cyan
    ssh "$SERVER_USER@$SERVER_HOST" "mkdir -p $RemotePath"
    
    $Items = @(
        "analysis",
        "common",
        "config",
        "ingest",
        "ml",
        "docker-compose.yml",
        "requirements-analysis.txt",
        "README.md",
        "recover_system.py",
        "manage.ps1",
        ".env.example"
    )
    foreach ($Item in $Items) {
        if (Test-Path $Item) {
            Write-Host "Copia di $Item..."
            $dest = "$SERVER_USER@$SERVER_HOST" + ":" + "$RemotePath/"
            scp -r "$Item" "$dest"
        }
    }
    Write-Host "Deploy completato!" -ForegroundColor Green
}

function Show-Status {
    Write-Host "Stato Server $SERVER_HOST" -ForegroundColor Cyan
    ssh "$SERVER_USER@$SERVER_HOST" "cd blockchain-analysis; docker compose ps"
}

function Show-Logs {
    Write-Host "Log di $Worker su $SERVER_HOST" -ForegroundColor Cyan
    ssh "$SERVER_USER@$SERVER_HOST" "docker logs --tail 50 -f $Worker"
}

function Open-Ssh {
    ssh "$SERVER_USER@$SERVER_HOST"
}

function Run-Clean {
    Get-ChildItem -Recurse -Include "__pycache__", "*.pyc" | Remove-Item -Force -Recurse
    Write-Host "Pulizia completata" -ForegroundColor Green
}

function Start-Stack {
    Write-Host "Avvio stack Docker..." -ForegroundColor Cyan
    docker compose up -d
}

function Stop-Stack {
    Write-Host "Stop stack Docker..." -ForegroundColor Yellow
    docker compose down
}

function Restart-Stack {
    Write-Host "Restart stack Docker..." -ForegroundColor Yellow
    docker compose down
    docker compose up -d
}

function Run-Recovery {
    Write-Host "Avvio recovery system..." -ForegroundColor Cyan
    python recover_system.py
}

function Run-Sampling {
    Write-Host "Avvio campionamento storico..." -ForegroundColor Cyan
    python ingest/ops/run_sampling_campaign.py
}

function Run-AnalyzeYearly {
    Write-Host "Avvio analisi ML annuale..." -ForegroundColor Cyan
    python analysis/scripts/analyze_ml_yearly.py
}

function Run-CompareEras {
    Write-Host "Avvio confronto per ere..." -ForegroundColor Cyan
    python analysis/scripts/compare_eras.py
}

function Run-Thesis {
    Write-Host "Avvio analisi completa tesi..." -ForegroundColor Cyan
    python analysis/scripts/complete_thesis_analysis.py
}

function Run-Temporal {
    Write-Host "Avvio analisi temporale..." -ForegroundColor Cyan
    python analysis/scripts/temporal_analysis.py
}

# --- MAIN ---
switch ($Command) {
    'Setup'  { Run-Setup }
    'Deploy' { Run-Deploy }
    'Status' { Show-Status }
    'Logs'   { Show-Logs }
    'Ssh'    { Open-Ssh }
    'Clean'  { Run-Clean }
    'Start'  { Start-Stack }
    'Stop'   { Stop-Stack }
    'Restart'{ Restart-Stack }
    'Recover'{ Run-Recovery }
    'Sample' { Run-Sampling }
    'AnalyzeYearly' { Run-AnalyzeYearly }
    'CompareEras' { Run-CompareEras }
    'Thesis' { Run-Thesis }
    'Temporal' { Run-Temporal }
}

