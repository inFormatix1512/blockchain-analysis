param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Setup", "Deploy", "Status", "Logs", "Ssh", "Clean")]
    [string]$Command,
    
    [string]$Worker = "ingest_worker_1"
)

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
    
    $Items = @("analysis", "common", "config", "ingest", "ml", "scripts", "docker-compose.yml", "requirements-analysis.txt", "README.md", "manage.ps1")
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
    ssh "$SERVER_USER@$SERVER_HOST" "cd blockchain-analysis; docker-compose ps"
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

# --- MAIN ---
switch ($Command) {
    'Setup'  { Run-Setup }
    'Deploy' { Run-Deploy }
    'Status' { Show-Status }
    'Logs'   { Show-Logs }
    'Ssh'    { Open-Ssh }
    'Clean'  { Run-Clean }
}

