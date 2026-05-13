<#
Infra helper: infra_up.ps1

Automates LocalStack S3 bucket creation, DB migrations, and starting the full Docker Compose stack.

Run from repository root in PowerShell (requires Docker, AWS CLI, and docker-compose):
  .\venv\Scripts\Activate.ps1  # optional for environment context
  .\scripts\infra_up.ps1

# Edit .env before running. This script will NOT commit secrets.
#>

Param()

function Assert-CommandExists {
    param([string]$cmd)
    $null = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($?) { return $true }
    Write-Error "$cmd is not available in PATH. Install it and retry."
    exit 1
}

# Basic checks
Assert-CommandExists -cmd docker
Assert-CommandExists -cmd aws

if (-not (Test-Path -Path .env)) {
    Write-Error ".env not found. Copy .env.example -> .env and populate secrets before running."
    exit 1
}

Write-Host "Starting LocalStack (S3) for bucket provisioning..."
docker compose up -d localstack

Write-Host "Waiting for LocalStack to be ready..."
$max = 60; $i=0; while ($i -lt $max) {
    try {
        $r = Invoke-RestMethod -Uri http://localhost:4566/_localstack/health -TimeoutSec 2
        if ($r.services -and $r.services.s3 -eq 'running') { Write-Host 'LocalStack S3 OK'; break }
    } catch { }
    Start-Sleep -Seconds 2; $i++
}
if ($i -ge $max) { Write-Warning 'LocalStack did not become ready in time; proceed anyway or check logs.' }

Write-Host "Create S3 bucket deepcoin-reports in LocalStack (if not exists)..."
try {
    aws --endpoint-url http://localhost:4566 s3 ls s3://deepcoin-reports 2>$null
    if ($LASTEXITCODE -ne 0) {
        aws --endpoint-url http://localhost:4566 s3 mb s3://deepcoin-reports
        Write-Host 'Created S3 bucket deepcoin-reports'
    } else { Write-Host 'S3 bucket already exists' }
} catch {
    Write-Warning 'aws CLI not able to create bucket; ensure AWS CLI v2 configured and endpoint is reachable.'
}

Write-Host "Running DB migrations (migrator)..."
docker compose run --rm migrator

Write-Host "Starting full stack (this may take a few minutes)..."
docker compose up --build -d

Write-Host "Waiting for API health endpoint to return OK..."
$healthUrl = 'http://127.0.0.1:8000/api/health'
$max=60; $i=0; while ($i -lt $max) {
    try {
        $r = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
        if ($r.status -eq 'healthy') { Write-Host 'API healthy'; break }
    } catch { }
    Start-Sleep -Seconds 3; $i++
}
if ($i -ge $max) { Write-Warning 'API health did not become healthy in time; check api logs (docker compose logs -f api).' }

Write-Host "Open Grafana and Prometheus in your browser (if available):"
Start-Process 'http://localhost:3001'
Start-Process 'http://localhost:9090'

Write-Host "Done. Tail logs with: docker compose logs -f api prometheus grafana nginx"
