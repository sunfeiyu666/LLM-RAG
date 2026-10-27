$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw 'Docker was not found. Install Docker Desktop first.' }
    & docker info --format '{{.ServerVersion}}'
    if ($LASTEXITCODE -ne 0) { throw 'Docker engine is unavailable. Start Docker Desktop with Linux containers, then retry.' }
    & docker compose version
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose is unavailable.' }
    Write-Host 'Building and starting RAG. The first build downloads Python packages.'
    & docker compose -f compose.yaml up -d --build --wait --wait-timeout 240
    if ($LASTEXITCODE -ne 0) {
        & docker compose -f compose.yaml logs --tail 60
        throw 'Startup failed. Check the messages above and Docker/README.md.'
    }
    $config = (& docker compose -f compose.yaml config --format json | Out-String | ConvertFrom-Json)
    $port = $config.services.'rag-web'.ports[0].published
    $url = "http://localhost:$port"
    Write-Host "Ready: $url"
    Start-Process $url
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
