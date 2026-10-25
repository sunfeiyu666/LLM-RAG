param([string]$PythonExe = '')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) {
    $pythonCandidate = Join-Path (Split-Path -Parent $projectRoot) 'LLM-RAG-local\.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $pythonCandidate) { $PythonExe = $pythonCandidate } else { $PythonExe = 'python' }
}
Push-Location -LiteralPath $projectRoot
try {
    foreach ($taskScript in @('tools/build_reviewed_facts.py', 'tools/build_correctness_review.py', 'tools/build_failure_reviews.py', 'tools/summarize_t7.py', 'tools/verify_t7.py')) {
        & $PythonExe -X utf8 $taskScript
        if ($LASTEXITCODE -ne 0) { throw "T7 reproduction failed: $taskScript" }
    }
    Write-Host 'T7 reports, figures and verification completed.'
} finally { Pop-Location }
