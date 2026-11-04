# Build a source-only bundle using an explicit allowlist; no credentials, data or venv.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Bundle = Join-Path $ProjectRoot 'outputs\deploy\voiceguard-source.tar.gz'
New-Item -ItemType Directory -Force (Split-Path -Parent $Bundle) | Out-Null
Push-Location $ProjectRoot
try {
    & tar -czf $Bundle --exclude=__pycache__ --exclude=*.pyc `
        vguard scripts tools frontend deploy tests third_party docs platform README.md requirements.txt .dockerignore .gitignore
    if ($LASTEXITCODE -ne 0) { throw 'Packaging failed' }
    Get-FileHash -LiteralPath $Bundle -Algorithm SHA256 | Format-List
    Write-Output $Bundle
} finally { Pop-Location }
