# 后台推送重试器：网络抖动时自动重试 git push，直到成功。
# 用法：pwsh -File tools/git_push_retry.ps1
$ErrorActionPreference = 'Continue'
Set-Location 'D:\Voice Guard'
$env:GIT_TERMINAL_PROMPT = '0'
for ($i = 1; $i -le 12; $i++) {
    Write-Output "push attempt $i $(Get-Date -Format HH:mm:ss)"
    git push 2>&1 | Select-Object -Last 2
    if ($LASTEXITCODE -eq 0) { Write-Output "PUSH OK"; exit 0 }
    Start-Sleep -Seconds 180
}
Write-Output "PUSH FAILED after 12 attempts"
exit 1
