param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("validate", "build-input-dataset", "push-input-dataset", "update-input-dataset", "build-kernel", "push-kernel", "status", "logs", "download")]
    [string]$Action,
    [string]$Message = "Update production input"
)

$ErrorActionPreference = "Stop"
$KaggleRoot = $PSScriptRoot
$ProductionRoot = Split-Path -Parent $KaggleRoot
$WorkspaceRoot = $ProductionRoot
$Python = Join-Path $WorkspaceRoot ".venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = Join-Path (Split-Path -Parent $WorkspaceRoot) ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
$Settings = Get-Content -LiteralPath (Join-Path $KaggleRoot "settings.json") -Raw | ConvertFrom-Json

switch ($Action) {
    "update-input-dataset" {
        & $Python (Join-Path $KaggleRoot "build.py") build-input-dataset
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m kaggle datasets version -p (Join-Path $KaggleRoot "build\input-dataset") -m $Message -q
    }
    "validate" { & $Python (Join-Path $KaggleRoot "build.py") validate }
    "build-input-dataset" { & $Python (Join-Path $KaggleRoot "build.py") build-input-dataset }
    "push-input-dataset" {
        & $Python (Join-Path $KaggleRoot "build.py") build-input-dataset
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m kaggle datasets create -p (Join-Path $KaggleRoot "build\input-dataset") -q
    }
    "build-kernel" { & $Python (Join-Path $KaggleRoot "build.py") build-kernel }
    "push-kernel" {
        & $Python (Join-Path $KaggleRoot "build.py") build-kernel
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m kaggle kernels push -p (Join-Path $KaggleRoot "build\kernel") --accelerator $Settings.accelerator
    }
    "status" { & $Python -m kaggle kernels status "$($Settings.username)/$($Settings.kernel_slug)" }
    "logs" { & $Python -m kaggle kernels logs "$($Settings.username)/$($Settings.kernel_slug)" }
    "download" {
        $Download = Join-Path $KaggleRoot "downloads"
        New-Item -ItemType Directory -Force -Path $Download | Out-Null
        & $Python -m kaggle kernels output "$($Settings.username)/$($Settings.kernel_slug)" -p $Download -o
    }
}

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
