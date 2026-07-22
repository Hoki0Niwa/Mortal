# Mortal self-play worker (for the main PC). client.py retries on its own when
# the server is unreachable, so no restart loop is needed here.
# Run one window per worker:
#   powershell -ExecutionPolicy Bypass -File start_worker.ps1
#   powershell -ExecutionPolicy Bypass -File start_worker.ps1 -TrainPlayProfile second
param(
    [string]$TrainPlayProfile = 'default',
    [string]$PythonExe = 'python',
    [string]$ConfigPath = (Join-Path $PSScriptRoot 'config.toml')
)

$py = $PythonExe
$dir = $PSScriptRoot
$cfg = (Resolve-Path -LiteralPath $ConfigPath).Path

$host.UI.RawUI.WindowTitle = "mortal-worker-$TrainPlayProfile"
$env:TRAIN_PLAY_PROFILE = $TrainPlayProfile
$env:MORTAL_CFG = $cfg
Set-Location $dir
& $py "$dir\client.py"
