# Mortal self-play worker (for the main PC). client.py retries on its own when
# the server is unreachable, so no restart loop is needed here.
# Run one window per worker:
#   powershell -ExecutionPolicy Bypass -File start_worker.ps1
#   powershell -ExecutionPolicy Bypass -File start_worker.ps1 -TrainPlayProfile second
param(
    [string]$TrainPlayProfile = 'default'
)

# Adjust these two paths on the main PC.
$py  = 'C:\Users\user\miniconda3\envs\mortal\python.exe'
$dir = 'C:\Users\user\git\Mortal\mortal'

$host.UI.RawUI.WindowTitle = "mortal-worker-$TrainPlayProfile"
$env:TRAIN_PLAY_PROFILE = $TrainPlayProfile
Set-Location $dir
& $py "$dir\client.py"
