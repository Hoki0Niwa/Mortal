# Launch the resident Mortal stack (server + trainer + client) on this PC.
# Each process gets its own window with an auto-restart loop for 24/7 operation.
# Run from the mortal/ directory with the conda env activated:
#   conda activate mortal
#   powershell -ExecutionPolicy Bypass -File start_all.ps1
#   powershell -ExecutionPolicy Bypass -File start_all.ps1 -TrainerRayonThreads 2 -TrainerOmpThreads 2
param(
    [int]$TrainerRayonThreads = 0,
    [int]$TrainerOmpThreads = 0
)

$py  = 'python'             # resolves to the activated conda env's python
$dir = (Get-Location).Path  # current directory (run this from mortal/)

function Start-Loop {
    param([string]$Title, [string]$Script, [string]$Pre = '')
    $cmd = "`$host.UI.RawUI.WindowTitle = '$Title'; $Pre while (`$true) { & '$py' '$Script'; Write-Host '[$Title] exited, restarting in 5s...'; Start-Sleep 5 }"
    Start-Process powershell -WorkingDirectory $dir -ArgumentList '-NoExit', '-Command', $cmd
}

Start-Loop -Title 'mortal-server'  -Script "$dir\server.py"
Start-Sleep 5
$trainerPre = ''
if ($TrainerRayonThreads -gt 0) {
    $trainerPre += "`$env:RAYON_NUM_THREADS = '$TrainerRayonThreads'; "
} else {
    $trainerPre += "Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue; "
}
if ($TrainerOmpThreads -gt 0) {
    $trainerPre += "`$env:OMP_NUM_THREADS = '$TrainerOmpThreads'; "
} else {
    $trainerPre += "Remove-Item Env:OMP_NUM_THREADS -ErrorAction SilentlyContinue; "
}

Start-Loop -Title 'mortal-trainer' -Script "$dir\train.py" -Pre $trainerPre
Start-Sleep 5
Start-Loop -Title 'mortal-client'  -Script "$dir\client.py"

Write-Host 'started: mortal-server, mortal-trainer, mortal-client'
