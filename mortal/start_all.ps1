# Launch the resident Mortal stack (server + trainer + client) on this PC.
# Each process gets its own window with an auto-restart loop for 24/7 operation.
# Run:  powershell -ExecutionPolicy Bypass -File start_all.ps1

$py  = 'C:\Users\user\miniconda3\envs\mortal\python.exe'
$dir = 'C:\Users\user\git\Mortal\mortal'

function Start-Loop {
    param([string]$Title, [string]$Script, [string]$Pre = '')
    $cmd = "`$host.UI.RawUI.WindowTitle = '$Title'; $Pre while (`$true) { & '$py' '$Script'; Write-Host '[$Title] exited, restarting in 5s...'; Start-Sleep 5 }"
    Start-Process powershell -WorkingDirectory $dir -ArgumentList '-NoExit', '-Command', $cmd
}

Start-Loop -Title 'mortal-server'  -Script "$dir\server.py"
Start-Sleep 5
# cap trainer thread pools so its parsing bursts steal fewer cores from the client
Start-Loop -Title 'mortal-trainer' -Script "$dir\train.py" -Pre "`$env:RAYON_NUM_THREADS = '2'; `$env:OMP_NUM_THREADS = '2';"
Start-Sleep 5
Start-Loop -Title 'mortal-client'  -Script "$dir\client.py"

Write-Host 'started: mortal-server, mortal-trainer, mortal-client'
