# Stop all Mortal worker clients on this machine (safe at any time: a killed
# worker only loses its current unfinished session).
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'client\.py' } |
    ForEach-Object {
        Write-Host "stopping pid $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force
    }
