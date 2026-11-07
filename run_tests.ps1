# Run the test suite (no GPU, network or real dataset needed).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
& .\.venv\Scripts\python.exe -m pytest @args
exit $LASTEXITCODE
