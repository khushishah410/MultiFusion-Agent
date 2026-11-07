# Start the FastAPI backend on http://127.0.0.1:8000 (docs at /docs).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
& .\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload
