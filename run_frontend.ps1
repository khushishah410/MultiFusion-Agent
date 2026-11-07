# Start the Streamlit UI on http://127.0.0.1:8501 (backend must be running).
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
& .\.venv\Scripts\python.exe -m streamlit run app\frontend\frontend_ui.py --server.address 127.0.0.1 --server.port 8501
