# Run the Streamlit app using the project's preferred Python environment if available.
# Usage: .\run_app.ps1

$preferredPython = "C:/Users/ontha/Python/envs/diabetic_readmission_env/python.exe"

if (Test-Path $preferredPython) {
    Write-Host "Using preferred Python: $preferredPython"
    & $preferredPython -m streamlit run .\app.py
} else {
    Write-Host "Preferred Python not found. Using active environment's streamlit (make sure streamlit is installed)."
    streamlit run .\app.py
}
