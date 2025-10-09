# Diabetes Readmission Prediction

Quick instructions to run the Streamlit app for the Diabetes Readmission Prediction project.

Prerequisites
- Python 3.8+ and a virtual environment (recommended).
- Install dependencies (see `requirements.txt`).

Run the app (recommended)
- From PowerShell in the project root:
	- `.\\run_app.ps1` — uses the Python executable at `C:/Users/ontha/Python/envs/diabetic_readmission_env/python.exe` if present, otherwise calls `streamlit run` in your active environment.

Run directly with Streamlit CLI
- Activate your environment and run:
	- `streamlit run .\app.py`

Run from VS Code (F5)
- A `tasks.json` and `launch.json` are included in `.vscode/` to run the Streamlit CLI when you press F5.

Notes
- The repository currently contains a minimal `app.py` (UI only). If you train and save a model (for example as a pickle/joblib file under `Results/Model/`), the app can be extended to load it and run predictions.
- If you add or update dependencies, run `pip install -r requirements.txt` in your environment.

