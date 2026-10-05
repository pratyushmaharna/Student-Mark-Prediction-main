# Student Marks Predictor + Weak Area Detector

This repository converts a Jupyter notebook into a Streamlit web application that:

- Loads and combines the UCI Student Performance datasets (`student-mat.csv`, `student-por.csv`).
- Performs preprocessing and feature engineering.
- Trains regression and classification models and saves artifacts.
- Predicts a student's expected final marks, pass probability, and detects weak areas with recommendations.
- Provides interactive visualizations and an exportable student report.

## Files

- `streamlit_app.py` — Main Streamlit application.
- `data_processing.py` — Data loading and preprocessing utilities.
- `train_models.py` — Training and evaluation logic (saves artifacts to `artifacts/`).
- `predict.py` — Loads saved models and exposes `predict_student_outcome`.
- `requirements.txt` — Python dependencies.
- `student-mat.csv`, `student-por.csv` — UCI datasets (include these in the project root).

## Quick start (local)

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place `student-mat.csv` and `student-por.csv` in the project root (they are included here).

4. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

5. Optional: train models from the command line (the Streamlit UI also exposes a button):

```bash
python train_models.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. On Streamlit Cloud, create a new app and point it to this GitHub repo and branch.
3. Set the main file to `streamlit_app.py` and (if needed) add a requirement to install the packages from `requirements.txt`.

Notes:
- The app expects the CSV files to be present in the repo root. To make the deployment fully automated, include the CSVs in the repository or modify the app to download them at runtime.
- Artifacts (trained models and feature importances) are saved in the `artifacts/` folder.

## Next steps

- Improve report export (PDF/HTML), add authentication, and integrate persistent storage for reports.
- Add CI to run basic linting and tests.
