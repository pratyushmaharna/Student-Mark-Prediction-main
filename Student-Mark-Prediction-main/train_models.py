"""
Train and evaluate subject-specific models using the prepared data.

This module:
- regression: RandomForestRegressor
- classification: RandomForestClassifier

It saves one regression and one classification model per subject
into the `artifacts/` folder.
"""

from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from data_processing import load_and_prepare_data


def train_and_evaluate(base_dir: Path | str = Path(__file__).resolve().parent):
    base_dir = Path(base_dir)
    prepared = load_and_prepare_data(base_dir)
    df = prepared["df"]

    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    subjects = df["subject"].unique()

    for subj in subjects:
        df_subj = df[df["subject"] == subj]

        # Features and targets
        X = df_subj.drop(columns=[c for c in ["G3", "G3_100", "pass_fail"] if c in df_subj.columns])
        y_reg = df_subj["G3"]
        y_clf = df_subj["pass_fail"]

        tree_preprocessor = prepared["tree_preprocessor"]

        # Regression model
        reg_model = Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestRegressor(n_estimators=200, random_state=42))
        ])
        reg_model.fit(X, y_reg)
        joblib.dump(reg_model, artifacts_dir / f"{subj}_regression_model.joblib")

        # Classification model
        clf_model = Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
        clf_model.fit(X, y_clf)
        joblib.dump(clf_model, artifacts_dir / f"{subj}_classification_model.joblib")

        print(f"✅ Models for {subj} saved in artifacts/")

    print("🎉 All subject-specific models trained and saved successfully.")


if __name__ == "__main__":
    train_and_evaluate()
