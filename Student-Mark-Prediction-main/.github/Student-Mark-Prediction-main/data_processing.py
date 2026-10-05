"""
Data loading and preprocessing helpers.

Provides functions to load the student dataset with subject column,
perform feature engineering, build preprocessing pipelines and perform train/test split.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_and_prepare_data(base_dir: Path | str = Path(__file__).resolve().parent):
    base_dir = Path(base_dir)
    mat_path = base_dir / "student-mat-with-subject.csv"

    if not mat_path.exists():
        raise FileNotFoundError(
            f"Expected file student-mat-with-subject.csv in {base_dir}."
        )

    # Load dataset (must include G1, G2, G3)
    df = pd.read_csv(mat_path)

    # Feature engineering
    if "G3" not in df.columns:
        raise ValueError("❌ ERROR: Dataset missing 'G3' column (final grade).")

    df["G3_100"] = df["G3"] * 5
    df["pass_fail"] = (df["G3"] >= 10).astype(int)

    max_absences = max(df["absences"].max(), 1)
    df["attendance_proxy_pct"] = (1 - (df["absences"] / max_absences)) * 100
    df["attendance_proxy_pct"] = df["attendance_proxy_pct"].clip(lower=0, upper=100)

    # Prepare feature matrix and targets
    columns_to_drop_from_features = ["G3", "G3_100", "pass_fail"]
    feature_df = df.drop(columns=[c for c in columns_to_drop_from_features if c in df.columns])

    X = feature_df.copy()
    y_reg = df["G3"].copy()
    y_clf = df["pass_fail"].copy()

    # Explicit feature lists
    categorical_features = [
        "subject", "sex", "school", "address", "famsize", "Pstatus",
        "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup",
        "paid", "activities", "nursery", "higher", "internet", "romantic"
    ]

    numeric_features = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3"
    ]

    # ✅ Filter out any columns missing in the dataset
    categorical_features = [f for f in categorical_features if f in X.columns]
    numeric_features = [f for f in numeric_features if f in X.columns]

    # Preprocessing pipelines
    numeric_linear_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")),
               ("scaler", StandardScaler())]
    )
    numeric_tree_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    linear_preprocessor = ColumnTransformer(
        transformers=[("num", numeric_linear_pipeline, numeric_features),
                      ("cat", categorical_pipeline, categorical_features)]
    )
    tree_preprocessor = ColumnTransformer(
        transformers=[("num", numeric_tree_pipeline, numeric_features),
                      ("cat", categorical_pipeline, categorical_features)]
    )

    # Train/test split
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Default template
    def build_default_student_template(reference_df):
        template = {}
        for col in reference_df.columns:
            if pd.api.types.is_numeric_dtype(reference_df[col]):
                template[col] = float(reference_df[col].median())
            else:
                template[col] = reference_df[col].mode().iloc[0]
        return template

    default_student_template = build_default_student_template(X)

    return {
        "df": df,
        "X": X,
        "y_reg": y_reg,
        "y_clf": y_clf,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "linear_preprocessor": linear_preprocessor,
        "tree_preprocessor": tree_preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "X_train_clf": X_train_clf,
        "X_test_clf": X_test_clf,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "default_student_template": default_student_template,
    }
def map_study_hours_per_day_to_uci_studytime(hours_per_day):
    weekly_hours = hours_per_day * 7
    if weekly_hours < 2:
        return 1
    elif weekly_hours <= 5:
        return 2
    elif weekly_hours <= 10:
        return 3
    else:
        return 4


def map_attendance_pct_to_absences(attendance_pct, max_reasonable_absences=30):
    attendance_pct = float(np.clip(attendance_pct, 0, 100))
    absences = round((100 - attendance_pct) / 100 * max_reasonable_absences)
    return int(absences)


def map_previous_test_score_100_to_grade_20(score_100):
    score_100 = float(np.clip(score_100, 0, 100))
    return int(round(score_100 / 5))


def map_internet_usage_to_uci_internet(internet_usage):
    internet_usage = str(internet_usage).strip().lower()
    if internet_usage in {"none", "no", "not available"}:
        return "no"
    return "yes"


def build_student_row_from_simple_inputs(reference_template, study_hours_per_day=4, attendance_pct=82,
                                         previous_test_score=65, assignment_submitted="Yes",
                                         sleep_hours=6, internet_usage="Medium", subject="OS"):
    row = reference_template.copy()
    # Accept subject directly (OS, DIP, SE, FAI, FLAT)
    row["subject"] = subject
    row["studytime"] = map_study_hours_per_day_to_uci_studytime(study_hours_per_day)
    row["absences"] = map_attendance_pct_to_absences(attendance_pct)
    row["G1"] = map_previous_test_score_100_to_grade_20(previous_test_score)
    row["G2"] = map_previous_test_score_100_to_grade_20(previous_test_score)
    row["internet"] = map_internet_usage_to_uci_internet(internet_usage)

    # Keep extra inputs for reference
    row["_input_assignment_submitted"] = assignment_submitted
    row["_input_sleep_hours"] = sleep_hours
    row["_input_attendance_pct"] = attendance_pct
    row["_input_study_hours_per_day"] = study_hours_per_day
    row["_input_previous_test_score_100"] = previous_test_score
    row["_input_internet_usage"] = internet_usage

    return pd.DataFrame([row])
