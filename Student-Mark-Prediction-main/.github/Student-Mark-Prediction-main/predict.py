from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from data_processing import load_and_prepare_data, build_student_row_from_simple_inputs

# Delay importing heavy training routine until required (helps on import-time in Streamlit deploy)
try:
    from train_models import train_and_evaluate
except Exception:
    train_and_evaluate = None


BASE_DIR = Path(__file__).resolve().parent


def _load_artifacts():
    artifacts_dir = BASE_DIR / "artifacts"
    # Check required artifact files
    required = [
        artifacts_dir / "best_regression_model.joblib",
        artifacts_dir / "best_classification_model.joblib",
        artifacts_dir / "regression_feature_importance.joblib",
    ]

    missing = any(not p.exists() for p in required)
    if missing:
        print("Model artifacts not found. Attempting to train models to generate artifacts (may take a few minutes)...")
        if train_and_evaluate is None:
            raise FileNotFoundError(
                "Model artifacts are missing and training function is not available to create them."
            )
        train_and_evaluate(BASE_DIR)

    # Load artifacts
    try:
        reg_model = joblib.load(artifacts_dir / "best_regression_model.joblib")
        clf_model = joblib.load(artifacts_dir / "best_classification_model.joblib")
        reg_importance_df = joblib.load(artifacts_dir / "regression_feature_importance.joblib")
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}") from e

    return reg_model, clf_model, reg_importance_df


reg_model, clf_model, reg_importance_df = _load_artifacts()

prepared = load_and_prepare_data(BASE_DIR)
X_columns = prepared["X"].columns
max_absences = max(prepared["df"]["absences"].max(), 1)

actionable_features = {
    "attendance_proxy_pct": "Low attendance / too many absences",
    "absences": "High number of absences",
    "studytime": "Low study time",
    "G1": "Low previous test score (G1)",
    "G2": "Low previous test score (G2)",
    "failures": "Past academic failures",
    "traveltime": "Long travel time",
    "internet": "No home internet access",
    "higher": "Low higher-education aspiration",
    "paid": "No extra paid classes / support",
}

importance_lookup = dict(zip(reg_importance_df["feature"], np.maximum(reg_importance_df["importance_mean"], 0)))


def compute_attendance_proxy_from_absences(absences, max_reference=max_absences):
    value = (1 - (absences / max_reference)) * 100
    return float(np.clip(value, 0, 100))


def get_risk_level(predicted_marks_100):
    if predicted_marks_100 < 50:
        return "High"
    elif predicted_marks_100 < 70:
        return "Medium"
    return "Low"


def feature_weakness_score(feature_name, value):
    if feature_name == "attendance_proxy_pct":
        if value >= 85:
            return 0.0
        elif value >= 75:
            return 0.4
        elif value >= 60:
            return 0.7
        return 1.0

    if feature_name == "absences":
        if value <= 2:
            return 0.0
        elif value <= 5:
            return 0.3
        elif value <= 10:
            return 0.7
        return 1.0

    if feature_name == "studytime":
        if value >= 4:
            return 0.0
        elif value == 3:
            return 0.2
        elif value == 2:
            return 0.6
        return 1.0

    if feature_name in {"G1", "G2"}:
        if value >= 15:
            return 0.0
        elif value >= 12:
            return 0.3
        elif value >= 10:
            return 0.6
        return 1.0

    if feature_name == "failures":
        if value == 0:
            return 0.0
        elif value == 1:
            return 0.5
        return 1.0

    if feature_name == "traveltime":
        if value <= 1:
            return 0.0
        elif value == 2:
            return 0.3
        elif value == 3:
            return 0.6
        return 1.0

    if feature_name == "internet":
        return 1.0 if str(value).lower() == "no" else 0.0

    if feature_name == "higher":
        return 1.0 if str(value).lower() == "no" else 0.0

    if feature_name == "paid":
        return 0.4 if str(value).lower() == "no" else 0.0

    return 0.0


def get_top_reasons_for_low_score(student_row_df, top_n=3):
    student_row = student_row_df.iloc[0].to_dict()

    if "attendance_proxy_pct" not in student_row:
        student_row["attendance_proxy_pct"] = compute_attendance_proxy_from_absences(student_row.get("absences", 0))

    scored_reasons = []
    for feature, readable_reason in actionable_features.items():
        if feature in student_row:
            weakness = feature_weakness_score(feature, student_row[feature])
            importance = importance_lookup.get(feature, 0.01)
            score = weakness * importance
            scored_reasons.append((readable_reason, feature, score, student_row[feature]))

    scored_reasons = sorted(scored_reasons, key=lambda x: x[2], reverse=True)
    meaningful = [item for item in scored_reasons if item[2] > 0]
    if not meaningful:
        return ["No major weak area detected from the available UCI features."]
    return [item[0] for item in meaningful[:top_n]]


def generate_recommendations_from_reasons(reasons):
    recommendation_map = {
        "Low attendance / too many absences": "Try to improve class attendance and reduce missed lectures.",
        "High number of absences": "Track missed classes weekly and revise the topics covered in those classes.",
        "Low study time": "Create a fixed daily study schedule and increase focused study time gradually.",
        "Low previous test score (G1)": "Revise the first-term weak topics and practice more solved examples.",
        "Low previous test score (G2)": "Work on second-term revision and identify where mistakes are repeating.",
        "Past academic failures": "Spend extra time on foundational concepts before moving to advanced topics.",
        "Long travel time": "Use commute time for light revision, flashcards, or audio learning.",
        "No home internet access": "Download notes in advance or use offline learning materials when possible.",
        "Low higher-education aspiration": "Set a clear academic goal because goals often improve consistency.",
        "No extra paid classes / support": "Consider mentorship, peer study groups, or low-cost learning support.",
    }
    recommendations = []
    for reason in reasons:
        if reason in recommendation_map:
            recommendations.append(recommendation_map[reason])
    return recommendations


def predict_student_outcome(student_input_df):
    model_input = student_input_df[X_columns].copy()
    helper_df = model_input.copy()
    helper_df["attendance_proxy_pct"] = helper_df["absences"].apply(compute_attendance_proxy_from_absences)

    predicted_g3 = float(reg_model.predict(model_input)[0])
    predicted_pass = int(clf_model.predict(model_input)[0])
    if hasattr(clf_model, "predict_proba"):
        pass_probability = float(clf_model.predict_proba(model_input)[0, 1])
    else:
        pass_probability = float(predicted_pass)

    predicted_marks_100 = np.clip(predicted_g3 * 5, 0, 100)
    risk_level = get_risk_level(predicted_marks_100)

    weak_reasons = get_top_reasons_for_low_score(helper_df, top_n=3)
    recommendations = generate_recommendations_from_reasons(weak_reasons)

    result = {
        "predicted_final_marks_out_of_20": round(predicted_g3, 2),
        "predicted_final_marks_out_of_100": round(predicted_marks_100, 2),
        "predicted_pass_flag": predicted_pass,
        "pass_probability": round(pass_probability, 4),
        "risk_level": risk_level,
        "top_reasons_for_low_score": weak_reasons,
        "recommendations": recommendations,
    }
    return result


def generate_html_report(student_input_df, result: dict) -> str:
        """Return a simple HTML report for a student prediction result."""
        student = student_input_df.iloc[0].to_dict()
        reasons_html = "".join(f"<li>{r}</li>" for r in result.get("top_reasons_for_low_score", []))
        recs_html = "".join(f"<li>{r}</li>" for r in result.get("recommendations", []))

        html = f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>Student Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 24px; }}
                h1 {{ color: #2b6cb0 }}
                .metrics {{ display:flex; gap:24px }}
                .box {{ border:1px solid #ddd; padding:12px; border-radius:6px }}
            </style>
        </head>
        <body>
            <h1>Student Marks Predictor — Report</h1>
            <p><strong>Predicted marks (out of 100):</strong> {result['predicted_final_marks_out_of_100']}</p>
            <p><strong>Pass probability:</strong> {result['pass_probability']}</p>
            <p><strong>Risk level:</strong> {result['risk_level']}</p>

            <h2>Top reasons for lower score</h2>
            <ul>
                {reasons_html}
            </ul>

            <h2>Recommendations</h2>
            <ul>
                {recs_html}
            </ul>

            <h2>Student input (mapped to model features)</h2>
            <div class="box">
                <pre>{pd.Series(student).to_json(orient='split')}</pre>
            </div>

            <footer style="margin-top:24px;color:#666;font-size:0.9em">Generated by Student Marks Predictor</footer>
        </body>
        </html>
        """
        return html
