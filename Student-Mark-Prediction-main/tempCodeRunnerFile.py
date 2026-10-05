import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
import traceback
from data_processing import load_and_prepare_data, build_student_row_from_simple_inputs
from train_models import train_and_evaluate
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Student Marks Predictor", layout="wide", page_icon=":mortar_board:")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent

def ensure_dirs():
    for d in ("data", "models", "reports"):
        p = BASE_DIR / d
        p.mkdir(parents=True, exist_ok=True)

@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found at {path}")
    return pd.read_csv(path)

@st.cache_data
def get_prepared_data():
    return load_and_prepare_data(BASE_DIR)

def main():
    ensure_dirs()
    st.sidebar.title("Dataset & Settings")
    # Only keep student-mat.csv, remove Portuguese
    dataset_choice = st.sidebar.selectbox("Select dataset", ("student-mat.csv",))
    st.sidebar.markdown("Place CSVs in the project root. Re-run if files change.")

    df_mat = None
    try:
        df_mat = load_csv("student-mat.csv")
    except Exception:
        df_mat = None

    prepared = None
    try:
        prepared = get_prepared_data()
    except Exception as e:
        logger.exception("Data preparation failed")

    tabs = st.tabs(["Data", "Train & Models", "Predict", "Visualize", "Reports"])

    # Data tab
    with tabs[0]:
        st.header("Data Overview")
        if dataset_choice == "student-mat.csv" and df_mat is not None:
            st.subheader("student-mat.csv — Preview")
            st.dataframe(df_mat.head(10))
            st.write("Shape:", df_mat.shape)

        st.markdown("---")
        if prepared is not None:
            st.subheader("Prepared data overview")
            st.write("Combined dataframe shape:", prepared["df"].shape)
            st.write("Feature matrix X shape:", prepared["X"].shape)
            st.write("Regression target (y_reg) size:", prepared["y_reg"].shape)
            st.write("Classification target (y_clf) size:", prepared["y_clf"].shape)

            with st.expander("Numeric features"):
                st.write(prepared["numeric_features"])
            with st.expander("Categorical features"):
                st.write(prepared["categorical_features"])

            st.subheader("Default student template (median/mode)")
            st.write(prepared["default_student_template"])    

        st.markdown("---")
        if df_mat is not None:
            with st.expander("student-mat summary"):
                st.write(df_mat.describe(include='all'))

        st.info("If CSV file is missing, put student-mat.csv in the project root.")

    # Train & Models tab
    with tabs[1]:
        st.header("Train & Evaluate Models")
        if st.button("Train & Evaluate Models"):
            with st.spinner("Training models (this may take a while)..."):
                try:
                    results = train_and_evaluate(BASE_DIR)
                    st.success("Training complete. Models saved to artifacts/.")

                    st.subheader("Regression Results")
                    st.dataframe(results["regression_results_df"])

                    st.subheader("Classification Results")
                    st.dataframe(results["classification_results_df"])

                    st.subheader("Top regression feature importances")
                    st.dataframe(results["reg_importance_df"].head(15))

                    st.write("Artifacts folder:", str(results["artifacts_dir"]))
                except Exception as e:
                    logger.exception("Training failed")
                    st.error(f"Training failed: {e}")
        else:
            st.write("Click the button to (re)train models and update artifacts.")

    # Predict tab
    with tabs[2]:
        st.header("Student Prediction Demo")
        if prepared is None:
            st.error("Data not prepared — ensure CSV is present and preprocessing ran.")
        else:
            with st.form(key="student_input_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=4.0)
                    attendance = st.slider("Attendance %", 0, 100, 82)
                with col2:
                    prev_score = st.number_input("Previous test score (0-100)", min_value=0, max_value=100, value=65)
                    assignment = st.selectbox("Assignment submitted?", ("Yes", "No"))
                with col3:
                    sleep_hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=6.0)
                    internet_usage = st.selectbox("Internet access", ("Yes", "No"))

                # Replace subjects with OS, DIP, SE, FAI, FLAT
                subject = st.selectbox("Subject", ("OS", "DIP", "SE", "FAI", "FLAT"))
                submit = st.form_submit_button("Predict")

            if submit:
                try:
                    template = prepared["default_student_template"]
                    student_df = build_student_row_from_simple_inputs(
                        reference_template=template,
                        study_hours_per_day=study_hours,
                        attendance_pct=attendance,
                        previous_test_score=prev_score,
                        assignment_submitted=assignment,
                        sleep_hours=sleep_hours,
                        internet_usage=internet_usage,
                        subject=subject,
                    )

                    import predict as predict_mod
                    result = predict_mod.predict_student_outcome(student_df)

                    st.metric("Predicted Marks (out of 100)", result["predicted_final_marks_out_of_100"])
                    st.metric("Pass Probability", f"{result['pass_probability'] * 100:.1f}%")
                    st.write("Risk level:", result["risk_level"]) 

                    st.subheader("Top Reasons for Lower Score")
                    for i, r in enumerate(result["top_reasons_for_low_score"], start=1):
                        st.write(f"{i}. {r}")

                    st.subheader("Recommendations")
                    for i, rec in enumerate(result["recommendations"], start=1):
                        st.write(f"{i}. {rec}")

                    report_text = "\n".join([
                        "Student Marks Predictor Report",
                        "===============================",
                        f"Predicted marks (out of 100): {result['predicted_final_marks_out_of_100']}",
                        f"Predicted pass probability: {result['pass_probability']}",
                        f"Risk level: {result['risk_level']}",
                        "\nTop reasons:",
                    ] + [f"- {r}" for r in result["top_reasons_for_low_score"]] + ["\nRecommendations:"] + [f"- {rec}" for rec in result["recommendations"]])

                    st.download_button("Download report (txt)", report_text, file_name="student_report.txt")
                    try:
                        html_report = predict_mod.generate_html_report(student_df, result)
                        st.download_button("Download report (HTML)", html_report, file_name="student_report.html", mime="text/html")
                    except Exception:
                        pass

                except Exception as e:
                    logger.exception("Prediction failed")
                    st.error(f"Prediction failed: {e}")

    # Visualize tab
    with tabs[3]:
        st.header("Visualizations")
        if prepared is None:
            st.error("Data not prepared — ensure CSV is present and preprocessing ran.")
        else:
            df_all = prepared["df"]

            with st.expander("Show visualizations options"):
                show_corr = st.checkbox("Correlation heatmap", value=True)
                show_distributions = st.checkbox("Distributions (G3_100)", value=True)
                show_feature_importance = st.checkbox("Feature importance (regression)", value=True)
                show_scatter = st.checkbox("Custom scatter plot", value=True)

            @st.cache_data
            def compute_corr(df):
                return df.select_dtypes(include=["number"]).corr()

            if show_corr:
                st.subheader("Correlation heatmap (numeric features)")
                corr = compute_corr(df_all)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0)
                st.pyplot(fig)

            if show_distributions:
                st.subheader("Distribution of Final Marks (G3_100)")
                fig, ax = plt.subplots()
                sns.histplot(df_all["G3_100"], bins=20, kde=True, ax=ax)
                ax.set_xlabel("Final Marks (0-100)")
                st.pyplot(fig)

            if show_feature_importance:
                st.subheader("Top regression feature importances")
                try:
                    reg_imp = joblib.load(BASE_DIR / "artifacts" / "regression_feature_importance.joblib")
                    top_imp = reg_imp.head(15).sort_values(by="importance_mean", ascending=True)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(top_imp["feature"], top_imp["importance_mean"])
                    ax.set_xlabel("Permutation importance")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not load feature importances: {e}")

            if show_scatter:
                st.subheader("Custom scatter plot")
                numeric_cols = df_all.select_dtypes(include=["number"]).columns.tolist()
                x_col = st.selectbox("X axis", numeric_cols, index=numeric_cols.index("G1") if "G1" in numeric_cols else 0)
                y_col = st.selectbox("Y axis", numeric_cols, index=numeric_cols.index("G3_100") if "G3_100" in numeric_cols else 0)
                color_by = st.selectbox("Color by (categorical)", (None,) + tuple(prepared["categorical_features"]))

                fig, ax = plt.subplots()
                if color_by and color_by in df_all.columns:
                    sns.scatterplot(data=df_all, x=x_col, y=y_col, hue=color_by, ax=ax)
                else:
                    sns.scatterplot(data=df_all, x=x_col, y=y_col, ax=ax)
                st.pyplot(fig)

    # Reports tab
    with tabs[4]:
        st.header("Reports & Artifacts")
        try:
            artifacts = list((BASE_DIR / "artifacts").iterdir())
            st.write("Saved artifacts:")
            for p in artifacts:
                st.write('-', p.name)
        except Exception:
            st.write("No artifacts found. Train models to generate artifacts.")

if __name__ == "__main__":
    main()
