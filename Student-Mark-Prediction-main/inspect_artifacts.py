import joblib
import pandas as pd

def main():
    base = '.'
    try:
        reg_df = joblib.load(base + '/artifacts/regression_feature_importance.joblib')
        print('--- Top regression feature importances ---')
        print(reg_df.head(15).to_string(index=False))
    except Exception as e:
        print('Could not load regression_feature_importance.joblib:', e)

    try:
        # We don't save the metrics DataFrames; load models exist
        from pathlib import Path
        p = Path(base) / 'artifacts' / 'best_regression_model.joblib'
        print('\nSaved regression model exists at:', p)
    except Exception as e:
        print('Error checking artifacts folder:', e)

if __name__ == '__main__':
    main()
